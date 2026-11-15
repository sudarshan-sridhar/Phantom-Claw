"""The request pipeline, shared by the server and the command line demo.

Order (contract): request, Merlin (only when no agent is named), plan from the agent's LLM,
Merlin's check, speech, say, laptop tools, arm skill. Agent states throughout, all idle at the end.
The arm is the body of whoever is speaking: a line with no arm skill gets a `gesture` in the
speaker's style for the length of the clip.
"""
from __future__ import annotations

import asyncio
import importlib
import os
import random
import sys
import threading
import time
from datetime import datetime
from typing import Any, Awaitable, Callable

from . import cast as cast_mod
from . import merlin, providers, voice
from .cast import AGENT_IDS, CAST

Emit = Callable[[dict], Awaitable[None]]


def clock_now() -> datetime:
    """Laptop clock, or PHANTOM_CLOCK=HH:MM for testing Escanor's moods."""
    fake = os.environ.get("PHANTOM_CLOCK", "").strip()
    now = datetime.now()
    if fake:
        try:
            h, m = fake.split(":")
            return now.replace(hour=int(h), minute=int(m))
        except ValueError:
            pass
    return now


def warm_claude() -> None:
    """The anthropic package takes a few seconds to import the first time. Do that in the
    background at start so the first routing call isn't the one that pays for it."""
    if "anthropic" in sys.modules:
        return

    def _go():
        try:
            importlib.import_module("anthropic")
        except ImportError:
            pass

    threading.Thread(target=_go, name="warm-claude", daemon=True).start()


class _Run:
    """Per-request state: the chain of gestures so each waits for the previous one."""

    def __init__(self):
        self.chain: asyncio.Task | None = None
        self.last_gesture: dict | None = None


class Pipeline:
    def __init__(
        self,
        arm: Any,
        emit: Emit,
        *,
        speak: bool = True,
        use_arm: bool = True,
        run_tools: bool = True,
        pace_audio: bool = True,
        rng: random.Random | None = None,
        now_fn: Callable[[], datetime] = clock_now,
        models: dict | None = None,
    ):
        self.arm = arm
        self.emit = emit
        self.speak_enabled = speak
        self.use_arm = use_arm
        self.run_tools = run_tools
        self.pace_audio = pace_audio
        self.rng = rng or random.Random()
        self.now_fn = now_fn
        self.models = models
        self.states: dict[str, str] = {cid: "idle" for cid in CAST}
        self._left_out = False   # the last arm result left the arm out of rest (stopped or error)
        # The last look at the desk, kept a few seconds so one request doesn't look twice.
        self._scene: dict | None = None
        try:
            self.scene_ttl = float(os.environ.get("PHANTOM_SCENE_TTL", "6"))
        except ValueError:
            self.scene_ttl = 6.0
        self._stop_req = False   # set by request_stop(); a desk plan checks it between steps
        warm_claude()

    def request_stop(self) -> None:
        """The stop button: the arm holds where it is (the server calls arm.stop) and a desk plan
        runs no further steps."""
        self._stop_req = True

    # ------------------------------------------------------------ helpers

    async def set_state(self, agent: str, state: str) -> None:
        self.states[agent] = state
        await self.emit({"type": "agent_state", "agent": agent, "state": state})

    async def error(self, where: str, message: str) -> None:
        await self.emit({"type": "error", "where": where, "message": message[:300]})

    async def speak(self, agent: str, text: str) -> dict:
        """Make the clip. Returns {"audio_url", "duration_s"}; on failure audio_url is None."""
        if not self.speak_enabled:
            return {"audio_url": None, "duration_s": voice.estimate_seconds(text)}
        try:
            r = await voice.tts(agent, text)
            return {"audio_url": r["audio_url"], "duration_s": r["duration_s"]}
        except Exception as e:  # noqa: BLE001 - speech must never stop the pipeline
            await self.error("tts", f"{agent}: {type(e).__name__}: {e}")
            return {"audio_url": None, "duration_s": voice.estimate_seconds(text)}

    def _arm_busy(self) -> bool:
        try:
            return bool(self.arm.status().get("busy"))
        except Exception:  # noqa: BLE001
            return False

    def _needs_rest(self) -> bool:
        """True when the arm was left out of rest by a stop or an error. The arm's own at_rest
        wins when it reports one; otherwise the last result decides."""
        try:
            st = self.arm.status()
        except Exception:  # noqa: BLE001
            st = {}
        if st.get("held"):
            return False      # carrying something it picked up: the arm itself decides what may run
        at_rest = st.get("at_rest")
        if at_rest is not None:
            return not at_rest
        return self._left_out

    def held_name(self) -> str | None:
        """The item the arm is carrying (status()["held"]), or None."""
        try:
            held = self.arm.status().get("held") if self.arm is not None else None
        except Exception:  # noqa: BLE001
            return None
        if isinstance(held, dict):
            return held.get("name") or "item"
        return held or None

    async def _run_skill(self, agent: str, skill: str, params: dict, auto: bool = False,
                         extra: dict | None = None) -> dict:
        loop = asyncio.get_running_loop()

        def on_phase(phase: str) -> None:
            ev = {"type": "arm", "agent": agent, "skill": skill, "phase": phase, **(extra or {})}
            if auto:
                ev["auto"] = True
            loop.call_soon_threadsafe(lambda: asyncio.ensure_future(self.emit(ev)))

        try:
            res = await asyncio.to_thread(self.arm.run, skill, dict(params), on_phase)
        except Exception as e:  # noqa: BLE001
            await self.error("arm", f"{skill}: {type(e).__name__}: {e}")
            res = {"ok": False, "result": "error", "detail": str(e)[:200]}
        await asyncio.sleep(0)  # let the last phase events go out first
        if res.get("result") in ("stopped", "error"):
            self._left_out = True
        elif skill == "rest" and res.get("ok"):
            self._left_out = False
        elif res.get("ok"):
            self._left_out = False   # every other skill that finished ends at rest
        return {**res, "skill": skill}

    async def run_arm(self, agent: str, skill: str, params: dict, extra: dict | None = None) -> dict:
        """Run one arm skill in a worker thread, sending `arm` events for each phase.

        After a stop or an error the arm is left where it stopped. Before the next skill it is
        sent to rest first (events carry "auto": true, the result carries "auto_rest"). A
        gesture is not worth the wait, so it is skipped after an automatic rest."""
        if not self.use_arm or self.arm is None:
            return {"ok": False, "result": "skipped", "detail": "arm not used", "skill": skill}
        if self._arm_busy():
            return {"ok": False, "result": "busy", "detail": "arm busy", "skill": skill}
        auto = None
        if skill != "rest" and self._needs_rest():
            await self.emit({"type": "arm", "agent": agent, "skill": "rest", "phase": "auto_rest", "auto": True})
            auto = await self._run_skill(agent, "rest", {}, auto=True)
            if not auto.get("ok"):
                await self.error("arm", f"could not go back to rest before {skill}: {auto.get('detail')}")
                return {"ok": False, "result": "error", "detail": f"could not go back to rest first: {auto.get('detail')}",
                        "skill": skill, "auto_rest": auto}
            if skill == "gesture":
                return {"ok": False, "result": "skipped", "detail": "went back to rest instead of this gesture",
                        "skill": skill, "auto_rest": auto}
        res = await self._run_skill(agent, skill, params, extra=extra)
        if auto is not None:
            res["auto_rest"] = auto
        return res

    # ------------------------------------------------------------ the desk: looking and plans of steps

    def scene_fresh(self) -> bool:
        return self._scene is not None and time.monotonic() - self._scene["t"] <= self.scene_ttl

    def forget_scene(self) -> None:
        self._scene = None

    def _keep_scene(self, res: dict) -> None:
        if res.get("result") == "done" and isinstance(res.get("scene"), list):
            self._scene = {"t": time.monotonic(), "res": {"ok": True, "result": "done", "scene": res["scene"],
                                                         "person": res.get("person"), "detail": res.get("detail", "")}}

    async def look_scene(self, agent: str | None = None, force: bool = False) -> dict:
        """What the overhead camera sees: {"ok", "result", "scene", "person", "ms", "cached"}.

        result is "done", "no_picture", "not_found" (Gemini failed), "no_arm" (no arm, or one
        that can't look) or an arm error. A fresh look from the last few seconds is reused."""
        if not force and self.scene_fresh():
            return {**self._scene["res"], "ms": 0, "cached": True}
        if not self.use_arm or self.arm is None:
            return {"ok": False, "result": "no_arm", "scene": None, "ms": 0, "cached": False}
        t0 = time.monotonic()
        while True:
            try:
                res = await asyncio.to_thread(self.arm.run, "look", {}, lambda _p: None)
            except Exception as e:  # noqa: BLE001
                res = {"ok": False, "result": "error", "detail": f"{type(e).__name__}: {e}"[:200]}
            if res.get("result") != "busy" or time.monotonic() - t0 > 6:
                break
            await asyncio.sleep(0.2)     # a gesture is finishing
        ms = int((time.monotonic() - t0) * 1000)
        result = res.get("result")
        if result == "error" and "unknown" in str(res.get("detail", "")):
            result = "no_arm"            # an older arm without the look skill: plan without a scene
        self._keep_scene(res)
        out = {"ok": result == "done" and isinstance(res.get("scene"), list), "result": result,
               "scene": res.get("scene"), "person": res.get("person"), "detail": str(res.get("detail", ""))[:200],
               "ms": ms, "cached": False}
        items = [{k: it.get(k) for k in ("name", "x_cm", "y_cm", "dist_cm", "tier", "graspable")}
                 for it in (out["scene"] or [])]
        await self.emit({"type": "scene", "agent": agent, "result": result, "items": items, "ms": ms})
        return out

    def _item_dist(self, name: str | None, decision: dict | None) -> float | None:
        scenes = []
        if self._scene:
            scenes.append(self._scene["res"].get("scene") or [])
        if decision and decision.get("scene"):
            scenes.append(decision["scene"])
        for sc in scenes:
            for it in sc:
                if it.get("name") == name and it.get("dist_cm") is not None:
                    return float(it["dist_cm"])
        return None

    async def run_plan(self, agent: str, steps: list[dict], decision: dict | None = None) -> dict:
        """Run a desk plan step by step and react to what comes back.

        pick missed: one more try after a fresh look, then give up. not_found, out_of_reach,
        not_taught, not_graspable, no_picture: stop and say why. hand_over not_taken: put it back.
        Any other failure while holding: put it back, then rest. stopped: nothing more moves.
        Returns {"ok", "result", "skill": "plan", "steps", "item", "say"}; say is the line that
        tells what really happened (may be empty)."""
        n = len(steps)
        manner = cast_mod.ARM_MANNER.get(agent, cast_mod.ARM_MANNER["gowther"])
        await self.emit({"type": "arm_plan", "agent": agent, "phase": "start", "steps": steps})
        done: list[dict] = []
        outcome, item, dist = None, None, None
        retried = False

        async def step(i: int, skill: str, params: dict, **flags) -> dict:
            info = {"step": i, "steps": n, **flags}
            if skill == "look" and self.scene_fresh() and not flags.get("retry"):
                res = {"ok": True, "result": "done", "detail": "used the look from a moment ago", "cached": True}
            else:
                res = await self.run_arm(agent, skill, params, extra=info)
                if skill == "look":
                    self._keep_scene(res)
                elif skill in ("pick", "place", "hand_over", "rest"):
                    self.forget_scene()          # things have moved
            rec = {"step": i, "skill": skill, "params": params, "result": res.get("result"),
                   "ok": bool(res.get("ok")), "detail": str(res.get("detail", ""))[:200], **flags}
            if res.get("cached"):
                rec["cached"] = True
            if res.get("auto_rest"):
                rec["auto_rest"] = True
            done.append(rec)
            await self.emit({"type": "arm_step", "agent": agent, **rec})
            return res

        async def put_back(i: int) -> None:
            if self._stop_req or not self.held_name():
                return
            back = await step(i, "place", {"where": "back", "pace": manner["pace"], "approach": manner["approach"]},
                              recovery=True)
            if not back.get("ok") and back.get("result") != "stopped" and self.held_name():
                await step(i, "rest", {}, recovery=True)

        i = 0
        while i < n:
            if self._stop_req:
                outcome = "stopped"
                break
            skill, params = steps[i]["skill"], dict(steps[i]["params"])
            tgt = params.get("target") if isinstance(params.get("target"), str) else None
            if skill in ("pick", "point_at") and tgt:
                item, dist = tgt, self._item_dist(tgt, decision)
            res = await step(i, skill, params)
            r = res.get("result")
            if skill == "pick" and res.get("held"):
                item = res["held"].get("name") or item
                dist = res["held"].get("dist_cm", dist)
            if res.get("ok"):
                i += 1
                continue
            if skill == "pick" and r == "missed" and not retried and not self._stop_req:
                retried = True
                self.forget_scene()
                await step(i, "look", {"target": tgt} if tgt else {}, retry=True)
                continue                         # the same pick again
            if r == "stopped":
                outcome = "stopped"
                break
            if skill == "hand_over" and r == "not_taken":
                await put_back(i)
                outcome = "not_taken"
                break
            refusal = r in ("missed", "not_found", "out_of_reach", "not_taught", "not_graspable", "no_picture")
            outcome = r if refusal else "failed"
            if self.held_name():
                await put_back(i)
                if not refusal:
                    outcome = "put_back"
            break
        else:
            last = next((d for d in reversed(done) if d["ok"] and d["skill"] != "look"), None)
            outcome = {"hand_over": "handed_over", "place": "placed", "point_at": "pointed"}.get(
                last["skill"] if last else "", "done")

        ok = outcome in ("handed_over", "placed", "pointed", "done")
        if outcome == "no_picture":
            line = cast_mod.NO_PICTURE_LINES.get(agent, "")
        else:
            line = cast_mod.arm_follow(agent, outcome, item, dist) if outcome != "done" else ""
        line = merlin.clean_follow(agent, line) if line else ""
        await self.emit({"type": "arm_plan", "agent": agent, "phase": "done", "result": outcome, "ok": ok,
                         "steps": done})
        detail = "; ".join(f"{d['skill']}: {d['result']}" for d in done)
        return {"ok": ok, "result": outcome, "detail": detail, "skill": "plan", "steps": done, "item": item,
                "say": line}

    async def _gesture(self, agent: str, seconds: float, set_states: bool) -> dict:
        seconds = round(min(10.0, max(1.5, float(seconds))), 2)
        if set_states:
            await self.set_state(agent, "speaking")
        t0 = time.monotonic()
        res = await self.run_arm(agent, "gesture", {"style": agent, "seconds": seconds})
        if res.get("result") != "done" and self.pace_audio:
            # Gesture skipped (busy or no arm): still wait for the line so the order holds.
            left = seconds - (time.monotonic() - t0)
            if left > 0:
                await asyncio.sleep(left)
        if set_states:
            await self.set_state(agent, "idle")
        return res

    def queue_gesture(self, run: _Run, agent: str, seconds: float, set_states: bool = False) -> asyncio.Task:
        """Start a gesture after any earlier one in this request has finished."""
        prev = run.chain

        async def _go():
            if prev is not None:
                try:
                    await prev
                except Exception:  # noqa: BLE001
                    pass
            res = await self._gesture(agent, seconds, set_states)
            run.last_gesture = res
            return res

        run.chain = asyncio.ensure_future(_go())
        return run.chain

    async def wait_gestures(self, run: _Run) -> None:
        if run.chain is not None:
            try:
                await run.chain
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------ the pipeline

    def resolve_named(self, text: str, agent: str | None) -> str | None:
        if agent:
            a = agent.strip().lower()
            if a in AGENT_IDS:
                return a
            found = merlin.find_named_agent(a)
            if found in AGENT_IDS:
                return found
        # The first of the five named in the text. "Merlin, ask Ban to..." still means Ban.
        for found in merlin.names_in(text):
            if found in AGENT_IDS:
                return found
        return None

    @staticmethod
    def addressed_merlin(text: str, agent: str | None) -> bool:
        """True when the owner spoke to Merlin herself (she's named first, or picked in the overlay)."""
        if agent and agent.strip().lower() == "merlin":
            return True
        names = merlin.names_in(text)
        return bool(names) and names[0] == "merlin"

    async def make_plan(self, chosen: str, text: str, group: str | None = None, lead: str | None = None,
                        look: dict | None = None) -> dict:
        """LLM plan plus Merlin's check. Returns plan, issues, replaced, provider info, decision.

        group is "lead" or "helper" when everyone was asked; a helper only chips in a line.
        look is what the camera saw (look_scene) for a desk job or a desk question."""
        decision = cast_mod.decide(chosen, text, self.now_fn())
        if group:
            decision["group"] = group
            decision["lead"] = lead or chosen
        if decision.get("desk") and group != "helper":
            decision["held"] = self.held_name()
            if look is not None and look.get("result") == "no_picture":
                decision["no_picture"] = True
            elif look is not None and look.get("ok"):
                decision["scene"] = look.get("scene")
                decision["person"] = look.get("person")
            if decision.get("no_picture"):
                # Said plainly, nothing moves, no LLM needed.
                plan = {"agent": chosen, "say": cast_mod.NO_PICTURE_LINES[chosen], "arm": None, "laptop": []}
                cast_mod.remember(chosen, text, "done")
                return {"plan": plan, "issues": ["no picture from the overhead camera"], "replaced": False,
                        "provider": {}, "decision": decision}
        info: dict = {}
        try:
            raw, info = await providers.get_json(
                CAST[chosen]["provider"],
                cast_mod.system_prompt(chosen),
                cast_mod.user_prompt(chosen, text, decision),
                agent=chosen,
                models=self.models,
            )
            if info.get("fallback"):
                await self.error("provider", f"{chosen}: fell through to {info['provider']}: " + "; ".join(info["errors"]))
        except providers.ProviderError as e:
            await self.error("provider", f"{chosen}: {e}")
            raw = None
        plan, issues, replaced = merlin.check_plan(raw, chosen, decision)
        if replaced:
            await self.error("check", f"{chosen}: plan replaced: " + "; ".join(issues))
        if group != "helper":
            cast_mod.remember(chosen, text, "done" if replaced else cast_mod.outcome_of(decision))
        return {"plan": plan, "issues": issues, "replaced": replaced, "provider": info, "decision": decision}

    # ------------------------------------------------------------ tools and the line that carries the result

    async def run_one_tool(self, call: dict, agent: str) -> dict:
        from backend import tools

        try:
            return await tools.run_tool(call, agent, self.emit)
        except Exception as e:  # noqa: BLE001 - a broken tool must not stop the request
            await self.error("tool", f"{call.get('tool')}: {type(e).__name__}: {e}")
            return {"ok": False, "tool": call.get("tool"), "detail": str(e)[:200]}

    @staticmethod
    def follow_line(agent: str, call: dict, r: dict) -> str | None:
        """The second line for one tool result, or None when there's nothing to add.

        Answers from lookup, read_screen and clipboard are already in the character's voice (the
        tool asked for that), so they only get tidied. Simple facts use a template, no LLM."""
        t = call.get("tool")
        if r.get("dry") and not r.get("answer"):
            return None
        if r.get("refused"):
            return merlin.clean_follow(agent, cast_mod.follow_template(agent, "refused"))
        if r.get("not_found"):
            return merlin.clean_follow(agent, cast_mod.follow_template(agent, "not_found", r["not_found"]))
        if t in ("lookup", "read_screen", "clipboard"):
            if r.get("empty"):
                return merlin.clean_follow(agent, cast_mod.follow_template(agent, "empty"))
            if r.get("answer"):
                return merlin.clean_follow(agent, r["answer"])
            return merlin.clean_follow(agent, cast_mod.follow_template(agent, "failed"))
        if t == "tell":
            if r.get("ok") and r.get("value"):
                return merlin.clean_follow(agent, cast_mod.follow_template(agent, r["what"], r["value"]))
            return merlin.clean_follow(agent, cast_mod.follow_template(agent, "failed"))
        if t == "screenshot" and r.get("ok") and r.get("where"):
            return merlin.clean_follow(agent, cast_mod.follow_template(agent, "saved", r["where"]))
        if not r.get("ok") and t != "timer":
            return merlin.clean_follow(agent, cast_mod.follow_template(agent, "failed"))
        return None

    def follow_ups(self, agent: str, plan: dict, results: list[dict]) -> list[str]:
        from backend import tools

        lines: list[str] = []
        # If the main thing opened, a missing extra isn't worth a line.
        opened = any(r.get("ok") for c, r in zip(plan["laptop"], results) if c["tool"] in tools.OPENERS)
        for call, r in zip(plan["laptop"], results):
            if opened and call["tool"] in tools.OPENERS and not r.get("ok"):
                continue
            line = self.follow_line(agent, call, r)
            if line and line not in lines:
                lines.append(line)
        return lines[:2]

    async def say_line(self, run: _Run, agent: str, text: str, gesture: bool) -> dict:
        """Speak one more line after the earlier ones in this request have played."""
        clip = await self.speak(agent, text)
        await self.wait_gestures(run)
        await self.set_state(agent, "speaking")
        await self.emit({"type": "say", "agent": agent, "text": text, "audio_url": clip["audio_url"],
                         "duration_s": clip["duration_s"]})
        if gesture:
            self.queue_gesture(run, agent, clip["duration_s"])
        return {"text": text, **clip}

    # ------------------------------------------------------------ Merlin answering herself

    async def ask_merlin(self, text: str) -> tuple[str, str | None, bool]:
        """Merlin's own answer when spoken to. Returns (line, hand_to, ok); ok is False when her
        LLM didn't answer in time, and then the normal routing takes over."""
        try:
            raw, _info = await asyncio.wait_for(
                providers.get_json(CAST["merlin"]["provider"], merlin.MERLIN_SYSTEM,
                                   f'The owner said to you: "{text}"', agent="merlin", models=self.models),
                merlin.MERLIN_TIMEOUT_S,
            )
        except (providers.ProviderError, asyncio.TimeoutError) as e:
            await self.error("provider", f"merlin: {type(e).__name__}: {str(e)[:150]}")
            return "", None, False
        line, hand_to = merlin.check_merlin_reply(raw)
        return line, hand_to, isinstance(raw, dict)

    async def merlin_only(self, run: _Run, line: str) -> dict:
        clip = await self.speak("merlin", line)
        out = {"type": "merlin", "chosen": None, "say": line, "audio_url": clip["audio_url"],
               "duration_s": clip["duration_s"], "argued": []}
        await self.emit(out)
        for cid in AGENT_IDS:
            await self.set_state(cid, "idle")
        self.queue_gesture(run, "merlin", clip["duration_s"], set_states=True)
        await self.wait_gestures(run)
        plan = {"agent": "merlin", "say": line, "arm": None, "laptop": []}
        return {"agent": "merlin", "say": line, "plan": plan, "audio_url": clip["audio_url"],
                "duration_s": clip["duration_s"], "arm_result": run.last_gesture,
                "merlin": {k: v for k, v in out.items() if k != "type"}, "provider": None,
                "check": {"issues": [], "replaced": False}, "tools": [], "followups": [], "group": []}

    # ------------------------------------------------------------ the pipeline

    async def pick_by_kind(self, text: str) -> tuple[str, str, str]:
        """Who takes an unnamed job. Returns (agent, why, source), source "claude" or "rules".

        A request asked again goes back to whoever dodged it, and the bottle goes to Ban; both are
        fixed rules. Everything else is a small question to Claude, checked against the cast. If
        Claude is slow, down or names nobody valid, the fixed rules decide, so this never blocks."""
        again = cast_mod.pending_repeat_agent(text)
        if again:
            return again, "asked again", "rules"
        if cast_mod.is_bottle_job(text):
            return "ban", "the bottle", "rules"
        desk = cast_mod.desk_kind(text)
        if desk == "question":
            return "gowther", "a question about the desk", "rules"
        # A desk job's look takes about 3 s anyway, so Claude gets a little longer to answer there.
        cap = merlin.ROUTE_TIMEOUT_S + (merlin.DESK_ROUTE_EXTRA_S if desk == "job" else 0.0)
        try:
            raw, _info = await asyncio.wait_for(
                providers.get_json(merlin.ROUTE_PROVIDER, merlin.ROUTE_SYSTEM, merlin.route_prompt(text),
                                   agent="merlin", models=self.models),
                cap,
            )
            got = merlin.check_route_reply(raw)
            if got:
                return got[0], got[1], "claude"
            await self.error("route", "routing answer named nobody valid, used the fixed rules")
        except asyncio.TimeoutError:
            await self.error("route", f"routing took over {cap:.1f} s, used the fixed rules")
        except providers.ProviderError as e:
            await self.error("route", f"routing failed, used the fixed rules: {str(e)[:150]}")
        agent, why = merlin.job_kind(text)
        if desk == "job" and agent is None:
            return (*merlin.desk_rule(text, self.rng), "rules")
        return merlin.choose_by_role(text, self.rng), why or "nobody in particular", "rules"

    async def route(self, text: str, agent: str | None, force_argue: bool | None) -> dict:
        """Who takes it and whether Merlin talks first. Merlin's own LLM answer is handled in handle()."""
        named = self.resolve_named(text, agent)
        to_merlin = self.addressed_merlin(text, agent)
        r = {"named": named, "chosen": named, "mline": None, "argued": [], "group": [], "ask_merlin": False,
             "reason": "named" if named else None, "source": "named" if named else None}
        if named:
            if to_merlin:
                r["mline"] = merlin.merlin_handoff(named, self.rng)
            return r
        kind_agent, _why = merlin.job_kind(text)
        # Spoken to directly, Merlin decides herself whether to answer or hand it on. A repeat and
        # the bottle still follow the fixed rules.
        if to_merlin and not cast_mod.pending_repeat_agent(text) and not cast_mod.is_bottle_job(text):
            r["ask_merlin"] = True
            return r
        if merlin.is_everyone(text):
            lead, why, source = await self.pick_by_kind(text)
            r.update(chosen=lead, group=merlin.pick_group(lead, self.rng), reason=why, source=source,
                     mline=self.rng.choice(merlin.GROUP_LINES).format(name=CAST[lead]["name"]))
            return r
        argue = force_argue if force_argue is not None else (
            not to_merlin and kind_agent is None and merlin.should_argue(self.rng))
        if argue:
            argued, chosen, mline = merlin.argue(self.rng)
            r.update(chosen=chosen, argued=argued, mline=mline, reason="argued", source="argument")
        else:
            chosen, why, source = await self.pick_by_kind(text)
            r.update(chosen=chosen, mline=merlin.merlin_handoff(chosen, self.rng), reason=why, source=source)
        return r

    async def handle(self, text: str, agent: str | None = None, force_argue: bool | None = None) -> dict:
        text = (text or "").strip()
        run = _Run()
        await self.emit({"type": "request", "text": text, "named_agent": self.resolve_named(text, agent)})
        merlin_out: dict | None = None
        self._stop_req = False
        timing: dict = {}
        t_start = time.monotonic()
        try:
            for cid in AGENT_IDS:
                await self.set_state(cid, "listening")
            # A desk job or a desk question looks at the desk first, at the same time as the routing.
            desk = None if cast_mod.is_bottle_job(text) else cast_mod.desk_kind(text)
            scene_task = None
            if desk and self.use_arm and self.arm is not None:
                scene_task = asyncio.ensure_future(self.look_scene())
            route = await self.route(text, agent, force_argue)
            timing["route_ms"] = int((time.monotonic() - t_start) * 1000)

            if route["ask_merlin"]:
                await self.set_state("merlin", "thinking")
                line, hand_to, answered = await self.ask_merlin(text)
                if answered and not hand_to:
                    return await self.merlin_only(run, line)
                if answered:
                    route.update(chosen=hand_to, mline=line, reason="Merlin handed it on", source="merlin")
                else:
                    chosen, why, source = await self.pick_by_kind(text)
                    route.update(chosen=chosen, mline=merlin.merlin_handoff(chosen, self.rng), reason=why, source=source)

            chosen = route["chosen"]
            group = route["group"]
            merlin.note_chosen(chosen)
            if desk == "job":
                merlin.note_arm_holder(chosen)

            async def lead_plan() -> dict:
                look = None
                if scene_task is not None:
                    look = await scene_task
                    timing["look_ms"] = look.get("ms")
                    timing["look_ready_ms"] = int((time.monotonic() - t_start) * 1000)
                r = await self.make_plan(chosen, text, "lead" if group else None, look=look)
                r["look"] = look
                timing["plan_ready_ms"] = int((time.monotonic() - t_start) * 1000)
                return r

            plan_task = asyncio.ensure_future(lead_plan())
            helper_tasks = [asyncio.ensure_future(self.make_plan(h, text, "helper", chosen)) for h in group[1:]]

            if route["mline"] is None:
                for cid in AGENT_IDS:
                    if cid != chosen:
                        await self.set_state(cid, "idle")
                await self.set_state(chosen, "thinking")
            else:
                await self.set_state("merlin", "thinking")
                argued, mline = route["argued"], route["mline"]
                # The chosen agent is already thinking while Merlin and the others talk.
                clips = await asyncio.gather(*[self.speak(a["agent"], a["say"]) for a in argued], self.speak("merlin", mline))
                for a, c in zip(argued, clips[:-1]):
                    a.update(c)
                mclip = clips[-1]
                merlin_out = {
                    "type": "merlin",
                    "chosen": chosen,
                    "say": mline,
                    "audio_url": mclip["audio_url"],
                    "duration_s": mclip["duration_s"],
                    "argued": argued,
                }
                await self.emit(merlin_out)
                busy = {chosen, *group, *[a["agent"] for a in argued]}
                for cid in AGENT_IDS:
                    if cid not in busy:
                        await self.set_state(cid, "idle")
                await self.set_state("merlin", "idle")
                # One gesture per line, in order: argument lines, then Merlin.
                for a in argued:
                    self.queue_gesture(run, a["agent"], a["duration_s"], set_states=True)
                self.queue_gesture(run, "merlin", mclip["duration_s"], set_states=True)
                await self.set_state(chosen, "thinking")

            result = await plan_task
            plan = result["plan"]

            # Tools that come back with something to say start now, so they run while the first
            # line is being made and played. The rest run once the line is out.
            from backend import tools

            early = {}
            if self.run_tools:
                early = {i: asyncio.ensure_future(self.run_one_tool(c, chosen))
                         for i, c in enumerate(plan["laptop"]) if c["tool"] in tools.ANSWERS}
            clip = await self.speak(chosen, plan["say"])
            await self.wait_gestures(run)  # Merlin and the argument finish first

            await self.set_state(chosen, "speaking")
            await self.emit(
                {"type": "say", "agent": chosen, "text": plan["say"], "audio_url": clip["audio_url"], "duration_s": clip["duration_s"]}
            )
            timing["first_say_ms"] = int((time.monotonic() - t_start) * 1000)
            gesture_task = None
            if plan["arm"] is None:
                gesture_task = self.queue_gesture(run, chosen, clip["duration_s"])

            tool_results: list[dict] = []
            followups: list[dict] = []
            if self.run_tools:
                for i, call in enumerate(plan["laptop"]):
                    tool_results.append(await early[i] if i in early else await self.run_one_tool(call, chosen))
                for line in self.follow_ups(chosen, plan, tool_results):
                    followups.append(await self.say_line(run, chosen, line, gesture=plan["arm"] is None))
                    if plan["arm"] is None:
                        gesture_task = run.chain
            look = result.get("look")
            if result["decision"].get("desk") == "question" and look and look.get("ok") and not result["replaced"]:
                # The answer to "what's on my desk", read from the camera's list, no motion.
                line = merlin.clean_follow(chosen, cast_mod.scene_answer(chosen, text, look.get("scene")))
                followups.append(await self.say_line(run, chosen, line, gesture=plan["arm"] is None))
                if plan["arm"] is None:
                    gesture_task = run.chain

            helpers = []
            for h, task in zip(group[1:], helper_tasks):
                hres = await task
                await self.say_line(run, h, hres["plan"]["say"], gesture=True)
                helpers.append({"agent": h, "say": hres["plan"]["say"]})

            arm_result = None
            if isinstance(plan["arm"], list):
                await self.set_state(chosen, "has_arm")
                t_arm = time.monotonic()
                arm_result = await self.run_plan(chosen, plan["arm"], result["decision"])
                timing["arm_ms"] = int((time.monotonic() - t_arm) * 1000)
                if arm_result.get("say"):
                    # the line that says what really happened
                    followups.append(await self.say_line(run, chosen, arm_result["say"], gesture=True))
            elif plan["arm"]:
                await self.set_state(chosen, "has_arm")
                skill = plan["arm"]["skill"]
                params = {k: v for k, v in plan["arm"].items() if k != "skill"}
                arm_result = await self.run_arm(chosen, skill, params)
            else:
                await self.wait_gestures(run)
                if gesture_task is not None:
                    arm_result = run.last_gesture

            return {
                "agent": chosen,
                "say": plan["say"],
                "plan": plan,
                "audio_url": clip["audio_url"],
                "duration_s": clip["duration_s"],
                "arm_result": arm_result,
                "merlin": {k: v for k, v in merlin_out.items() if k != "type"} if merlin_out else None,
                "provider": result["provider"].get("provider"),
                "check": {"issues": result["issues"], "replaced": result["replaced"]},
                "tools": tool_results,
                "followups": followups,
                "group": helpers,
                "routed": {"why": route["reason"], "by": route["source"]},
                "timing": timing,
            }
        except Exception as e:  # noqa: BLE001
            await self.error("pipeline", f"{type(e).__name__}: {e}")
            raise
        finally:
            await self.wait_gestures(run)
            for cid in CAST:
                if self.states.get(cid) != "idle":
                    await self.set_state(cid, "idle")
