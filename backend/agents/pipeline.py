"""The request pipeline, shared by the server and the command line demo.

Order (contract): request, Merlin (only when no agent is named), plan from the agent's LLM,
Merlin's check, speech, say, laptop tools, arm skill. Agent states throughout, all idle at the end.
The arm is the body of whoever is speaking: a line with no arm skill gets a `gesture` in the
speaker's style for the length of the clip.
"""
from __future__ import annotations

import asyncio
import os
import random
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

    async def run_arm(self, agent: str, skill: str, params: dict) -> dict:
        """Run one arm skill in a worker thread, sending `arm` events for each phase."""
        if not self.use_arm or self.arm is None:
            return {"ok": False, "result": "skipped", "detail": "arm not used", "skill": skill}
        if self._arm_busy():
            return {"ok": False, "result": "busy", "detail": "arm busy", "skill": skill}
        loop = asyncio.get_running_loop()

        def on_phase(phase: str) -> None:
            ev = {"type": "arm", "agent": agent, "skill": skill, "phase": phase}
            loop.call_soon_threadsafe(lambda: asyncio.ensure_future(self.emit(ev)))

        try:
            res = await asyncio.to_thread(self.arm.run, skill, dict(params), on_phase)
        except Exception as e:  # noqa: BLE001
            await self.error("arm", f"{skill}: {type(e).__name__}: {e}")
            res = {"ok": False, "result": "error", "detail": str(e)[:200]}
        await asyncio.sleep(0)  # let the last phase events go out first
        return {**res, "skill": skill}

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
        found = merlin.find_named_agent(text)
        return found if found in AGENT_IDS else None

    async def make_plan(self, chosen: str, text: str) -> dict:
        """LLM plan plus Merlin's check. Returns plan, issues, replaced, provider info, decision."""
        decision = cast_mod.decide(chosen, text, self.now_fn())
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
        cast_mod.remember(chosen, text, "done" if replaced else cast_mod.outcome_of(decision))
        return {"plan": plan, "issues": issues, "replaced": replaced, "provider": info, "decision": decision}

    async def handle(self, text: str, agent: str | None = None, force_argue: bool | None = None) -> dict:
        text = (text or "").strip()
        run = _Run()
        named = self.resolve_named(text, agent)
        await self.emit({"type": "request", "text": text, "named_agent": named})
        merlin_out: dict | None = None
        try:
            for cid in AGENT_IDS:
                await self.set_state(cid, "listening")

            plan_task: asyncio.Task
            if named:
                chosen = named
                merlin.note_chosen(chosen)
                for cid in AGENT_IDS:
                    if cid != chosen:
                        await self.set_state(cid, "idle")
                await self.set_state(chosen, "thinking")
                plan_task = asyncio.ensure_future(self.make_plan(chosen, text))
            else:
                await self.set_state("merlin", "thinking")
                argue = merlin.should_argue(self.rng) if force_argue is None else force_argue
                if argue:
                    argued, chosen, mline = merlin.argue(self.rng)
                else:
                    argued, chosen = [], merlin.choose_by_role(text, self.rng)
                    mline = merlin.merlin_handoff(chosen, self.rng)
                merlin.note_chosen(chosen)
                # The chosen agent starts thinking while Merlin and the others talk.
                plan_task = asyncio.ensure_future(self.make_plan(chosen, text))
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
                for cid in AGENT_IDS:
                    if cid != chosen and cid not in [a["agent"] for a in argued]:
                        await self.set_state(cid, "idle")
                await self.set_state("merlin", "idle")
                # One gesture per line, in order: argument lines, then Merlin.
                for a in argued:
                    self.queue_gesture(run, a["agent"], a["duration_s"], set_states=True)
                self.queue_gesture(run, "merlin", mclip["duration_s"], set_states=True)
                await self.set_state(chosen, "thinking")

            result = await plan_task
            plan = result["plan"]
            clip = await self.speak(chosen, plan["say"])
            await self.wait_gestures(run)  # Merlin and the argument finish first

            await self.set_state(chosen, "speaking")
            await self.emit(
                {"type": "say", "agent": chosen, "text": plan["say"], "audio_url": clip["audio_url"], "duration_s": clip["duration_s"]}
            )
            gesture_task = None
            if plan["arm"] is None:
                gesture_task = self.queue_gesture(run, chosen, clip["duration_s"])

            tool_results = []
            if self.run_tools:
                from backend import tools

                for call in plan["laptop"]:
                    try:
                        r = await tools.run_tool(call, chosen, self.emit)
                    except Exception as e:  # noqa: BLE001
                        r = {"ok": False, "tool": call.get("tool"), "detail": str(e)[:200]}
                        await self.error("tool", f"{call.get('tool')}: {e}")
                    tool_results.append(r)
                    if call["tool"] == "lookup" and r.get("answer"):
                        aclip = await self.speak(chosen, r["answer"])
                        await self.wait_gestures(run)
                        await self.emit(
                            {"type": "say", "agent": chosen, "text": r["answer"], "audio_url": aclip["audio_url"], "duration_s": aclip["duration_s"]}
                        )
                        if plan["arm"] is None:
                            gesture_task = self.queue_gesture(run, chosen, aclip["duration_s"])

            arm_result = None
            if plan["arm"]:
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
            }
        except Exception as e:  # noqa: BLE001
            await self.error("pipeline", f"{type(e).__name__}: {e}")
            raise
        finally:
            await self.wait_gestures(run)
            for cid in CAST:
                if self.states.get(cid) != "idle":
                    await self.set_state(cid, "idle")
