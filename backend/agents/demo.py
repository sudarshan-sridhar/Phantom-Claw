"""Run one request through every agent, without the server, and print each line and plan.

    python -m backend.agents.demo "hand me the bottle"
    python -m backend.agents.demo "hand me the bottle" --clock 12:00 --twice
    python -m backend.agents.demo "please hand me the bottle" --clock 12:00 --agents escanor
    python -m backend.agents.demo "open notepad" --dry-tools
    python -m backend.agents.demo --arm "hand me something to drink" "King, give me the Sharpie"

With --arm every request goes through the whole pipeline in order, on the pretend arm (FakeArm,
never the real one) with dry laptop tools, and prints who took it, the line, the steps, what each
step returned, the follow-up line and the times. PHANTOM_FAKE_TIERS=tall,can,low makes every grip
tier count as taught; PHANTOM_FAKE_TAKEN=0 means nobody takes a handover.

It first prints who Merlin would give the request to. With --dry-tools each plan's laptop calls
also run in dry-run mode (they work out what they'd open, nothing is opened) and the second line
that carries a result is printed too.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime

from . import cast as cast_mod
from .cast import AGENT_IDS, CAST
from .pipeline import Pipeline, clock_now


def _step_str(s: dict) -> str:
    p = s.get("params") or {}
    what = p.get("target") or p.get("where")
    if isinstance(what, dict):
        what = what.get("near") and f"near {what['near']} ({what.get('side')})" or f"({what.get('x_cm')}, {what.get('y_cm')})"
    knobs = ", ".join(f"{k} {p[k]}" for k in ("pace", "approach", "lift_cm_style", "wait_s") if k in p)
    return f"{s['skill']}" + (f" {what}" if what else "") + (f" [{knobs}]" if knobs else "")


async def run_on_arm(args, now_fn, models) -> None:
    """Every request in order through the whole pipeline, on the pretend arm only."""
    import time

    from backend.arm import FakeArm   # never RealArm here

    from backend import tools
    tools.set_dry_run(True)
    events: list[dict] = []

    async def emit(ev: dict) -> None:
        events.append(ev)

    arm = FakeArm(time_scale=args.arm_scale)
    p = Pipeline(arm, emit, speak=args.voice, run_tools=True, pace_audio=False, now_fn=now_fn, models=models or None)
    cast_mod.reset_memory()
    try:
        import anthropic  # noqa: F401 - slow first import, done before the timed routing call
    except ImportError:
        pass
    print(f"pretend arm, taught tiers {sorted(arm.taught)}, time scale {args.arm_scale}, nobody takes: {not arm.hand_taken}")
    import copy

    from backend.arm.fake import FAKE_SCENE
    for text in args.text:
        if args.fresh_desk:
            arm.scene_items = copy.deepcopy(FAKE_SCENE)    # every request starts from the same desk
        events.clear()
        t0 = time.monotonic()
        res = await p.handle(text, None, force_argue=False if not args.argue else None)
        total = time.monotonic() - t0
        plan = res["plan"]
        print(f'\n> "{text}"')
        m = res.get("merlin")
        print(f"  {CAST[res['agent']]['name']} ({(res.get('routed') or {}).get('by')}: {(res.get('routed') or {}).get('why')})"
              + (f"; Merlin: {m['say']}" if m else ""))
        print(f"  says: {res['say']}")
        arm_plan = plan["arm"]
        if isinstance(arm_plan, list):
            print("  steps: " + "; ".join(_step_str(s) for s in arm_plan))
            ar = res["arm_result"]
            print("  ran:   " + "; ".join(f"{s['skill']} {s['result']}" + (" (retry)" if s.get("retry") else "")
                                          + (" (cached)" if s.get("cached") else "")
                                          + (" (recovery)" if s.get("recovery") else "") for s in ar["steps"]))
            print(f"  result: {ar['result']}")
        elif arm_plan:
            print(f"  arm: {json.dumps(arm_plan)} -> {(res['arm_result'] or {}).get('result')}")
        else:
            print("  arm: none")
        for f in res["followups"]:
            print(f"  then: {f['text']}")
        if res["check"]["issues"]:
            print(f"  check: {'; '.join(res['check']['issues'])}")
        t = res.get("timing") or {}
        print(f"  times: route {t.get('route_ms')} ms, look {t.get('look_ms')} ms (ready at {t.get('look_ready_ms')}), "
              f"plan ready {t.get('plan_ready_ms')} ms, first line {t.get('first_say_ms')} ms, arm {t.get('arm_ms')} ms, "
              f"whole request {total:.1f} s")
        for e in events:
            if e.get("type") == "error":
                print(f"  error event: {e['where']}: {e['message']}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="+", help="the request (with --arm, several requests in order)")
    ap.add_argument("--arm", action="store_true", help="run each request through the whole pipeline on the pretend arm")
    ap.add_argument("--arm-scale", type=float, default=0.2, help="pretend arm time scale (1.0 is real time)")
    ap.add_argument("--argue", action="store_true", help="with --arm, let the argument happen as it would live")
    ap.add_argument("--fresh-desk", action="store_true", help="with --arm, put the six items back before each request")
    ap.add_argument("--clock", help="fake laptop time HH:MM for Escanor")
    ap.add_argument("--agents", default=",".join(AGENT_IDS))
    ap.add_argument("--twice", action="store_true", help="ask each agent a second time")
    ap.add_argument("--voice", action="store_true", help="also make the speech clips")
    ap.add_argument("--dry-tools", action="store_true", help="run the laptop calls in dry-run mode and show the result lines")
    ap.add_argument("--routed", action="store_true", help="only the agent Merlin picks")
    ap.add_argument("--gemini-model", help="override the gemini model (to test fall through)")
    ap.add_argument("--openai-model", help="override the openai model (to test fall through)")
    args = ap.parse_args()

    def now_fn() -> datetime:
        if args.clock:
            h, m = args.clock.split(":")
            return datetime.now().replace(hour=int(h), minute=int(m))
        return clock_now()

    models = {}
    if args.gemini_model:
        models["gemini"] = args.gemini_model
    if args.openai_model:
        models["openai"] = args.openai_model

    if args.arm:
        await run_on_arm(args, now_fn, models)
        return
    args.text = args.text[0]

    if args.dry_tools:
        from backend import tools
        tools.set_dry_run(True)

    errors: list[dict] = []

    async def emit(ev: dict) -> None:
        if ev.get("type") == "error":
            errors.append(ev)

    p = Pipeline(None, emit, speak=args.voice, use_arm=False, run_tools=False, now_fn=now_fn, models=models or None)
    cast_mod.reset_memory()
    print(f'Request: "{args.text}"  clock {now_fn().strftime("%H:%M")}  escanor mood: {cast_mod.escanor_mood(now_fn().hour + now_fn().minute / 60)}')

    try:
        import anthropic  # noqa: F401 - slow first import, done before the timed routing call
    except ImportError:
        pass
    route = await p.route(args.text, None, force_argue=False)
    if route["ask_merlin"]:
        line, hand_to, _answered = await p.ask_merlin(args.text)
        print(f"Merlin answers herself: {line}" + (f"  (hands it to {hand_to})" if hand_to else ""))
        if hand_to:
            route["chosen"] = hand_to
    elif route["named"]:
        print(f"Named: {CAST[route['named']]['name']} takes it" + (f"; Merlin: {route['mline']}" if route["mline"] else ""))
    else:
        group = f", with {', '.join(CAST[a]['name'] for a in route['group'][1:])} chipping in" if route["group"] else ""
        print(f"Unnamed: Merlin gives it to {CAST[route['chosen']]['name']}{group} ({route['reason']}, by {route['source']}); "
              f"Merlin: {route['mline']}")
    print()

    agents = [a.strip() for a in args.agents.split(",") if a.strip() in AGENT_IDS]
    if args.routed and route.get("chosen"):
        agents = [route["chosen"]]
    rounds = 2 if args.twice else 1
    for rnd in range(rounds):
        if rounds > 1:
            print(f"--- ask number {rnd + 1} ---")
        results = await asyncio.gather(*[p.make_plan(a, args.text) for a in agents])
        for a, r in zip(agents, results):
            plan = r["plan"]
            prov = r["provider"]
            via = f"{prov.get('provider')} {prov.get('model')}{' (fallback)' if prov.get('fallback') else ''}, {prov.get('ms')} ms" if prov else "no provider"
            mark = "  <- Merlin's pick" if a == route.get("chosen") else ""
            print(f"{CAST[a]['name']:<10} [{via}]{mark}")
            print(f"  says: {plan['say']}")
            print(f"  arm:  {json.dumps(plan['arm'])}")
            print(f"  laptop: {json.dumps(plan['laptop'])}")
            if r["issues"]:
                print(f"  check: {'; '.join(r['issues'])}{' (REPLACED)' if r['replaced'] else ''}")
            if args.dry_tools and plan["laptop"]:
                results_ = [await p.run_one_tool(c, a) for c in plan["laptop"]]
                for c, tr in zip(plan["laptop"], results_):
                    what = tr.get("would") or tr.get("value") or tr.get("answer") or tr.get("detail") or ("ok" if tr.get("ok") else "failed")
                    print(f"  tool {c['tool']}: {what}")
                for line in p.follow_ups(a, plan, results_):
                    print(f"  then says: {line}")
            if args.voice:
                clip = await p.speak(a, plan["say"])
                print(f"  audio: {clip['audio_url']} ({clip['duration_s']} s)")
        print()
    for e in errors:
        print(f"error event: {e['where']}: {e['message']}")


if __name__ == "__main__":
    asyncio.run(main())
