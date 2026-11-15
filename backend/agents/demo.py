"""Run one request through every agent, without the server, and print each line and plan.

    python -m backend.agents.demo "hand me the bottle"
    python -m backend.agents.demo "hand me the bottle" --clock 12:00 --twice
    python -m backend.agents.demo "please hand me the bottle" --clock 12:00 --agents escanor
    python -m backend.agents.demo "open notepad" --dry-tools

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


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
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
