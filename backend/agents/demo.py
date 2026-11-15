"""Run one request through every agent, without the server, and print each line and plan.

    python -m backend.agents.demo "hand me the bottle"
    python -m backend.agents.demo "hand me the bottle" --clock 12:00 --twice
    python -m backend.agents.demo "please hand me the bottle" --clock 12:00 --agents escanor
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime

from . import cast as cast_mod
from .cast import AGENT_IDS, CAST
from .pipeline import Pipeline, clock_now


async def _noop(_ev: dict) -> None:
    pass


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--clock", help="fake laptop time HH:MM for Escanor")
    ap.add_argument("--agents", default=",".join(AGENT_IDS))
    ap.add_argument("--twice", action="store_true", help="ask each agent a second time")
    ap.add_argument("--voice", action="store_true", help="also make the speech clips")
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

    errors: list[dict] = []

    async def emit(ev: dict) -> None:
        if ev.get("type") == "error":
            errors.append(ev)

    p = Pipeline(None, emit, speak=args.voice, use_arm=False, run_tools=False, now_fn=now_fn, models=models or None)
    cast_mod.reset_memory()
    agents = [a.strip() for a in args.agents.split(",") if a.strip() in AGENT_IDS]
    print(f'Request: "{args.text}"  clock {now_fn().strftime("%H:%M")}  escanor mood: {cast_mod.escanor_mood(now_fn().hour + now_fn().minute / 60)}\n')

    rounds = 2 if args.twice else 1
    for rnd in range(rounds):
        if rounds > 1:
            print(f"--- ask number {rnd + 1} ---")
        results = await asyncio.gather(*[p.make_plan(a, args.text) for a in agents])
        for a, r in zip(agents, results):
            plan = r["plan"]
            prov = r["provider"]
            via = f"{prov.get('provider')} {prov.get('model')}{' (fallback)' if prov.get('fallback') else ''}, {prov.get('ms')} ms" if prov else "no provider"
            print(f"{CAST[a]['name']:<10} [{via}]")
            print(f"  says: {plan['say']}")
            print(f"  arm:  {json.dumps(plan['arm'])}")
            print(f"  laptop: {json.dumps(plan['laptop'])}")
            if r["issues"]:
                print(f"  check: {'; '.join(r['issues'])}{' (REPLACED)' if r['replaced'] else ''}")
            if args.voice:
                clip = await p.speak(a, plan["say"])
                print(f"  audio: {clip['audio_url']} ({clip['duration_s']} s)")
        print()
    for e in errors:
        print(f"error event: {e['where']}: {e['message']}")


if __name__ == "__main__":
    asyncio.run(main())
