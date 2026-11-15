"""Laptop tools: note, todo, timer, open_url, lookup. None of them sends, deletes or pays for anything."""
from __future__ import annotations

import asyncio
import os
import re
import webbrowser
from datetime import datetime
from typing import Awaitable, Callable

from backend.agents.config import NOTES_FILE
from backend.agents import providers
from backend.agents.cast import CAST

Emit = Callable[[dict], Awaitable[None]]

_timers: set[asyncio.Task] = set()


def _append(line: str) -> None:
    NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
    new = not NOTES_FILE.exists()
    with NOTES_FILE.open("a", encoding="utf-8") as f:
        if new:
            f.write("# Notes\n\n")
        f.write(line + "\n")


async def note(agent: str, text: str, emit: Emit, kind: str = "note") -> dict:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    text = re.sub(r"\s+", " ", text).strip()
    line = f"- [ ] {text} ({agent}, {stamp})" if kind == "todo" else f"- {stamp} ({agent}) {text}"
    await asyncio.to_thread(_append, line)
    await emit({"type": "note", "agent": agent, "text": text, "kind": kind})
    return {"ok": True, "tool": kind}


async def timer(agent: str, seconds: int, label: str, emit: Emit) -> dict:
    async def _run():
        await asyncio.sleep(seconds)
        await emit({"type": "timer_done", "agent": agent, "label": label})

    task = asyncio.create_task(_run())
    _timers.add(task)
    task.add_done_callback(_timers.discard)
    return {"ok": True, "tool": "timer", "seconds": seconds}


async def open_url(agent: str, url: str) -> dict:
    if not re.match(r"^https?://", url, re.I):
        return {"ok": False, "tool": "open_url", "detail": "only http or https"}
    if os.environ.get("PHANTOM_NO_BROWSER") == "1":
        return {"ok": True, "tool": "open_url", "detail": "skipped (PHANTOM_NO_BROWSER)"}
    ok = await asyncio.to_thread(webbrowser.open, url)
    return {"ok": bool(ok), "tool": "open_url"}


async def lookup(agent: str, query: str) -> dict:
    """Ask the agent's own LLM for a two sentence answer. No web access."""
    c = CAST[agent]
    system = (
        f"You are {c['name']}. Ego: {c['ego']}. {c['notes']} Answer the question in at most two short sentences, "
        f"in character, from what you already know. No web access. Reply as JSON: {{\"answer\": \"...\"}}"
    )
    try:
        obj, info = await providers.get_json(c["provider"], system, query, agent=None)
        answer = str(obj.get("answer") or "").strip()[:300]
        return {"ok": bool(answer), "tool": "lookup", "answer": answer, "provider": info["provider"]}
    except providers.ProviderError as e:
        return {"ok": False, "tool": "lookup", "detail": str(e)[:200]}


async def run_tool(call: dict, agent: str, emit: Emit) -> dict:
    t = call.get("tool")
    if t in ("note", "todo"):
        return await note(agent, call["text"], emit, kind=t)
    if t == "timer":
        return await timer(agent, int(call["seconds"]), call.get("label") or "timer", emit)
    if t == "open_url":
        return await open_url(agent, call["url"])
    if t == "lookup":
        return await lookup(agent, call["query"])
    return {"ok": False, "tool": t, "detail": "unknown tool"}
