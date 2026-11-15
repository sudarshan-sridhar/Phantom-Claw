"""The first five tools: note, todo, timer, open_url and lookup."""
from __future__ import annotations

import asyncio
import os
import re
import webbrowser
from datetime import datetime

from backend.agents import providers
from backend.agents.cast import CAST, FOLLOW_RULES
from backend.agents.config import NOTES_FILE

from .common import Emit, fail, is_dry, ok, would

_timers: set[asyncio.Task] = set()


def _append(line: str) -> None:
    NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
    new = not NOTES_FILE.exists()
    with NOTES_FILE.open("a", encoding="utf-8") as f:
        if new:
            f.write("# Notes\n\n")
        f.write(line + "\n")


async def note(agent: str, text: str, emit: Emit, kind: str = "note") -> dict:
    """Add a line (or a to-do item) to backend/notes.md and tell the overlay. Only ever appends."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    text = re.sub(r"\s+", " ", text).strip()
    line = f"- [ ] {text} ({agent}, {stamp})" if kind == "todo" else f"- {stamp} ({agent}) {text}"
    if is_dry():
        return would(kind, f"add to notes.md: {line}")
    await asyncio.to_thread(_append, line)
    await emit({"type": "note", "agent": agent, "text": text, "kind": kind})
    return ok(kind)


async def timer(agent: str, seconds: int, label: str, emit: Emit) -> dict:
    """Start a timer. When it ends a timer_done event goes out."""
    if is_dry():
        return would("timer", f"start a {seconds} s timer called {label!r}", seconds=seconds)

    async def _run():
        await asyncio.sleep(seconds)
        await emit({"type": "timer_done", "agent": agent, "label": label})

    task = asyncio.create_task(_run())
    _timers.add(task)
    task.add_done_callback(_timers.discard)
    return ok("timer", seconds=seconds)


def open_address(tool: str, url: str) -> dict:
    """Open an http or https address in the default browser (shared by the web tools)."""
    if not re.match(r"^https?://", url, re.I):
        return fail(tool, "only http or https")
    if is_dry():
        return would(tool, f"open {url} in the browser", url=url)
    if os.environ.get("PHANTOM_NO_BROWSER") == "1":
        return ok(tool, detail="skipped (PHANTOM_NO_BROWSER)", url=url)
    from .apps import _windows, show_window

    before = _windows()
    opened = bool(webbrowser.open(url))
    shown = False
    if opened:
        # the page opens as a tab in a browser that may be behind other windows: bring it up
        try:
            shown = show_window(before, "brave chrome edge firefox opera", wait_s=4.0)
        except Exception:  # noqa: BLE001 - the page still opened
            shown = False
    return {"ok": opened, "tool": tool, "url": url, "shown": shown}


async def open_url(agent: str, url: str) -> dict:
    """Open a web page in the default browser."""
    return await asyncio.to_thread(open_address, "open_url", url)


async def lookup(agent: str, query: str) -> dict:
    """Ask the agent's own LLM for a short answer, already said the way that character talks.

    No web access, so it is only what the model knows.
    """
    c = CAST[agent]
    system = (
        f"You are {c['name']}. Ego: {c['ego']}. {c['notes']} Answer the question from what you already know, "
        f"in at most two short sentences, the way you talk. No web access, so if you aren't sure of a number "
        f"or a fact, say you aren't sure instead of making one up. {FOLLOW_RULES} "
        f"Reply as JSON: {{\"answer\": \"...\"}}"
    )
    try:
        obj, info = await providers.get_json(c["provider"], system, query, agent=None)
        answer = str(obj.get("answer") or "").strip()[:300]
        return {"ok": bool(answer), "tool": "lookup", "answer": answer, "provider": info["provider"]}
    except providers.ProviderError as e:
        return fail("lookup", str(e)[:200])
