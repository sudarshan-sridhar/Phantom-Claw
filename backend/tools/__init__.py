"""Laptop tools, all in one registry.

TOOLS below is the only list of what a plan may call. Merlin's plan check uses `check_call` to
clean each call, and the pipeline uses `run_tool` to do it. Nothing here deletes, moves or
overwrites a file, sends anything, pays for anything, types or reads a password, shuts the
machine down, changes a setting or installs anything.

Dry run (`set_dry_run(True)` or PHANTOM_DRY_TOOLS=1) works out what each tool would do and
does none of it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from . import apps, ask, basic, desk
from .common import Emit, is_dry, set_dry_run
from .safety import FAMILY, forbidden, wanted_families, wanted_tools

__all__ = ["TOOLS", "OPENERS", "ANSWERS", "check_call", "run_tool", "set_dry_run", "is_dry", "forbidden",
           "wanted_tools", "wanted_families", "FAMILY", "Emit"]


@dataclass(frozen=True)
class Tool:
    fields: dict[str, int] = field(default_factory=dict)  # text field -> max length
    kind: str = "action"   # "opener" opens something, "answer" produces something to say, "action" otherwise
    help: str = ""


TOOLS: dict[str, Tool] = {
    "note": Tool({"text": 500}, "action", "save a note"),
    "todo": Tool({"text": 500}, "action", "save a to-do item"),
    "timer": Tool({"label": 80}, "action", "start a timer, seconds 1 to 3600"),
    "open_url": Tool({"url": 2000}, "opener", "open an http or https page"),
    "lookup": Tool({"query": 300}, "answer", "answer a factual question from what you know"),
    "open_app": Tool({"name": 80}, "opener", "open an installed program by name"),
    "open_folder": Tool({"name": 60}, "opener", "open Downloads, Documents, Desktop, Pictures, Videos, Music, "
                                                "Screenshots or the project folder"),
    "open_file": Tool({"query": 120}, "opener", "open the newest file in Downloads, Documents or Desktop whose name has these words"),
    "web_search": Tool({"query": 200}, "opener", "open a web search"),
    "youtube": Tool({"query": 200}, "opener", "open a YouTube search"),
    "play_music": Tool({"query": 200}, "opener", "open Spotify on a search"),
    "media": Tool({}, "action", "press a media key: play_pause, next, previous, volume_up, volume_down, mute"),
    "type_text": Tool({"text": desk.TYPE_MAX}, "action", "type text into the window in front (dictation)"),
    "read_screen": Tool({"question": 300}, "answer", "look at the screen and answer a question about it"),
    "clipboard": Tool({"question": 300}, "answer", "read the copied text and answer about it or sum it up"),
    "tell": Tool({}, "answer", "the time, date, battery level, or which window is in front"),
    "show_desktop": Tool({}, "action", "minimise everything"),
    "screenshot": Tool({}, "action", "save a screenshot in Pictures, Screenshots"),
}
OPENERS = {n for n, t in TOOLS.items() if t.kind == "opener"}
ANSWERS = {n for n, t in TOOLS.items() if t.kind == "answer"}

_DEFAULTS = {"read_screen": {"question": "What's on my screen?"}, "clipboard": {"question": "Sum it up."}}


def _text(v: Any, limit: int) -> str:
    return re.sub(r"\s+", " ", str(v or "")).strip()[:limit]


def check_call(call: Any, issues: list[str]) -> dict | None:
    """Return a clean copy of one tool call, or None (with a note in issues) if it can't be used."""
    if not isinstance(call, dict) or call.get("tool") not in TOOLS:
        issues.append(f"dropped unknown tool {call!r}"[:120])
        return None
    t = call["tool"]
    spec = TOOLS[t]
    out: dict[str, Any] = {"tool": t}
    for name, limit in spec.fields.items():
        out[name] = _text(call.get(name), limit) if t != "type_text" else desk.clean_typing(call.get(name))
    if t == "timer":
        try:
            secs = int(min(3600, max(1, float(call.get("seconds")))))
        except (TypeError, ValueError):
            secs = 60
        return {"tool": t, "seconds": secs, "label": out["label"] or "timer"}
    if t == "open_url":
        if not re.match(r"^https?://[^\s]+$", out["url"], re.I):
            issues.append("dropped open_url that was not http or https")
            return None
        return out
    if t == "media":
        action = re.sub(r"[\s/-]+", "_", str(call.get("action") or "").strip().lower())
        action = {"pause": "play_pause", "play": "play_pause", "resume": "play_pause", "skip": "next",
                  "prev": "previous", "back": "previous", "louder": "volume_up", "quieter": "volume_down",
                  "unmute": "mute"}.get(action, action)
        if action not in desk.MEDIA_KEYS:
            issues.append(f"dropped media action {action!r}")
            return None
        return {"tool": t, "action": action}
    if t == "tell":
        kind = desk.tell_kind(call.get("what"))
        if kind is None:
            issues.append(f"dropped tell {call.get('what')!r}")
            return None
        return {"tool": t, "what": kind}
    if t in _DEFAULTS:
        for k, v in _DEFAULTS[t].items():
            out[k] = out[k] or v
        return out
    if t in ("show_desktop", "screenshot"):
        return out
    if t == "open_folder" and apps.folder_key(out["name"]) is None:
        issues.append(f"dropped open_folder {out['name']!r}: not one of the known folders")
        return None
    # everything else needs its one text field filled in
    if any(not out[k] for k in spec.fields):
        issues.append(f"dropped {t} with nothing in it")
        return None
    return out


async def run_tool(call: dict, agent: str, emit: Emit) -> dict:
    """Do one checked tool call. Returns a result dict with ok, tool and whatever it found."""
    t = call.get("tool")
    if t in ("note", "todo"):
        return await basic.note(agent, call["text"], emit, kind=t)
    if t == "timer":
        return await basic.timer(agent, int(call["seconds"]), call.get("label") or "timer", emit)
    if t == "open_url":
        return await basic.open_url(agent, call["url"])
    if t == "lookup":
        return await basic.lookup(agent, call["query"])
    if t == "open_app":
        return await apps.open_app(agent, call["name"])
    if t == "open_folder":
        return await apps.open_folder(agent, call["name"])
    if t == "open_file":
        return await apps.open_file(agent, call["query"])
    if t == "web_search":
        return await apps.web_search(agent, call["query"])
    if t == "youtube":
        return await apps.youtube(agent, call["query"])
    if t == "play_music":
        return await apps.play_music(agent, call["query"])
    if t == "media":
        return await desk.media(agent, call["action"])
    if t == "type_text":
        return await desk.type_text(agent, call["text"])
    if t == "read_screen":
        return await ask.read_screen(agent, call.get("question") or "")
    if t == "clipboard":
        return await ask.clipboard(agent, call.get("question") or "")
    if t == "tell":
        return await desk.tell(agent, call["what"])
    if t == "show_desktop":
        return await desk.show_desktop(agent)
    if t == "screenshot":
        return await desk.screenshot(agent)
    return {"ok": False, "tool": t, "detail": "unknown tool"}


def tool_list_for_prompt() -> str:
    """The tool lines the agent prompts show, built from the registry so they never drift."""
    fields = {
        "note": '"text": "..."', "todo": '"text": "..."', "timer": '"seconds": N, "label": "..."',
        "open_url": '"url": "https://..."', "lookup": '"query": "..."', "open_app": '"name": "spotify"',
        "open_folder": '"name": "downloads"', "open_file": '"query": "part of the file name"',
        "web_search": '"query": "..."', "youtube": '"query": "..."', "play_music": '"query": "lo-fi beats"',
        "media": '"action": "play_pause"', "type_text": '"text": "..."', "read_screen": '"question": "..."',
        "clipboard": '"question": "..."', "tell": '"what": "time" | "date" | "battery" | "window"',
        "show_desktop": "", "screenshot": "",
    }
    lines = []
    for name, spec in TOOLS.items():
        f = fields.get(name, "")
        lines.append(f'- {{"tool": "{name}"{", " + f if f else ""}}}   {spec.help}')
    return "\n".join(lines)

