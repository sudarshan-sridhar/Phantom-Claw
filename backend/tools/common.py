"""Shared bits for the laptop tools: the dry-run switch and the result shapes."""
from __future__ import annotations

import os
from typing import Awaitable, Callable

Emit = Callable[[dict], Awaitable[None]]

# Dry run: every tool works out what it would do (which shortcut, which file, which address)
# but nothing is opened, typed, pressed or captured. The demo and the tests use it.
_STATE = {"dry": os.environ.get("PHANTOM_DRY_TOOLS", "").strip() == "1"}


def set_dry_run(on: bool) -> None:
    _STATE["dry"] = bool(on)


def is_dry() -> bool:
    return _STATE["dry"]


def ok(tool: str, **extra) -> dict:
    return {"ok": True, "tool": tool, **extra}


def fail(tool: str, detail: str, **extra) -> dict:
    return {"ok": False, "tool": tool, "detail": detail, **extra}


def would(tool: str, what: str, **extra) -> dict:
    """What a dry run reports instead of doing the thing."""
    return {"ok": True, "tool": tool, "dry": True, "would": what, **extra}
