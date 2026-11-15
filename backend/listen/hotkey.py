"""The Ctrl+Shift shortcut: press the two modifier keys together, nothing else, let go quickly.

The `keyboard` package installs a low level Windows hook, which does not need admin rights.
Any other key pressed during the chord (Ctrl+Shift+Space, Ctrl+Shift+T and so on) cancels it,
so normal shortcuts keep working.
"""
from __future__ import annotations

import time
from typing import Callable

from . import config

_CTRL = {"ctrl", "left ctrl", "right ctrl", "control"}
_SHIFT = {"shift", "left shift", "right shift"}


class ChordDetector:
    """Feed it key events; it calls on_chord when a clean Ctrl+Shift press and release happens."""

    def __init__(self, on_chord: Callable[[], None], window_ms: int = config.CHORD_MS, clock=time.monotonic):
        self.on_chord = on_chord
        self.window = window_ms / 1000
        self.clock = clock
        self.down: set[str] = set()
        self.t_first = 0.0
        self.both = False
        self.spoiled = False

    def event(self, name: str, is_down: bool) -> None:
        name = (name or "").lower()
        kind = "ctrl" if name in _CTRL else "shift" if name in _SHIFT else None
        if kind is None:
            if is_down and self.down:
                self.spoiled = True
            return
        if is_down:
            if not self.down:
                self.t_first = self.clock()
                self.both = False
                self.spoiled = False
            self.down.add(kind)  # held keys repeat, the set ignores that
            if self.down == {"ctrl", "shift"}:
                self.both = True
        else:
            self.down.discard(kind)
            if not self.down:
                clean = self.both and not self.spoiled and self.clock() - self.t_first <= self.window
                self.both = False
                if clean:
                    self.on_chord()


def start_hotkey(on_chord: Callable[[], None]) -> tuple[bool, str]:
    """Hook the keyboard. Returns (ok, message)."""
    try:
        import keyboard
    except Exception as e:  # noqa: BLE001
        return False, f"keyboard package not importable ({type(e).__name__})"
    det = ChordDetector(on_chord)

    def hook(ev):
        det.event(ev.name or "", ev.event_type == "down")

    try:
        keyboard.hook(hook)
    except Exception as e:  # noqa: BLE001
        return False, f"keyboard hook failed ({type(e).__name__}: {str(e)[:100]})"
    return True, "Ctrl+Shift (tap both, nothing else) toggles listening"
