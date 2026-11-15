"""A small pretend arm with the contract interface, used only when backend.arm is not importable yet.

No hardware, no serial port. It sleeps for roughly realistic durations and reports phases.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

_SKILLS = {"grip_bottle", "point", "half_reach", "wave", "rest", "gesture"}


class StandinArm:
    def __init__(self, speed: float = 1.0):
        self._lock = threading.Lock()
        self._busy = False
        self._stop = threading.Event()
        self.speed = speed  # divide durations by this (tests use a large number)
        self.log: list[tuple[str, dict]] = []

    def _sleep(self, s: float) -> bool:
        """Sleep s seconds; return False if stopped."""
        return not self._stop.wait(max(0.0, s / self.speed))

    def run(self, skill: str, params: dict, on_phase: Callable[[str], None]) -> dict:
        if skill not in _SKILLS:
            on_phase("error")
            return {"ok": False, "result": "error", "detail": f"unknown skill {skill}"}
        with self._lock:
            if self._busy:
                return {"ok": False, "result": "busy", "detail": "another skill is running"}
            self._busy = True
            self._stop.clear()
        self.log.append((skill, dict(params or {})))
        try:
            pace = float((params or {}).get("pace", 0.5))
            f = 1.6 - pace  # slower pace, longer moves
            steps: list[tuple[str, float]]
            if skill == "grip_bottle":
                hold = float(params.get("hold_s", 3))
                steps = [("start", 0.2), ("at_target", 2.5 * f)]
                if params.get("look_first"):
                    steps.append(("at_target", 1.0))
                steps += [("gripped", 1.0 * f), ("lifted", 1.0 * f + hold), ("put_back", 1.5 * f), ("returning", 1.5 * f)]
            elif skill == "point":
                steps = [("start", 0.2), ("at_target", 1.5 * f), ("returning", 2.0 + 1.2 * f)]
            elif skill == "half_reach":
                steps = [("start", 0.2), ("returning", 1.2 * f)]
            elif skill == "wave":
                steps = [("start", 0.2), ("at_target", 1.5), ("returning", 5.0)]
            elif skill == "gesture":
                secs = min(10.0, max(1.5, float(params.get("seconds", 2))))
                steps = [("start", 0.0), ("at_target", 0.4), ("returning", max(0.0, secs - 0.8))]
            else:  # rest
                steps = [("start", 0.0), ("returning", 0.3)]
            for phase, wait in steps:
                on_phase(phase)
                if not self._sleep(wait):
                    on_phase("done")
                    return {"ok": False, "result": "stopped", "detail": "stopped"}
            if not self._sleep(0.4):
                return {"ok": False, "result": "stopped", "detail": "stopped"}
            on_phase("done")
            return {"ok": True, "result": "done", "detail": f"pretend {skill}"}
        finally:
            with self._lock:
                self._busy = False

    def stop(self) -> None:
        self._stop.set()

    def status(self) -> dict:
        return {"connected": False, "busy": self._busy, "torque": False, "joints": {}, "standin": True}


def make_arm() -> StandinArm:
    return StandinArm()
