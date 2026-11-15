"""A small pretend arm with the contract interface, used only when backend.arm is not importable yet.

No hardware, no serial port. It sleeps for roughly realistic durations and reports phases.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

_SKILLS = {"grip_bottle", "fetch_bottle", "point", "half_reach", "wave", "rest", "gesture",
           "point_at", "pick", "place", "hand_over"}

# What the stand-in "sees" for look (the same six things as FakeArm's pretend desk).
_SCENE = [
    {"name": "water bottle", "x_cm": 2.0, "y_cm": 30.0, "dist_cm": 30.1, "bearing": 3.8, "tier": "tall", "graspable": True},
    {"name": "juice can", "x_cm": -12.0, "y_cm": 24.0, "dist_cm": 26.8, "bearing": -26.6, "tier": "can", "graspable": True},
    {"name": "red bull can", "x_cm": 14.0, "y_cm": 26.0, "dist_cm": 29.5, "bearing": 28.3, "tier": "can", "graspable": True},
    {"name": "cup", "x_cm": -6.0, "y_cm": 36.0, "dist_cm": 36.5, "bearing": -9.5, "tier": "can", "graspable": True},
    {"name": "sharpie", "x_cm": 9.0, "y_cm": 34.0, "dist_cm": 35.2, "bearing": 14.8, "tier": "low", "graspable": True,
     "axis_deg": 30.0},
    {"name": "scissors", "x_cm": -15.0, "y_cm": 31.0, "dist_cm": 34.4, "bearing": -25.8, "tier": "low", "graspable": True,
     "axis_deg": -60.0},
]


class StandinArm:
    def __init__(self, speed: float = 1.0):
        self._lock = threading.Lock()
        self._busy = False
        self._stop = threading.Event()
        self.speed = speed  # divide durations by this (tests use a large number)
        self.log: list[tuple[str, dict]] = []    # every skill that moves (look is not logged)
        self.held: dict | None = None

    def _sleep(self, s: float) -> bool:
        """Sleep s seconds; return False if stopped."""
        return not self._stop.wait(max(0.0, s / self.speed))

    def run(self, skill: str, params: dict, on_phase: Callable[[str], None]) -> dict:
        if skill == "look":
            # no motion: the pretend camera answers at once
            on_phase("start")
            on_phase("seen")
            on_phase("done")
            return {"ok": True, "result": "done", "detail": "pretend look", "scene": [dict(i) for i in _SCENE],
                    "person": {"x_cm": 56.0, "y_cm": 21.5, "bearing": 69.0}}
        if skill not in _SKILLS:
            on_phase("error")
            return {"ok": False, "result": "error", "detail": f"unknown skill {skill}"}
        if skill in ("place", "hand_over") and self.held is None:
            return {"ok": False, "result": "nothing_held", "detail": "not holding anything"}
        if self.held is not None and skill == "gesture":
            return {"ok": True, "result": "done", "detail": "gesture skipped while holding", "skipped": True}
        if self.held is not None and skill not in ("place", "hand_over", "rest"):
            return {"ok": False, "result": "hands_full", "detail": "holding something"}
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
            elif skill == "fetch_bottle":
                wait = float(params.get("wait_s", 3))
                steps = [("start", 0.2), ("seen", 3.0), ("at_target", 3.0 * f)]
                if params.get("look_first"):
                    steps.append(("at_target", 1.0))
                steps += [("gripped", 0.8), ("lifted", 1.9 * f), ("carrying", 2.0 * f), ("offering", 0.6 + wait / 2),
                          ("handed_over", 1.5 * f), ("returning", 4.5 * f)]
            elif skill == "point":
                steps = [("start", 0.2), ("at_target", 1.5 * f), ("returning", 2.0 + 1.2 * f)]
            elif skill == "half_reach":
                steps = [("start", 0.2), ("returning", 1.2 * f)]
            elif skill == "wave":
                steps = [("start", 0.2), ("at_target", 1.5), ("returning", 5.0)]
            elif skill == "gesture":
                secs = min(10.0, max(1.5, float(params.get("seconds", 2))))
                steps = [("start", 0.0), ("at_target", 0.4), ("returning", max(0.0, secs - 0.8))]
            elif skill == "point_at":
                steps = [("start", 0.2), ("seen", 0.0), ("at_target", 2.0 * f), ("returning", 2.0 * f)]
            elif skill == "pick":
                steps = [("start", 0.2), ("seen", 0.0), ("at_target", 3.0 * f), ("gripped", 1.0), ("lifted", 1.5 * f)]
            elif skill == "place":
                steps = [("start", 0.2), ("seen", 0.0), ("carrying", 2.0 * f), ("at_target", 1.5 * f),
                         ("put_back", 0.8), ("returning", 3.0 * f)]
            elif skill == "hand_over":
                steps = [("start", 0.2), ("seen", 0.0), ("carrying", 2.0 * f), ("offering", 1.5),
                         ("handed_over", 1.0), ("returning", 3.0 * f)]
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
            if skill == "pick":
                name = str((params or {}).get("target") or "item")
                self.held = {"name": name}
                return {"ok": True, "result": "holding", "detail": f"pretend pick {name}", "held": {"name": name}}
            if skill in ("place", "hand_over", "rest"):
                was, self.held = self.held, None
                if skill == "place":
                    return {"ok": True, "result": "placed", "detail": "pretend place"}
                if skill == "hand_over":
                    return {"ok": True, "result": "handed_over", "detail": f"pretend hand over {was['name']}"}
            return {"ok": True, "result": "handed_over" if skill == "fetch_bottle" else "done", "detail": f"pretend {skill}"}
        finally:
            with self._lock:
                self._busy = False

    def stop(self) -> None:
        self._stop.set()

    def status(self) -> dict:
        return {"connected": False, "busy": self._busy, "torque": False, "joints": {}, "standin": True,
                "held": dict(self.held) if self.held else None}


def make_arm() -> StandinArm:
    return StandinArm()
