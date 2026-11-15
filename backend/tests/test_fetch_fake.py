"""fetch_bottle and the auto-rest on FakeArm. No hardware, no cameras."""
import threading
import time

import pytest

from backend.arm import FakeArm
from backend.arm import skills as S

SCALE = 0.002

HANDED = ["start", "seen", "at_target", "gripped", "lifted", "carrying", "offering", "handed_over", "returning", "done"]
PUT_BACK = ["start", "seen", "at_target", "gripped", "lifted", "carrying", "offering", "put_back", "returning", "done"]
MISSED = ["start", "seen", "at_target", "missed", "returning", "done"]
TURN_FAILED = ["start", "seen", "at_target", "gripped", "lifted", "carrying", "put_back", "returning", "done"]


def run(arm, skill, params=None):
    phases = []
    out = arm.run(skill, params or {}, phases.append)
    return out, phases


@pytest.mark.parametrize("outcome,phases,ok", [
    ("handed_over", HANDED, True),
    ("put_back", PUT_BACK, True),
    ("missed", MISSED, False),
    ("turn_failed", TURN_FAILED, False),
    ("no_bottle", ["start", "done"], False),
    ("out_of_reach", ["start", "seen", "done"], False),
])
def test_each_outcome(outcome, phases, ok):
    arm = FakeArm(time_scale=SCALE, fetch_outcome=outcome)
    out, got = run(arm, "fetch_bottle", {"pace": 0.6, "wait_s": 3})
    assert out["result"] == outcome and out["ok"] is ok
    assert got == phases
    assert set(got) <= set(S.PHASES)
    st = arm.status()
    assert st["at_rest"] is True and st["holding"] is False


def test_miss_grip_switch_also_misses_fetch():
    out, phases = run(FakeArm(time_scale=SCALE, miss_grip=True), "fetch_bottle")
    assert out["result"] == "missed" and phases == MISSED


def test_wrist_only_fallback():
    arm = FakeArm(time_scale=SCALE, overhead=False)
    out, phases = run(arm, "fetch_bottle", {"pace": 0.3, "wait_s": 5, "look_first": True})
    assert out["result"] == "handed_over" and phases == HANDED
    assert "middle grip" in out["detail"] and "cannot be measured" in out["detail"]
    arm.fetch_outcome = "no_bottle"
    out, phases = run(arm, "fetch_bottle")
    assert out["result"] == "no_bottle" and phases == ["start", "returning", "done"]
    arm.wrist_fallback = False
    out, phases = run(arm, "fetch_bottle")
    assert out["result"] == "no_bottle" and "fallback is off" in out["detail"] and phases == ["start", "done"]


def test_timings_follow_pace():
    for pace in (0.0, 1.0):
        for outcome in ("handed_over", "put_back", "missed", "turn_failed"):
            arm = FakeArm(time_scale=0.0, fetch_outcome=outcome)
            spent = []
            arm._wait = spent.append          # add up the pretend time instead of sleeping
            run(arm, "fetch_bottle", {"pace": pace, "wait_s": 2})
            want = S.fetch_nominal(pace, 2, False, outcome)
            assert abs(sum(spent) - want) < 0.2 * want, (pace, outcome, sum(spent), want)
    assert S.fetch_stretch(1.0) == 1.0 and abs(S.fetch_stretch(0.0) - 1.8) < 1e-9


@pytest.mark.parametrize("stop_at", ["start", "seen", "at_target", "gripped", "lifted", "carrying", "offering",
                                     "handed_over", "returning"])
def test_stop_at_each_stage_then_auto_rest(stop_at):
    arm = FakeArm(time_scale=SCALE)
    phases = []

    def on(p):
        phases.append(p)
        if p == stop_at:
            arm.stop()
    out = arm.run("fetch_bottle", {"pace": 1.0}, on)
    assert out["result"] == "stopped" and out["ok"] is False
    assert phases[-1] == stop_at and "done" not in phases
    holding = stop_at in ("gripped", "lifted", "carrying", "offering")
    assert arm.status()["holding"] is holding and arm.status()["at_rest"] is False
    # the next skill goes to rest first by itself; holding the bottle it puts it down first
    out, ph = run(arm, "wave")
    assert out["result"] == "done" and out["auto_rest"] is True
    assert ph[0] == "auto_rest" and ph[1:] == ["start", "at_target", "returning", "done"]
    assert ("put the bottle down" in out["detail"]) is holding
    assert arm.status()["holding"] is False


def test_rest_puts_a_held_bottle_down_first():
    arm = FakeArm(time_scale=SCALE)
    arm.run("fetch_bottle", {}, lambda p: arm.stop() if p == "carrying" else None)
    out, phases = run(arm, "rest")
    assert out["result"] == "done" and "put the bottle down" in out["detail"]
    assert phases == ["start", "put_back", "returning", "done"]
    out, phases = run(arm, "rest")
    assert out["detail"] == "at rest" and phases == ["start", "returning", "done"]


def test_busy_during_fetch():
    arm = FakeArm(time_scale=0.01)
    box = {}
    th = threading.Thread(target=lambda: box.update(out=arm.run("fetch_bottle", {"pace": 1.0}, None)))
    th.start()
    time.sleep(0.05)
    assert arm.run("gesture", {"style": "ban", "seconds": 2}, None)["result"] == "busy"
    assert arm.run("rest", {}, None)["result"] == "busy"
    arm.stop()
    th.join(5)
    assert box["out"]["result"] == "stopped"


def test_error_also_auto_rests():
    arm = FakeArm(time_scale=SCALE)

    def boom(*a, **k):
        raise RuntimeError("stalled at {1: 2200}")
    arm._skill_point = boom
    out, phases = run(arm, "point")
    assert out["result"] == "error" and phases == ["error"]
    del arm._skill_point
    out, phases = run(arm, "point")
    assert out["result"] == "done" and out["auto_rest"] is True and phases[0] == "auto_rest"
