"""Tests for FakeArm and the skill checks. No hardware. Run from the project root:
    C:\\Users\\sudar\\.conda\\envs\\lerobot\\python.exe -m pytest backend/tests/test_arm_fake.py -q
"""
import threading
import time

import pytest

from backend.arm import FakeArm, SKILLS, STYLES, make_arm, nominal_duration, validate
from backend.arm import skills as S

SCALE = 0.002   # 1 s of arm time takes 2 ms


def run(arm, skill, params=None):
    phases = []
    out = arm.run(skill, params or {}, phases.append)
    return out, phases


def run_in_thread(arm, skill, params=None):
    box = {"phases": []}
    th = threading.Thread(target=lambda: box.update(out=arm.run(skill, params or {}, box["phases"].append)))
    th.start()
    return th, box


GRIP = ["start", "at_target", "gripped", "lifted", "put_back", "returning", "done"]
MOVE = ["start", "at_target", "returning", "done"]


@pytest.mark.parametrize("skill,params,expected", [
    ("grip_bottle", {"pace": 0.5, "hold_s": 5}, GRIP),
    ("grip_bottle", {"pace": 0.2, "hold_s": 4, "look_first": True}, GRIP),
    ("grip_bottle", {"pace": 1.0, "hold_s": 3, "flourish": True}, GRIP),
    ("point", {"pace": 0.8}, MOVE),
    ("half_reach", {"pace": 0.0}, MOVE),
    ("wave", {}, MOVE),
    ("rest", {}, ["start", "returning", "done"]),
] + [("gesture", {"style": s, "seconds": 3}, MOVE) for s in STYLES])
def test_phase_order(skill, params, expected):
    out, phases = run(FakeArm(time_scale=SCALE), skill, params)
    assert out == {"ok": True, "result": "done", "detail": out["detail"]}
    assert phases == expected
    assert set(phases) <= set(S.PHASES)


def test_missed_grip():
    arm = FakeArm(time_scale=SCALE, miss_grip=True)
    out, phases = run(arm, "grip_bottle", {"pace": 0.5})
    assert out["result"] == "missed" and out["ok"] is False
    assert phases == ["start", "at_target", "missed", "returning", "done"]
    arm.miss_grip = False
    out, _ = run(arm, "grip_bottle", {})
    assert out["result"] == "done"


def test_busy_refused():
    arm = FakeArm(time_scale=0.02)
    th, box = run_in_thread(arm, "wave")
    time.sleep(0.05)
    assert arm.status()["busy"] is True
    out, phases = run(arm, "point", {})
    assert out["result"] == "busy" and out["ok"] is False and phases == []
    out, _ = run(arm, "gesture", {"style": "ban", "seconds": 2})
    assert out["result"] == "busy"
    th.join(5)
    assert box["out"]["result"] == "done"
    assert arm.status()["busy"] is False


@pytest.mark.parametrize("skill,params", [
    ("grip_bottle", {"hold_s": 8}),
    ("gesture", {"style": "merlin", "seconds": 10}),
])
def test_stop_holds_then_needs_rest(skill, params):
    arm = FakeArm(time_scale=0.02)
    th, box = run_in_thread(arm, skill, params)
    time.sleep(0.08)
    t = time.time()
    arm.stop()
    th.join(5)
    assert time.time() - t < 0.2, "stop must be quick"
    assert box["out"]["result"] == "stopped" and box["out"]["ok"] is False
    assert "done" not in box["phases"] and "returning" not in box["phases"]
    # not at rest any more: everything but rest is refused
    out, phases = run(arm, "point", {})
    assert out["result"] == "error" and "rest" in out["detail"]
    assert phases == ["error"]
    arm.time_scale = SCALE
    out, _ = run(arm, "rest")
    assert out["result"] == "done"
    out, _ = run(arm, "point", {})
    assert out["result"] == "done"


def test_stop_when_idle_does_nothing_lasting():
    arm = FakeArm(time_scale=SCALE)
    arm.stop()
    out, _ = run(arm, "point", {})
    assert out["result"] == "done"


def test_unknown_skill():
    out, phases = run(FakeArm(time_scale=SCALE), "throw_bottle", {})
    assert out["result"] == "error" and phases == []


def test_broken_listener_does_not_break_motion():
    def bad(_):
        raise RuntimeError("socket closed")
    out = FakeArm(time_scale=SCALE).run("point", {}, bad)
    assert out["result"] == "done"
    assert FakeArm(time_scale=SCALE).run("rest", {}, None)["result"] == "done"


def test_validate_clamps():
    assert validate("grip_bottle", {"pace": 5, "hold_s": 20}) == \
        {"pace": 1.0, "hold_s": 8.0, "flourish": False, "look_first": False}
    assert validate("grip_bottle", {"pace": -1, "hold_s": 0, "flourish": "true", "look_first": 1}) == \
        {"pace": 0.0, "hold_s": 1.0, "flourish": True, "look_first": True}
    assert validate("grip_bottle", {"pace": "fast", "hold_s": float("nan"), "flourish": "maybe"}) == \
        {"pace": 0.5, "hold_s": 5.0, "flourish": False, "look_first": False}
    assert validate("grip_bottle", {"pace": True}) ["pace"] == 0.5
    assert validate("point", {"pace": "0.3", "hold_s": 4, "skill": "point"}) == {"pace": 0.3}
    assert validate("wave", {"pace": 1}) == {}
    assert validate("rest", None) == {}
    assert validate("gesture", {"style": " King ", "seconds": 30}) == {"style": "king", "seconds": 10.0}
    assert validate("gesture", {"style": "zeldris", "seconds": 0.2}) == {"style": "gowther", "seconds": 1.5}
    assert set(SKILLS) == {"grip_bottle", "point", "half_reach", "wave", "rest", "gesture"}
    with pytest.raises(ValueError):
        validate("dance", {})


def test_make_arm_is_fake(monkeypatch):
    monkeypatch.delenv("PHANTOM_ARM", raising=False)
    assert isinstance(make_arm(), FakeArm)
    monkeypatch.setenv("PHANTOM_ARM", "fake")
    assert isinstance(make_arm(time_scale=SCALE), FakeArm)


def test_status_shape():
    st = FakeArm().status()
    assert {"connected", "busy", "torque", "joints"} <= set(st)
    assert set(st["joints"]) == {"shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"}


def test_durations_are_realistic():
    assert 12 <= nominal_duration("grip_bottle", {"pace": 0.5, "hold_s": 0}) <= 16   # plus hold_s (clamped to 1)
    assert 9.5 <= nominal_duration("grip_bottle", {"pace": 1.0, "hold_s": 1}) - 1 <= 11
    assert nominal_duration("grip_bottle", {"pace": 0.0}) > nominal_duration("grip_bottle", {"pace": 1.0})
    assert 7 <= nominal_duration("point", {}) <= 9
    assert 5 <= nominal_duration("half_reach", {}) <= 7
    assert 13 <= nominal_duration("wave", {}) <= 15
    assert 4.5 <= nominal_duration("rest", {}) <= 5.5
    for style in STYLES:
        for sec in (1.5, 4, 10):
            extra = nominal_duration("gesture", {"style": style, "seconds": sec}) - sec
            assert 0 <= extra <= 4.0, (style, sec, extra)


def test_fake_takes_its_time():
    arm = FakeArm(time_scale=0.01)
    t = time.time()
    run(arm, "point", {"pace": 0.5})
    assert 0.06 <= time.time() - t <= 0.3


# gesture plans: the limits from the contract change
@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("seconds", [1.5, 3, 6, 10])
def test_gesture_plan_limits(style, seconds):
    g = S.gesture_plan(style, seconds)
    rest, photo = S.REST_POSE, S.PHOTO_POSE
    pose = g["pose"]
    assert 0 < g["f"] <= 1.0
    if style in ("elizabeth", "king"):
        assert g["f"] <= 0.5
    # every pose between rest and the photo pose for shoulder and elbow
    assert rest[2] <= pose[2] <= photo[2]
    assert photo[3] <= pose[3] <= rest[3]
    assert abs(pose[4] - g["flex_anchor"]) <= 180
    # rise and return no faster than the caps
    assert (pose[2] - rest[2]) / ((1 - S.LEAD) * g["rise_s"]) <= S.RISE_V[2] + 1
    assert (rest[3] - pose[3]) / g["rise_s"] <= S.RISE_V[3] + 1
    base = {**pose, 5: 2030, 6: g["grip"]}
    steps = S.gesture_steps(style, g["talk_s"], base, g["flex_anchor"])
    assert steps == sorted(steps, key=lambda s: (s[0], s[1]))
    assert all(0 <= t <= g["talk_s"] for t, *_ in steps)
    for t, j, goal, speed, acc in steps:
        assert 40 <= speed <= 1000 and 1 <= acc <= 20
        if j == 1:
            assert abs(goal - photo[1]) <= 250
        elif j == 2:
            assert rest[2] <= goal <= photo[2]
        elif j == 3:
            assert photo[3] <= goal <= rest[3]
        elif j == 4:
            assert abs(goal - g["flex_anchor"]) <= 180
        elif j == 5:
            assert 1100 <= goal <= 2960
        else:
            assert 1310 <= goal <= 1900
    # every joint the style moves ends back at the talking pose
    last = {}
    for t, j, goal, *_ in steps:
        last[j] = goal
    for j, goal in last.items():
        if j != 6:
            assert goal == S.gesture_clamp(j, base[j], g["flex_anchor"])
