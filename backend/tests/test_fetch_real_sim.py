"""RealArm.fetch_bottle, rest and gestures on a simulated servo bus with pretend cameras.

Nothing here opens a serial port or a camera: SimBus stands in for scservo_sdk and FakeVision
for backend.vision. The checks: results and phases, the order of moves against
setup/tools/fetch_bottle.py, every goal inside the calibrated ranges, torque never switched off,
stop at every stage holds, and rest from any of those poses (putting a held bottle down first).
"""
import json
from pathlib import Path

import pytest

from backend.arm import skills as S
from backend.tests.simbus import CAP, make_real

SETUP = Path(__file__).resolve().parents[2] / "setup"
BT = json.load(open(SETUP / "bottle_teach.json"))
REST = {1: 2009, 2: 946, 3: 3190, 4: 2632}
OPEN, CLOSED, LIFT_UP = 1790, 1310, 170


def posture(r):
    """The script's posture(), written out again here so the test does not lean on the port."""
    m, f = BT["mid"], BT["far"]
    t = min(1.0, max(0.0, (r - m["r_cm"]) / (f["r_cm"] - m["r_cm"])))
    mix = lambda a, b: {k: round(a[k] + t * (b[k] - a[k])) for k in ("lift", "elbow", "wrist_flex")}
    return mix(m["via"], f["via"]), mix(m["grasp"], f["grasp"])


def script_moves(outcome, r=30.0, pan_b=1990, pan_p=2620, middle=False):
    """The goals of every move setup/tools/fetch_bottle.py makes, in order, for one outcome."""
    if middle:
        k = ("lift", "elbow", "wrist_flex")
        via, grasp = {x: BT["mid"]["via"][x] for x in k}, {x: BT["mid"]["grasp"][x] for x in k}
    else:
        via, grasp = posture(r)
    far = BT["far"]["grasp"]
    g = BT["mid"]["grasp"]
    tuck = {"lift": g["lift"] - 230, "elbow": g["elbow"], "wrist_flex": g["wrist_flex"]}
    three = lambda p: ("blend", ((2, p["lift"]), (3, p["elbow"]), (4, p["wrist_flex"])))
    rest = ("blend", ((3, REST[3]), (2, REST[2]), (1, REST[1]), (4, REST[4]), (6, CLOSED)))
    moves = [("blend", ((3, via["elbow"]), (2, via["lift"]), (1, pan_b), (4, via["wrist_flex"]), (6, OPEN))),
             three(grasp), ("close",)]
    if outcome == "missed":
        return moves + [("blend", ((6, OPEN),)), three(via), rest]
    moves += [three(tuck), ("blend", ((1, pan_p), (2, far["lift"] - 120), (3, far["elbow"]), (4, far["wrist_flex"])))]
    if outcome == "turn_failed":
        return moves + [three(tuck), ("blend", ((3, g["elbow"]), (4, g["wrist_flex"]))), ("blend", ((2, g["lift"]),)),
                        ("blend", ((6, OPEN),)), ("blend", ((2, g["lift"] - 220),)), rest]
    moves.append(("offer",))
    if outcome == "handed_over":
        return moves + [("blend", ((6, OPEN),)), ("blend", ((2, far["lift"] - 300),)), rest]
    return moves + [three(tuck), ("blend", ((1, pan_b),)),
                    ("blend", ((2, grasp["lift"] - LIFT_UP), (3, grasp["elbow"]), (4, grasp["wrist_flex"]))),
                    ("blend", ((2, grasp["lift"]),)), ("blend", ((6, OPEN),)), three(via), rest]


def moves_of(arm):
    out = []
    for m in arm.trace:
        if m[0] == "offer":
            out.append(("offer",))
        elif m[0] != "wrist_sweep":
            out.append(m)
    return out


def run(arm, skill="fetch_bottle", params=None, on=None):
    phases = []

    def cb(p):
        phases.append(p)
        if on:
            on(p)
    return arm.run(skill, params or {"pace": 1.0, "wait_s": 3}, cb), phases


def safe(bus):
    assert bus.bad_goals() == [], "a goal outside the calibrated range"
    assert bus.torque_off_writes() == [], "torque was switched off"


def at_rest(bus):
    return all(abs(bus.pos(i) - REST[i]) <= 20 for i in REST)


HANDED = ["start", "seen", "at_target", "gripped", "lifted", "carrying", "offering", "handed_over", "returning", "done"]
PUT_BACK = HANDED[:7] + ["put_back", "returning", "done"]


def test_handed_over_by_tug_matches_the_script():
    arm, bus, vis, clock = make_real()

    def on(p):
        if p == "offering":   # a person pulls the bottle a second after it is held out
            bus.at(clock.t + 1.0, lambda: setattr(bus.s[6], "ext", 15.0))
    out, phases = run(arm, on=on)
    assert out["result"] == "handed_over" and out["ok"] and "tug" in out["detail"]
    assert phases == HANDED
    assert moves_of(arm) == script_moves("handed_over")
    safe(bus)
    assert at_rest(bus) and not bus.holding
    # the base ran at full torque for the fetch and is back to 500 at rest
    base = [v for _, i, a, v in bus.writes if i == 1 and a == 48]
    assert base[-2:] == [1000, 500]


def test_handed_over_when_the_wrist_camera_sees_a_hand():
    arm, bus, vis, clock = make_real(hand_after=1.5)
    out, phases = run(arm)
    assert out["result"] == "handed_over" and "hand" in out["detail"]
    assert phases == HANDED and vis.watchers and vis.watchers[0].stopped
    safe(bus)


def test_nobody_takes_it_so_it_goes_back():
    arm, bus, vis, clock = make_real()
    t0 = clock.t
    out, phases = run(arm, params={"pace": 1.0, "wait_s": 4})
    assert out["result"] == "put_back" and out["ok"] and "4 s" in out["detail"]
    assert phases == PUT_BACK
    assert moves_of(arm) == script_moves("put_back")
    safe(bus)
    assert at_rest(bus) and not bus.holding
    assert clock.t - t0 < S.fetch_nominal(1.0, 4, False, "put_back") + 8


def test_missed_grip():
    arm, bus, vis, clock = make_real(grip_ok=False)
    out, phases = run(arm)
    assert out["result"] == "missed" and not out["ok"]
    assert phases == ["start", "seen", "at_target", "missed", "returning", "done"]
    assert moves_of(arm) == script_moves("missed")
    safe(bus)
    assert at_rest(bus)


def test_turn_that_cannot_finish_puts_it_down_and_goes_home():
    arm, bus, vis, clock = make_real()
    bus.s[1].block = (0, 2200)          # something in the way of the base at pan 2200
    out, phases = run(arm)
    assert out["result"] == "turn_failed" and "2200" in out["detail"]
    assert phases == ["start", "seen", "at_target", "gripped", "lifted", "carrying", "put_back", "returning", "done"]
    assert moves_of(arm) == script_moves("turn_failed")
    safe(bus)
    assert at_rest(bus) and not bus.holding


def test_not_seen_and_out_of_reach_do_not_move():
    arm, bus, vis, clock = make_real(r_cm=None)
    out, phases = run(arm)
    assert out["result"] == "no_bottle" and phases == ["start", "done"]
    assert moves_of(arm) == []
    arm, bus, vis, clock = make_real(r_cm=52.0)
    out, phases = run(arm)
    assert out["result"] == "out_of_reach" and "52 cm" in out["detail"] and phases == ["start", "seen", "done"]
    assert moves_of(arm) == []
    safe(bus)


def test_person_not_seen_goes_to_the_right():
    arm, bus, vis, clock = make_real(person_pan=None)
    out, _ = run(arm, on=lambda p: p == "offering" and bus.at(clock.t + 0.8, lambda: setattr(bus.s[1], "ext", 20.0)))
    assert out["result"] == "handed_over" and "person not seen" in out["detail"]
    pan_p = round(651.8986469044033 * 1.2 + 2101.1990247998265)
    assert moves_of(arm) == script_moves("handed_over", pan_p=pan_p)


@pytest.mark.parametrize("pace", [0.0, 0.5, 1.0])
def test_pace_stretches_the_legs(pace):
    arm, bus, vis, clock = make_real()
    t0 = clock.t
    out, _ = run(arm, params={"pace": pace, "wait_s": 2, "look_first": pace == 0.0})
    assert out["result"] == "put_back"
    want = S.fetch_nominal(pace, 2, pace == 0.0, "put_back")
    assert 0.8 * want < clock.t - t0 < want + 10, (pace, clock.t - t0, want)
    safe(bus)


# wrist camera only
def test_wrist_only_finds_bottle_and_person_and_uses_the_middle_grip():
    arm, bus, vis, clock = make_real(overhead=False, wrist_bottle_pan=2150, wrist_person_pan=2700)
    out, phases = run(arm, on=lambda p: p == "offering" and bus.at(clock.t + 1.0, lambda: setattr(bus.s[6], "ext", 12.0)))
    assert out["result"] == "handed_over" and phases == HANDED
    assert "wrist camera only" in out["detail"] and "middle grip" in out["detail"] and "cannot be measured" in out["detail"]
    trace = list(arm.trace)
    i = trace.index(("wrist_sweep",))
    assert trace[i + 1] == ("blend", ((4, 2150),))          # wrist tilted up to look across the desk
    sweep = [m for m in trace[i + 1:] if m[0] == "blend" and len(m[1]) == 1 and m[1][0][0] == 1]
    pans = [REST[1]] + [m[1][0][1] for m in sweep]
    steps = [abs(b - a) for a, b in zip(pans, pans[1:])]
    assert steps[0] <= 400 and all(s <= 300 for s in steps[1:]), steps   # small steps only
    pan_b = moves_of(arm)[len(sweep) + 1][1][2][1]          # pan of the out leg
    assert abs(pan_b - 2150) <= 15
    out_leg = moves_of(arm)[len(sweep) + 1:]
    pan_p = out_leg[4][1][0][1]
    assert abs(pan_p - 2700) <= 15
    assert out_leg == script_moves("handed_over", pan_b=pan_b, pan_p=pan_p, middle=True)
    safe(bus)
    assert at_rest(bus)


def test_wrist_only_person_not_found_goes_45_degrees_right():
    arm, bus, vis, clock = make_real(overhead=False, wrist_bottle_pan=1800, wrist_person_pan=None)
    out, _ = run(arm)
    assert out["result"] == "put_back" and "45 degrees right" in out["detail"]
    assert len(vis.wrist_asks) <= 5 + 3                      # five stops, three centring tries at most
    safe(bus)
    assert at_rest(bus)


def test_wrist_only_nothing_seen_returns_to_rest():
    arm, bus, vis, clock = make_real(overhead=False, wrist_bottle_pan=None)
    out, phases = run(arm)
    assert out["result"] == "no_bottle" and phases == ["start", "returning", "done"]
    assert at_rest(bus)
    arm, bus, vis, clock = make_real(overhead=False, wrist_picture=False)
    out, phases = run(arm)
    assert out["result"] == "no_bottle" and "no camera gives a picture" in out["detail"]
    assert at_rest(bus)


def test_wrist_fallback_can_be_switched_off():
    arm, bus, vis, clock = make_real(overhead=False)
    arm.wrist_fallback = False
    out, phases = run(arm)
    assert out["result"] == "no_bottle" and "fallback is off" in out["detail"] and vis.wrist_asks == []
    assert moves_of(arm) == []


# stop and rest
STAGES = ["start", "seen", "at_target", "gripped", "lifted", "carrying", "offering", "handed_over", "returning"]


@pytest.mark.parametrize("stage", STAGES)
def test_stop_at_each_stage_holds_then_rest_is_safe(stage):
    arm, bus, vis, clock = make_real()

    def on(p):
        if p == "offering" and stage in ("handed_over", "returning"):
            bus.at(clock.t + 0.8, lambda: setattr(bus.s[6], "ext", 15.0))
        if p == stage:
            arm.stop()
    out, phases = run(arm, on=on)
    assert out["result"] == "stopped" and phases[-1] == stage
    # held where it was: no goal written after the stop except the hold itself
    held = {i: bus.s[i].goal for i in range(1, 6)}
    clock.sleep(2.0)
    assert all(abs(bus.pos(i) - held[i]) <= 25 for i in range(1, 5)), (stage, {i: bus.pos(i) for i in range(1, 5)}, held)
    holding = bus.holding
    # at_target: the jaws were already closing, and a stop leaves the gripper goal alone, so they
    # close on the cap and hold it
    assert holding is (stage in ("at_target", "gripped", "lifted", "carrying", "offering"))
    safe(bus)
    # rest from here, by asking for rest directly
    out, ph = run(arm, "rest", {})
    assert out["result"] == "done", out
    assert ("put the bottle down" in out["detail"]) is holding
    assert ph == (["start", "put_back", "returning", "done"] if holding else ["start", "returning", "done"])
    safe(bus)
    assert at_rest(bus) and not bus.holding


@pytest.mark.parametrize("stage", ["at_target", "carrying"])
def test_arm_auto_rests_before_the_next_skill(stage):
    arm, bus, vis, clock = make_real()
    run(arm, on=lambda p: p == stage and arm.stop())
    out, ph = run(arm, "point", {"pace": 1.0})
    assert out["result"] == "done" and out["auto_rest"] is True and "rest first" in out["detail"]
    assert ph == ["auto_rest", "start", "at_target", "returning", "done"]
    safe(bus)
    assert at_rest(bus) and not bus.holding


def test_busy_on_the_real_arm():
    import threading
    arm, bus, vis, clock = make_real()
    got = {}

    def on(p):
        if p == "carrying":
            got["busy"] = arm.run("gesture", {"style": "ban", "seconds": 2}, None)
            t = threading.Thread(target=lambda: got.update(rest=arm.run("rest", {}, None)))
            t.start()
            t.join()
    out, _ = run(arm, on=on)
    assert got["busy"]["result"] == "busy" and got["rest"]["result"] == "busy"
    assert out["result"] == "put_back"


def test_rest_lifts_open_jaws_clear_before_folding():
    arm, bus, vis, clock = make_real(grip_ok=False)
    # stop during the final approach: jaws open around the cap, arm reaching out
    run(arm, on=lambda p: p == "seen" and bus.at(clock.t + 4.0, arm.stop))
    lift_before = bus.pos(2)
    arm.trace.clear()
    out, _ = run(arm, "rest", {})
    first = list(arm.trace)[0]
    assert first[0] == "blend" and len(first[1]) == 1 and first[1][0][0] == 2   # shoulder only, straight up
    assert abs(first[1][0][1] - (lift_before - 200)) <= 20
    assert at_rest(bus)


# gestures: the whole move lasts about as long as the line
@pytest.mark.parametrize("style", S.STYLES)
@pytest.mark.parametrize("seconds", [1.5, 3.0, 5.0, 8.0])
def test_gesture_on_the_bus_fits_the_line(style, seconds):
    arm, bus, vis, clock = make_real()
    t0 = clock.t
    out, phases = run(arm, "gesture", {"style": style, "seconds": seconds})
    took = clock.t - t0
    assert out["result"] == "done" and phases == ["start", "at_target", "returning", "done"]
    assert took - seconds <= 1.5, (style, seconds, took)
    g = S.gesture_plan(style, seconds)
    assert abs(took - (g["rise_s"] + g["talk_s"] + g["back_s"])) <= 0.5, (style, seconds, took)
    safe(bus)
