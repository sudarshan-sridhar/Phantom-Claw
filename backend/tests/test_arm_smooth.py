"""Smooth motion on RealArm's simulated bus: the motion trace, trace_report's numbers, and that
pick, place and hand_over stop only where they mean to (a picture, the jaws, the offer), write one
goal per joint per leg, and always with a soft ramp. No hardware."""
import json
import shutil
from pathlib import Path

import pytest

from backend.arm import desk
from backend.arm import trace_report as R
from backend.arm.real import STOP_ACC
from backend.tests.simbus import BASE_FILES, make_real

DESK = [{"name": "water bottle", "x_cm": 2.0, "y_cm": 30.0, "tier": "tall"},
        {"name": "cup", "x_cm": -8.0, "y_cm": 26.0, "tier": "can", "width_cm": 8.0}]
LIVE = Path(__file__).resolve().parents[2] / "setup"
ON_PURPOSE = {"look", "look_first", "aim", "picture", "settle", "grip", "release", "offer", "point", "hold"}


def traced(seq, sag=0.0, setup_dir=None, items=DESK, **kw):
    """Run skills in a row with the trace on. Returns (results, arm trace, truth trace, bus, vis)."""
    arm, bus, vis, clock = make_real(scene_items=items, setup_dir=setup_dir, **kw)
    for j in (2, 3):
        bus.s[j].sag = sag
    bus.truth = []
    arm.trace_start()
    res = []
    for sk, p in seq:
        if sk == "hand_over":
            vis.hand_after = 0.5
        res.append(arm.run(sk, p, None)["result"])
    tr = arm.trace_stop()
    return res, tr, bus.truth_trace(tr), bus, vis, arm


@pytest.mark.parametrize("pace,sag", [(1.0, 0.0), (0.0, 0.0), (1.0, 30.0), (0.5, 60.0)])
@pytest.mark.parametrize("second", ["place", "hand_over"])
def test_pick_then_place_or_hand_over_stop_only_on_purpose(pace, sag, second):
    p2 = {"where": {"x_cm": -10, "y_cm": 30}} if second == "place" else {"wait_s": 3}
    res, tr, truth, bus, _, _ = traced([("pick", {"target": "water bottle", "pace": pace}),
                                        (second, {**p2, "pace": pace})], sag=sag)
    assert res == ["holding", "placed" if second == "place" else "handed_over"]
    assert len(tr["rows"]) > 300 and {"skill", "phase", "leg", "goal", "hold", "end"} <= {e[0] for e in tr["events"]}
    for t in (tr, truth):                 # what the arm read, and where the servos really were
        rep = R.analyse(t)
        assert R.unplanned(rep) == [], (pace, sag, R.unplanned(rep))
        by = {sk["skill"]: sk for sk in rep["skills"]}
        whys = {s["why"] for sk in rep["skills"] for s in sk["arm_stops"]}
        assert whys <= ON_PURPOSE
        assert "grip" in {s["why"] for s in by["pick"]["arm_stops"]}
        assert ("release" if second == "place" else "offer") in {s["why"] for s in by[second]["arm_stops"]}
        # one goal per joint per leg (two with the settle push before the jaws close or open)
        for sk in rep["skills"]:
            assert max(sk["max_goals_per_leg"].values()) <= 2, sk["max_goals_per_leg"]


@pytest.mark.parametrize("skill,params", [("wave", {}), ("rest", {})] + [
    ("gesture", {"style": s, "seconds": sec}) for s in ("escanor", "elizabeth", "ban", "king", "gowther", "merlin")
    for sec in (2.0, 6.0)])
def test_wave_rest_and_gestures_never_stop_and_go(skill, params):
    arm, bus, vis, clock = make_real()
    if skill == "rest":                  # from a stop half way out
        arm.run("point", {"pace": 1.0}, lambda p: p == "at_target" and arm.stop())
    bus.truth = []
    arm.trace_start()
    assert arm.run(skill, params, None)["result"] == "done"
    tr = arm.trace_stop()
    for t in (tr, bus.truth_trace(tr)):
        rep = R.analyse(t)
        assert R.unplanned(rep) == [], R.unplanned(rep)
        assert {s["why"] for s in rep["skills"][0]["arm_stops"]} <= {"pose"}


def test_gesture_keys_fill_their_gap():
    from backend.arm import skills as S
    for d, gap, acc in ((400, 1.6, 8), (90, 0.8, 6), (110, 2.1, 5)):
        v = S.fill_speed(d, gap, acc)
        assert abs(S.move_time(d, d / v, acc) - gap) < 0.05      # arrives as the next key starts
    assert S.fill_speed(500, 0.2, 14) == round(500 / 0.16)       # too short to arrive: the old speed


def test_every_goal_has_a_soft_ramp():
    res, tr, _, bus, _, _ = traced([("pick", {"target": "water bottle", "pace": 0.5}),
                                    ("hand_over", {"wait_s": 3, "pace": 0.5}),
                                    ("gesture", {"style": "ban", "seconds": 3}), ("wave", {}), ("rest", {})])
    assert res == ["holding", "handed_over", "done", "done", "done"]
    acc = {}
    for _, i, a, v in bus.writes:
        if a == 41:
            assert 1 <= v <= STOP_ACC, (i, v)
            acc[i] = v
        elif a == 42:
            assert acc.get(i, 0) >= 1, f"goal to {i} with no ramp set"


def test_loaded_and_slow_legs_ramp_softer():
    arm, bus, vis, clock = make_real(scene_items=DESK)
    assert arm._leg_acc(1.0) == 7 and arm._leg_acc(1.8) == 4
    assert arm._leg_acc(1.0, loaded=True) == 5 and arm._leg_acc(1.8, loaded=True) == 3
    # within a leg the slower joints get a softer ramp, so they all finish together
    arm.run("pick", {"target": "water bottle", "pace": 1.0}, None)
    out = arm.run("place", {"where": {"x_cm": -10, "y_cm": 30}, "pace": 1.0}, None)
    assert out["result"] == "placed"
    accs = [v for _, i, a, v in bus.writes if a == 41 and i in (1, 2, 3, 4)]
    assert min(accs) >= 1 and len(set(accs)) > 2


def test_small_aim_corrections_are_soft_and_go_together(tmp_path):
    for f in BASE_FILES:
        shutil.copy(LIVE / f, tmp_path / f)
    spot = lambda r, lift: {"r_cm": r, "grasp": {"lift": lift, "elbow": 2700, "wrist_flex": 2173},
                            "via": {"lift": lift - 150, "elbow": 2700, "wrist_flex": 2173}}
    (tmp_path / "grip_teach.json").write_text(json.dumps(
        {"low": {"near": spot(22, 1750), "far": spot(38, 2400), "open": 1700, "closed": 1370}}))
    items = [{"name": "sharpie", "x_cm": 8.0, "y_cm": 28.0, "tier": "low", "axis_deg": 40.0}]
    arm, bus, vis, clock = make_real(scene_items=items, setup_dir=tmp_path)
    arm.aim_setting = "low"
    vis.aim_readings = [{"angle_deg": 35.0, "offset_px": -120.0}, {"angle_deg": 10.0, "offset_px": -60.0},
                        {"angle_deg": 1.0, "offset_px": 5.0}]
    bus.truth = []
    arm.trace_start()
    out = arm.run("pick", {"target": "sharpie", "pace": 1.0}, None)
    tr = arm.trace_stop()
    assert out["result"] == "holding"
    # a big turn alone, then a small turn and the pan from the same picture in one move, then lined up
    assert [(bool(r["d_roll"]), bool(r["d_pan"])) for r in out["aim"]] == [(True, False), (True, True), (False, False)]
    legs = [sorted(j for j, _ in e[2]) for e in tr["events"] if e[0] == "leg"]
    assert [1, 5] in legs
    for t in (tr, bus.truth_trace(tr)):
        rep = R.analyse(t)
        assert R.unplanned(rep) == []
        ph = rep["skills"][0]["phases"]
        assert ph["aiming 2"][1]["peak"] < 150                  # 43 ticks of pan: a nudge, not a twitch
        assert ph["aiming 2"][5]["peak"] < 300
    # every correction takes 0.5 to 1.2 s, longer for a bigger one
    assert desk.aim_move_s(d_pan=40) == pytest.approx(0.66) and desk.aim_move_s(d_roll=100) < desk.aim_move_s(d_roll=500) == 1.2
    assert desk.aim_move_s(d_roll=100, d_pan=80) == 0.8 and desk.aim_move_s(d_reach_cm=3.0) == 0.8


def test_aim_step_small_turn_brings_the_pan_along():
    s = desk.aim_step({"angle_deg": 10.0, "offset_px": -60.0}, 2016, 2066, 29.5)
    assert s["roll"] == 2016 - 114 and s["pan"] == 2066 - 43 and "and 60 px left" in s["why"]
    s = desk.aim_step({"angle_deg": 10.0, "offset_px": -10.0}, 2016, 2066, 29.5)   # pan within tolerance
    assert s["roll"] is not None and s["pan"] is None
    s = desk.aim_step({"angle_deg": 20.0, "offset_px": -60.0}, 2016, 2066, 29.5)   # a big turn moves the picture
    assert s["roll"] is not None and s["pan"] is None


def test_trace_report_finds_a_stop_and_go():
    rows, t = [], 0.0
    pos = 2000.0
    while t < 3.0:                       # 1 s at 200 t/s, 0.5 s still, 1 s at 200 t/s, then still
        rows.append([round(t, 4), int(pos), 946, 3190, 2632, 2030, 1310])
        pos += 200 / 30 if (t < 1.0 or 1.5 <= t < 2.5) else 0
        t += 1 / 30
    trace = {"joints": {"1": "shoulder_pan"}, "rows": rows,
             "events": [["skill", 0.0, "pick"], ["leg", 0.0, [[1, 2400]]], ["goal", 0.0, 1, 2400], ["end", 3.0, "pick"]]}
    rep = R.analyse(trace)
    sk = rep["skills"][0]
    assert sk["phases"]["start"][1]["stops"] == 1 and 0.3 < sk["phases"]["start"][1]["longest_pause_s"] < 0.7
    assert 150 < sk["phases"]["start"][1]["peak"] < 250
    assert len(R.unplanned(rep)) == 2           # the joint and the whole arm
    trace["events"].insert(2, ["hold", 1.1, "aim"])
    assert R.unplanned(R.analyse(trace)) == []
    # a new goal in the stillness makes it a pause between two legs, not a mid-leg stop
    trace["events"] = [e for e in trace["events"] if e[0] != "hold"] + [["goal", 1.4, 1, 2400]]
    st = R.analyse(trace)["skills"][0]["phases"]["start"][1]
    assert st["stops"] == 0 and st["pauses"] == 1


def test_try_real_trace_file(tmp_path, monkeypatch, capsys):
    import backend.arm.real as real
    from backend.arm import try_real
    arm, bus, vis, clock = make_real(scene_items=DESK)         # the simulated bus, made before the swap
    monkeypatch.setattr(real, "RealArm", lambda **kw: arm)
    path = tmp_path / "run.json"
    assert try_real.main(["pick", "place", "--target", "water bottle", "--pace", "1", "--yes", "--trace", str(path)]) == 0
    trace = json.loads(path.read_text())
    assert trace["hz"] == 30.0 and len(trace["rows"]) > 300 and trace["joints"]["1"] == "shoulder_pan"
    assert R.main([str(path)]) == 0
    printed = capsys.readouterr().out
    assert "trace_report" in printed and "skill pick" in printed and "0 unplanned stops" in printed
