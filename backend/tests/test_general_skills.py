"""The general building blocks: look, point_at, pick, place, hand_over.

No network, no camera, no arm: Gemini answers are canned, FakeArm pretends, and RealArm runs on
the simulated bus (backend/tests/simbus.py).
"""
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from backend.arm import FakeArm, validate
from backend.arm import desk
from backend.arm import skills as S
from backend.tests.simbus import clean_setup, make_real
from backend.vision import TableMap, Taught, load_tiers
from backend.vision.scene import axis_angle, match_name, parse_scene, scene_prompt, tier_for

SETUP = Path(__file__).resolve().parents[2] / "setup"
SCALE = 0.001
REST = {1: 2009, 2: 946, 3: 3190, 4: 2632}

CANNED = {
    "items": [
        {"name": "Water Bottle", "box": [300, 450, 600, 520], "height": "tall", "graspable": True},
        {"name": "red bull can", "box": [420, 600, 560, 650], "height": "medium", "graspable": "true"},
        {"name": "Sharpie", "box": [500, 300, 540, 420], "height": "low", "ends": [[505, 305], [535, 415]]},
        {"name": "robot arm", "box": [0, 400, 400, 600], "height": "tall", "graspable": False},
        {"name": "paper marker", "box": [800, 100, 900, 200], "height": "low"},
        {"name": "stapler", "box": [600, 700, 680, 820], "height": "low", "graspable": False},
        {"name": "half a box", "box": [1, 2]},
        "nonsense",
    ],
    "person": [[0, 0, 200, 300]],
    "match": 1,
}


# scene parsing and matching
def test_scene_from_a_canned_gemini_answer():
    tm = TableMap()
    s = parse_scene(CANNED, tm, want="something to drink")
    names = [i["name"] for i in s["items"]]
    assert names == ["water bottle", "red bull can", "sharpie", "stapler"]      # arm and markers ignored
    by = {i["name"]: i for i in s["items"]}
    assert [by[n]["tier"] for n in names] == ["tall", "can", "low", "low"]
    assert by["water bottle"]["tier_from"] == "name" and by["stapler"]["tier_from"] == "gemini"
    assert by["stapler"]["graspable"] is False and by["sharpie"]["graspable"] is True
    # the bottle sits where fetch_bottle would put it
    old = tm.locate({"bottle": [300, 450, 600, 520], "person": None})["bottle"]
    assert by["water bottle"]["dist_cm"] == pytest.approx(old["r_cm"], abs=0.01)
    assert abs(by["water bottle"]["pan"] - old["pan"]) <= 1
    b = by["water bottle"]
    assert b["dist_cm"] == pytest.approx(math.hypot(b["x_cm"], b["y_cm"]), abs=0.02)
    assert by["sharpie"]["axis_deg"] is not None and -90 < by["sharpie"]["axis_deg"] <= 90
    assert by["water bottle"]["axis_deg"] is None
    assert s["person"] is not None and "pan" in s["person"]
    assert s["match"]["name"] == "red bull can"


def test_scene_survives_odd_answers():
    tm = TableMap()
    assert parse_scene("nonsense", tm) == {"items": [], "person": None, "match": None}
    s = parse_scene([{"name": "cup", "box": [400, 400, 500, 480]}], tm, want="the cup")
    assert s["items"][0]["tier"] == "can" and s["match"]["name"] == "cup"
    s = parse_scene({"items": CANNED["items"], "match": None}, tm, want="the sharpie")
    assert s["match"]["name"] == "sharpie"        # Gemini said nothing fits; the plain name still does
    s = parse_scene({"items": CANNED["items"], "match": "stapler"}, tm, want="the grey thing")
    assert s["match"]["name"] == "stapler"


@pytest.mark.parametrize("name,tier", [
    ("water bottle", "tall"), ("juice bottle", "tall"), ("juice can", "can"), ("Red Bull can", "can"),
    ("red bull", "can"), ("cup", "can"), ("paper cups", "can"), ("sharpie", "low"), ("black marker", "low"),
    ("pen", "low"), ("scissors", "low"),
])
def test_known_names_pick_the_tier(name, tier):
    assert tier_for(name, "tall" if tier == "low" else "low") == (tier, "name")


def test_unknown_names_use_the_height_guess():
    assert tier_for("stapler", "low") == ("low", "gemini")
    assert tier_for("lamp", "tall") == ("tall", "gemini")
    assert tier_for("box", "medium") == ("can", "gemini")


def test_match_name():
    tm = TableMap()
    items = parse_scene(CANNED, tm)["items"]
    assert match_name(items, "the sharpie")["name"] == "sharpie"
    assert match_name(items, "my pen")["name"] == "sharpie"          # synonym
    assert match_name(items, "the bottle")["name"] == "water bottle"
    assert match_name(items, "my phone") is None
    assert match_name(items, "") is None


def test_prompt_carries_the_spoken_name():
    assert '"the cup"' in scene_prompt("the cup") and "match" in scene_prompt("the cup")
    assert "match" not in scene_prompt(None)


def test_find_asks_gemini_once_with_the_name(monkeypatch):
    from backend.vision import Vision, cameras, gemini, markers
    asked = []
    monkeypatch.setattr(cameras, "grab_overhead", lambda *a, **k: np.zeros((72, 128, 3), np.uint8))
    monkeypatch.setattr(markers, "refresh", lambda table, img, path=None: 0)
    monkeypatch.setattr(gemini, "ask_json", lambda img, prompt, **k: asked.append(prompt) or CANNED)
    v = Vision()
    assert v.find("something to drink")["name"] == "red bull can"
    assert len(asked) == 1 and "something to drink" in asked[0]
    s = v.scene()
    assert len(s["items"]) == 4 and s["markers_used"] == 0
    monkeypatch.setattr(cameras, "grab_overhead", lambda *a, **k: None)
    assert v.scene() is None and v.find("cup") is None


def test_axis_convention_and_jaw_roll():
    assert axis_angle(0, 1) == 0 and axis_angle(0, -1) == 0      # along the line away from the base
    assert axis_angle(1, 0) == 90 and axis_angle(1, 1) == pytest.approx(45) and axis_angle(-1, 1) == pytest.approx(-45)
    assert desk.jaw_roll(10.0, 10.0) == desk.ROLL_ZERO            # lying along the line of sight
    r = desk.jaw_roll(40.0, 10.0)
    assert r == round(desk.ROLL_ZERO + desk.ROLL_SIGN * math.radians(30) * desk.ROLL_TICKS_PER_RAD)
    for axis in range(-90, 91, 15):
        assert desk.ROLL_RANGE[0] <= desk.jaw_roll(axis, -50.0) <= desk.ROLL_RANGE[1]
    assert desk.jaw_roll(90.0, 0.0) == desk.jaw_roll(-90.0, 0.0)   # an axis has no head or tail


def test_cup_is_gripped_at_the_rim_nearest_the_arm():
    tm = TableMap()
    from backend.vision.scene import grip_point
    cup = {"name": "cup", **tm.point(0.0, 30.0), "width_cm": 8.0}
    at = grip_point(cup, tm)
    assert at["dist_cm"] == pytest.approx(26.0, abs=0.05) and at["rim_offset_cm"] == 4.0
    assert abs(at["bearing"] - cup["bearing"]) < 0.01
    can = {"name": "juice can", **tm.point(0.0, 30.0), "width_cm": 7.0}
    assert grip_point(can, tm)["dist_cm"] == pytest.approx(30.0)


# taught grips by tier
def spot(r, lift, elbow=2700, flex=2173, **kw):
    return {"r_cm": r, "grasp": {"lift": lift, "elbow": elbow, "wrist_flex": flex},
            "via": {"lift": lift - 150, "elbow": elbow, "wrist_flex": flex}, **kw}


@pytest.fixture
def setup_dir(tmp_path):
    for f in ("my_follower_calibration_2026-11-14.json", "bottle_pick.json", "bottle_teach.json", "table_map.json"):
        shutil.copy(SETUP / f, tmp_path / f)
    return tmp_path


def teach(d, data):
    (Path(d) / "grip_teach.json").write_text(json.dumps(data))


def test_tall_tier_is_the_fetch_bottle_blend(setup_dir):
    tiers = load_tiers(setup_dir)
    T = Taught(setup_dir / "bottle_teach.json")
    tall = tiers["tall"]
    for r in (18, 23.2, 27.5, 31, 39.3, 45):
        assert tall.posture(r)[:2] == T.posture(r)[:2]
    assert (tall.min_r, tall.max_r) == (T.min_r, T.max_r)
    assert tall.tuck() == T.tuck() and tall.open == 1790 and tall.closed is None
    assert not tiers["can"].available and not tiers["low"].available


def test_blend_works_with_two_three_and_four_spots(setup_dir):
    teach(setup_dir, {"low": {"near": spot(20, 1700), "far": spot(40, 2500), "open": 1700, "closed": 1370},
                      "can": {"near": spot(20, 1600), "middle": spot(28, 1900), "far": spot(36, 2200),
                              "farthest": spot(44, 2600), "open": 1950, "closed": 1600}})
    tiers = load_tiers(setup_dir)
    low, can = tiers["low"], tiers["can"]
    assert low.ring() == {"min_cm": 16.0, "max_cm": 42.0, "spots": ["near", "far"]}
    assert can.ring() == {"min_cm": 16.0, "max_cm": 46.0, "spots": ["near", "middle", "far", "farthest"]}
    assert low.posture(30)[1]["lift"] == 2100
    assert can.posture(24)[1]["lift"] == 1750 and can.posture(40)[1]["lift"] == 2400
    assert can.posture(10)[1]["lift"] == 1600 and can.posture(60)[1]["lift"] == 2600
    assert (low.open, low.closed, can.open) == (1700, 1370, 1950)
    # three spots, given in any order, are sorted by distance
    teach(setup_dir, {"low": {"far": spot(40, 2500), "near": spot(20, 1700), "middle": spot(30, 2000)}})
    assert load_tiers(setup_dir)["low"].posture(25)[1]["lift"] == 1850


def test_tall_spots_from_grip_teach_widen_the_ring(setup_dir):
    teach(setup_dir, {"tall": {"farthest": spot(45, 2800)}})
    tall = load_tiers(setup_dir)["tall"]
    assert tall.ring()["max_cm"] == 47.0 and tall.ring()["spots"] == ["middle", "far", "farthest"]


def test_a_spot_without_a_distance_does_not_count(setup_dir):
    teach(setup_dir, {"low": {"near": {**spot(20, 1700), "r_cm": None}}})
    assert not load_tiers(setup_dir)["low"].available


# parameter checks
def test_validate_the_new_skills():
    assert validate("pick", {"target": " the  cup ", "approach": "Look First", "lift_cm_style": "HIGH"}) == \
        {"target": "the cup", "pace": 0.5, "approach": "look_first", "lift_cm_style": "high"}
    assert validate("pick", {"target": {"x_cm": "3", "y_cm": 30}, "look_first": True})["approach"] == "look_first"
    assert validate("pick", {"target": {"x_cm": 3}, "lift_cm_style": "sky"}) == \
        {"target": None, "pace": 0.5, "approach": "direct", "lift_cm_style": "normal"}
    assert validate("place", {})["where"] == "back"
    assert validate("place", {"where": {"x_cm": 1, "y_cm": 2}})["where"] == {"x_cm": 1.0, "y_cm": 2.0}
    assert validate("place", {"where": {"near": "cup", "side": "Left"}})["where"] == {"near": "cup", "side": "left"}
    assert validate("place", {"where": {"near": "cup", "side": "under"}})["where"] == {"near": "cup", "side": "front"}
    assert validate("place", {"where": "the mug"})["where"] == {"near": "the mug", "side": "front"}
    assert validate("hand_over", {"wait_s": 30, "pace": -2}) == {"wait_s": 6.0, "pace": 0.0}
    assert validate("look", {}) == {"target": None}


def test_place_next_to_another_item_gives_sensible_coordinates():
    held = {"name": "water bottle", "x_cm": 2.0, "y_cm": 30.0}
    cup = {"name": "cup", "x_cm": -6.0, "y_cm": 36.0}
    look = lambda n: cup if "cup" in n else None
    for side, want in (("right", (4.0, 36.0)), ("left", (-16.0, 36.0)), ("front", (-6.0, 26.0)), ("behind", (-6.0, 46.0))):
        x, y, words = desk.place_centre({"near": "cup", "side": side}, held, look)
        assert (x, y) == want and side in words
    assert desk.place_centre("back", held, look)[:2] == (2.0, 30.0)
    with pytest.raises(desk.Refuse) as e:
        desk.place_centre({"near": "phone", "side": "right"}, held, look)
    assert e.value.result == "not_found"


def test_durations():
    assert 2 <= S.nominal_duration("look", {}) <= 4
    assert 9 <= S.nominal_duration("pick", {"pace": 0.5}) <= 20
    assert S.nominal_duration("pick", {"pace": 0}) > S.nominal_duration("pick", {"pace": 1})
    assert 10 <= S.nominal_duration("place", {}) <= 25
    assert 10 <= S.nominal_duration("hand_over", {"wait_s": 3}) <= 25
    assert S.general_nominal("pick", validate("pick", {}), "not_taught") == S.FETCH_LOOK_S


# FakeArm
def run(arm, skill, params=None):
    phases = []
    out = arm.run(skill, params or {}, phases.append)
    return out, phases


def test_fake_look_lists_the_desk():
    out, ph = run(FakeArm(time_scale=SCALE), "look", {"target": "something to drink"})
    assert out["ok"] and out["result"] == "done" and ph == ["start", "seen", "done"]
    assert {i["name"] for i in out["scene"]} == {"water bottle", "juice can", "red bull can", "cup", "sharpie", "scissors"}
    assert all({"x_cm", "y_cm", "dist_cm", "bearing", "tier", "graspable"} <= set(i) for i in out["scene"])
    assert out["match"] is not None and out["person"]["pan"] > 2009


def test_fake_pick_then_place_then_state_cleared():
    arm = FakeArm(time_scale=SCALE)
    out, ph = run(arm, "pick", {"target": "the water bottle", "pace": 0.3})
    assert out["result"] == "holding" and out["ok"] and ph == ["start", "seen", "at_target", "gripped", "lifted", "done"]
    st = arm.status()
    assert st["held"]["name"] == "water bottle" and st["holding"] and st["at_rest"] is False
    assert "water bottle" not in {i["name"] for i in run(arm, "look")[0]["scene"]}
    out, ph = run(arm, "place", {"where": {"x_cm": 0, "y_cm": 28}})
    assert out["result"] == "placed" and out["ok"]
    assert ph == ["start", "seen", "carrying", "at_target", "put_back", "returning", "done"]
    st = arm.status()
    assert st["held"] is None and not st["holding"] and st["at_rest"] is True
    seen = {i["name"]: i for i in run(arm, "look")[0]["scene"]}
    assert (seen["water bottle"]["x_cm"], seen["water bottle"]["y_cm"]) == (0.0, 28.0)
    assert run(arm, "place", {})[0]["result"] == "nothing_held"
    assert run(arm, "hand_over", {})[0]["result"] == "nothing_held"


def test_fake_place_next_to_the_cup():
    arm = FakeArm(time_scale=SCALE)
    run(arm, "pick", {"target": "bottle"})
    out, _ = run(arm, "place", {"where": {"near": "cup", "side": "right"}})
    assert out["result"] == "placed" and out["placed_at"] == {"x_cm": 4.0, "y_cm": 36.0}


def test_fake_hand_over_not_taken_keeps_holding():
    arm = FakeArm(time_scale=SCALE)
    arm.hand_taken = False
    run(arm, "pick", {"target": "bottle"})
    out, ph = run(arm, "hand_over", {"wait_s": 2})
    assert out["result"] == "not_taken" and not out["ok"] and ph == ["start", "seen", "carrying", "offering", "done"]
    assert arm.status()["held"]["name"] == "water bottle" and arm.status()["holding"]
    arm.hand_taken = True
    out, ph = run(arm, "hand_over", {"wait_s": 2})
    assert out["result"] == "handed_over" and ph[-3:] == ["handed_over", "returning", "done"]
    assert arm.status()["held"] is None and arm.status()["at_rest"]


def test_fake_refusals_do_not_move():
    arm = FakeArm(time_scale=SCALE)
    before = dict(arm.status()["joints"])
    out, ph = run(arm, "pick", {"target": "sharpie"})
    assert out["result"] == "not_taught" and "teach_grip.py low" in out["detail"] and ph == ["start", "seen", "done"]
    assert run(arm, "pick", {"target": "my phone"})[0]["result"] == "not_found"
    arm.scene_items.append({"name": "laptop", "x_cm": 0.0, "y_cm": 55.0, "tier": "tall", "graspable": True})
    arm.scene_items.append({"name": "mug", "x_cm": -20.0, "y_cm": -12.0, "tier": "tall", "graspable": True})
    arm.scene_items.append({"name": "box", "x_cm": 10.0, "y_cm": 25.0, "tier": "tall", "graspable": False})
    out, _ = run(arm, "pick", {"target": "laptop"})
    assert out["result"] == "out_of_reach" and "55 cm" in out["detail"]
    out, _ = run(arm, "pick", {"target": "mug"})
    assert out["result"] == "out_of_reach" and "round to the side" in out["detail"]
    assert run(arm, "pick", {"target": "box"})[0]["result"] == "not_graspable"
    assert run(arm, "point_at", {"target": "laptop"})[0]["result"] == "out_of_reach"
    assert arm.status()["joints"] == before and arm.status()["at_rest"]
    arm.overhead = False
    assert run(arm, "look")[0]["result"] == "no_picture"
    assert run(arm, "pick", {"target": "bottle"})[0]["result"] == "no_picture"


def test_fake_with_the_low_tier_taught():
    arm = FakeArm(time_scale=SCALE)
    arm.taught = {"tall", "can", "low"}
    out, _ = run(arm, "pick", {"target": "sharpie"})
    assert out["result"] == "holding" and "wrist roll" in out["detail"]
    assert arm.status()["joints"]["wrist_roll"] != 2030


def test_fake_while_holding():
    arm = FakeArm(time_scale=SCALE)
    run(arm, "pick", {"target": "bottle"})
    out, ph = run(arm, "gesture", {"style": "ban", "seconds": 3})
    assert out["result"] == "done" and out["skipped"] and ph == []
    out, ph = run(arm, "wave")
    assert out["result"] == "hands_full" and ph == []
    assert run(arm, "pick", {"target": "cup"})[0]["result"] == "hands_full"
    out, ph = run(arm, "place", {"where": {"x_cm": 0, "y_cm": 70}})
    assert out["result"] == "out_of_reach" and "still holding" in out["detail"]
    assert arm.status()["held"] is not None
    out, ph = run(arm, "rest")
    assert "put the water bottle down" in out["detail"] and ph == ["start", "put_back", "returning", "done"]
    assert arm.status()["held"] is None and not arm.status()["holding"]


def test_fake_point_at():
    out, ph = run(FakeArm(time_scale=SCALE), "point_at", {"target": "red bull"})
    assert out["result"] == "done" and ph == ["start", "seen", "at_target", "returning", "done"]
    assert out["target"]["name"] == "red bull can"
    out, _ = run(FakeArm(time_scale=SCALE), "point_at", {"target": {"x_cm": 5, "y_cm": 30}})
    assert out["result"] == "done"


def test_fake_stop_during_pick_then_rest_puts_it_down():
    import threading
    import time
    arm = FakeArm(time_scale=0.02)
    box = {}
    th = threading.Thread(target=lambda: box.update(out=arm.run("pick", {"target": "bottle"}, lambda p: p == "gripped" and arm.stop())))
    th.start()
    th.join(10)
    assert box["out"]["result"] == "stopped"
    arm.time_scale = SCALE
    out, ph = run(arm, "point_at", {"target": "cup"})
    assert out["auto_rest"] and ph[0] == "auto_rest" and out["result"] == "done"
    assert not arm.status()["holding"]


# RealArm on the simulated bus
DESK = [{"name": "water bottle", "x_cm": 2.0, "y_cm": 30.0, "tier": "tall"},
        {"name": "cup", "x_cm": -8.0, "y_cm": 26.0, "tier": "can", "width_cm": 8.0},
        {"name": "sharpie", "x_cm": 8.0, "y_cm": 28.0, "tier": "low", "axis_deg": 40.0},
        {"name": "laptop", "x_cm": 0.0, "y_cm": 58.0, "tier": "low"}]


def rrun(arm, skill, params=None):
    phases = []
    return arm.run(skill, params or {}, phases.append), phases


def safe(bus):
    assert bus.bad_goals() == [] and bus.torque_off_writes() == []


def at_rest(bus):
    return all(abs(bus.pos(i) - REST[i]) <= 20 for i in REST)


def test_real_pick_then_place_back():
    arm, bus, vis, clock = make_real(scene_items=DESK)
    out, ph = rrun(arm, "pick", {"target": "the water bottle", "pace": 1.0})
    assert out["result"] == "holding" and ph == ["start", "seen", "at_target", "gripped", "lifted", "done"]
    assert bus.holding and arm.status()["held"]["name"] == "water bottle" and not arm.status()["at_rest"]
    tall = load_tiers(clean_setup())["tall"]
    tuck = tall.carry("normal")
    assert abs(bus.pos(2) - tuck["lift"]) <= 120 and abs(bus.pos(3) - tuck["elbow"]) <= 120
    pan_pick = bus.pos(1)
    out, ph = rrun(arm, "place", {"where": "back", "pace": 1.0})
    assert out["result"] == "placed" and ph == ["start", "seen", "carrying", "at_target", "put_back", "returning", "done"]
    assert "auto_rest" not in out
    assert not bus.holding and at_rest(bus) and arm.status()["held"] is None
    # it went down at the same pan it was picked from
    downs = [m for m in arm.trace if m[0] == "blend" and m[1] and m[1][0][0] == 1 and len(m[1]) == 1]
    assert abs(downs[-1][1][0][1] - pan_pick) <= 3
    safe(bus)


def test_real_refusals_do_not_move():
    arm, bus, vis, clock = make_real(scene_items=DESK)
    for target, res in (("sharpie", "not_taught"), ("cup", "not_taught"), ("laptop", "not_taught"), ("phone", "not_found")):
        arm.trace.clear()
        out, ph = rrun(arm, "pick", {"target": target})
        assert out["result"] == res, (target, out)
        assert [m for m in arm.trace if m[0] in ("blend", "close")] == []
    arm, bus, vis, clock = make_real(scene_items=[{"name": "water bottle", "x_cm": 0.0, "y_cm": 50.0, "tier": "tall"}])
    out, ph = rrun(arm, "pick", {"target": "bottle"})
    assert out["result"] == "out_of_reach" and ph == ["start", "seen", "done"] and list(arm.trace) == []
    arm, bus, vis, clock = make_real(overhead=False)
    assert rrun(arm, "look")[0]["result"] == "no_picture"
    assert rrun(arm, "place")[0]["result"] == "nothing_held"
    assert bus.torque_off_writes() == []


def test_real_low_tier_sets_the_wrist_roll(setup_dir):
    teach(setup_dir, {"low": {"near": spot(22, 1750), "far": spot(38, 2400), "open": 1700, "closed": 1370}})
    arm, bus, vis, clock = make_real(scene_items=DESK, setup_dir=setup_dir)
    out, ph = rrun(arm, "pick", {"target": "sharpie", "pace": 1.0})
    assert out["result"] == "holding", out
    first = next(m for m in arm.trace if m[0] == "blend")
    goals = dict(first[1])
    item = vis.scene("sharpie")["match"]
    assert goals[5] == desk.jaw_roll(40.0, item["bearing"]) and goals[6] == 1700
    out, _ = rrun(arm, "place", {"where": {"near": "cup", "side": "behind"}, "pace": 1.0})
    assert out["result"] == "placed" and at_rest(bus)
    last_roll = [dict(m[1])[5] for m in arm.trace if m[0] == "blend" and 5 in dict(m[1])]
    assert abs(last_roll[-1] - 2030) <= 5          # the wrist turned back after letting go
    safe(bus)


def test_real_hand_over_not_taken_keeps_holding_then_taken():
    arm, bus, vis, clock = make_real(scene_items=DESK)
    rrun(arm, "pick", {"target": "bottle", "pace": 1.0})
    out, ph = rrun(arm, "hand_over", {"wait_s": 2, "pace": 1.0})
    assert out["result"] == "not_taken" and ph == ["start", "seen", "carrying", "offering", "done"]
    assert bus.holding and arm.status()["held"] is not None and vis.watchers[-1].stopped
    vis.hand_after = 0.5
    out, ph = rrun(arm, "hand_over", {"wait_s": 3, "pace": 1.0})
    assert out["result"] == "handed_over" and "hand" in out["detail"]
    assert not bus.holding and at_rest(bus) and arm.status()["held"] is None
    safe(bus)


def test_real_while_holding_only_the_carry_skills_run():
    arm, bus, vis, clock = make_real(scene_items=DESK)
    rrun(arm, "pick", {"target": "bottle", "pace": 1.0})
    n = len(arm.trace)
    assert rrun(arm, "gesture", {"style": "king", "seconds": 2})[0]["skipped"]
    assert rrun(arm, "fetch_bottle")[0]["result"] == "hands_full"
    assert len(arm.trace) == n and bus.holding
    out, ph = rrun(arm, "rest")
    assert "put the water bottle down" in out["detail"] and at_rest(bus) and not bus.holding
    safe(bus)


def test_real_stop_during_carry_then_auto_rest():
    arm, bus, vis, clock = make_real(scene_items=DESK)
    out, _ = rrun(arm, "pick", {"target": "bottle", "pace": 1.0})
    arm_out = {}

    def on(p):
        if p == "carrying":
            arm.stop()
    arm_out = arm.run("place", {"where": {"x_cm": -10, "y_cm": 30}}, on)
    assert arm_out["result"] == "stopped"
    out, ph = rrun(arm, "point_at", {"target": "cup", "pace": 1.0})
    assert out["auto_rest"] and ph[0] == "auto_rest" and out["result"] == "done"
    assert not bus.holding and at_rest(bus)
    safe(bus)


def test_real_fetch_bottle_still_works_after_general_skills():
    from backend.tests.test_fetch_real_sim import script_moves, moves_of
    arm, bus, vis, clock = make_real(scene_items=DESK)
    rrun(arm, "point_at", {"target": "cup", "pace": 1.0})
    arm.trace.clear()
    out, ph = rrun(arm, "fetch_bottle", {"pace": 1.0, "wait_s": 3})
    assert out["result"] == "put_back" and moves_of(arm) == script_moves("put_back")


# try_real --plan moves nothing
@pytest.mark.parametrize("argv", [
    ["look", "--plan"], ["point_at", "--target", "cup", "--plan"],
    ["pick", "place", "--target", "red bull", "--near", "cup", "--side", "right", "--plan"],
    ["pick", "hand_over", "--target", "10,30", "--approach", "look_first", "--lift-style", "high", "--plan"],
])
def test_try_real_plan_for_the_new_skills(argv, capsys, monkeypatch):
    import backend.arm.real as real
    from backend.arm import try_real
    monkeypatch.setattr(real, "RealArm", lambda *a, **k: pytest.fail("--plan must not open the port"))
    assert try_real.main(argv) == 0
    text = capsys.readouterr().out
    assert "steps:" in text and "tier tall" in text
