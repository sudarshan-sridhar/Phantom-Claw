"""Pick's last aim with the wrist camera: the picture reader (OpenCV on the saved samples, Gemini
answers canned), the correction sums, and the loop on FakeArm and on RealArm's simulated bus.
No network, no camera, no arm."""
import json
import threading
from pathlib import Path

import numpy as np
import pytest

from backend.arm import FakeArm, desk
from backend.tests.simbus import BASE_FILES, make_real
from backend.vision import wrist_aim as W

SAMPLES = Path(__file__).resolve().parents[2] / "setup" / "wrist_samples"
LIVE = Path(__file__).resolve().parents[2] / "setup"
SCALE = 0.001


def sample(name):
    import cv2
    img = cv2.imread(str(SAMPLES / name))
    if img is None:
        pytest.skip(f"no sample picture {name}")
    return img


# the reader
@pytest.mark.parametrize("name,lo,hi,off_lo,off_hi", [
    ("hover.jpg", 25, 45, None, None),             # pen about 35 degrees clockwise, jaws closed
    ("roll_minus.jpg", -6, 6, -135, -85),          # upright, about 100 px left of the jaw centre
    ("pan_adj.jpg", -6, 6, -90, -45),              # upright, about 60 px left
    ("roll_plus.jpg", 40, 70, None, None),         # turned the wrong way, further from vertical
])
def test_opencv_reads_the_samples(name, lo, hi, off_lo, off_hi):
    r = W.read_cv(sample(name), "long")
    assert r is not None and r["confidence"] >= W.AIM["cv_min_conf"], r
    assert lo <= r["angle_deg"] <= hi, r
    if off_lo is not None:
        assert off_lo <= r["offset_px"] <= off_hi and r["between_jaws"], r


def test_read_aim_uses_opencv_when_sure_and_does_not_ask_gemini():
    asked = []
    r = W.read_aim(sample("roll_minus.jpg"), name="sharpie", ask=lambda *a, **k: asked.append(1))
    assert r["source"] == "opencv" and asked == []


def test_read_aim_falls_back_to_gemini_then_gives_up():
    rng = np.random.default_rng(1)
    desk_only = np.clip(rng.normal(150, 4, (360, 640, 3)), 0, 255).astype(np.uint8)   # nothing on it
    canned = W.line_reading((400, 50), (380, 300), "gemini", 0.9)
    r = W.read_aim(desk_only, name="sharpie", ask=lambda *a, **k: canned)
    assert r["source"] == "gemini" and abs(r["angle_deg"]) < 6
    log = {}
    assert W.read_aim(desk_only, name="sharpie", ask=lambda *a, **k: None, log=log) is None
    assert "opencv" in log and "gemini" in log

    def broken(*a, **k):
        raise RuntimeError("503")
    assert W.read_aim(desk_only, name="sharpie", ask=broken) is None


def test_parse_gemini_answers():
    # ends straight up the picture, left of the jaws
    r = W.parse_gemini({"found": True, "box": [190, 460, 1000, 600], "ends": [[190, 520], [1000, 552]],
                        "long": True, "confidence": 0.95}, "long")
    assert abs(r["angle_deg"]) < 6 and -125 < r["offset_px"] < -95 and r["confidence"] == 0.95
    # top end to the right: clockwise, positive
    r = W.parse_gemini({"box": [148, 216, 1000, 671], "ends": [[151, 632], [1000, 216]], "long": True}, "long")
    assert 35 < r["angle_deg"] < 45
    # ends answered as small boxes are read by their middles; a big "end" box is thrown away
    r = W.parse_gemini({"box": [151, 222, 1000, 670], "ends": [[151, 600, 171, 640], [981, 219, 1000, 252]]}, "long")
    assert r["angle_deg"] is not None
    r = W.parse_gemini({"box": [151, 222, 1000, 670], "ends": [[151, 574, 999, 670], [981, 219, 1000, 252]],
                        "confidence": 0.95}, "long")
    assert r["angle_deg"] is None and r["confidence"] < W.AIM["gemini_min_conf"]   # a long item with no angle
    # a round item: only the centre
    r = W.parse_gemini({"box": [300, 600, 600, 800], "ends": None, "long": False, "confidence": 0.8}, "round")
    assert r["angle_deg"] is None and r["x_at_jaws"] == pytest.approx(448.0)
    assert W.parse_gemini({"found": False}) is None and W.parse_gemini("nonsense") is None


def test_shape_for():
    assert W.shape_for("sharpie") == "long" and W.shape_for("earbuds case") == "round"
    assert W.shape_for("bottle cap") == "round" and W.shape_for("stapler") is None


# the correction sums
def test_aim_step_roll_first_then_pan():
    A = desk.WRIST_AIM
    s = desk.aim_step({"angle_deg": 35.0, "offset_px": -110.0}, 2016, 2066, 29.5)
    assert s["roll"] == 2016 - 398 and s["pan"] is None and not s["done"]     # roll only, the measured fix
    s = desk.aim_step({"angle_deg": 2.0, "offset_px": -110.0}, 1618, 2066, 29.5)
    assert s["roll"] is None and s["pan"] == 2066 - 79 and not s["capped"]    # 110 / 1.4
    s = desk.aim_step({"angle_deg": 80.0, "offset_px": 0.0}, 2016, 2066, 29.5)
    assert s["d_roll"] == -A["max_roll_step"] and s["capped"]
    s = desk.aim_step({"angle_deg": 0.0, "offset_px": 400.0}, 2016, 2066, 29.5)
    assert s["d_pan"] == A["max_pan_step"] and s["capped"]
    s = desk.aim_step({"angle_deg": -5.0, "offset_px": 20.0}, 2016, 2066, 29.5)
    assert s["done"] and s["roll"] is None and s["pan"] is None
    s = desk.aim_step({"angle_deg": None, "offset_px": -60.0}, 2016, 2066, 29.5)   # round item: pan at once
    assert s["pan"] == 2066 - 43
    # further out the camera swings further per tick, so the same offset needs fewer ticks
    assert abs(desk.aim_step({"angle_deg": 0, "offset_px": -60.0}, 2016, 2066, 44.0)["d_pan"]) < 43
    # the wrist at its limit: go on to the pan instead of looping
    s = desk.aim_step({"angle_deg": -30.0, "offset_px": -60.0}, desk.ROLL_RANGE[1], 2066, 29.5)
    assert s["roll"] is None and s["pan"] is not None


def test_aim_tiers_setting(monkeypatch):
    monkeypatch.delenv("PHANTOM_WRIST_AIM", raising=False)
    assert desk.aim_tiers() == ("low",)
    assert desk.aim_tiers("all") == ("tall", "can", "low") and desk.aim_tiers("off") == ()
    monkeypatch.setenv("PHANTOM_WRIST_AIM", "can,low")
    assert desk.aim_tiers() == ("can", "low")


# FakeArm
def frun(arm, skill, params=None):
    phases = []
    return arm.run(skill, params or {}, phases.append), phases


def low_fake():
    arm = FakeArm(time_scale=SCALE)
    arm.taught = {"tall", "can", "low"}
    arm.aim_setting = "low"
    return arm


def aiming(ph):
    return [p for p in ph if p.startswith("aiming")]


def test_fake_default_pretend_item_converges():
    arm = low_fake()
    out, ph = frun(arm, "pick", {"target": "sharpie"})
    assert out["result"] == "holding" and "wrist aim" in out["detail"]
    assert [bool(r["d_roll"]) for r in out["aim"]] == [True, False, False]
    assert out["aim"][-1]["done"] and len(aiming(ph)) == 3
    assert ph.index("seen") < ph.index(aiming(ph)[0]) and ph.index(aiming(ph)[-1]) < ph.index("at_target")


def test_fake_scripted_converges_in_two_rounds():
    arm = low_fake()
    arm.wrist_reader = [{"angle_deg": 35.0, "offset_px": -110.0, "source": "opencv"},
                        {"angle_deg": 1.0, "offset_px": -70.0, "source": "opencv"},
                        {"angle_deg": 0.5, "offset_px": 4.0, "source": "opencv"}]
    out, ph = frun(arm, "pick", {"target": "sharpie"})
    assert out["result"] == "holding"
    (n1, roll1, pan1), (n2, roll2, pan2), (n3, roll3, pan3) = arm.wrist_reads
    assert roll2 == roll1 - 398 and pan2 == pan1               # round 1: roll only
    assert roll3 == roll2 and pan3 < pan2                      # round 2: pan only
    assert out["aim"][2]["done"]


def test_fake_huge_correction_is_capped():
    arm = low_fake()
    arm.wrist_reader = [{"angle_deg": 89.0, "offset_px": 0.0}, {"angle_deg": 0.0, "offset_px": -600.0},
                        {"angle_deg": 0.0, "offset_px": 0.0}]
    out, _ = frun(arm, "pick", {"target": "sharpie"})
    assert out["aim"][0]["d_roll"] == -500 and out["aim"][0]["capped"]
    assert out["aim"][1]["d_pan"] == -80 and out["aim"][1]["capped"] and "capped" in out["detail"]


def test_fake_unreadable_carries_on_with_the_overhead_aim():
    arm = low_fake()
    arm.wrist_reader = [None]
    out, ph = frun(arm, "pick", {"target": "sharpie"})
    assert out["result"] == "holding" and out["aim"] == [{"round": 1, "unread": True,
                                                          "why": "neither OpenCV nor Gemini was sure"}]
    assert len(arm.wrist_reads) == 1 and "could not read" in out["detail"]


def test_fake_stop_during_the_aim():
    arm = FakeArm(time_scale=0.02)
    arm.taught = {"low"}
    arm.aim_setting = "low"
    out = arm.run("pick", {"target": "sharpie"}, lambda p: p.startswith("aiming") and arm.stop())
    assert out["result"] == "stopped" and not arm.status()["holding"]


def test_fake_tall_does_not_aim_unless_asked():
    arm = FakeArm(time_scale=SCALE)
    out, ph = frun(arm, "pick", {"target": "bottle"})
    assert "aim" not in out and aiming(ph) == []
    arm = FakeArm(time_scale=SCALE)
    arm.aim_setting = "all"
    out, ph = frun(arm, "pick", {"target": "bottle"})
    assert out["result"] == "holding" and out["aim"] and aiming(ph)


# RealArm on the simulated bus
DESK = [{"name": "sharpie", "x_cm": 8.0, "y_cm": 28.0, "tier": "low", "axis_deg": 40.0},
        {"name": "water bottle", "x_cm": 2.0, "y_cm": 30.0, "tier": "tall"}]


@pytest.fixture
def low_setup(tmp_path):
    import shutil
    for f in BASE_FILES:
        shutil.copy(LIVE / f, tmp_path / f)
    spot = lambda r, lift: {"r_cm": r, "grasp": {"lift": lift, "elbow": 2700, "wrist_flex": 2173},
                            "via": {"lift": lift - 150, "elbow": 2700, "wrist_flex": 2173}}
    (tmp_path / "grip_teach.json").write_text(json.dumps(
        {"low": {"near": spot(22, 1750), "far": spot(38, 2400), "open": 1700, "closed": 1370}}))
    return tmp_path


def real_low(low_setup, readings, **kw):
    arm, bus, vis, clock = make_real(scene_items=DESK, setup_dir=low_setup, **kw)
    arm.aim_setting = "low"
    vis.aim_readings = readings
    return arm, bus, vis, clock


def rrun(arm, skill, params=None):
    phases = []
    return arm.run(skill, params or {}, phases.append), phases


def test_real_aim_converges_in_two_rounds(low_setup):
    arm, bus, vis, clock = real_low(low_setup, [{"angle_deg": 35.0, "offset_px": -120.0, "source": "opencv"},
                                                {"angle_deg": 1.0, "offset_px": -70.0, "source": "gemini"},
                                                {"angle_deg": 0.5, "offset_px": 5.0, "source": "opencv"}])
    out, ph = rrun(arm, "pick", {"target": "sharpie", "pace": 1.0})
    assert out["result"] == "holding", out
    (_, roll1, pan1, name, path1), (_, roll2, pan2, _, _), (_, roll3, pan3, _, _) = vis.aim_looks
    assert name == "sharpie" and path1.endswith("last_wrist_aim_1.jpg") and str(low_setup) in path1
    assert abs((roll2 - roll1) + 398) <= 12 and abs(pan2 - pan1) <= 6
    assert abs(roll3 - roll2) <= 6 and -60 <= pan3 - pan2 <= -40      # 70 px at 29 cm, about 51 ticks
    assert [r["done"] for r in out["aim"]] == [False, False, True] and "wrist aim" in out["detail"]
    marks = aiming(ph)
    assert len(marks) == 3 and "wrist roll -" in marks[0] and "pan -" in marks[1]
    assert ph.index("seen") < ph.index(marks[0]) and ph.index(marks[-1]) < ph.index("at_target")
    assert vis.closed >= 1                                         # the wrist camera let go afterwards
    # the jaws went down with the corrected roll and pan
    assert abs(bus.pos(5) - roll2) <= 12
    assert bus.bad_goals() == [] and bus.torque_off_writes() == []
    # place puts the wrist back to where it was before the pick
    out, _ = rrun(arm, "place", {"where": "back", "pace": 1.0})
    assert out["result"] == "placed" and abs(bus.pos(5) - 2030) <= 15


def test_real_aim_caps_a_huge_correction(low_setup):
    arm, bus, vis, clock = real_low(low_setup, [{"angle_deg": 85.0, "offset_px": 0.0},
                                                {"angle_deg": 0.0, "offset_px": -900.0}])
    out, ph = rrun(arm, "pick", {"target": "sharpie", "pace": 1.0})
    assert out["result"] == "holding"
    (_, roll1, pan1, _, _), (_, roll2, pan2, _, _), (_, roll3, pan3, _, _) = vis.aim_looks
    assert abs((roll2 - roll1) + 500) <= 12 and abs((pan3 - pan2) + 80) <= 6
    assert out["aim"][0]["capped"] and out["aim"][1]["capped"]
    assert out["aim"][2]["unread"]                 # the script ran out: the third picture is unreadable
    assert bus.bad_goals() == []


def test_real_aim_unreadable_keeps_the_overhead_aim(low_setup):
    arm, bus, vis, clock = real_low(low_setup, None)
    out, ph = rrun(arm, "pick", {"target": "sharpie", "pace": 1.0})
    assert out["result"] == "holding" and len(vis.aim_looks) == 1 and out["aim"][0]["unread"]
    assert "could not read" in aiming(ph)[0]
    first = next(m for m in arm.trace if m[0] == "blend")
    assert abs(bus.pos(5) - dict(first[1])[5]) <= 12      # the roll the overhead aim chose
    # a vision object with no wrist_aim at all (an old one) is the same as unreadable
    arm, bus, vis, clock = make_real(scene_items=DESK, setup_dir=low_setup)
    arm.aim_setting = "low"
    vis.wrist_aim = None
    out, ph = rrun(arm, "pick", {"target": "sharpie", "pace": 1.0})
    assert out["result"] == "holding" and out["aim"][0]["unread"]


def test_real_stop_during_the_aim(low_setup):
    arm, bus, vis, clock = real_low(low_setup, [{"angle_deg": 35.0, "offset_px": -120.0}] * 3)
    out = arm.run("pick", {"target": "sharpie", "pace": 1.0}, lambda p: p.startswith("aiming") and arm.stop())
    assert out["result"] == "stopped" and not bus.holding
    # a stop while the picture is being read (it takes real time) is noticed too
    arm, bus, vis, clock = real_low(low_setup, [{"angle_deg": 35.0, "offset_px": -120.0}] * 3)
    vis.aim_delay = 0.5
    box = {}
    th = threading.Thread(target=lambda: box.update(out=arm.run("pick", {"target": "sharpie", "pace": 1.0})))
    th.start()
    import time
    for _ in range(200):
        if getattr(vis, "aim_looks", None):
            break
        time.sleep(0.01)
    arm.stop()
    th.join(10)
    assert box["out"]["result"] == "stopped" and len(vis.aim_looks) == 1
    out, ph = rrun(arm, "rest")
    assert out["result"] == "done" and bus.torque_off_writes() == []


# compact items: reach and jaw opening (the earbuds case miss on 2026-11-15)
def compact(cy, x=452.0, width=120.0, bottom=None, conf=0.9):
    r = W.point_reading(x, "gemini", conf)
    h = width * 0.8
    return W.extents(r, x - width / 2, cy - h / 2, x + width / 2, cy + h / 2 if bottom is None else bottom)


def test_reader_measures_a_cut_off_wide_case():
    # the miss picture as Gemini should answer it: x 310 to 515, from y 250 down past the bottom edge
    r = W.parse_gemini({"found": True, "box": [694, 484, 1000, 805], "ends": None, "long": False,
                        "confidence": 0.8}, "round", opening=1750)
    assert r["compact"] and r["touches_bottom"] and 200 <= r["width_px"] <= 210 and r["between_jaws"]
    assert W.jaw_tips(1750) == (300.0, 520.0) and W.jaw_tips(2400)[1] > 640
    # a compact OpenCV blob is never sure enough alone, so Gemini is asked
    assert W.AIM["compact_cv_conf"] < W.AIM["cv_min_conf"]
    # a long item's Gemini box carries no compact fields: long items aim as before
    assert "compact" not in W.parse_gemini({"box": [100, 500, 900, 560], "ends": None, "confidence": 0.9}, "long")


def test_aim_step_reach_for_compact_items():
    A = desk.WRIST_AIM
    s = desk.aim_step(compact(300), 2016, 2066, 30.0, reach_range=(18, 40))
    assert s["roll"] is None and s["pan"] is None and s["reach"] == pytest.approx(30 - 65 * 0.035, abs=0.01)
    s = desk.aim_step(compact(120), 2016, 2066, 30.0)                 # far up the picture: move out, capped
    assert s["d_reach"] == A["max_reach_step_cm"] and s["capped"]
    s = desk.aim_step(compact(330, bottom=360), 2016, 2066, 30.0)     # cut off at the bottom edge: back
    assert s["d_reach"] == -A["max_reach_step_cm"] and "bottom edge" in s["why"]
    s = desk.aim_step(compact(250), 2016, 2066, 30.0)                 # within 30 px of the target row
    assert s["done"] and "row 250" in s["why"]
    s = desk.aim_step(compact(300, x=560), 2016, 2066, 30.0)          # pan comes before reach
    assert s["pan"] is not None and s["reach"] is None
    s = desk.aim_step(compact(330, bottom=360), 2016, 2066, 19.0, reach_range=(18, 40))   # at the end of the reach
    assert s["d_reach"] == pytest.approx(-1.0) and desk.aim_step(compact(330, bottom=360), 2016, 2066, 18.0,
                                                                   reach_range=(18, 40))["done"]
    # a long reading never moves the reach, whatever its numbers
    assert desk.aim_step({"angle_deg": 0.0, "offset_px": 0.0, "centre_y": 100}, 2016, 2066, 30.0)["done"]


def test_jaw_opening_from_width():
    assert desk.jaw_opening(80.0, 1750) is None                        # a pen: the opening will do
    o = desk.jaw_opening(205.0, 1750)                                  # the earbuds case
    assert o == 1750 + round((205 - 220) / 0.5) + desk.WRIST_AIM["open_margin_ticks"]
    assert W.jaw_gap(o) >= 205 + 50
    assert desk.jaw_opening(600.0, 1750) == desk.WRIST_AIM["open_max"] <= 2600


def test_compact_items_start_at_the_taught_roll():
    from backend.vision.table import GripTier
    spot = lambda r: {"r_cm": r, "roll": 2016, "rel_axis_deg": 0.0, "pan": 2066,
                      "via": {"lift": 1900, "elbow": 2700, "wrist_flex": 2173},
                      "grasp": {"lift": 2050, "elbow": 2700, "wrist_flex": 2173}}
    tiers = {"low": GripTier("low", [spot(22), spot(38)], open_value=1750, closed=1448)}
    case = {"name": "earbuds case", "tier": "low", "x_cm": -14, "y_cm": 26, "dist_cm": 29.5, "bearing": -28,
            "pan": 2300, "axis_deg": -30.0}
    assert desk.pick_plan(case, tiers, (734, 3360))["roll"] == 2016
    pen = {**case, "name": "sharpie"}
    assert desk.pick_plan(pen, tiers, (734, 3360))["roll"] != 2016


def test_fake_compact_too_near_then_corrected_in_two_rounds():
    arm = low_fake()
    arm.scene_items.append({"name": "earbuds case", "x_cm": 0.0, "y_cm": 30.0, "tier": "low", "graspable": True,
                            "axis_deg": -30.0})
    arm.wrist_reader = [compact(300), compact(240)]
    out, ph = frun(arm, "pick", {"target": "earbuds case"})
    assert out["result"] == "holding", out
    assert [r["done"] for r in out["aim"]] == [False, True] and out["aim"][0]["d_reach"] < -2
    assert "reach -2.3 cm" in out["detail"] and "reach" in aiming(ph)[0]


def test_real_compact_too_near_then_corrected(low_setup):
    desk_items = DESK + [{"name": "earbuds case", "x_cm": 4.0, "y_cm": 30.0, "tier": "low", "axis_deg": -30.0}]
    arm, bus, vis, clock = make_real(scene_items=desk_items, setup_dir=low_setup)
    arm.aim_setting = "low"
    lifts = []

    def read(n, roll, pan):
        lifts.append(bus.pos(2))
        return [compact(300), compact(240)][n - 1] if n <= 2 else None
    vis.aim_readings = read
    out, ph = rrun(arm, "pick", {"target": "earbuds case", "pace": 1.0})
    assert out["result"] == "holding", out
    assert len(lifts) == 2 and lifts[1] < lifts[0] - 20        # moved in toward the base (lower lift)
    assert [r["done"] for r in out["aim"]] == [False, True] and abs(out["aim"][0]["d_reach"] + 2.28) < 0.05
    assert vis.aim_openings[0] is not None                      # the reader is told where the jaws are
    assert bus.bad_goals() == []


def test_real_cut_off_moves_back_then_looks_again(low_setup):
    desk_items = DESK + [{"name": "earbuds case", "x_cm": 4.0, "y_cm": 30.0, "tier": "low"}]
    arm, bus, vis, clock = make_real(scene_items=desk_items, setup_dir=low_setup)
    arm.aim_setting = "low"
    vis.aim_readings = [compact(330, bottom=360), compact(320, bottom=360), compact(235)]
    out, ph = rrun(arm, "pick", {"target": "earbuds case", "pace": 1.0})
    assert out["result"] == "holding"
    assert [r["d_reach"] for r in out["aim"][:2]] == [-3.0, -3.0] and out["aim"][2]["done"]
    assert len(vis.aim_looks) == 3 and "bottom edge" in aiming(ph)[0]


def test_real_wide_item_opens_wider_and_is_held(low_setup):
    desk_items = DESK + [{"name": "earbuds case", "x_cm": 4.0, "y_cm": 30.0, "tier": "low"}]
    arm, bus, vis, clock = make_real(scene_items=desk_items, setup_dir=low_setup)
    arm.aim_setting = "low"
    bus.cap = 1720                       # a 5 cm case stops the jaws far out (the pen stopped at 1448)
    vis.aim_readings = [compact(240, width=205.0)]
    out, ph = rrun(arm, "pick", {"target": "earbuds case", "pace": 1.0})
    assert out["result"] == "holding", out
    want = desk.jaw_opening(205.0, 1700)
    assert out["aim"][-1]["opened"] == want > 1720 and "jaws opened" in out["detail"]
    assert any("opening the jaws" in p for p in ph)
    assert abs(out["held"]["x_cm"] - 4.0) < 0.01 and arm.status()["holding"]
    assert bus.bad_goals() == []
    # without opening wider the jaws start inside the case and push it away: a miss
    arm, bus, vis, clock = make_real(scene_items=desk_items, setup_dir=low_setup)
    arm.aim_setting = "low"
    bus.cap = 1720
    vis.aim_readings = [compact(240, width=60.0)]
    out, _ = rrun(arm, "pick", {"target": "earbuds case", "pace": 1.0})
    assert out["result"] == "missed"


def test_real_compact_closed_on_nothing_is_still_a_miss(low_setup):
    desk_items = DESK + [{"name": "earbuds case", "x_cm": 4.0, "y_cm": 30.0, "tier": "low"}]
    arm, bus, vis, clock = make_real(scene_items=desk_items, setup_dir=low_setup, grip_ok=False)
    arm.aim_setting = "low"
    vis.aim_readings = [compact(240, width=205.0)]
    out, _ = rrun(arm, "pick", {"target": "earbuds case", "pace": 1.0})
    assert out["result"] == "missed" and not bus.holding


def test_real_tall_does_not_aim_by_default(low_setup):
    arm, bus, vis, clock = make_real(scene_items=DESK, setup_dir=low_setup)
    arm.aim_setting = None
    out, ph = rrun(arm, "pick", {"target": "water bottle", "pace": 1.0})
    assert out["result"] == "holding" and "aim" not in out and not getattr(vis, "aim_looks", None)
