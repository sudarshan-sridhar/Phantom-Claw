"""backend.vision: box parsing, desk mapping and the taught grips (offline), plus the saved
teaching photos through Gemini (live, PHANTOM_LIVE=1). No camera is opened."""
import os
from pathlib import Path

import numpy as np
import pytest

from backend.vision import TableMap, Taught, WRIST, parse_box, parse_boxes, wrist_pan_for

SETUP = Path(__file__).resolve().parents[2] / "setup"
live = pytest.mark.skipif(os.environ.get("PHANTOM_LIVE") != "1", reason="set PHANTOM_LIVE=1 to call Gemini")


@pytest.mark.parametrize("raw,want", [
    ([100, 200, 300, 400], [100.0, 200.0, 300.0, 400.0]),
    ([[100, 200, 300, 400]], [100.0, 200.0, 300.0, 400.0]),
    ({"box_2d": [1, 2, 3, 4]}, [1.0, 2.0, 3.0, 4.0]),
    ([{"box": [1, 2, 3, 4]}], [1.0, 2.0, 3.0, 4.0]),
    ({"bottle": [5, 6, 7, 8]}, [5.0, 6.0, 7.0, 8.0]),
    (None, None), ([1, 2, 3], None), ("100,200,300,400", None), ([True, 2, 3, 4], None), ({}, None),
])
def test_parse_box_odd_shapes(raw, want):
    assert parse_box(raw) == want


def test_parse_boxes_list_answer():
    assert parse_boxes([{"bottle": [1, 2, 3, 4], "person": None}]) == {"bottle": [1.0, 2.0, 3.0, 4.0], "person": None}
    assert parse_boxes("nonsense") == {"bottle": None, "person": None}


def test_table_map_matches_opencv():
    import cv2
    tm = TableMap()
    H = np.array(tm.H)
    for x, y in ((640, 360), (100, 650), (1200, 80)):
        want = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float64), H)[0, 0]
        assert np.allclose(tm.to_table(x, y), want)
    seen = tm.locate({"bottle": [400, 450, 600, 520], "person": None})
    assert "person" not in seen and 0 < seen["bottle"]["r_cm"] < 80


def test_taught_blend():
    T = Taught()
    via, grasp, t = T.posture(T.mid["r_cm"])
    assert t == 0 and grasp == {k: T.mid["grasp"][k] for k in grasp}
    via, grasp, t = T.posture(100)
    assert t == 1 and via == {k: T.far["via"][k] for k in via}
    assert T.in_reach(23.2) and T.in_reach(39.3) and not T.in_reach(15) and not T.in_reach(45)
    assert T.tuck() == {"lift": 1425, "elbow": 2700, "wrist_flex": 2173}
    assert T.middle()[1] == {"lift": 1655, "elbow": 2700, "wrist_flex": 2173}


def test_wrist_centring_maths():
    assert wrist_pan_for([300, 460, 700, 540], 2000, 652)[0] == pytest.approx(2000)
    right, _ = wrist_pan_for([300, 800, 700, 900], 2000, 652)
    assert (right > 2000) == (WRIST["pan_sign"] > 0)


@live
@pytest.mark.parametrize("name,r,pan", [("mid", 23.2, 1932), ("far", 39.3, 2050)])
def test_saved_teaching_photos(name, r, pan):
    import cv2
    from backend.vision import Vision
    seen = Vision().locate(cv2.imread(str(SETUP / f"teach_bottle_{name}_overhead.jpg")))
    assert abs(seen["bottle"]["r_cm"] - r) < 1.5 and abs(seen["bottle"]["pan"] - pan) < 25
