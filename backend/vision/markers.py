"""Keep the desk map right when the overhead phone gets moved.

The four ArUco markers are taped to the desk, so their places on the desk never change. Only
where they land in the picture changes when the phone is bumped or put back after charging.
So we remember each marker's four corners in desk centimetres once, and on every overhead
picture we work the pixel-to-desk mapping out again from whichever markers are visible.

    python -m backend.vision.markers --save     remember the corners from the current picture
    python -m backend.vision.markers            show what the camera sees and how far the map moved

The desk frame is the one the table mapping was built in: the origin is the centre of marker 0,
x runs to the right in the picture and y runs away from the arm.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

SETUP = Path(__file__).resolve().parents[2] / "setup"
SIDE = 8.0   # marker side in cm
_SQUARE = np.array([[-SIDE / 2, SIDE / 2], [SIDE / 2, SIDE / 2], [SIDE / 2, -SIDE / 2], [-SIDE / 2, -SIDE / 2]])
_DETECTOR = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters())


def find(img):
    """{marker id: its four corners in pixels} for every marker in the picture."""
    corners, ids, _ = _DETECTOR.detectMarkers(img)
    if ids is None:
        return {}
    return {int(i): c[0].astype(np.float64) for c, i in zip(corners, ids.flatten())}


def _to_desk(H, pts):
    return cv2.perspectiveTransform(np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2), H).reshape(-1, 2)


def fit_from_scratch(px):
    """Build the desk frame from the markers alone, anchored on marker 0, treating every marker
    as a rigid 8 cm square. Returns (H, {id: corners in cm}). Needs marker 0."""
    if 0 not in px:
        raise ValueError("marker 0 has to be visible to rebuild the desk frame")
    H, _ = cv2.findHomography(px[0], _SQUARE)
    desk = {}
    for _ in range(4):
        src, dst = [], []
        for m in sorted(px):
            t = _to_desk(H, px[m])
            c = t.mean(0)
            if m == 0:
                fit = _SQUARE.copy()
            else:
                turn = [(np.arctan2(*(t[k] - c)[::-1]) - np.arctan2(*_SQUARE[k][::-1]) + np.pi) % (2 * np.pi) - np.pi for k in range(4)]
                a = float(np.mean(turn))
                R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
                fit = _SQUARE @ R.T + c
            desk[m] = fit
            src.append(px[m]); dst.append(fit)
        H, _ = cv2.findHomography(np.vstack(src), np.vstack(dst))
    return H, desk


def refresh(table, img, path=None):
    """Update table.H from the markers in this picture. Returns how many markers were used
    (0 means the old mapping was kept)."""
    try:
        with open(path or SETUP / "table_map.json") as f:
            known = json.load(f).get("marker_corners_cm") or {}
        px = find(img)
        use = [m for m in px if str(m) in known]
        if len(use) < 2:
            return 0
        src = np.vstack([px[m] for m in use])
        dst = np.vstack([np.array(known[str(m)]) for m in use])
        H, _ = cv2.findHomography(src, dst)
        if H is None:
            return 0
        table.H = H.tolist()
        return len(use)
    except Exception:
        return 0   # a bad picture must never stop the arm from using the last good mapping


if __name__ == "__main__":
    from . import cameras
    img = cv2.imread(sys.argv[-1]) if sys.argv[-1].lower().endswith((".jpg", ".png")) else cameras.grab_overhead(cameras.OVERHEAD_INDEX)
    if img is None:
        raise SystemExit("The overhead camera gives no picture.")
    px = find(img)
    print("markers seen:", sorted(px))
    p = SETUP / "table_map.json"
    tm = json.load(open(p))
    old_H = np.array(tm["homography_px_to_cm"])
    if "--save" in sys.argv:
        H, desk = fit_from_scratch(px)
        known = tm.get("marker_corners_cm") or {}
        known.update({str(m): c.tolist() for m, c in desk.items()})
        tm["marker_corners_cm"] = known
        tm["homography_px_to_cm"] = H.tolist()
        tm["image_size"] = [img.shape[1], img.shape[0]]
        json.dump(tm, open(p, "w"), indent=1)
        print("saved corners for markers", sorted(int(k) for k in known), "and the new mapping")
    else:
        H, _ = fit_from_scratch(px)
    for m in sorted(px):
        c = px[m].mean(0)
        a, b = _to_desk(old_H, [c])[0], _to_desk(H, [c])[0]
        print(f"  marker {m}: desk position now x {b[0]:6.1f} y {b[1]:6.1f} cm; the old mapping would have put it {np.linalg.norm(a - b):.1f} cm away")
