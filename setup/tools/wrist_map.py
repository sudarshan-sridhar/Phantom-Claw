"""Teach the gripper camera where the desk is, so the arm can find things with no overhead camera.

From one fixed look-down pose the camera sees the desk ahead of the gripper. Turning the base
only turns that view, so one mapping from picture to "centimetres ahead and to the side of where
the arm is facing" is good for every base angle. This script finds that mapping from the paper
markers, whose places on the desk are known: it turns the base through a sweep, finds marker
corners in each gripper picture, and fits the mapping. Saved to setup/wrist_map.json.

Run from the project root with the area clear (the arm moves):
    python setup\\tools\\wrist_map.py
"""
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SETUP = ROOT / "setup"
PY = sys.executable
POSE = {"lift": 1151, "elbow": 2552, "wrist_flex": 2863}      # the look-down pose
PANS = [1250, 1350, 1450, 1550, 1700, 1850, 2000, 2300, 2450, 2600, 2750]


def main():
    from backend.vision import cameras, markers

    tm = json.loads((SETUP / "table_map.json").read_text())
    base, pan_zero, tpr = tm["base_cm"], tm["pan_zero"], tm["ticks_per_rad"]
    corners_cm = {int(k): v for k, v in tm["marker_corners_cm"].items()}
    cam = cameras.WristCamera()
    px, cm, seen = [], [], {}
    size = None
    try:
        for pan in PANS:
            subprocess.run([PY, str(SETUP / "tools" / "move.py"), "2.5",
                            f"p:{pan},{POSE['lift']},{POSE['elbow']},{POSE['wrist_flex']}"],
                           capture_output=True, text=True)
            time.sleep(0.5)
            f = cam.frame()
            if f is None:
                continue
            size = [f.shape[1], f.shape[0]]
            cv2.imwrite(str(SETUP / f"wrist_map_{pan}.jpg"), cv2.resize(f, (640, 360)))
            h = (pan - pan_zero) / tpr
            found = markers.find(f)
            for mid, pts in (found or {}).items():
                if int(mid) not in corners_cm:
                    continue
                seen[pan] = seen.get(pan, []) + [int(mid)]
                for (u, v), (tx, ty) in zip(pts, corners_cm[int(mid)]):
                    x, y = tx - base[0], ty - base[1]
                    fwd = x * math.sin(h) + y * math.cos(h)
                    lat = x * math.cos(h) - y * math.sin(h)
                    px.append([float(u), float(v)])
                    cm.append([lat, fwd])
    finally:
        cam.close()
        subprocess.run([PY, str(SETUP / "tools" / "move.py"), "4", "rest"], capture_output=True, text=True)
    print("markers seen at each base angle:", seen)
    if len(px) < 8:
        print(f"only {len(px)} marker corners seen; need at least 8 (two markers). Nothing saved.")
        return
    P, C = np.array(px, np.float32), np.array(cm, np.float32)
    # every marker counts: the near one pins down the near desk, the far one the far desk
    H, mask = cv2.findHomography(P, C, 0)
    back = cv2.perspectiveTransform(P.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(back - C, axis=1)
    keep = np.ones(len(px), bool)
    for i in range(0, len(px), 4):
        print("  corners", i, "to", i + 3, "error cm", [round(float(e), 1) for e in err[i:i + 4]], "kept", keep[i:i + 4].tolist())
    print(f"{len(px)} corners, {int(keep.sum())} kept; error on kept corners: median {np.median(err[keep]):.2f} cm, "
          f"worst {err[keep].max():.2f} cm")
    out = {"note": "gripper camera picture (upright, pixels) to [cm to the right, cm ahead] of where the arm faces, "
                   "from the look-down pose. Made by setup/tools/wrist_map.py.",
           "pose": POSE, "image_size": size, "homography_px_to_cm": H.tolist(),
           "corners": len(px), "median_error_cm": round(float(np.median(err[keep])), 2),
           "made": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (SETUP / "wrist_map.json").write_text(json.dumps(out, indent=1))
    print("saved setup/wrist_map.json")


if __name__ == "__main__":
    main()
