"""Desk geometry and the taught grips.

TableMap turns a Gemini box in the overhead picture into a distance and a base (pan) angle,
using the marker mapping in setup/table_map.json. Taught holds the hand-taught bottle grips
from setup/bottle_teach.json and blends them by distance. Both are the maths from
setup/tools/fetch_bottle.py, unchanged.
"""
import json
import math
from pathlib import Path

SETUP = Path(__file__).resolve().parents[2] / "setup"

JAW_OFFSET = 85      # the camera bearing reads this many ticks right of the pan that grips (from the 3 teachings)


def _perspective(H, x, y):
    """The same as cv2.perspectiveTransform for one point, so this file needs no OpenCV."""
    u = H[0][0] * x + H[0][1] * y + H[0][2]
    v = H[1][0] * x + H[1][1] * y + H[1][2]
    w = H[2][0] * x + H[2][1] * y + H[2][2]
    return u / w, v / w


class TableMap:
    def __init__(self, path=None):
        with open(path or SETUP / "table_map.json") as f:
            tm = json.load(f)
        self.H = tm["homography_px_to_cm"]
        self.base = tm["base_cm"]
        self.pan_zero = tm["pan_zero"]
        self.tpr = tm["ticks_per_rad"]
        self.image_size = tm["image_size"]

    def to_table(self, x_px, y_px):
        return _perspective(self.H, x_px, y_px)

    def pan_for_bearing(self, rad):
        return self.pan_zero + self.tpr * rad

    def locate(self, boxes):
        """boxes: {"bottle": [ymin, xmin, ymax, xmax] or None, "person": ...} in 0 to 1000 units.
        Returns {"boxes", "bottle": {r_cm, bearing_deg, pan}, "person": {bearing_deg, pan}};
        bottle and person are left out when not seen."""
        sx, sy = self.image_size[0] / 1000.0, self.image_size[1] / 1000.0
        out = {"boxes": boxes}
        b = boxes.get("bottle")
        if b:
            ymin, xmin, ymax, xmax = b
            # where the bottle stands: the bottom of the box, a little up from the very edge
            tx, ty = self.to_table((xmin + xmax) / 2 * sx, (ymax - 0.07 * (ymax - ymin)) * sy)
            tx, ty = tx - self.base[0], ty - self.base[1]
            a = math.atan2(tx, ty)
            out["bottle"] = {"r_cm": math.hypot(tx, ty), "bearing_deg": math.degrees(a),
                             "pan": self.pan_for_bearing(a) - JAW_OFFSET}
        p = boxes.get("person")
        if p:
            ymin, xmin, ymax, xmax = p
            tx, ty = self.to_table((xmin + xmax) / 2 * sx, (ymin + ymax) / 2 * sy)
            a = math.atan2(tx - self.base[0], ty - self.base[1])
            out["person"] = {"bearing_deg": math.degrees(a), "pan": self.pan_for_bearing(a)}
        return out


class Taught:
    """The taught middle and far grips, blended by distance."""

    def __init__(self, path=None):
        with open(path or SETUP / "bottle_teach.json") as f:
            self.bt = json.load(f)
        self.mid, self.far = self.bt["mid"], self.bt["far"]
        self.min_r = self.mid["r_cm"] - 4
        self.max_r = self.far["r_cm"] + 2

    def in_reach(self, r_cm):
        return self.min_r <= r_cm <= self.max_r

    def posture(self, r_cm):
        """(via, grasp, t): the out pose above the cap and the grasp pose, each
        {"lift", "elbow", "wrist_flex"}, and the blend t (0 middle, 1 far)."""
        m, f = self.mid, self.far
        t = (r_cm - m["r_cm"]) / (f["r_cm"] - m["r_cm"])
        t = max(0.0, min(1.0, t))

        def mix(a, b):
            return {k: round(a[k] + t * (b[k] - a[k])) for k in ("lift", "elbow", "wrist_flex")}
        return mix(m["via"], f["via"]), mix(m["grasp"], f["grasp"]), t

    def middle(self):
        """The taught middle grip as it is (used when the distance cannot be measured)."""
        keys = ("lift", "elbow", "wrist_flex")
        return {k: self.mid["via"][k] for k in keys}, {k: self.mid["grasp"][k] for k in keys}, 0.0

    def tuck(self):
        """The taught middle grip, raised: known to clear the table with the bottle in the jaws."""
        g = self.mid["grasp"]
        return {"lift": g["lift"] - 230, "elbow": g["elbow"], "wrist_flex": g["wrist_flex"]}

    def table_lift_tucked(self):
        """Shoulder value that stands the bottle on the table at the tuck."""
        return self.mid["grasp"]["lift"]

    def far_grasp(self):
        return dict(self.far["grasp"])
