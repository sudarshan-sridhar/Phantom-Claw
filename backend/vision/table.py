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

JAW_CM = 4.7          # the jaws grip this far to one side of the line the base turns on
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

    def jaw_ticks(self, r_cm):
        """How far the base must turn past an item's bearing to bring the jaws onto it. The jaws sit
        to one side, so the nearer the item, the bigger the turn (85 ticks at 35 cm, 130 at 23 cm).
        One fixed number put the jaws beside a bottle standing close in."""
        return self.tpr * math.atan(JAW_CM / max(12.0, r_cm))

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
                             "pan": self.pan_for_bearing(a) - self.jaw_ticks(math.hypot(tx, ty))}
        p = boxes.get("person")
        if p:
            ymin, xmin, ymax, xmax = p
            tx, ty = self.to_table((xmin + xmax) / 2 * sx, (ymin + ymax) / 2 * sy)
            a = math.atan2(tx - self.base[0], ty - self.base[1])
            out["person"] = {"bearing_deg": math.degrees(a), "pan": self.pan_for_bearing(a)}
        return out

    # any item on the desk (the general skills). Positions are in the arm frame: cm from the base,
    # x to the right in the picture, y away from the arm.
    def _arm_xy(self, x_px, y_px):
        tx, ty = self.to_table(x_px, y_px)
        return tx - self.base[0], ty - self.base[1]

    def point(self, x, y):
        """A desk point in the arm frame to distance, bearing and the pan that lines the jaws up with it."""
        a = math.atan2(x, y)
        return {"x_cm": round(x, 2), "y_cm": round(y, 2), "dist_cm": round(math.hypot(x, y), 2),
                "bearing": round(math.degrees(a), 2), "pan": int(round(self.pan_for_bearing(a) - self.jaw_ticks(math.hypot(x, y))))}

    def place_box(self, box, tier="tall"):
        """Where an item in this box is on the desk. Standing things (tall, can): the bottom of the
        box, a little up from the edge, as for the bottle. Lying things (low): the box centre.
        Also its width across the picture at that point (width_cm)."""
        sx, sy = self.image_size[0] / 1000.0, self.image_size[1] / 1000.0
        ymin, xmin, ymax, xmax = box
        yb = (ymax - 0.07 * (ymax - ymin)) if tier != "low" else (ymin + ymax) / 2
        x, y = self._arm_xy((xmin + xmax) / 2 * sx, yb * sy)
        out = self.point(x, y)
        xl, yl = self._arm_xy(xmin * sx, yb * sy)
        xr, yr = self._arm_xy(xmax * sx, yb * sy)
        out["width_cm"] = round(math.hypot(xr - xl, yr - yl), 1)
        return out

    def place_person(self, box):
        sx, sy = self.image_size[0] / 1000.0, self.image_size[1] / 1000.0
        ymin, xmin, ymax, xmax = box
        x, y = self._arm_xy((xmin + xmax) / 2 * sx, (ymin + ymax) / 2 * sy)
        a = math.atan2(x, y)
        return {"box": box, "x_cm": round(x, 2), "y_cm": round(y, 2), "bearing": round(math.degrees(a), 2),
                "pan": int(round(self.pan_for_bearing(a)))}

    def axis_of(self, box, ends=None):
        """Long-axis angle on the desk (see backend/vision/scene.py for the convention). From the two
        ends Gemini gave, mapped onto the desk; without them, along the longer side of the box."""
        from .scene import axis_angle
        sx, sy = self.image_size[0] / 1000.0, self.image_size[1] / 1000.0
        if ends:
            (y1, x1), (y2, x2) = ends
        else:
            ymin, xmin, ymax, xmax = box
            yc, xc = (ymin + ymax) / 2, (xmin + xmax) / 2
            if (xmax - xmin) * sx >= (ymax - ymin) * sy:
                (y1, x1), (y2, x2) = (yc, xmin), (yc, xmax)
            else:
                (y1, x1), (y2, x2) = (ymin, xc), (ymax, xc)
        ax, ay = self._arm_xy(x1 * sx, y1 * sy)
        bx, by = self._arm_xy(x2 * sx, y2 * sy)
        return axis_angle(bx - ax, by - ay)

class WristTable(TableMap):
    """The desk as the gripper camera sees it from its look-down pose, for one base angle.
    Used when the overhead camera gives no picture. setup/wrist_map.json (made by
    setup/tools/wrist_map.py) turns a pixel into centimetres ahead and to the right of where
    the arm is facing; the base angle then turns that into the usual arm frame."""

    def __init__(self, pan, setup_dir=None):
        d = Path(setup_dir) if setup_dir else SETUP
        super().__init__(d / "table_map.json")
        with open(d / "wrist_map.json") as f:
            wm = json.load(f)
        self.Hw = wm["homography_px_to_cm"]
        self.image_size = wm["image_size"]
        self.pose = wm["pose"]
        self.heading = (pan - self.pan_zero) / self.tpr

    def _arm_xy(self, x_px, y_px):
        lat, fwd = _perspective(self.Hw, x_px, y_px)
        h = self.heading
        return fwd * math.sin(h) + lat * math.cos(h), fwd * math.cos(h) - lat * math.sin(h)


def wrist_map(setup_dir=None):
    """The saved gripper camera map, or None when it has not been made."""
    p = (Path(setup_dir) if setup_dir else SETUP) / "wrist_map.json"
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


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


# Taught grips by tier, for the general skills (pick, place). fetch_bottle keeps using Taught.
SPOTS = ("near", "middle", "far", "farthest")
POSE_KEYS = ("lift", "elbow", "wrist_flex")
TALL_OPEN, TALL_CLOSED = 1790, 1478       # what fetch_bottle opens to and where the cap stops the jaws
CARRY_LIFT = {"low": -190, "normal": -230, "high": -320}   # shoulder ticks above the nearest taught grasp
VIA_LIFT = 170                            # a spot taught without a pose above it: this much shoulder up


def _pose(v):
    if isinstance(v, dict) and all(isinstance(v.get(k), (int, float)) and not isinstance(v.get(k), bool) for k in POSE_KEYS):
        return {k: int(round(v[k])) for k in POSE_KEYS}
    return None


class GripTier:
    """The taught grips of one height tier, blended by distance. Works with 1 to 4 taught spots:
    they are sorted by distance and blended piecewise between neighbours (with two spots this is
    exactly Taught.posture). The reachable ring is the nearest spot minus 4 cm to the farthest plus
    2 cm, the same margins fetch_bottle uses."""

    def __init__(self, name, spots, open_value=TALL_OPEN, closed=None, source=""):
        self.name = name
        self.spots = sorted(spots, key=lambda s: s["r_cm"])
        self.open = int(open_value)
        self.closed = int(closed) if closed is not None else None
        self.source = source
        self.available = bool(self.spots)
        self.min_r = self.spots[0]["r_cm"] - 4 if self.spots else None
        self.max_r = self.spots[-1]["r_cm"] + 2 if self.spots else None

    def in_reach(self, r_cm):
        return self.available and self.min_r <= r_cm <= self.max_r

    def posture(self, r_cm):
        """(via, grasp, t) as Taught.posture: t is the blend inside the segment it falls in."""
        s = self.spots
        if len(s) == 1:
            return dict(s[0]["via"]), dict(s[0]["grasp"]), 0.0
        if r_cm <= s[0]["r_cm"]:
            a, b = s[0], s[1]
        elif r_cm >= s[-1]["r_cm"]:
            a, b = s[-2], s[-1]
        else:
            a, b = next((p, q) for p, q in zip(s, s[1:]) if p["r_cm"] <= r_cm <= q["r_cm"])
        span = b["r_cm"] - a["r_cm"]
        t = 1.0 if span < 1e-6 else max(0.0, min(1.0, (r_cm - a["r_cm"]) / span))

        def mix(p, q):
            return {k: round(p[k] + t * (q[k] - p[k])) for k in POSE_KEYS}
        return mix(a["via"], b["via"]), mix(a["grasp"], b["grasp"]), t

    def base_grasp(self):
        """The nearest taught grasp: the tuck and the put-down recovery are built on it."""
        return dict(self.spots[0]["grasp"])

    def carry(self, style="normal"):
        g = self.base_grasp()
        return {"lift": g["lift"] + CARRY_LIFT.get(style, CARRY_LIFT["normal"]), "elbow": g["elbow"],
                "wrist_flex": g["wrist_flex"]}

    def tuck(self):
        return self.carry("normal")

    def table_lift(self):
        return self.base_grasp()["lift"]

    def far_grasp(self):
        return dict(self.spots[-1]["grasp"])

    def roll_zero(self, sign, default):
        """Wrist roll that closes the jaws across an item lying along the arm's line of sight,
        worked out from spots taught on a lying item (each keeps its roll and the item's angle
        relative to the bearing). Falls back to the constant when none have that."""
        got = [s["roll"] - sign * math.radians(s["rel_axis_deg"]) * 4096 / (2 * math.pi)
               for s in self.spots if s.get("roll") is not None and s.get("rel_axis_deg") is not None]
        return sum(got) / len(got) if got else default

    def ring(self):
        if not self.available:
            return None
        return {"min_cm": round(self.min_r, 1), "max_cm": round(self.max_r, 1),
                "spots": [s["spot"] for s in self.spots]}

    def summary(self):
        return {"tier": self.name, "available": self.available, "ring": self.ring(), "open": self.open,
                "closed": self.closed, "source": self.source}


def _spot(name, v):
    if not isinstance(v, dict):
        return None
    r = v.get("r_cm")
    grasp = _pose(v.get("grasp"))
    if not isinstance(r, (int, float)) or isinstance(r, bool) or grasp is None:
        return None
    via = _pose(v.get("via")) or {**grasp, "lift": grasp["lift"] - VIA_LIFT}
    out = {"spot": name, "r_cm": float(r), "via": via, "grasp": grasp}
    for k in ("pan", "roll", "rel_axis_deg", "gripper_open", "gripped_at"):
        if isinstance(v.get(k), (int, float)) and not isinstance(v.get(k), bool):
            out[k] = float(v[k])
    return out


def load_tiers(setup_dir=None):
    """{"tall": GripTier, "can": ..., "low": ...}. tall is the fetch_bottle grips from
    bottle_teach.json (mid and far; near was taught at the same distance and is left out, as
    fetch_bottle does), plus any tall spots in grip_teach.json (a taught spot of the same name
    replaces it). can and low come only from grip_teach.json; a tier with no taught spot is there
    but not available."""
    d = Path(setup_dir) if setup_dir else SETUP
    T = Taught(d / "bottle_teach.json")
    tall = {"middle": _spot("middle", T.mid), "far": _spot("far", T.far)}
    extra = {}
    p = d / "grip_teach.json"
    if p.exists():
        try:
            with open(p) as f:
                extra = json.load(f)
        except Exception:
            extra = {}
        if isinstance(extra.get("tiers"), dict):
            extra = extra["tiers"]
    tiers = {}
    for name in ("tall", "can", "low") + tuple(k for k in extra if k not in ("tall", "can", "low", "note")):
        t = extra.get(name) if isinstance(extra.get(name), dict) else {}
        spots = dict(tall) if name == "tall" else {}
        for s in SPOTS:
            sp = _spot(s, t.get(s))
            if sp:
                spots[s] = sp
        opened = t.get("open", t.get("grip"))
        closed = t.get("closed")
        if name == "tall":
            opened = TALL_OPEN if opened is None else opened
            closed = None if closed is None else closed     # None: fetch_bottle's own miss check
        tiers[name] = GripTier(name, [s for s in spots.values() if s], open_value=opened or TALL_OPEN, closed=closed,
                               source="bottle_teach.json" + (" + grip_teach.json" if name == "tall" and t else "")
                               if name == "tall" else "grip_teach.json")
    return tiers
