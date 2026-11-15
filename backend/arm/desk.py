"""Desk sums shared by RealArm and FakeArm for the general skills (look, point_at, pick, place,
hand_over): reach checks, where to put something down, and the wrist roll for lying items.

Desk positions are in the arm frame used by backend/vision/scene.py: cm from the arm base,
x to the right in the overhead picture, y straight away from the arm.
"""
import math

NEAR_GAP_CM = 10.0         # place {"near": item, "side": ...}: this far from the item's centre
NEAREST_CM = 8.0           # a target given as a desk position picks the item within this of it
SIDES = ("left", "right", "front", "behind")   # as seen from the arm: front is toward the arm

# Wrist roll for lying items. At ROLL_ZERO the jaws are taken to close across an item whose long
# side points along the arm's line of sight (from the base to the item); turning the item by an
# angle turns the roll by the same angle times ROLL_SIGN. Both are first guesses, not checked on
# the arm yet: if the jaws come down along the item instead of across it, add 1024 to ROLL_ZERO
# (90 degrees); if they turn the wrong way as the item turns, flip ROLL_SIGN. A low tier taught
# with setup/tools/teach_grip.py on a lying item works its own zero out from the teaching.
ROLL_ZERO = 2030           # the roll at rest
ROLL_SIGN = 1
ROLL_TICKS_PER_RAD = 4096 / (2 * math.pi)
ROLL_RANGE = (1110, 2950)  # inside what the camera cable allows (1100 to 2960)


class Refuse(Exception):
    """A skill that will not move: result name and a plain reason."""

    def __init__(self, result, detail, **extra):
        super().__init__(detail)
        self.result, self.detail, self.extra = result, detail, extra


def wrap90(a):
    while a > 90:
        a -= 180
    while a <= -90:
        a += 180
    return a


def jaw_roll(axis_deg, bearing_deg, zero=ROLL_ZERO, sign=ROLL_SIGN):
    """Wrist roll that closes the jaws across the short side of an item lying at axis_deg (desk
    frame) seen at bearing_deg. The jaws are symmetric, so the turn is kept within 90 degrees,
    and moved by half a turn if that falls outside the roll range."""
    rel = wrap90(axis_deg - bearing_deg)
    roll = zero + sign * math.radians(rel) * ROLL_TICKS_PER_RAD
    lo, hi = ROLL_RANGE
    if roll < lo:
        roll += 2048
    elif roll > hi:
        roll -= 2048
    return int(round(max(lo, min(hi, roll))))


def label(target):
    if isinstance(target, dict):
        if "x_cm" in target:
            return f"the spot at x {target['x_cm']:.0f} cm, y {target['y_cm']:.0f} cm"
        if "near" in target:
            return f"the {target['side']} side of the {target['near']}"
    return f"the {target}" if target else "that"


def nearest(items, x, y, within=NEAREST_CM):
    best = min(items or [], key=lambda it: math.hypot(it["x_cm"] - x, it["y_cm"] - y), default=None)
    if best is None or math.hypot(best["x_cm"] - x, best["y_cm"] - y) > within:
        return None
    return best


def reach_problem(pos, min_r, max_r, pan_range, what="it"):
    """None when the arm can reach this point, else the reason. pos: {"dist_cm", "pan"}."""
    r, pan = pos["dist_cm"], pos["pan"]
    if not (min_r <= r <= max_r):
        return f"{what} is {r:.0f} cm from the arm; I can only reach between {min_r:.0f} and {max_r:.0f} cm"
    if not (pan_range[0] <= pan <= pan_range[1]):
        return f"{what} is too far round to the side (pan {pan:.0f}; I can turn from {pan_range[0]} to {pan_range[1]})"
    return None


def pick_plan(item, tiers, pan_range):
    """Tier, poses and pan for picking this item, or Refuse (not_graspable, not_taught, out_of_reach)."""
    name, tier_name = item["name"], item.get("tier") or "tall"
    if not item.get("graspable", True):
        raise Refuse("not_graspable", f"the {name} looks too big for my gripper")
    tier = tiers.get(tier_name)
    if tier is None or not tier.available:
        raise Refuse("not_taught", f"the {name} needs the {tier_name} grip and nobody has taught it yet "
                                   f"(setup/tools/teach_grip.py {tier_name} near|middle|far|farthest)")
    at = item.get("grip_at") or item
    problem = reach_problem(at, tier.min_r, tier.max_r, pan_range, what=f"the {name}")
    if problem:
        raise Refuse("out_of_reach", problem)
    via, grasp, t = tier.posture(at["dist_cm"])
    roll = None
    if tier_name == "low" and item.get("axis_deg") is not None:
        roll = jaw_roll(item["axis_deg"], at["bearing"], zero=tier.roll_zero(ROLL_SIGN, ROLL_ZERO))
    return {"tier": tier, "via": via, "grasp": grasp, "t": t, "pan": int(at["pan"]), "roll": roll,
            "dist_cm": at["dist_cm"]}


def offset(x, y, side, gap=NEAR_GAP_CM):
    dx, dy = {"left": (-gap, 0.0), "right": (gap, 0.0), "front": (0.0, -gap), "behind": (0.0, gap)}[side]
    return x + dx, y + dy


def place_centre(where, held, look):
    """(x, y, words) where the held item's centre should go. where: "back", {"x_cm", "y_cm"} or
    {"near": name, "side": side}. look(name) returns the matching item (or raises Refuse)."""
    if where in (None, "", "back"):
        return held["x_cm"], held["y_cm"], "back where it was"
    if isinstance(where, dict) and "x_cm" in where:
        return float(where["x_cm"]), float(where["y_cm"]), "at " + label(where)
    if isinstance(where, dict) and "near" in where:
        other = look(where["near"])
        if other is None:
            raise Refuse("not_found", f"I cannot see the {where['near']} on the desk; still holding the {held['name']}")
        x, y = offset(other["x_cm"], other["y_cm"], where["side"])
        return x, y, f"{where['side']} of the {other['name']}"
    raise Refuse("error", f"I do not understand where to put it: {where!r}")


def jaws_for_centre(x, y, held, point):
    """Where the jaws go to put the held item's centre at (x, y): the same spot, except a cup,
    which is held by its wall, so the jaws stay that wall's offset nearer the arm."""
    off = held.get("rim_offset_cm") or 0.0
    d = math.hypot(x, y)
    if off and d > 1e-6:
        s = max(0.0, d - off) / d
        return point(x * s, y * s)
    return point(x, y)
