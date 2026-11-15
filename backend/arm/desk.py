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


# Last aim with the wrist camera, for pick (measured on the arm on 2026-11-15 with the king size
# Sharpie; adjust here after the next real run). From the hover pose the wrist picture is read by
# backend/vision/wrist_aim.py (the jaw gap centre and the jaw row live there, in AIM). Then:
#   roll_new = roll_now - angle_deg * roll_ticks_per_deg     (angle: clockwise from vertical)
#   pan_new  = pan_now + offset_px / pan_px_per_tick          (offset: item x minus jaw centre x)
# Subtracting 400 ticks of roll turned a pen seen 35 degrees clockwise to vertical. Turning the
# base 35 ticks lower moved the pen about 50 px right in the picture at 29.5 cm reach (1.4 px per
# tick); the gain grows with reach (the camera swings further per tick), so it is scaled by
# dist_cm / pan_gain_ref_cm when pan_gain_scales is on.
WRIST_AIM = {
    "tiers": ("low",),           # tiers that aim with the wrist camera by default (PHANTOM_WRIST_AIM=all|low|off)
    "rounds": 3,                 # at most this many looks
    "angle_tol_deg": 6.0,        # within this, the roll is left alone
    "offset_tol_px": 25.0,       # within this (640 wide picture), the pan is left alone
    "roll_ticks_per_deg": 4096 / 360.0,   # 11.38
    "roll_sign": -1,             # roll_new = roll_now + roll_sign * angle * ticks_per_deg
    "pan_px_per_tick": 1.4,
    "pan_gain_ref_cm": 29.5,
    "pan_gain_scales": True,
    "pan_sign": 1,               # pan_new = pan_now + pan_sign * offset_px / gain
    "max_roll_step": 500,        # ticks in one correction, never more
    "max_pan_step": 80,
    "settle_s": 0.3,             # wait after a move before the next picture
    "roll_speed": 900,           # ticks per second for the correction moves (pace stretches them a little)
    "pan_speed": 150,
    "min_move_s": 0.4,
}


def aim_step(reading, roll_now, pan_now, dist_cm=None, cfg=WRIST_AIM, roll_range=ROLL_RANGE, pan_range=None):
    """What one wrist reading asks for. The roll is corrected first and alone, since turning the
    wrist turns the picture: the pan is only corrected once the angle is within tolerance (or the
    item is round). Returns {"done", "roll", "pan", "d_roll", "d_pan", "capped", "why"}; roll and
    pan are the new goals, or None when that joint stays."""
    out = {"done": False, "roll": None, "pan": None, "d_roll": 0, "d_pan": 0, "capped": False, "why": ""}
    ang, off = reading.get("angle_deg"), reading.get("offset_px")
    angle_off = ang is not None and abs(ang) > cfg["angle_tol_deg"]
    offset_off = off is not None and abs(off) > cfg["offset_tol_px"]
    if angle_off:
        want = cfg["roll_sign"] * ang * cfg["roll_ticks_per_deg"]
        step = max(-cfg["max_roll_step"], min(cfg["max_roll_step"], want))
        out["capped"] = abs(step) < abs(want) - 0.5
        new = int(round(roll_now + step))
        new = max(roll_range[0], min(roll_range[1], new))
        if new != int(roll_now):
            out.update(roll=new, d_roll=new - int(roll_now), why=f"turned {ang:+.0f} deg from the jaws")
            return out
        out["capped"] = False           # the wrist is at its limit: go on to the pan
    if offset_off:
        gain = cfg["pan_px_per_tick"]
        if cfg.get("pan_gain_scales") and dist_cm:
            gain *= max(0.4, min(2.5, dist_cm / cfg["pan_gain_ref_cm"]))
        want = cfg["pan_sign"] * off / gain
        step = max(-cfg["max_pan_step"], min(cfg["max_pan_step"], want))
        out["capped"] = abs(step) < abs(want) - 0.5
        new = int(round(pan_now + step))
        if pan_range is not None:
            new = max(pan_range[0], min(pan_range[1], new))
        side = "left" if off < 0 else "right"
        out.update(pan=new, d_pan=new - int(pan_now), why=f"{abs(off):.0f} px {side} of the jaw centre")
        return out
    out["done"] = True
    bits = ([f"within {abs(ang):.0f} deg"] if ang is not None else ["round"]) + \
           ([f"{abs(off):.0f} px"] if off is not None else [])
    out["why"] = "lined up (" + ", ".join(bits) + ")"
    return out


def aim_tiers(setting=None):
    """Which tiers aim with the wrist camera: from the arm's own setting, else PHANTOM_WRIST_AIM
    ("low", the default; "all"; "off"), else WRIST_AIM["tiers"]."""
    import os
    s = setting if setting is not None else os.environ.get("PHANTOM_WRIST_AIM", "")
    if isinstance(s, (tuple, list, set)):
        return tuple(s)
    s = str(s).strip().lower()
    if s in ("all", "on", "1", "true"):
        return ("tall", "can", "low")
    if s in ("off", "0", "none", "false"):
        return ()
    if s:
        return tuple(t.strip() for t in s.split(",") if t.strip())
    return tuple(WRIST_AIM["tiers"])


def aim_words(rounds):
    """One line for the result detail from the rounds the aim loop kept."""
    if not rounds:
        return ""
    parts = []
    for r in rounds:
        if r.get("unread"):
            parts.append(f"look {r['round']}: could not read the wrist picture")
            continue
        src = r.get("source") or "?"
        if r.get("done"):
            parts.append(f"look {r['round']} ({src}): {r['why']}")
        elif r.get("d_roll"):
            parts.append(f"look {r['round']} ({src}): {r['why']}, wrist roll {r['d_roll']:+d}" + (" (capped)" if r.get("capped") else ""))
        elif r.get("d_pan"):
            parts.append(f"look {r['round']} ({src}): {r['why']}, pan {r['d_pan']:+d}" + (" (capped)" if r.get("capped") else ""))
        else:
            parts.append(f"look {r['round']} ({src}): {r['why']}")
    return "wrist aim: " + "; ".join(parts)


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
