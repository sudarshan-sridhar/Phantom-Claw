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
#
# Reach, for compact items only (a case, a cap: the reader gives "compact" and where it sits). A
# long item lying along the reach crosses the jaw row wherever the hover is, so the distance does
# not matter for it; a compact one has to sit just above the jaw row or the jaws close on air (the
# earbuds case miss on 2026-11-15: cut off at the bottom edge between the jaw tips). So the hover
# moves along the taught blend, using the tier poses for dist_cm + d:
#   d_cm = -(centre_y - reach_target_y) * reach_cm_per_px    (lower in the picture: nearer the arm)
# and an item cut off by the bottom edge moves back toward the base by max_reach_step_cm. The
# target row (y 235 of 360, just above the jaw row at 290) is a first guess. The gain comes from the
# jaw gap: about 220 px for roughly 5 cm at hover height, so about 0.035 cm per px (also a guess).
# Each round corrects one thing: the roll, else the pan, else the reach.
#
# Jaw opening: an item wider than wide_ratio of the jaw gap at the present opening (the gap model
# is in backend/vision/wrist_aim.py AIM: 220 px at 1750 ticks, growing 0.5 px per tick) makes the
# jaws open to ref + (width_px - ref_gap) / px_per_tick + open_margin_ticks before going down,
# never above open_max (the calibrated range is 1305 to 2788).
WRIST_AIM = {
    "tiers": ("low",),           # tiers that aim with the wrist camera by default (PHANTOM_WRIST_AIM=all|low|off)
    "rounds": 4,                 # at most this many looks
    "angle_tol_deg": 9.0,        # within this, the roll is left alone
    "offset_tol_px": 40.0,       # within this (640 wide picture), the pan is left alone
    "roll_ticks_per_deg": 4096 / 360.0,   # 11.38
    "roll_sign": -1,             # roll_new = roll_now + roll_sign * angle * ticks_per_deg
    "pan_px_per_tick": 1.4,
    "pan_gain_ref_cm": 29.5,
    "pan_gain_scales": True,
    "pan_sign": 1,               # pan_new = pan_now + pan_sign * offset_px / gain
    "max_roll_step": 500,        # ticks in one correction, never more
    "max_pan_step": 80,
    "settle_s": 0.3,             # longest wait after a move before the next picture
    "settle_min_s": 0.12,        # shortest one: then the picture is taken as soon as the joints read still
    "together_deg": 15.0,        # a turn this small barely moves the picture: the pan from the same picture goes with it
    # correction move time, seconds: (shortest, units per extra second, longest), roll and pan in ticks,
    # reach in cm. A small correction is slow and soft instead of a twitch at full speed; pace stretches
    # them a little. RealArm gives every joint of the move the same ramp, so they start and stop together.
    "move_s": {"roll": (0.5, 600.0, 1.2), "pan": (0.5, 250.0, 0.8), "reach": (0.5, 10.0, 0.8)},
    "move_acc_max": 16,          # acceleration register cap for those moves (x100 ticks/s^2)
    "min_move_s": 0.4,           # the jaw opening move
    "reach_target_y": 235.0,     # a compact item's centre row should sit here (first guess)
    "reach_cm_per_px": 0.035,    # cm of reach per px of that row (first guess, from the jaw gap)
    "reach_tol_px": 30.0,        # within this, the reach is left alone
    "max_reach_step_cm": 3.0,    # cm in one correction, never more; also the step back when cut off
    "wide_ratio": 0.6,           # wider than this share of the jaw gap: open wider before going down
    "open_margin_ticks": 120,    # extra opening on top of the item's width (about 60 px)
    "open_max": 2600,            # never open further than this
}


def aim_step(reading, roll_now, pan_now, dist_cm=None, cfg=WRIST_AIM, roll_range=ROLL_RANGE, pan_range=None,
             reach_range=None):
    """What one wrist reading asks for. The roll is corrected first, since turning the wrist turns
    the picture: the pan is only corrected once the angle is within tolerance (or the item is round),
    except that a small turn (together_deg or less) barely moves the picture, so the pan read from the
    same picture goes with it in one move. The reach (compact items only) comes once both are lined
    up. Returns {"done", "roll",
    "pan", "reach", "d_roll", "d_pan", "d_reach", "capped", "why"}; roll and pan are the new
    goals and reach the new distance in cm, or None when that stays."""
    out = {"done": False, "roll": None, "pan": None, "reach": None, "d_roll": 0, "d_pan": 0, "d_reach": 0.0,
           "capped": False, "why": ""}
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
            if not (offset_off and abs(ang) <= cfg.get("together_deg", 0.0)):
                return out
        else:
            out["capped"] = False       # the wrist is at its limit: go on to the pan
    if offset_off:
        gain = cfg["pan_px_per_tick"]
        if cfg.get("pan_gain_scales") and dist_cm:
            gain *= max(0.4, min(2.5, dist_cm / cfg["pan_gain_ref_cm"]))
        want = cfg["pan_sign"] * off / gain
        step = max(-cfg["max_pan_step"], min(cfg["max_pan_step"], want))
        capped = abs(step) < abs(want) - 0.5
        new = int(round(pan_now + step))
        if pan_range is not None:
            new = max(pan_range[0], min(pan_range[1], new))
        side = "left" if off < 0 else "right"
        why = f"{abs(off):.0f} px {side} of the jaw centre"
        if out["roll"] is not None:     # a small turn and the pan together
            why, capped = f"{out['why']} and {why}", capped or out["capped"]
        out.update(pan=new, d_pan=new - int(pan_now), why=why, capped=capped)
        return out
    out["capped"] = False
    reach = reach_want(reading, cfg)
    if reach is not None and dist_cm is not None:
        want, why = reach
        step = max(-cfg["max_reach_step_cm"], min(cfg["max_reach_step_cm"], want))
        out["capped"] = abs(step) < abs(want) - 0.05
        new = dist_cm + step
        if reach_range is not None:
            new = max(reach_range[0], min(reach_range[1], new))
        if abs(new - dist_cm) >= 0.1:
            out.update(reach=round(new, 2), d_reach=round(new - dist_cm, 2), why=why)
            return out
        out["capped"] = False           # at the end of the taught reach: carry on as it is
    out["done"] = True
    bits = ([f"within {abs(ang):.0f} deg"] if ang is not None else ["round"]) + \
           ([f"{abs(off):.0f} px"] if off is not None else []) + \
           ([f"row {reading['centre_y']:.0f}"] if reading.get("compact") and reading.get("centre_y") is not None else [])
    out["why"] = "lined up (" + ", ".join(bits) + ")"
    return out


def aim_move_s(d_roll=0, d_pan=0, d_reach_cm=0.0, cfg=WRIST_AIM):
    """Seconds for one correction move: a bigger one takes longer, a small one is slow rather than
    a twitch. With several joints the longest time wins and they all use it."""
    t = 0.0
    for d, key in ((abs(d_roll or 0), "roll"), (abs(d_pan or 0), "pan"), (abs(d_reach_cm or 0.0), "reach")):
        if d:
            lo, per, hi = cfg["move_s"][key]
            t = max(t, min(hi, lo + d / per))
    return t or cfg["move_s"]["pan"][0]


def reach_want(reading, cfg=WRIST_AIM):
    """(cm to move the hover out, why) for a compact item that does not sit at the target row, or
    None. Negative: back toward the base."""
    if not reading.get("compact") or reading.get("centre_y") is None:
        return None
    if reading.get("touches_bottom"):
        return -cfg["max_reach_step_cm"], "cut off by the bottom edge: too near, moving back"
    dy = reading["centre_y"] - cfg["reach_target_y"]
    if abs(dy) <= cfg["reach_tol_px"]:
        return None
    where = "too near, moving back" if dy > 0 else "too far, moving out"
    return -dy * cfg["reach_cm_per_px"], f"centre row {reading['centre_y']:.0f}, {abs(dy):.0f} px off: {where}"


def jaw_opening(width_px, open_now, cfg=WRIST_AIM):
    """Gripper ticks to open to before going down for an item width_px wide, or None when the
    present opening will do (the item is at most wide_ratio of the jaw gap)."""
    from backend.vision.wrist_aim import AIM, jaw_gap
    if not width_px or open_now is None:
        return None
    if width_px <= cfg["wide_ratio"] * jaw_gap(open_now):
        return None
    ref = AIM["jaw_open_ref"]
    want = ref + (width_px - jaw_gap(ref)) / AIM["jaw_px_per_tick"] + cfg["open_margin_ticks"]
    want = int(round(min(cfg["open_max"], want)))
    return want if want > open_now + 5 else None


def is_compact(name, rounds=()):
    """True for an item that is not long and thin: named so (an earbuds case, a cap), or read as
    compact by the wrist camera on any look."""
    from backend.vision.wrist_aim import shape_for
    if shape_for(name) == "round":
        return True
    return any(r.get("compact") for r in rounds or ())


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
        if r.get("opened"):
            parts.append(f"jaws opened to {r['opened']} for a {r['width_px']:.0f} px wide item")
            continue
        src = r.get("source") or "?"
        if r.get("done"):
            parts.append(f"look {r['round']} ({src}): {r['why']}")
        elif r.get("d_roll"):
            parts.append(f"look {r['round']} ({src}): {r['why']}, wrist roll {r['d_roll']:+d}"
                         + (f" and pan {r['d_pan']:+d}" if r.get("d_pan") else "") + (" (capped)" if r.get("capped") else ""))
        elif r.get("d_pan"):
            parts.append(f"look {r['round']} ({src}): {r['why']}, pan {r['d_pan']:+d}" + (" (capped)" if r.get("capped") else ""))
        elif r.get("d_reach"):
            parts.append(f"look {r['round']} ({src}): {r['why']}, reach {r['d_reach']:+.1f} cm"
                         + (" (capped)" if r.get("capped") else ""))
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
    if tier_name == "low" and is_compact(name):
        # a round or compact item has no long side for the overhead angle to mean anything (the
        # earbuds case got roll 1450 from a -30 deg "angle" and missed): start at the taught roll
        rolls = [s["roll"] for s in tier.spots if s.get("roll") is not None]
        roll = int(round(sum(rolls) / len(rolls))) if rolls else None
    elif tier_name == "low" and item.get("axis_deg") is not None:
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
