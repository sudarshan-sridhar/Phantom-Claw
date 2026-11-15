"""Arm skill list, parameter checks and the timing shared by FakeArm and RealArm.

Follows docs/contract.md, sections "Arm skills" and "Skill server interface".
"""
import math
from typing import Callable, Protocol

PHASES = ("auto_rest", "start", "seen", "at_target", "gripped", "lifted", "carrying", "offering", "handed_over",
          "put_back", "missed", "returning", "done", "error")
RESULTS = ("done", "handed_over", "put_back", "missed", "no_bottle", "out_of_reach", "turn_failed",
           "stopped", "error", "busy",
           # the general skills (look, point_at, pick, place, hand_over)
           "holding", "placed", "not_found", "no_picture", "not_taught", "not_graspable", "nothing_held",
           "not_taken", "hands_full")
OK_RESULTS = ("done", "handed_over", "put_back", "holding", "placed")   # the skill did its job

# skill -> {parameter: default}. Parameters not listed here are dropped by validate().
SKILLS = {
    "grip_bottle": {"pace": 0.5, "hold_s": 5.0, "flourish": False, "look_first": False},
    "fetch_bottle": {"pace": 0.5, "wait_s": 3.0, "look_first": False},
    "point": {"pace": 0.5},
    "half_reach": {"pace": 0.5},
    "wave": {},
    "rest": {},
    "gesture": {"style": "gowther", "seconds": 3.0},
    # general building blocks: any item on the desk, no fixed routine (backend/arm/README.md)
    "look": {"target": None},
    "point_at": {"target": None, "pace": 0.5},
    "pick": {"target": None, "pace": 0.5, "approach": "direct", "lift_cm_style": "normal"},
    "place": {"where": "back", "pace": 0.5, "approach": "direct"},
    "hand_over": {"wait_s": 3.0, "pace": 0.5},
}
GENERAL = ("look", "point_at", "pick", "place", "hand_over")
RANGES = {"pace": (0.0, 1.0), "hold_s": (1.0, 8.0), "wait_s": (2.0, 6.0), "seconds": (1.5, 10.0)}
CHOICES = {"approach": ("direct", "look_first"), "lift_cm_style": ("low", "normal", "high")}
SIDES = ("left", "right", "front", "behind")
STYLES = ("escanor", "elizabeth", "ban", "king", "gowther", "merlin")


class Arm(Protocol):
    def run(self, skill: str, params: dict, on_phase: Callable[[str], None]) -> dict: ...
    def stop(self) -> None: ...
    def status(self) -> dict: ...


def _num(v, default, lo, hi):
    if v is None or isinstance(v, bool):
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if math.isnan(f):
        return default
    return max(lo, min(hi, f))


def _flag(v, default):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
        return v != 0
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "on", "1"):
            return True
        if s in ("false", "no", "off", "0", ""):
            return False
    return default


def validate(skill, params=None):
    """Return the skill's parameters with defaults filled in and numbers clamped to the
    contract's ranges. Unknown parameters are dropped. Raises ValueError for an unknown skill."""
    if skill not in SKILLS:
        raise ValueError(f"unknown arm skill: {skill!r}")
    params = params if isinstance(params, dict) else {}
    if "approach" in SKILLS[skill] and "approach" not in params and _flag(params.get("look_first"), False):
        params = {**params, "approach": "look_first"}     # the fetch_bottle name for the same manner
    out = {}
    for key, default in SKILLS[skill].items():
        v = params.get(key, default)
        if key == "style":
            s = v.strip().lower() if isinstance(v, str) else ""
            out[key] = s if s in STYLES else default
        elif key in RANGES:
            out[key] = _num(v, default, *RANGES[key])
        elif key in CHOICES:
            s = v.strip().lower().replace(" ", "_") if isinstance(v, str) else ""
            out[key] = s if s in CHOICES[key] else default
        elif key == "target":
            out[key] = _target(v)
        elif key == "where":
            out[key] = _where(v)
        else:
            out[key] = _flag(v, default)
    return out


def _xy(v):
    if isinstance(v, dict):
        x, y = _num(v.get("x_cm"), None, -200, 200), _num(v.get("y_cm"), None, -200, 200)
        if x is not None and y is not None:
            return {"x_cm": x, "y_cm": y}
    return None


def _name(v):
    if isinstance(v, str):
        s = " ".join(v.strip().split())[:60]
        return s or None
    return None


def _target(v):
    """An item name (str) or a desk position {"x_cm", "y_cm"}; None when neither."""
    return _xy(v) or _name(v)


def _where(v):
    """"back", {"x_cm", "y_cm"} or {"near": name, "side": left|right|front|behind}. A plain name
    means in front of that item (the side nearer the arm)."""
    if v is None:
        return "back"
    if isinstance(v, str):
        s = _name(v)
        if s is None or s.lower() in ("back", "where it was", "put it back", "original"):
            return "back"
        return {"near": s, "side": "front"}
    if isinstance(v, dict):
        xy = _xy(v)
        if xy:
            return xy
        near = _name(v.get("near"))
        if near:
            side = v.get("side").strip().lower() if isinstance(v.get("side"), str) else ""
            return {"near": near, "side": side if side in SIDES else "front"}
    return "back"


def result(res, detail="", **extra):
    return {"ok": res in OK_RESULTS, "result": res, "detail": detail, **extra}


# Timing. grip_bottle.py used 4.2 s legs, a 1.0 s lift and a 1.2 s lower.
SLOW_LEG_S, FAST_LEG_S = 6.5, 3.4      # pace 0.0 and pace 1.0, rest to grasp and back
BASE_LEG_S = 4.2
POINT_LEG = 0.6                        # point legs as a fraction of the grip leg (via is closer than grasp)
HALF_LEG = 0.5                         # half_reach legs as a fraction of the grip leg
HALF_FRACTION = 0.6                    # half_reach goes this far from rest toward the via pose
LOOK_LEG1, LOOK_LEG2 = 0.7, 0.45       # look_first: rest to via, then via to grasp
LOOK_PAUSE_S = 1.0
POINT_HOLD_S, HALF_HOLD_S = 2.0, 1.0
GRIPPER_OPEN_S = 1.2                   # gripper opens during the first 1.2 s of the out leg
CLOSE_S = 1.5                          # about how long close_gripper takes (open to the cap at speed 300)
RELEASE_S = 0.8                        # open the gripper (0.6 s) plus the 0.2 s pause
WAVE_LEG_S = 3.8                       # rest to the photo pose and back
WAVE_MOVE_S = 6.0                      # the gif_thumbnail timeline (4.5 s) plus its 1.5 s tail
REST_LEG_S = 4.2

# Flourish: wrist roll offsets (ticks) from its current value, as (time s, offset). Two full rolls.
FLOURISH = [(0.0, 300), (0.9, -300), (2.15, 300), (3.4, -300), (4.65, 0)]
FLOURISH_SPEED, FLOURISH_ACC = 800, 16
FLOURISH_S = 5.5                       # the flourish takes this long; the hold is at least this long


def leg_time(pace):
    return SLOW_LEG_S - (SLOW_LEG_S - FAST_LEG_S) * pace


def lift_time(pace):
    return 1.0 * leg_time(pace) / BASE_LEG_S


def lower_time(pace):
    return 1.2 * leg_time(pace) / BASE_LEG_S


def hold_time(hold_s, flourish):
    return max(hold_s, FLOURISH_S) if flourish else hold_s


# fetch_bottle, the numbers from setup/tools/fetch_bottle.py. At pace 1.0 every leg takes exactly
# the script's time; lower paces stretch every arm leg, up to 1.8 times at pace 0. Gripper
# actions and the short pauses keep the script's times.
FETCH_OUT_S = 3.3            # rest to just above the cap, all joints in one move
FETCH_OPEN_S = 1.2           # the gripper opens during the first part of the out leg
FETCH_APPROACH_S = 1.0       # the short final approach onto the cap
FETCH_LOOK_FIRST_S = 1.0     # look_first: pause above the cap before the final approach
FETCH_TUCK_S = 1.9           # lift and pull the bottle in close
FETCH_TURN_MIN_S, FETCH_TURN_RATE = 1.8, 400.0   # base turn: at least 1.8 s, at most 400 ticks/s
FETCH_REACH_S = 1.5          # reach out toward the person in the last half of the turn
FETCH_REST_S = 4.5           # back to rest
FETCH_VIA_S = 1.2            # back up above the cap (missed grip, after the put back)
FETCH_ABOVE_SPOT_S = 1.6     # put back: down to just above the spot
FETCH_SET_S = 1.2            # put back: stand it on the table
FETCH_CLEAR_S = 1.0          # after a handover, lift clear of the hand
FETCH_RELEASE_S = 0.6        # open the gripper to let go on the table
FETCH_HANDOVER_OPEN_S = 0.4  # open the gripper when the person takes it
FETCH_HANDOVER_WAIT_S = 0.8  # give the hand time to come away before moving
FETCH_AFTER_OPEN_S = 0.2
FETCH_OFFER_SETTLE_S = 0.35  # let the arm settle before taking the tug reference
RECOVER_TUCK_S, RECOVER_ALIGN_S, RECOVER_TABLE_S, RECOVER_CLEAR_S = 1.5, 1.0, 1.6, 1.2
FETCH_CLOSE_S = 0.8          # about how long the close takes (speed 500 onto the cap)
FETCH_LOOK_S = 3.0           # about how long the overhead picture and Gemini take
WRIST_LOOK_S = 1.6           # one wrist-camera stop of the sweep: small move plus Gemini


def fetch_stretch(pace):
    """1.0 at pace 1 (the script's timings), 1.8 at pace 0."""
    return 1.8 - 0.8 * max(0.0, min(1.0, pace))


def fetch_turn_s(pan_from, pan_to, pace):
    return max(FETCH_TURN_MIN_S, abs(pan_to - pan_from) / FETCH_TURN_RATE) * fetch_stretch(pace)


def fetch_nominal(pace, wait_s, look_first, outcome="handed_over", turn_ticks=600):
    """About how long fetch_bottle takes for one outcome, in seconds."""
    k = fetch_stretch(pace)
    t = FETCH_LOOK_S
    if outcome in ("no_bottle", "out_of_reach"):
        return t
    t += FETCH_OUT_S * k + FETCH_APPROACH_S * k + FETCH_CLOSE_S + (FETCH_LOOK_FIRST_S if look_first else 0.0)
    if outcome == "missed":
        return t + FETCH_RELEASE_S + FETCH_VIA_S * k + FETCH_REST_S * k
    t += FETCH_TUCK_S * k
    turn = max(FETCH_TURN_MIN_S, turn_ticks / FETCH_TURN_RATE) * k
    if outcome == "turn_failed":
        return t + 0.5 * turn + 1.2 + (RECOVER_TUCK_S + RECOVER_ALIGN_S + RECOVER_TABLE_S + RECOVER_CLEAR_S) * k \
            + FETCH_RELEASE_S + FETCH_AFTER_OPEN_S + FETCH_REST_S * k
    t += max(turn, 0.5 * turn + FETCH_REACH_S * k) + 0.3 + FETCH_OFFER_SETTLE_S
    if outcome == "handed_over":
        return t + 0.5 * wait_s + FETCH_HANDOVER_OPEN_S + FETCH_HANDOVER_WAIT_S + FETCH_CLEAR_S * k + FETCH_REST_S * k
    return (t + wait_s + FETCH_TUCK_S * k + turn + FETCH_ABOVE_SPOT_S * k + FETCH_SET_S * k + FETCH_RELEASE_S
            + FETCH_AFTER_OPEN_S + FETCH_VIA_S * k + FETCH_REST_S * k)


# The general skills reuse the fetch_bottle legs and the same pace stretch.
POINT_AT_RAISE = 120         # point_at: shoulder this much above the pose over the item, so nothing is touched
POINT_AT_HOLD_S = 2.0
TURN_TICKS = 600             # a typical base turn, for the time estimates


def general_nominal(skill, p, outcome=None):
    """About how long one of the general skills takes, in seconds, for one outcome."""
    k = fetch_stretch(p.get("pace", 0.5))
    look_first = p.get("approach") == "look_first"
    turn = max(FETCH_TURN_MIN_S, TURN_TICKS / FETCH_TURN_RATE) * k
    if skill == "look":
        return FETCH_LOOK_S
    if skill == "point_at":
        look = 0.0 if isinstance(p.get("target"), dict) else FETCH_LOOK_S
        if outcome in ("not_found", "out_of_reach", "no_picture"):
            return look
        return look + FETCH_OUT_S * k + POINT_AT_HOLD_S + FETCH_REST_S * k
    if skill == "pick":
        if outcome in ("not_found", "out_of_reach", "no_picture", "not_taught", "not_graspable"):
            return FETCH_LOOK_S
        t = FETCH_LOOK_S + FETCH_OUT_S * k + (FETCH_LOOK_FIRST_S if look_first else 0.0) + FETCH_APPROACH_S * k + FETCH_CLOSE_S
        if outcome == "missed":
            return t + FETCH_RELEASE_S + FETCH_VIA_S * k + FETCH_REST_S * k
        return t + FETCH_TUCK_S * k
    if skill == "place":
        look = FETCH_LOOK_S if isinstance(p.get("where"), dict) and "near" in p["where"] else 0.0
        if outcome in ("not_found", "out_of_reach", "nothing_held"):
            return look
        return (look + turn + FETCH_ABOVE_SPOT_S * k + (FETCH_LOOK_FIRST_S if look_first else 0.0) + FETCH_SET_S * k
                + FETCH_RELEASE_S + FETCH_AFTER_OPEN_S + FETCH_VIA_S * k + FETCH_REST_S * k)
    if skill == "hand_over":
        if outcome == "nothing_held":
            return 0.0
        t = max(turn, 0.5 * turn + FETCH_REACH_S * k) + 0.3 + FETCH_OFFER_SETTLE_S
        if outcome == "not_taken":
            return t + p.get("wait_s", 3.0) + FETCH_TUCK_S * k
        return (t + 0.5 * p.get("wait_s", 3.0) + FETCH_HANDOVER_OPEN_S + FETCH_HANDOVER_WAIT_S + FETCH_CLEAR_S * k
                + FETCH_REST_S * k)
    raise ValueError(skill)


def nominal_duration(skill, params=None):
    """About how long a skill takes, in seconds, without stops."""
    p = validate(skill, params)
    if skill in GENERAL:
        return general_nominal(skill, p)
    if skill == "fetch_bottle":
        return fetch_nominal(p["pace"], p["wait_s"], p["look_first"])
    if skill == "grip_bottle":
        t = leg_time(p["pace"])
        out = (LOOK_LEG1 + LOOK_LEG2) * t + LOOK_PAUSE_S if p["look_first"] else t
        return (out + CLOSE_S + lift_time(p["pace"]) + hold_time(p["hold_s"], p["flourish"])
                + lower_time(p["pace"]) + RELEASE_S + t)
    if skill == "point":
        return 2 * POINT_LEG * leg_time(p["pace"]) + POINT_HOLD_S
    if skill == "half_reach":
        return 2 * HALF_LEG * leg_time(p["pace"]) + HALF_HOLD_S
    if skill == "wave":
        return 2 * WAVE_LEG_S + WAVE_MOVE_S
    if skill == "gesture":
        g = gesture_plan(p["style"], p["seconds"])
        return g["rise_s"] + g["talk_s"] + g["back_s"]
    return REST_LEG_S + 0.8


# Gesture: the arm as the body of whoever is speaking (contract, "The arm is the body ...").
# Every talking pose lies on the joint-space line from rest to the photo pose, at fraction f
# (1.0 is the photo pose, the tallest one allowed). Moves during the line are timed single
# goals per joint, like gif_thumbnail.py, scaled to fit the line.
LEAD = 0.3                                     # the elbow leads the shoulder by this fraction
REST_POSE = {1: 2009, 2: 946, 3: 3190, 4: 2632}
PHOTO_POSE = {1: 2029, 2: 1639, 3: 2490, 4: 2223}
GRIP_CLOSED, JAW_SHUT, JAW_OPEN = 1310, 1400, 1900
GESTURE_PAN = (PHOTO_POSE[1] - 250, PHOTO_POSE[1] + 250)
GESTURE_ROLL = (1100, 2960)
GESTURE_FLEX_SPAN = 180
RISE_V = {2: 600, 3: 650}                      # rise and return speed caps, ticks/s, shoulder and elbow

# rise: pose fraction f. flex: wrist flex offset at the talking pose. grip: gripper goal while talking.
# period: seconds per cycle of keys (None = the keys span the whole line once).
# talk: fraction of the line the moves last (king flops back early).
# keys: (cycle fraction 0..1, joint, offset from the talking pose); joint 6 values are absolute.
GESTURE_STYLE = {
    "escanor": {"rise": 1.0, "flex": 100, "grip": GRIP_CLOSED, "period": 3.2, "talk": 1.0, "acc": 8,
                "vmax": {1: 350},
                "keys": [(0.0, 1, 200), (0.5, 1, -200)]},
    "elizabeth": {"rise": 0.45, "flex": 0, "grip": GRIP_CLOSED, "period": 2.4, "talk": 1.0, "acc": 6,
                  "vmax": {1: 150, 4: 250},
                  "keys": [(0.0, 1, 80), (0.0, 4, -90), (0.25, 4, 0), (0.5, 1, -80), (0.5, 4, -90), (0.75, 4, 0)]},
    "ban": {"rise": 0.85, "flex": 0, "grip": GRIP_CLOSED, "period": 1.0, "talk": 1.0, "acc": 14,
            "vmax": {1: 650, 4: 450, 6: 900},
            "keys": [(0.0, 1, 180), (0.0, 4, -120), (0.15, 6, JAW_SHUT), (0.35, 6, JAW_OPEN),
                     (0.5, 1, -180), (0.5, 4, 100), (0.65, 6, JAW_SHUT), (0.85, 6, JAW_OPEN)]},
    "king": {"rise": 0.4, "flex": 0, "grip": GRIP_CLOSED, "period": None, "talk": 0.75, "acc": 5,
             "vmax": {2: 200, 3: 200, 4: 200},
             "keys": [(0.0, 2, -70), (0.0, 3, 70), (0.35, 2, 40), (0.35, 4, -100), (0.65, 2, -70), (0.65, 4, 0)]},
    "gowther": {"rise": 0.75, "flex": 0, "grip": GRIP_CLOSED, "period": None, "talk": 1.0, "acc": 10,
                "vmax": {5: 300},
                "keys": [(0.35, 5, 250), (0.75, 5, 0)]},
    "merlin": {"rise": 0.8, "flex": 0, "grip": 1550, "period": 3.0, "talk": 1.0, "acc": 5,
               "vmax": {1: 200, 2: 150},
               "keys": [(0.0, 1, 120), (0.25, 2, -80), (0.5, 1, -120), (0.75, 2, 0)]},
}


# How long a servo move really takes. The servo ramps up and down at its acceleration
# register (x100 ticks/s^2), so a short leg asked to take 0.6 s can take twice that.
# The gesture planner uses this so the whole gesture fits the line.
GESTURE_LEG_ACC = 14                           # rise and return ramps (ban's talking moves use 14 too)
LEG_VMAX = {1: 500, 2: RISE_V[2], 3: RISE_V[3], 4: 500, 6: 900}
EDGE_FRAC, EDGE_MIN, EDGE_MAX = 0.25, 0.9, 1.8   # rise (and return) budget: a quarter of the line
TALK_MIN_S = 0.4
GESTURE_OVER_S = 0.4                           # the body may finish this much after the voice


def move_time(d, dur, acc):
    """Seconds for one servo to travel d ticks when its speed register is set for dur seconds."""
    d = abs(d)
    if d <= 2:
        return 0.0
    v = max(12, round(d / max(dur, 1e-3)))
    a = max(1, acc) * 100.0
    if d >= v * v / a:
        return d / v + v / a
    return 2 * math.sqrt(d / a)


def _leg_items(dists, dur, back):
    """(joint, start, duration) as RealArm._leg_out (back False) and _leg_back (back True) time them."""
    out = []
    for j, d in dists.items():
        if j == 2:
            out.append((j, d, 0.0 if back else LEAD * dur, (1 - LEAD) * dur))
        elif j == 6:
            out.append((j, d, 0.6 * dur, 0.4 * dur) if back else (j, d, 0.0, min(GRIPPER_OPEN_S, dur)))
        else:
            out.append((j, d, 0.0, dur))
    return out


def leg_real_time(dists, dur, back=False, acc=GESTURE_LEG_ACC):
    """Real seconds for one rise (back False) or return (back True) leg, and whether every joint
    stays under its speed cap."""
    real, ok = 0.0, True
    for j, d, ts, dj in _leg_items(dists, dur, back):
        if d > 2 and d / dj > LEG_VMAX.get(j, 500) + 1:
            ok = False
        real = max(real, ts + move_time(d, dj, acc))
    return real, ok


def _fit_leg(dists, budget, back):
    """Shortest real time for this leg inside the speed caps: returns (commanded s, real s)."""
    best = None
    for n in range(2, 41):
        c = budget * n / 20.0
        real, ok = leg_real_time(dists, c, back)
        if ok and (best is None or real < best[1] - 1e-9):
            best = (c, real)
    if best is None:
        c = 2 * budget
        while not leg_real_time(dists, c, back)[1]:
            c *= 1.25
        best = (c, leg_real_time(dists, c, back)[0])
    return best


def _pose_at(style, f, rest):
    st = GESTURE_STYLE[style]
    line = {i: round(rest[i] + f * (PHOTO_POSE[i] - rest[i])) for i in range(1, 5)}
    pose = dict(line)
    pose[4] = line[4] + st["flex"]
    return line, pose


def gesture_plan(style, seconds, rest=None):
    """Pose fraction, talking pose (joints 1 to 4) and timings for one line.

    The whole gesture (rise, talk, return) is planned to last about the line's length plus
    GESTURE_OVER_S. Rise and return each get about a quarter of the line; the pose height is
    then the tallest one the servos can really reach in that time, so a short line rises less
    instead of moving faster or running long. rise_s and back_s are the real expected times;
    rise_cmd and back_cmd are what the legs are asked for (the ramps add the difference)."""
    rest = rest or REST_POSE
    st = GESTURE_STYLE[style]
    seconds = max(RANGES["seconds"][0], min(RANGES["seconds"][1], float(seconds)))
    edge = max(EDGE_MIN, min(EDGE_MAX, EDGE_FRAC * seconds))
    grip_d = abs(st["grip"] - GRIP_CLOSED)

    def legs(f, with_grip=True):
        _, pose = _pose_at(style, f, rest)
        dists = {j: abs(pose[j] - rest[j]) for j in range(1, 5)}
        if with_grip:
            dists[6] = grip_d
        return _fit_leg(dists, edge, back=False), _fit_leg(dists, edge, back=True)

    def fits(f):
        # the gripper does not depend on the height, so only the arm joints decide it
        (_, up), (_, down) = legs(f, with_grip=False)
        return up <= edge + 1e-6 and down <= edge + 1e-6

    if fits(st["rise"]):
        f = st["rise"]
    else:
        lo, hi = 0.0, st["rise"]
        for _ in range(24):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if fits(mid) else (lo, mid)
        f = max(0.02, lo)
    line, pose = _pose_at(style, f, rest)
    (rise_cmd, rise_s), (back_cmd, back_s) = legs(f)
    talk_s = max(TALK_MIN_S, seconds + GESTURE_OVER_S - rise_s - back_s) * st["talk"]
    return {"style": style, "f": f, "pose": pose, "flex_anchor": line[4], "grip": st["grip"],
            "rise_s": round(rise_s, 3), "rise_cmd": round(rise_cmd, 3), "talk_s": round(talk_s, 3),
            "back_s": round(back_s, 3), "back_cmd": round(back_cmd, 3), "acc": GESTURE_LEG_ACC}


def gesture_clamp(j, g, flex_anchor, rest=None):
    """Gesture limits: pan within 250 of the photo pose, shoulder and elbow between rest and the
    photo pose, wrist flex within 180 of the pose value, wrist roll 1100 to 2960."""
    rest = rest or REST_POSE
    if j == 1:
        lo, hi = GESTURE_PAN
    elif j in (2, 3):
        lo, hi = sorted((rest[j], PHOTO_POSE[j]))
    elif j == 4:
        lo, hi = flex_anchor - GESTURE_FLEX_SPAN, flex_anchor + GESTURE_FLEX_SPAN
    elif j == 5:
        lo, hi = GESTURE_ROLL
    else:
        lo, hi = GRIP_CLOSED, JAW_OPEN
    return int(max(lo, min(hi, round(g))))


def fill_speed(d, gap, acc):
    """Cruise speed (ticks/s) for a move of d ticks to take the whole gap with the servo's ramp at
    acc, so a joint arrives as its next key starts instead of arriving early and standing still
    (stop and go). Too short a gap even for a pure ramp up and down: as fast as it can, the next
    key then turns it round on its ramp before it gets there."""
    a = max(1, acc) * 100.0
    disc = (a * gap) ** 2 - 4 * a * d
    if d <= 0:
        return 0
    if disc < 0:
        return round(d / (0.8 * gap))       # it never gets there anyway: the old speed, so the swing stays as big
    return round((a * gap - math.sqrt(disc)) / 2)


def gesture_steps(style, talk_s, base, flex_anchor, rest=None):
    """Timeline for the talking part: list of (time s, joint, goal, speed, acc), sorted by time.
    base: {joint: value at the talking pose} for joints 1 to 6. Ends with every moved joint back at base."""
    st = GESTURE_STYLE[style]
    period = st["period"]
    n = max(1, round(talk_s / period)) if period else 1
    per = talk_s / n
    end = max(0.0, talk_s - min(0.6, 0.15 * talk_s))
    raw = []
    for c in range(n):
        for u, j, v in st["keys"]:
            t = (c + u) * per
            if t < end - 0.05:        # keys that fall in the closing moment are dropped
                raw.append((t, j, v if j == 6 else base[j] + v))
    for j in sorted({j for _, j, _ in st["keys"]}):
        raw.append((end, j, base[j]))
    raw = [(t, j, gesture_clamp(j, g, flex_anchor, rest)) for t, j, g in raw]
    raw.sort(key=lambda s: (s[0], s[1]))
    steps, prev = [], dict(base)
    for k, (t, j, g) in enumerate(raw):
        nxt = next((t2 for t2, j2, _ in raw[k + 1:] if j2 == j and t2 > t), talk_s)
        gap = max(0.1, nxt - t)
        speed = fill_speed(abs(g - prev[j]), gap, st["acc"])
        speed = max(40, min(st["vmax"].get(j, 400), speed))
        steps.append((round(t, 3), j, g, speed, st["acc"]))
        prev[j] = g
    return steps
