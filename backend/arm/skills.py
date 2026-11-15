"""Arm skill list, parameter checks and the timing shared by FakeArm and RealArm.

Follows docs/contract.md, sections "Arm skills" and "Skill server interface".
"""
import math
from typing import Callable, Protocol

PHASES = ("start", "at_target", "gripped", "lifted", "put_back", "missed", "returning", "done", "error")
RESULTS = ("done", "missed", "stopped", "error", "busy")

# skill -> {parameter: default}. Parameters not listed here are dropped by validate().
SKILLS = {
    "grip_bottle": {"pace": 0.5, "hold_s": 5.0, "flourish": False, "look_first": False},
    "point": {"pace": 0.5},
    "half_reach": {"pace": 0.5},
    "wave": {},
    "rest": {},
    "gesture": {"style": "gowther", "seconds": 3.0},
}
RANGES = {"pace": (0.0, 1.0), "hold_s": (1.0, 8.0), "seconds": (1.5, 10.0)}
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
    out = {}
    for key, default in SKILLS[skill].items():
        v = params.get(key, default)
        if key == "style":
            s = v.strip().lower() if isinstance(v, str) else ""
            out[key] = s if s in STYLES else default
        elif key in RANGES:
            out[key] = _num(v, default, *RANGES[key])
        else:
            out[key] = _flag(v, default)
    return out


def result(res, detail=""):
    return {"ok": res == "done", "result": res, "detail": detail}


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


def nominal_duration(skill, params=None):
    """About how long a skill takes, in seconds, without stops."""
    p = validate(skill, params)
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


def gesture_plan(style, seconds, rest=None):
    """Rise and return time, pose fraction and the talking pose (joints 1 to 4) for one line."""
    rest = rest or REST_POSE
    st = GESTURE_STYLE[style]
    rise_s = max(1.0, min(1.8, 0.9 + 0.12 * seconds))     # short lines rise quickly
    f_cap = min(RISE_V[2] * (1 - LEAD) * rise_s / abs(PHOTO_POSE[2] - rest[2]),
                RISE_V[3] * rise_s / abs(PHOTO_POSE[3] - rest[3]))
    f = min(st["rise"], f_cap)                               # a short line rises less, never faster
    line = {i: round(rest[i] + f * (PHOTO_POSE[i] - rest[i])) for i in range(1, 5)}
    pose = dict(line)
    pose[4] = line[4] + st["flex"]
    return {"style": style, "f": f, "pose": pose, "flex_anchor": line[4], "grip": st["grip"],
            "rise_s": rise_s, "talk_s": seconds * st["talk"], "back_s": rise_s}


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
        speed = round(abs(g - prev[j]) / (0.8 * gap))
        speed = max(40, min(st["vmax"].get(j, 400), speed))
        steps.append((round(t, 3), j, g, speed, st["acc"]))
        prev[j] = g
    return steps
