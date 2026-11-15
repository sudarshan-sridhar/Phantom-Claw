"""FakeArm: the pretend arm. No hardware. Each skill sleeps for about as long as the real
one takes and reports the same phases in the same order as RealArm.

Rules shared with RealArm:
- one skill at a time; a second run while busy returns result "busy"
- stop() holds where it is; run returns "stopped"
- after a stop or an error the arm is not at rest; the next skill sends it to rest by itself
  first (phase "auto_rest", result key auto_rest True), then runs
- rest is safe from any pose; holding the bottle, it first stands it on the table and lets go

Switches for fetch_bottle (attributes, or environment variables when the server starts):
- fetch_outcome: "handed_over" (default), "put_back", "missed", "no_bottle", "out_of_reach",
  "turn_failed"                                   PHANTOM_FAKE_FETCH
- taken_by: "tug" or "hand", how a handover is noticed
- overhead: False pretends the overhead phone is off, so the wrist-camera sweep runs
                                                  PHANTOM_FAKE_OVERHEAD=0
- wrist_fallback: False turns the wrist-camera sweep off
- miss_grip: every grip misses (grip_bottle and fetch_bottle)  PHANTOM_FAKE_MISS=1

Switches for the general skills (look, point_at, pick, place, hand_over):
- scene_items: the pretend desk, a list of {"name", "x_cm", "y_cm", "tier", "graspable", "axis_deg"}
  (default: the owner's six items); person_bearing_deg: where the person sits (None: not seen)
- taught: which tiers have taught grips (default only "tall", as on the real arm today)
                                                  PHANTOM_FAKE_TIERS=tall,can,low
- hand_taken: False means nobody takes what hand_over holds out   PHANTOM_FAKE_TAKEN=0
- overhead: False also makes look, point_at and pick report no_picture
- aim_setting: which tiers line up with the wrist camera before going down (None reads
  PHANTOM_WRIST_AIM: "low" by default, "all", "off")
- wrist_reader: what the pretend wrist camera reads during that aim. None: a pretend item that
  starts a little turned and to the left and answers the corrections as the real one did. A list:
  scripted readings, one per look (None in it: the picture cannot be read). A function
  (look number, roll, pan) -> reading or None.
"""
import copy
import math
import os
import threading

from . import desk
from . import skills as S
from .desk import Refuse

NAMES = {1: "shoulder_pan", 2: "shoulder_lift", 3: "elbow_flex", 4: "wrist_flex", 5: "wrist_roll", 6: "gripper"}
REST = {1: 2009, 2: 946, 3: 3190, 4: 2632, 5: 2030, 6: 1310}
VIA = {1: 2032, 2: 1637, 3: 2610, 4: 2229}
GRASP = {1: 2012, 2: 1964, 3: 2475, 4: 1990}
HALF = {i: round(REST[i] + S.HALF_FRACTION * (VIA[i] - REST[i])) for i in VIA}
PHOTO = dict(S.PHOTO_POSE)
# fetch_bottle poses (the taught middle grip, the tuck, held out at full reach)
F_VIA = {1: 1932, 2: 1554, 3: 2656, 4: 2280}
F_GRASP = {1: 1932, 2: 1655, 3: 2700, 4: 2173}
F_TUCK = {2: 1425, 3: 2700, 4: 2173}
F_OUT = {1: 2620, 2: 2434, 3: 1840, 4: 2003}
WRIST_STOPS = 5

# The pretend desk for the general skills, in the arm frame (cm from the base, x right, y away).
FAKE_SCENE = [
    {"name": "water bottle", "x_cm": 2.0, "y_cm": 30.0, "tier": "tall", "graspable": True},
    {"name": "juice can", "x_cm": -12.0, "y_cm": 24.0, "tier": "can", "graspable": True},
    {"name": "red bull can", "x_cm": 14.0, "y_cm": 26.0, "tier": "can", "graspable": True},
    {"name": "cup", "x_cm": -6.0, "y_cm": 36.0, "tier": "can", "graspable": True, "width_cm": 8.0},
    {"name": "sharpie", "x_cm": 9.0, "y_cm": 34.0, "tier": "low", "graspable": True, "axis_deg": 30.0},
    {"name": "scissors", "x_cm": -15.0, "y_cm": 31.0, "tier": "low", "graspable": True, "axis_deg": -60.0},
]
PERSON_BEARING_DEG = 69.0          # about 1.2 rad to the right, where fetch_bottle holds things out
# Pretend taught grips: the real tall ones (bottle_teach.json mid and far), and made-up can and
# low ones at the same distances so the timing and the reach ring look like the real thing.
_TALL = [{"spot": "middle", "r_cm": 23.2, "via": {"lift": 1554, "elbow": 2656, "wrist_flex": 2280},
          "grasp": {"lift": 1655, "elbow": 2700, "wrist_flex": 2173}},
         {"spot": "far", "r_cm": 39.3, "via": {"lift": 2554, "elbow": 1598, "wrist_flex": 2287},
          "grasp": {"lift": 2554, "elbow": 1840, "wrist_flex": 2003}}]


def _lower(spots, d):
    return [{**s, "via": {**s["via"], "lift": s["via"]["lift"] + d}, "grasp": {**s["grasp"], "lift": s["grasp"]["lift"] + d}}
            for s in spots]


class _Geom:
    """TableMap.point without the file: the numbers from setup/table_map.json."""
    pan_zero, tpr, jaw = 2101.2, 651.9, 85

    def point(self, x, y):
        a = math.atan2(x, y)
        return {"x_cm": round(x, 2), "y_cm": round(y, 2), "dist_cm": round(math.hypot(x, y), 2),
                "bearing": round(math.degrees(a), 2), "pan": int(round(self.pan_zero + self.tpr * a - self.jaw))}

    def person(self, bearing_deg):
        a = math.radians(bearing_deg)
        return {"box": None, "x_cm": round(60 * math.sin(a), 1), "y_cm": round(60 * math.cos(a), 1),
                "bearing": bearing_deg, "pan": int(round(self.pan_zero + self.tpr * a))}


class _Stopped(Exception):
    pass


class FakeArm:
    def __init__(self, time_scale=1.0, miss_grip=None, fetch_outcome=None, overhead=None):
        self.time_scale = time_scale          # tests use a small number to run fast
        # simulate a missed grip on every grip; also settable at any time
        self.miss_grip = os.environ.get("PHANTOM_FAKE_MISS") == "1" if miss_grip is None else miss_grip
        self.fetch_outcome = fetch_outcome or os.environ.get("PHANTOM_FAKE_FETCH") or "handed_over"
        self.overhead = os.environ.get("PHANTOM_FAKE_OVERHEAD", "1") != "0" if overhead is None else overhead
        self.wrist_fallback = os.environ.get("PHANTOM_WRIST_FALLBACK", "1") != "0"
        self.taken_by = "tug"
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._torque = False
        self._at_rest = True
        self._holding = False
        self._joints = dict(REST)
        self._emit_cb = None
        self._mute = False
        # the general skills
        self.scene_items = copy.deepcopy(FAKE_SCENE)
        self.person_bearing_deg = PERSON_BEARING_DEG
        env_tiers = os.environ.get("PHANTOM_FAKE_TIERS")
        self.taught = set((env_tiers or "tall").replace(" ", "").split(","))
        self.hand_taken = os.environ.get("PHANTOM_FAKE_TAKEN", "1") != "0"
        self.geom = _Geom()
        self._held, self._carry = None, False
        self.aim_setting = None
        self.wrist_reader = None
        self.wrist_reads = []            # (look number, roll, pan) for each wrist look, for tests

    def tiers(self):
        from backend.vision.table import GripTier
        spots = {"tall": _TALL, "can": _lower(_TALL, 120), "low": _lower(_TALL, 260)}
        closed = {"tall": None, "can": 1560, "low": 1370}
        return {n: GripTier(n, spots[n] if n in self.taught else [], open_value=1790 if n != "can" else 1900,
                            closed=closed[n], source="fake") for n in ("tall", "can", "low")}

    # interface
    def run(self, skill, params=None, on_phase=None):
        try:
            p = S.validate(skill, params)
        except ValueError as e:
            return S.result("error", str(e))
        if not self._lock.acquire(blocking=False):
            return S.result("busy", "another skill is running")
        self._stop.clear()
        self._emit_cb = on_phase
        auto = None
        try:
            early = self._carry_gate(skill)
            if early is not None:
                return early
            if skill != "look":
                self._torque = True
                keep = self._carry and skill in ("place", "hand_over")
                if skill != "rest" and not keep and not self._at_rest:
                    auto = self._auto_rest()
                self._at_rest = False
            out = getattr(self, "_skill_" + skill)(**p)
            res, detail = out[0], out[1]
            extra = out[2] if len(out) > 2 else {}
            if skill != "look":
                self._at_rest = not self._carry     # every skill ends at rest, except while carrying
            if auto is not None:
                return S.result(res, f"went back to rest first ({auto}); {detail}", auto_rest=True, **extra)
            return S.result(res, detail, **extra)
        except _Stopped:
            self._carry = False
            self._at_rest = False
            return S.result("stopped", "stopped; holding where it is", **({"auto_rest": True} if auto else {}))
        except Exception as e:
            self._carry = False
            self._at_rest = False
            self._mute = False
            self._emit("error")
            return S.result("error", str(e), **({"auto_rest": True} if auto else {}))
        finally:
            self._mute = False
            self._emit_cb = None
            self._lock.release()

    def stop(self):
        self._stop.set()

    def status(self):
        return {"connected": True, "busy": self._lock.locked(), "torque": self._torque, "at_rest": self._at_rest,
                "holding": self._holding, "held": {k: v for k, v in self._held.items() if k != "tier_obj"} if self._carry and self._held else None,
                "joints": {NAMES[i]: v for i, v in self._joints.items()}, "fake": True}

    # helpers
    def _emit(self, name):
        cb = self._emit_cb
        if cb and not self._mute:
            try:
                cb(name)
            except Exception:
                pass   # a broken listener must not break the motion

    def _wait(self, s):
        if self._stop.wait(max(0.0, s) * self.time_scale):
            raise _Stopped()

    def _move(self, target, s):
        self._wait(s)
        self._joints.update(target)

    def _auto_rest(self):
        self._emit("auto_rest")
        self._mute = True
        try:
            _, detail = self._skill_rest()
        finally:
            self._mute = False
        self._at_rest = True
        return detail

    # skills; each returns (result, detail)
    def _skill_grip_bottle(self, pace, hold_s, flourish, look_first):
        t = S.leg_time(pace)
        self._emit("start")
        if look_first:
            self._move(VIA, S.LOOK_LEG1 * t)
            self._wait(S.LOOK_PAUSE_S)
            self._move(GRASP, S.LOOK_LEG2 * t)
        else:
            self._move(GRASP, t)
        self._joints[6] = 1902
        self._emit("at_target")
        self._wait(S.CLOSE_S)
        if self.miss_grip:
            self._joints[6] = 1315
            self._emit("missed")
            self._wait(0.6)
            res, detail = "missed", "the gripper closed on nothing"
        else:
            self._joints[6] = 1547
            self._holding = True
            self._emit("gripped")
            self._move({2: 1800}, S.lift_time(pace))
            self._emit("lifted")
            self._wait(S.hold_time(hold_s, flourish))
            self._move({2: GRASP[2]}, S.lower_time(pace))
            self._wait(S.RELEASE_S)
            self._joints[6] = 1902
            self._holding = False
            self._emit("put_back")
            res, detail = "done", "bottle lifted and put back"
        self._emit("returning")
        self._move(REST, t)
        self._emit("done")
        return res, detail

    def _skill_point(self, pace):
        leg = S.POINT_LEG * S.leg_time(pace)
        self._emit("start")
        self._move(VIA, leg)
        self._emit("at_target")
        self._wait(S.POINT_HOLD_S)
        self._emit("returning")
        self._move(REST, leg)
        self._emit("done")
        return "done", "pointed at the bottle"

    def _skill_half_reach(self, pace):
        leg = S.HALF_LEG * S.leg_time(pace)
        self._emit("start")
        self._move(HALF, leg)
        self._emit("at_target")
        self._wait(S.HALF_HOLD_S)
        self._emit("returning")
        self._move(REST, leg)
        self._emit("done")
        return "done", "reached part of the way"

    def _skill_wave(self):
        self._emit("start")
        self._move(PHOTO, S.WAVE_LEG_S)
        self._emit("at_target")
        self._wait(S.WAVE_MOVE_S)
        self._emit("returning")
        self._move(REST, S.WAVE_LEG_S)
        self._emit("done")
        return "done", "waved"

    def _put_down_here(self):
        """The recovery from fetch_bottle.py: tuck, stand the bottle on the table, let go, lift clear."""
        self._move(F_TUCK, S.RECOVER_TUCK_S + S.RECOVER_ALIGN_S)
        self._move({2: F_GRASP[2]}, S.RECOVER_TABLE_S)
        self._wait(S.FETCH_RELEASE_S)
        self._joints[6] = 1790
        self._holding = False
        self._emit("put_back")
        self._wait(S.FETCH_AFTER_OPEN_S)
        self._move({2: F_GRASP[2] - 220}, S.RECOVER_CLEAR_S)

    def _skill_rest(self):
        self._emit("start")
        at_rest = all(abs(self._joints[i] - REST[i]) <= 150 for i in range(1, 5))
        detail = "at rest"
        if self._holding and not at_rest:
            what = self._held["name"] if self._held else "bottle"
            if self._held:      # it is now on the table where the arm was
                x, y = self._held["x_cm"], self._held["y_cm"]
                self.scene_items.append({**self._scene_entry(self._held), "x_cm": x, "y_cm": y})
            self._put_down_here()
            detail = f"put the {what} down on the table where the arm was, then went to rest"
            self._emit("returning")
            self._move(REST, S.FETCH_REST_S)
        else:
            self._emit("returning")
            self._move(REST, 0.3 if at_rest else S.REST_LEG_S + 0.8)
        self._held, self._carry = None, False
        self._emit("done")
        return "done", detail

    def _skill_gesture(self, style, seconds):
        g = S.gesture_plan(style, seconds)
        self._emit("start")
        self._move(g["pose"], g["rise_s"])
        self._emit("at_target")
        self._wait(g["talk_s"])
        self._emit("returning")
        self._move(REST, g["back_s"])
        self._emit("done")
        return "done", f"gesture {style}"

    def _skill_fetch_bottle(self, pace, wait_s, look_first):
        k = S.fetch_stretch(pace)
        out = self.fetch_outcome
        self._emit("start")
        note = "bottle at 30.0 cm, pan 1990, blend 0.42 between middle and far; held out at pan 2620"
        if self.overhead:
            self._wait(S.FETCH_LOOK_S)
            if out == "no_bottle":
                self._emit("done")
                return "no_bottle", "I cannot see the bottle on the table"
            if out == "out_of_reach":
                self._emit("seen")
                self._emit("done")
                return "out_of_reach", "the bottle is 52 cm from the arm; I can only reach it between 19 and 41 cm"
        else:
            if not self.wrist_fallback:
                self._emit("done")
                return "no_bottle", "the overhead camera gives no picture (wrist fallback is off)"
            self._move({4: 2150}, 1.2 * k)
            self._wait(S.WRIST_LOOK_S * WRIST_STOPS * k + 2 * S.WRIST_LOOK_S)
            if out in ("no_bottle", "out_of_reach"):
                self._emit("returning")
                self._move(REST, S.FETCH_REST_S * k)
                self._emit("done")
                return "no_bottle", "the wrist camera could not find the bottle (overhead camera gives no picture)"
            note = ("wrist camera only: bottle centred at pan 1990; distance cannot be measured with one camera, "
                    "so it used the taught middle grip; held out at pan 2613 (person not seen, 45 degrees right)")
        self._emit("seen")
        self._joints[6] = 1790
        self._move(F_VIA, S.FETCH_OUT_S * k)
        if look_first:
            self._wait(S.FETCH_LOOK_FIRST_S)
        self._move(F_GRASP, S.FETCH_APPROACH_S * k)
        self._emit("at_target")
        self._wait(S.FETCH_CLOSE_S)
        if self.miss_grip or out == "missed":
            self._joints[6] = 1315
            self._emit("missed")
            self._wait(S.FETCH_RELEASE_S)
            self._move(F_VIA, S.FETCH_VIA_S * k)
            self._emit("returning")
            self._move(REST, S.FETCH_REST_S * k)
            self._emit("done")
            return "missed", f"the gripper closed on nothing. {note}"
        self._joints[6] = 1478
        self._holding = True
        self._emit("gripped")
        self._move(F_TUCK, S.FETCH_TUCK_S * k)
        self._emit("lifted")
        turn = S.fetch_turn_s(1990, F_OUT[1], pace)
        self._wait(0)
        self._emit("carrying")
        if out == "turn_failed":
            self._move({1: 2300}, 0.5 * turn + 1.2)
            self._put_down_here()
            self._emit("returning")
            self._move(REST, S.FETCH_REST_S * k)
            self._emit("done")
            return "turn_failed", f"could not finish the turn (stalled) at pan 2300; the bottle is on the table where the arm stopped. {note}"
        self._move(F_OUT, max(turn, 0.5 * turn + S.FETCH_REACH_S * k) + 0.3)
        self._emit("offering")
        self._wait(0.3 + S.FETCH_OFFER_SETTLE_S)
        if out == "handed_over":
            self._wait(min(wait_s, 1.0))
            self._joints[6] = 1790
            self._holding = False
            self._emit("handed_over")
            self._wait(S.FETCH_HANDOVER_OPEN_S + S.FETCH_HANDOVER_WAIT_S)
            self._move({2: F_OUT[2] - 300}, S.FETCH_CLEAR_S * k)
            res, detail = "handed_over", f"handed over ({self.taken_by}). {note}"
        else:
            self._wait(wait_s)
            self._move(F_TUCK, S.FETCH_TUCK_S * k)
            self._move({1: F_GRASP[1]}, turn)
            self._move(F_GRASP, (S.FETCH_ABOVE_SPOT_S + S.FETCH_SET_S) * k)
            self._wait(S.FETCH_RELEASE_S)
            self._joints[6] = 1790
            self._holding = False
            self._emit("put_back")
            self._wait(S.FETCH_AFTER_OPEN_S)
            self._move(F_VIA, S.FETCH_VIA_S * k)
            res, detail = "put_back", f"nobody took it in {wait_s:g} s; put it back. {note}"
        self._emit("returning")
        self._move(REST, S.FETCH_REST_S * k)
        self._emit("done")
        return res, detail

    # The general skills, with the same results, phases and refusals as RealArm.
    CARRY_OK = ("look", "place", "hand_over", "rest", "gesture")
    PAN_RANGE = (764, 3330)          # the calibrated base range, 30 ticks inside each end

    def _carry_gate(self, skill):
        carrying = self._carry and self._held is not None
        if skill in ("place", "hand_over") and not carrying:
            return S.result("nothing_held", "I am not holding anything I picked up")
        if carrying and skill == "gesture":
            return S.result("done", f"gesture skipped while holding the {self._held['name']}", skipped=True)
        if carrying and skill not in self.CARRY_OK:
            return S.result("hands_full", f"I am holding the {self._held['name']}; place it or hand it over first")
        return None

    @staticmethod
    def _scene_entry(item):
        return {k: item.get(k) for k in ("name", "x_cm", "y_cm", "tier", "graspable", "axis_deg", "width_cm")
                if item.get(k) is not None}

    def _items(self):
        from backend.vision.scene import grip_point
        out = []
        for e in self.scene_items:
            it = {"name": e["name"], "box": None, "tier": e.get("tier", "tall"), "tier_from": "name",
                  "graspable": e.get("graspable", True), "axis_deg": e.get("axis_deg"), "width_cm": e.get("width_cm")}
            it.update(self.geom.point(e["x_cm"], e["y_cm"]))
            it["grip_at"] = grip_point(it, self.geom)
            out.append(it)
        return out

    def _scene(self, name=None):
        if not self.overhead:
            raise Refuse("no_picture", "the overhead camera gives no picture")
        from backend.vision.scene import match_name
        self._wait(S.FETCH_LOOK_S)
        items = self._items()
        person = None if self.person_bearing_deg is None else self.geom.person(self.person_bearing_deg)
        return {"items": items, "person": person, "match": match_name(items, name) if name else None}

    def _find_target(self, target):
        if not target:
            raise Refuse("not_found", "no target was given")
        if isinstance(target, dict):
            s = self._scene()
            item = desk.nearest(s["items"], target["x_cm"], target["y_cm"])
        else:
            s = self._scene(name=target)
            item = s["match"]
        if item is None:
            raise Refuse("not_found", f"I cannot see {desk.label(target)} on the desk")
        return item, s

    def _skill_look(self, target):
        from backend.vision.scene import public
        self._emit("start")
        try:
            s = self._scene(name=target if isinstance(target, str) else None)
        except Refuse as r:
            self._emit("done")
            return r.result, r.detail, r.extra
        self._emit("seen")
        self._emit("done")
        items = [public(i) for i in s["items"]]
        extra = {"scene": items, "person": s["person"]}
        detail = f"I can see {len(items)} things: {', '.join(i['name'] for i in items) or 'nothing'}"
        if isinstance(target, str):
            extra["match"] = public(s["match"])
            detail += f"; {target!r} is the {s['match']['name']}" if s["match"] else f"; nothing fits {target!r}"
        return "done", detail, extra

    def _skill_point_at(self, target, pace):
        from backend.vision.scene import public
        k = S.fetch_stretch(pace)
        self._emit("start")
        try:
            if isinstance(target, dict):
                item = {"name": desk.label(target)[4:], **self.geom.point(target["x_cm"], target["y_cm"])}
            else:
                item, _ = self._find_target(target)
            self._emit("seen")
            tiers = [t for t in self.tiers().values() if t.available]
            lo, hi = min(t.min_r for t in tiers), max(t.max_r for t in tiers)
            problem = desk.reach_problem(item, lo, hi, self.PAN_RANGE, what=f"the {item['name']}")
            if problem:
                raise Refuse("out_of_reach", problem)
        except Refuse as r:
            self._emit("done")
            return r.result, r.detail, r.extra
        self._move({1: item["pan"], **{i: F_VIA[i] for i in (2, 3, 4)}}, S.FETCH_OUT_S * k)
        self._emit("at_target")
        self._wait(S.POINT_AT_HOLD_S)
        self._emit("returning")
        self._move(REST, S.FETCH_REST_S * k)
        self._emit("done")
        return "done", f"pointed at the {item['name']}, {item['dist_cm']:.0f} cm away", {"target": public(item)}

    def _skill_pick(self, target, pace, approach, lift_cm_style):
        from backend.vision.scene import public
        k = S.fetch_stretch(pace)
        self._emit("start")
        try:
            item, s = self._find_target(target)
            self._emit("seen")
            plan = desk.pick_plan(item, self.tiers(), self.PAN_RANGE)
        except Refuse as r:
            self._emit("done")
            return r.result, r.detail, r.extra
        tier, name = plan["tier"], item["name"]
        note = (f"the {name} at {plan['dist_cm']:.1f} cm, pan {plan['pan']}, {tier.name} grip blended {plan['t']:.2f}"
                + (f", wrist roll {plan['roll']} for its angle {item['axis_deg']:.0f} deg" if plan["roll"] is not None else ""))
        self._joints[6] = tier.open
        out = {1: plan["pan"], **{i: plan["via"][n] for i, n in ((2, "lift"), (3, "elbow"), (4, "wrist_flex"))}}
        if plan["roll"] is not None:
            out[5] = plan["roll"]
        self._move(out, S.FETCH_OUT_S * k)
        if approach == "look_first":
            self._wait(S.FETCH_LOOK_FIRST_S)
        aim = []
        if tier.name in desk.aim_tiers(self.aim_setting):
            aim = self._aim_with_wrist(plan, k)
        if aim:
            note += ". " + desk.aim_words(aim)
        aim_extra = {"aim": aim} if aim else {}
        self._move({i: plan["grasp"][n] for i, n in ((2, "lift"), (3, "elbow"), (4, "wrist_flex"))}, S.FETCH_APPROACH_S * k)
        self._emit("at_target")
        self._wait(S.FETCH_CLOSE_S)
        if self.miss_grip:
            self._joints[6] = 1315
            self._emit("missed")
            self._wait(S.FETCH_RELEASE_S)
            self._move(F_VIA, S.FETCH_VIA_S * k)
            self._emit("returning")
            self._move({**REST, 5: 2030}, S.FETCH_REST_S * k)
            self._emit("done")
            return "missed", f"the gripper closed on nothing. {note}", aim_extra
        grip = tier.closed or 1478
        self._joints[6] = grip
        self._holding = True
        self._emit("gripped")
        carry = tier.carry(lift_cm_style)
        self._move({2: carry["lift"], 3: carry["elbow"], 4: carry["wrist_flex"]}, S.FETCH_TUCK_S * k)
        self._emit("lifted")
        self.scene_items = [e for e in self.scene_items if e["name"] != name or (e["x_cm"], e["y_cm"]) != (item["x_cm"], item["y_cm"])]
        self._held = {**public(item), "grip": grip, "rim_offset_cm": item["grip_at"].get("rim_offset_cm"),
                      "width_cm": item.get("width_cm"), "tier_obj": tier, "person": s["person"]}
        self._carry = True
        self._emit("done")
        return "holding", f"holding the {name}, gripped at {grip}. {note}", {"held": public(item), **aim_extra}

    def _wrist_read(self, n, plan):
        roll, pan = self._joints.get(5, 2030), self._joints[1]
        self.wrist_reads.append((n, roll, pan))
        r = self.wrist_reader
        if callable(r):
            return r(n, roll, pan)
        if isinstance(r, (list, tuple)):
            return r[n - 1] if n - 1 < len(r) else None
        # the pretend item: lined up at a roll 250 ticks lower and a pan 30 ticks lower than planned
        if getattr(self, "_aim_truth", (None,))[0] is not plan:
            self._aim_truth = (plan, roll - 250, pan - 30)
        _, roll_ok, pan_ok = self._aim_truth
        W = desk.WRIST_AIM
        gain = W["pan_px_per_tick"] * (plan["dist_cm"] / W["pan_gain_ref_cm"] if W["pan_gain_scales"] else 1.0)
        return {"angle_deg": round((roll - roll_ok) / W["roll_ticks_per_deg"], 1),
                "offset_px": round((pan_ok - pan) * gain, 1), "confidence": 0.9, "source": "opencv"}

    def _aim_with_wrist(self, plan, k):
        """The same loop as RealArm._aim_with_wrist, on the pretend joints."""
        W = desk.WRIST_AIM
        stretch = 1.0 + 0.25 * (k - 1.0)
        rounds = []
        for n in range(1, W["rounds"] + 1):
            self._wait(W["settle_s"] + 0.1)                   # settle, then the picture and OpenCV
            reading = self._wrist_read(n, plan)
            if reading is None:
                rounds.append({"round": n, "unread": True, "why": "neither OpenCV nor Gemini was sure"})
                self._emit(f"aiming {n}: could not read the wrist picture; keeping the aim it has")
                break
            roll_now, pan_now = self._joints.get(5, 2030), self._joints[1]
            step = desk.aim_step(reading, roll_now, pan_now, plan["dist_cm"], pan_range=self.PAN_RANGE)
            rec = {"round": n, "source": reading.get("source"), "angle_deg": reading.get("angle_deg"),
                   "offset_px": reading.get("offset_px"), "confidence": reading.get("confidence"),
                   **{key: step[key] for key in ("done", "d_roll", "d_pan", "capped", "why")}}
            rounds.append(rec)
            if step["done"]:
                self._emit(f"aiming {n}: {step['why']}")
                break
            if step["roll"] is not None:
                self._emit(f"aiming {n}: {step['why']}, wrist roll {step['d_roll']:+d} to {step['roll']}"
                           + (" (capped)" if step["capped"] else ""))
                self._move({5: step["roll"]}, max(W["min_move_s"], abs(step["d_roll"]) / W["roll_speed"]) * stretch)
            elif step["pan"] is not None and step["d_pan"]:
                self._emit(f"aiming {n}: {step['why']}, pan {step['d_pan']:+d} to {step['pan']}"
                           + (" (capped)" if step["capped"] else ""))
                self._move({1: step["pan"]}, max(W["min_move_s"], abs(step["d_pan"]) / W["pan_speed"]) * stretch)
            else:
                rec["why"] += "; cannot turn further"
                self._emit(f"aiming {n}: {rec['why']}")
                break
        return rounds

    def _skill_place(self, where, pace, approach):
        k = S.fetch_stretch(pace)
        H = self._held
        tier, name = H["tier_obj"], H["name"]
        self._emit("start")
        try:
            def look(n):
                return self._scene(name=n)["match"]
            x, y, words = desk.place_centre(where, H, look)
            spot = desk.jaws_for_centre(x, y, H, self.geom.point)
            problem = desk.reach_problem(spot, tier.min_r, tier.max_r, self.PAN_RANGE, what="that spot")
            if problem:
                raise Refuse("out_of_reach", f"{problem}; still holding the {name}")
        except Refuse as r:
            self._emit("done")
            return r.result, r.detail, r.extra
        self._emit("seen")
        via, grasp, _ = tier.posture(spot["dist_cm"])
        self._emit("carrying")
        self._move({1: spot["pan"]}, S.fetch_turn_s(self._joints[1], spot["pan"], pace))
        self._move({2: grasp["lift"] - 170, 3: grasp["elbow"], 4: grasp["wrist_flex"]}, S.FETCH_ABOVE_SPOT_S * k)
        if approach == "look_first":
            self._wait(S.FETCH_LOOK_FIRST_S)
        self._move({2: grasp["lift"]}, S.FETCH_SET_S * k)
        self._emit("at_target")
        self._wait(S.FETCH_RELEASE_S)
        self._joints[6] = tier.open
        self._holding = False
        self.scene_items.append({**self._scene_entry(H), "x_cm": round(x, 2), "y_cm": round(y, 2)})
        self._held, self._carry = None, False
        self._emit("put_back")
        self._wait(S.FETCH_AFTER_OPEN_S)
        self._move({2: via["lift"], 3: via["elbow"], 4: via["wrist_flex"]}, S.FETCH_VIA_S * k)
        self._emit("returning")
        self._move({**REST, 5: 2030}, S.FETCH_REST_S * k)
        self._emit("done")
        return "placed", f"put the {name} down {words}, {spot['dist_cm']:.0f} cm from the arm", \
            {"placed_at": {"x_cm": round(x, 1), "y_cm": round(y, 1)}}

    def _skill_hand_over(self, wait_s, pace):
        k = S.fetch_stretch(pace)
        H = self._held
        name = H["name"]
        self._emit("start")
        person = H.get("person")
        pan_p = person["pan"] if person else self.geom.person(PERSON_BEARING_DEG)["pan"]
        pan_p = int(max(950, min(3250, pan_p)))
        self._emit("seen")
        turn = S.fetch_turn_s(self._joints[1], pan_p, pace)
        self._emit("carrying")
        self._move({**F_OUT, 1: pan_p}, max(turn, 0.5 * turn + S.FETCH_REACH_S * k) + 0.3)
        self._emit("offering")
        self._wait(0.3 + S.FETCH_OFFER_SETTLE_S)
        if self.hand_taken:
            self._wait(min(wait_s, 1.0))
            self._joints[6] = H["tier_obj"].open
            self._holding = False
            self._held, self._carry = None, False
            self._emit("handed_over")
            self._wait(S.FETCH_HANDOVER_OPEN_S + S.FETCH_HANDOVER_WAIT_S)
            self._move({2: F_OUT[2] - 300}, S.FETCH_CLEAR_S * k)
            self._emit("returning")
            self._move({**REST, 5: 2030}, S.FETCH_REST_S * k)
            self._emit("done")
            return "handed_over", f"handed over the {name} ({self.taken_by}) at pan {pan_p}"
        self._wait(wait_s)
        c = H["tier_obj"].carry("normal")
        self._move({2: c["lift"], 3: c["elbow"], 4: c["wrist_flex"]}, S.FETCH_TUCK_S * k)
        self._carry = True
        self._emit("done")
        return "not_taken", f"nobody took the {name} in {wait_s:g} s; still holding it in the carry pose at pan {pan_p}"
