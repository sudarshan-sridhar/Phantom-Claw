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
"""
import os
import threading

from . import skills as S

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
            self._torque = True
            if skill != "rest" and not self._at_rest:
                auto = self._auto_rest()
            self._at_rest = False
            res, detail = getattr(self, "_skill_" + skill)(**p)
            self._at_rest = True
            if auto is not None:
                return S.result(res, f"went back to rest first ({auto}); {detail}", auto_rest=True)
            return S.result(res, detail)
        except _Stopped:
            self._at_rest = False
            return S.result("stopped", "stopped; holding where it is", **({"auto_rest": True} if auto else {}))
        except Exception as e:
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
                "holding": self._holding, "joints": {NAMES[i]: v for i, v in self._joints.items()}, "fake": True}

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
            self._put_down_here()
            detail = "put the bottle down on the table where the arm was, then went to rest"
            self._emit("returning")
            self._move(REST, S.FETCH_REST_S)
        else:
            self._emit("returning")
            self._move(REST, 0.3 if at_rest else S.REST_LEG_S + 0.8)
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
