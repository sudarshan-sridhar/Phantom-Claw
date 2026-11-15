"""FakeArm: the pretend arm. No hardware. Each skill sleeps for about as long as the real
one takes and reports the same phases in the same order as RealArm.

Rules shared with RealArm:
- one skill at a time; a second run while busy returns result "busy"
- stop() holds where it is; run returns "stopped"
- after a stop or an error the arm is not at rest, and every skill except `rest` is refused
  with result "error" until `rest` has run
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


class _Stopped(Exception):
    pass


class FakeArm:
    def __init__(self, time_scale=1.0, miss_grip=None):
        self.time_scale = time_scale          # tests use a small number to run fast
        # simulate a missed grip on every grip_bottle; also settable at any time
        self.miss_grip = os.environ.get("PHANTOM_FAKE_MISS") == "1" if miss_grip is None else miss_grip
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._torque = False
        self._at_rest = True
        self._joints = dict(REST)
        self._emit_cb = None

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
        try:
            if skill != "rest" and not self._at_rest:
                raise RuntimeError("arm is not at rest; run rest first")
            self._torque = True
            res, detail = getattr(self, "_skill_" + skill)(**p)
            self._at_rest = True
            return S.result(res, detail)
        except _Stopped:
            self._at_rest = False
            return S.result("stopped", "stopped; holding where it is")
        except Exception as e:
            self._at_rest = False
            self._emit("error")
            return S.result("error", str(e))
        finally:
            self._emit_cb = None
            self._lock.release()

    def stop(self):
        self._stop.set()

    def status(self):
        return {"connected": True, "busy": self._lock.locked(), "torque": self._torque,
                "joints": {NAMES[i]: v for i, v in self._joints.items()}, "fake": True}

    # helpers
    def _emit(self, name):
        cb = self._emit_cb
        if cb:
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
            self._emit("gripped")
            self._move({2: 1800}, S.lift_time(pace))
            self._emit("lifted")
            self._wait(S.hold_time(hold_s, flourish))
            self._move({2: GRASP[2]}, S.lower_time(pace))
            self._wait(S.RELEASE_S)
            self._joints[6] = 1902
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

    def _skill_rest(self):
        self._emit("start")
        self._emit("returning")
        self._move(REST, S.REST_LEG_S + 0.8)
        self._emit("done")
        return "done", "at rest"

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
