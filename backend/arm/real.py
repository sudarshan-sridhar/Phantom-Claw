"""RealArm: the SO-101 follower on COM3, driven with scservo_sdk.

Built from the hand-tested scripts in setup/tools:
- grip_bottle.py: rd/wr with retries, torque_on with the range check, blend() (one goal per
  joint, servo speed and acceleration registers, the elbow leads the shoulder by 30 percent,
  the settle correction), close_gripper() stall detection, the numbers, and the
  stop-and-hold error handling.
- gif_thumbnail.py: timed single goals per joint for the wave (and the flourish and gesture).

Nothing here touches the serial port on import. The port opens when RealArm() is created.
Torque is never switched off by this code. Every skill ends at rest, holding.
"""
import json
import threading
import time
from pathlib import Path

import scservo_sdk as scs

from . import skills as S

SETUP = Path(__file__).resolve().parents[2] / "setup"

# Feetech STS3215 registers, the same addresses as setup/tools
TORQUE_EN, ACCEL, GOAL, SPEED, TORQUE_LIMIT, POSITION, CURRENT = 40, 41, 42, 46, 48, 56, 69

ACC = 7                                   # gentle ramps for the blended legs (x100 ticks/s^2)
TORQUE = {1: 500, 2: 1000, 3: 800, 4: 500, 5: 400, 6: 300}
CUR_MAX = {1: 800, 2: 1600, 3: 1400, 4: 800, 6: 700}
LEAD = S.LEAD                             # 0.3, the elbow leads the shoulder
CLOSED = S.GRIP_CLOSED                    # 1310
MISS_TICKS = 100                          # closed to within this of CLOSED means nothing in the jaws
ROLL_RANGE = (1100, 2960)                 # wrist roll, what the camera cable allows
PHOTO = dict(S.PHOTO_POSE)
PHOTO_GRIP = 1747
REST_TOL = 150                            # every joint 1 to 4 within this of rest counts as at rest
WAVE_SHUT, WAVE_OPEN = 1400, 1900


class _Stopped(Exception):
    pass


class RealArm:
    def __init__(self, port="COM3", baud=1_000_000, setup_dir=None, port_handler=None, packet_handler=None):
        d = Path(setup_dir) if setup_dir else SETUP
        with open(d / "my_follower_calibration_2026-11-14.json") as f:
            cal = json.load(f)
        with open(d / "bottle_pick.json") as f:
            pick = json.load(f)
        self.rng = {c["id"]: (c["range_min"], c["range_max"]) for c in cal.values()}
        self.names = {c["id"]: name for name, c in cal.items()}
        r = self.rng
        g, v = pick["grasp"], pick["via"]
        self.GRASP = {1: g["pan"], 2: g["lift"], 3: g["elbow"], 4: g["wrist_flex"]}
        self.VIA = {1: v["pan"], 2: v["lift"], 3: v["elbow"], 4: v["wrist_flex"]}
        self.REST = {1: 2009, 2: r[2][0] + 15, 3: r[3][1] - 4, 4: 2632}
        self.HALF = {i: round(self.REST[i] + S.HALF_FRACTION * (self.VIA[i] - self.REST[i])) for i in range(1, 5)}
        self.OPEN = pick["gripper_open"]
        self.LIFT = pick["lift"]["lift"]

        self._lock = threading.Lock()     # one skill at a time
        self._bus = threading.Lock()      # one user of the serial bus at a time
        self._stop = threading.Event()
        self._emit_cb = None
        self._joints = {}
        self._torque = False
        self._last_cur = 0.0

        self._port_name, self._baud = port, baud
        self._p, self._h = port_handler, packet_handler
        self.connected = False
        self.last_error = ""
        self._connect()

    # connection
    def _connect(self):
        try:
            if self._p is None:
                self._p = scs.PortHandler(self._port_name)
            if self._h is None:
                self._h = scs.PacketHandler(0)
            if not self._p.openPort():
                raise RuntimeError(f"could not open {self._port_name}")
            if not self._p.setBaudRate(self._baud):
                raise RuntimeError(f"could not set baud rate on {self._port_name}")
            self.connected = True
            self.last_error = ""
        except Exception as e:
            self.connected = False
            self.last_error = str(e)

    def close(self):
        """Close the port. Torque stays on and the servos keep holding."""
        with self._bus:
            try:
                if self._p is not None:
                    self._p.closePort()
            finally:
                self.connected = False

    # bus access, as grip_bottle.py
    def _rd(self, i, a, n=2):
        h, p = self._h, self._p
        for _ in range(6):
            v, c, _ = (h.read1ByteTxRx if n == 1 else h.read2ByteTxRx)(p, i, a)
            if c == scs.COMM_SUCCESS:
                return v
        raise RuntimeError(f"read {i}/{a} failed")

    def _wr(self, i, a, v, n=2):
        h, p = self._h, self._p
        for _ in range(6):
            c, _ = (h.write1ByteTxRx if n == 1 else h.write2ByteTxRx)(p, i, a, v)
            if c == scs.COMM_SUCCESS:
                return
        raise RuntimeError(f"write {i}/{a} failed")

    def _clamp(self, i, g):
        return max(self.rng[i][0], min(self.rng[i][1], g))

    def _torque_on(self):
        start = {i: self._rd(i, POSITION) for i in range(1, 7)}
        bad = [i for i in (1, 2, 3, 4, 6) if not (self.rng[i][0] - 25 <= start[i] <= self.rng[i][1] + 25)]
        if bad:
            raise RuntimeError(f"joints {bad} read out of range: {start}")
        for i in range(1, 7):
            self._wr(i, GOAL, self._clamp(i, start[i]))
            self._wr(i, TORQUE_LIMIT, TORQUE[i])
        for i in range(1, 7):
            self._wr(i, TORQUE_EN, 1, 1)

    # stop handling
    def _check_stop(self):
        if self._stop.is_set():
            raise _Stopped()

    def _wait(self, s):
        if self._stop.wait(max(0.0, s)):
            raise _Stopped()

    def _check_current(self, i):
        if i in CUR_MAX and (self._rd(i, CURRENT) & 0x7FFF) * 6.5 > CUR_MAX[i]:
            raise RuntimeError(f"ID {i} drew too much current")

    def _hold_here(self):
        """Hold every arm joint where it is, as grip_bottle.py does on any error. The gripper
        goal is left alone so a held bottle keeps its squeeze."""
        for i in (1, 2, 3, 4, 5):
            try:
                self._wr(i, GOAL, self._clamp(i, self._rd(i, POSITION)))
            except Exception:
                pass

    # motion, as grip_bottle.py
    def _blend(self, items, settle=True):
        """items: (id, goal, start_s, duration_s). One motion; each joint runs its own smooth ramp."""
        items = [(i, self._clamp(i, g), ts, dur) for i, g, ts, dur in items]
        pos = {i: self._rd(i, POSITION) for i, *_ in items}
        for i, g, ts, dur in items:
            self._wr(i, ACCEL, ACC, 1)
            self._wr(i, SPEED, max(12, round(abs(g - pos[i]) / dur)))
        sent, t0, last_move = set(), time.time(), time.time()
        end = max(ts + dur for _, _, ts, dur in items)
        while True:
            self._check_stop()
            t = time.time() - t0
            for i, g, ts, dur in items:
                if i not in sent and t >= ts:
                    self._wr(i, GOAL, g)
                    sent.add(i)
            time.sleep(0.03)
            now = {i: self._rd(i, POSITION) for i, *_ in items}
            self._joints.update(now)
            for i in now:
                self._check_current(i)
            if any(abs(now[i] - pos[i]) > 2 for i in now):
                last_move = time.time()
            pos = now
            done = len(sent) == len(items)
            if done and all(abs(now[i] - g) <= 12 for i, g, _, _ in items):
                return
            still = time.time() - last_move
            if done and t > end and still > 0.3 and all(abs(now[i] - g) <= 60 for i, g, _, _ in items):
                if settle:   # sagged a little short under its own weight: one small push, at low speed
                    for i, g, _, _ in items:
                        if i in (2, 3) and abs(g - now[i]) > 8:
                            self._wr(i, SPEED, 60)
                            self._wr(i, GOAL, self._clamp(i, g + (g - now[i])))
                    self._wait(0.4)
                return
            if done and still > 1.2:
                raise RuntimeError(f"stalled at {now}")
            if t > end + 6:
                raise RuntimeError("move took too long")

    def _close_gripper(self):
        self._wr(6, ACCEL, 10, 1)
        self._wr(6, SPEED, 300)
        self._wr(6, GOAL, CLOSED)
        t_start = time.time()
        last, t_move = self._rd(6, POSITION), time.time()
        while time.time() - t_move < 0.35:
            self._check_stop()
            time.sleep(0.03)
            pos = self._rd(6, POSITION)
            if abs(pos - last) > 2:
                last, t_move = pos, time.time()
            if time.time() - t_start > 5:
                raise RuntimeError("gripper did not settle")
        pos = self._rd(6, POSITION)
        self._joints[6] = pos
        if pos - CLOSED > MISS_TICKS:
            self._wr(6, GOAL, pos - 70)   # keep a light squeeze, the same force as before
            return pos
        return None

    def _leg_out(self, target, dur, grip=None, settle=True):
        """Rest side to a pose: the elbow runs the whole time, the shoulder starts after LEAD."""
        items = [(3, target[3], 0, dur), (2, target[2], LEAD * dur, (1 - LEAD) * dur),
                 (1, target[1], 0, dur), (4, target[4], 0, dur)]
        if grip is not None:
            items.append((6, grip, 0, min(S.GRIPPER_OPEN_S, dur)))
        self._blend(items, settle)

    def _leg_back(self, dur, roll=None):
        """Back to rest: all start together, the shoulder gets back early, the elbow finishes the fold.
        The gripper closes in the last 40 percent, unless it is squeezing something."""
        items = [(3, self.REST[3], 0, dur), (2, self.REST[2], 0, (1 - LEAD) * dur),
                 (1, self.REST[1], 0, dur), (4, self.REST[4], 0, dur)]
        g_pos, g_goal = self._rd(6, POSITION), self._rd(6, GOAL)
        holding = g_pos - CLOSED > MISS_TICKS and g_goal < g_pos - 40
        if not holding:
            items.append((6, CLOSED, 0.6 * dur, 0.4 * dur))
        if roll is not None:
            items.append((5, roll, 0, dur))
        self._blend(items, settle=False)
        return holding

    def _go(self, i, goal, speed, acc=12):
        """One timed goal, as gif_thumbnail.py: clamp 10 inside the range, then acc, speed, goal."""
        lo, hi = ROLL_RANGE if i == 5 else self.rng[i]
        goal = max(lo + 10, min(hi - 10, int(goal)))
        self._wr(i, ACCEL, acc, 1)
        self._wr(i, SPEED, int(speed))
        self._wr(i, GOAL, goal)

    def _watch_until(self, deadline, watch):
        """Wait until deadline, checking stop, and the current of the watched joints when there is time."""
        while True:
            self._check_stop()
            left = deadline - time.time()
            if left <= 0:
                return
            if watch and left > 0.08 and time.time() - self._last_cur >= 0.1:
                for i in watch:
                    self._check_current(i)
                self._last_cur = time.time()
                continue
            time.sleep(min(0.01, left))

    def _timeline(self, steps, tail, watch=()):
        """steps: (time s, id, goal, speed, acc), sorted by time."""
        t0 = time.time()
        for ts, i, g, sp, acc in steps:
            self._watch_until(t0 + ts, watch)
            self._go(i, g, sp, acc)
        last = steps[-1][0] if steps else 0.0
        self._watch_until(t0 + last + tail, watch)

    # run
    def _emit(self, name):
        cb = self._emit_cb
        if cb:
            try:
                cb(name)
            except Exception:
                pass   # a broken listener must not break the motion

    def _prepare(self, skill):
        if not self.connected:
            self._connect()
        if not self.connected:
            raise RuntimeError(f"arm not connected: {self.last_error}")
        pos = {i: self._rd(i, POSITION) for i in range(1, 5)}
        self._joints.update(pos)
        if skill != "rest":
            off = [i for i in range(1, 5) if abs(pos[i] - self.REST[i]) > REST_TOL]
            if off:
                raise RuntimeError(f"arm is not at rest (joints {off}); run rest first")
        if not all(self._rd(i, TORQUE_EN, 1) == 1 for i in range(1, 7)):
            self._torque_on()
        self._torque = True
        if skill != "gesture":   # gestures run with every spoken line, in time with the audio
            try:
                import winsound
                winsound.Beep(1200, 300)
            except Exception:
                pass

    def run(self, skill, params=None, on_phase=None):
        try:
            p = S.validate(skill, params)
        except ValueError as e:
            return S.result("error", str(e))
        if not self._lock.acquire(blocking=False):
            return S.result("busy", "another skill is running")
        try:
            with self._bus:
                self._stop.clear()
                self._emit_cb = on_phase
                try:
                    self._prepare(skill)
                    res, detail = getattr(self, "_skill_" + skill)(**p)
                    return S.result(res, detail)
                except _Stopped:
                    self._hold_here()
                    return S.result("stopped", "stopped; holding where it is")
                except Exception as e:
                    self._hold_here()
                    self._emit("error")
                    return S.result("error", str(e))
                except BaseException:
                    self._hold_here()
                    raise
                finally:
                    self._emit_cb = None
        finally:
            self._lock.release()

    def stop(self):
        """Hold where it is. Only sets a flag; the skill thread does the bus writes."""
        self._stop.set()

    def status(self):
        busy = self._lock.locked()
        if self.connected and not busy and self._bus.acquire(timeout=0.2):
            try:
                for i in range(1, 7):
                    self._joints[i] = self._rd(i, POSITION)
                self._torque = all(self._rd(i, TORQUE_EN, 1) == 1 for i in range(1, 7))
            except Exception as e:
                self.last_error = str(e)
            finally:
                self._bus.release()
        return {"connected": self.connected, "busy": busy, "torque": self._torque,
                "joints": {self.names.get(i, str(i)): v for i, v in sorted(self._joints.items())},
                "error": self.last_error}

    # skills; each returns (result, detail)
    def _skill_grip_bottle(self, pace, hold_s, flourish, look_first):
        t = S.leg_time(pace)
        self._emit("start")
        if look_first:
            self._leg_out(self.VIA, S.LOOK_LEG1 * t, grip=self.OPEN)
            self._wait(S.LOOK_PAUSE_S)
            self._leg_out(self.GRASP, S.LOOK_LEG2 * t)
        else:
            self._leg_out(self.GRASP, t, grip=self.OPEN)
        self._emit("at_target")
        grip = self._close_gripper()
        if grip is None:
            self._emit("missed")
            self._blend([(6, self.OPEN, 0, 0.6)], settle=False)
            res, detail = "missed", "the gripper closed on nothing"
        else:
            self._emit("gripped")
            self._blend([(2, self.LIFT, 0, S.lift_time(pace))])
            self._emit("lifted")
            if flourish:
                self._flourish(hold_s)
            else:
                self._wait(hold_s)
            self._blend([(2, self.GRASP[2], 0, S.lower_time(pace))])
            self._blend([(6, self.OPEN, 0, 0.6)], settle=False)
            self._wait(0.2)
            self._emit("put_back")
            res, detail = "done", f"bottle lifted and put back, gripped at {grip}"
        self._emit("returning")
        self._leg_back(t)
        self._emit("done")
        return res, detail

    def _flourish(self, hold_s):
        t0 = time.time()
        s5 = self._rd(5, POSITION)
        steps = [(ts, 5, s5 + d, S.FLOURISH_SPEED, S.FLOURISH_ACC) for ts, d in S.FLOURISH]
        self._timeline(steps, tail=S.FLOURISH_S - S.FLOURISH[-1][0])
        left = hold_s - (time.time() - t0)
        if left > 0:
            self._wait(left)

    def _skill_point(self, pace):
        leg = S.POINT_LEG * S.leg_time(pace)
        self._emit("start")
        self._leg_out(self.VIA, leg)
        self._emit("at_target")
        self._wait(S.POINT_HOLD_S)
        self._emit("returning")
        self._leg_back(leg)
        self._emit("done")
        return "done", "pointed at the bottle"

    def _skill_half_reach(self, pace):
        leg = S.HALF_LEG * S.leg_time(pace)
        self._emit("start")
        self._leg_out(self.HALF, leg)
        self._emit("at_target")
        self._wait(S.HALF_HOLD_S)
        self._emit("returning")
        self._leg_back(leg)
        self._emit("done")
        return "done", "reached part of the way"

    def _skill_wave(self):
        self._emit("start")
        self._leg_out(PHOTO, S.WAVE_LEG_S, grip=PHOTO_GRIP)
        self._emit("at_target")
        s = {i: self._rd(i, POSITION) for i in (1, 3, 4, 5, 6)}
        sh, op = WAVE_SHUT, WAVE_OPEN
        steps = [  # time s, joint, goal, speed: gif_thumbnail.py, unchanged
            (0.0, 1, s[1] + 250, 520), (0.0, 5, s[5] + 500, 950), (0.0, 4, s[4] - 150, 380), (0.0, 3, s[3] - 120, 320), (0.1, 6, sh, 900),
            (0.6, 6, op, 900),
            (1.0, 1, s[1] - 250, 620), (1.0, 5, s[5] - 500, 1150), (1.0, 4, s[4] + 80, 380), (1.0, 3, s[3], 320), (1.3, 6, sh, 900),
            (1.8, 6, op, 900),
            (2.0, 1, s[1] + 180, 560), (2.0, 5, s[5] + 400, 1050), (2.0, 4, s[4] - 120, 380), (2.0, 3, s[3] - 100, 320), (2.4, 6, sh, 900),
            (2.8, 6, op, 900),
            (3.0, 1, s[1] - 180, 520), (3.0, 5, s[5] - 400, 1050), (3.0, 4, s[4] + 60, 380), (3.0, 3, s[3], 320), (3.4, 6, sh, 900),
            (3.8, 6, op, 900),
            (4.0, 1, s[1], 400), (4.0, 5, s[5], 900), (4.0, 4, s[4], 300), (4.0, 3, s[3], 300), (4.5, 6, s[6], 600),
        ]
        self._timeline([(ts, i, g, sp, 12) for ts, i, g, sp in steps], tail=1.5, watch=(1, 3, 4))
        self._emit("returning")
        self._leg_back(S.WAVE_LEG_S, roll=s[5])
        self._emit("done")
        return "done", "waved"

    def _skill_rest(self):
        self._emit("start")
        self._emit("returning")
        holding = self._leg_back(S.REST_LEG_S)
        self._emit("done")
        return "done", "at rest; the gripper is still holding something" if holding else "at rest"

    def _skill_gesture(self, style, seconds):
        g = S.gesture_plan(style, seconds, self.REST)
        self._emit("start")
        self._leg_out(g["pose"], g["rise_s"], grip=g["grip"], settle=False)
        self._emit("at_target")
        base = dict(g["pose"])
        base[5] = self._rd(5, POSITION)
        base[6] = g["grip"]
        steps = S.gesture_steps(style, g["talk_s"], base, g["flex_anchor"], self.REST)
        tail = max(0.0, g["talk_s"] - (steps[-1][0] if steps else 0.0))
        self._timeline(steps, tail=tail, watch=(1, 2, 3, 4))
        self._emit("returning")
        self._leg_back(g["back_s"], roll=base[5])
        self._emit("done")
        return "done", f"gesture {style}"
