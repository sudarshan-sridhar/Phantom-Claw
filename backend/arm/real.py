"""RealArm: the SO-101 follower on COM3, driven with scservo_sdk.

Built from the hand-tested scripts in setup/tools:
- grip_bottle.py: rd/wr with retries, torque_on with the range check, blend() (one goal per
  joint, servo speed and acceleration registers, the elbow leads the shoulder by 30 percent,
  the settle correction), close_gripper() stall detection, the numbers, and the
  stop-and-hold error handling.
- fetch_bottle.py: the whole fetch_bottle skill, same numbers and same order of moves.
- gif_thumbnail.py: timed single goals per joint for the wave (and the flourish and gesture).

Nothing here touches the serial port on import. The port opens when RealArm() is created.
Torque is never switched off by this code. Every skill ends at rest, holding.
"""
import collections
import json
import math
import os
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

# fetch_bottle, from setup/tools/fetch_bottle.py
FETCH_OPEN = 1790                         # jaws open around the cap
FETCH_BASE_TORQUE = 1000                  # the base has to swing a loaded arm
FETCH_CUR_MAX = {1: 1300, 2: 1600, 3: 1400, 4: 800, 6: 700}
FETCH_TOL = 110                           # carrying a bottle, the elbow settles further short
FETCH_POLL = 0.01                         # arrival check interval (the script used 0.03)
MOVE_WINDOW = 0.05                        # about one loop of the script (0.03 s sleep plus the bus reads)
FETCH_CLOSE_SPEED = 500
LIFT_UP = 170                             # shoulder ticks to raise the bottle off the table
PERSON_PAN = (950, 3250)
TUG_TICKS, TUG_TICKS_GRIP = 14, 9


class _Stopped(Exception):
    pass


class _RealClock:
    @staticmethod
    def time():
        return time.time()

    @staticmethod
    def sleep(s):
        time.sleep(s)


def _wrist_fallback_default():
    return os.environ.get("PHANTOM_WRIST_FALLBACK", "1") != "0"


class RealArm:
    def __init__(self, port="COM3", baud=1_000_000, setup_dir=None, port_handler=None, packet_handler=None,
                 vision=None, wrist_fallback=None, clock=None, quiet=False):
        d = Path(setup_dir) if setup_dir else SETUP
        self.setup_dir = d
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

        from backend.vision.table import TableMap, Taught   # plain JSON maths, no camera code
        self.taught = Taught(d / "bottle_teach.json")
        self.table = TableMap(d / "table_map.json")
        self._vision = vision                 # made on first use, so the cameras load only for fetch_bottle
        self.wrist_fallback = _wrist_fallback_default() if wrist_fallback is None else wrist_fallback

        self._clock = clock or _RealClock()
        self._virtual = clock is not None     # tests drive a pretend clock together with a pretend bus
        self.quiet = quiet                    # no beeps
        self._lock = threading.Lock()     # one skill at a time
        self._bus = threading.Lock()      # one user of the serial bus at a time
        self._stop = threading.Event()
        self._emit_cb = None
        self._mute = False
        self._joints = {}
        self._torque = False
        self._last_cur = 0.0
        self._cur_max = dict(CUR_MAX)
        self._base_raised = False
        self.trace = collections.deque(maxlen=2000)   # every move, for the simulated-bus tests

        self._port_name, self._baud = port, baud
        self._p, self._h = port_handler, packet_handler
        self.connected = False
        self.last_error = ""
        self._connect()

    # time, so tests can run on a pretend clock
    def _now(self):
        return self._clock.time()

    def _sleep(self, s):
        self._clock.sleep(s)

    def _beep(self, freq, ms):
        if self.quiet:
            return
        try:
            import winsound
            winsound.Beep(freq, ms)
        except Exception:
            pass

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
        return max(self.rng[i][0], min(self.rng[i][1], int(g)))

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
        if self._virtual:
            self._check_stop()
            self._sleep(max(0.0, s))
            self._check_stop()
        elif self._stop.wait(max(0.0, s)):
            raise _Stopped()

    def _check_current(self, i):
        if i in self._cur_max and (self._rd(i, CURRENT) & 0x7FFF) * 6.5 > self._cur_max[i]:
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
    def _blend(self, items, settle=True, tol=60, acc=ACC, poll=0.03, window=None):
        """items: (id, goal, start_s, duration_s). One motion; each joint runs its own smooth ramp.
        tol: how far short a joint may stop and still count as there (fetch_bottle uses 110).
        window: None tests "has it moved 2 ticks" once a loop, as the scripts do. A number tests it
        over at least that many seconds and at least long enough for the slowest commanded joint to
        cover 4 ticks, so a slow move is not taken for a stall (fetch_bottle)."""
        items = [(i, self._clamp(i, g), ts, dur) for i, g, ts, dur in items]
        self.trace.append(("blend", tuple((i, g) for i, g, _, _ in items)))
        pos = {i: self._rd(i, POSITION) for i, *_ in items}
        speeds = {}
        for i, g, ts, dur in items:
            speeds[i] = max(12, round(abs(g - pos[i]) / dur))
            self._wr(i, ACCEL, acc, 1)
            self._wr(i, SPEED, speeds[i])
        if window is not None:
            moving = [speeds[i] for i, g, _, _ in items if abs(g - pos[i]) > 12]
            window = max(window, 4.0 / min(moving)) if moving else window
        sent, t0, last_move = set(), self._now(), self._now()
        ref, t_ref = dict(pos), self._now()
        end = max(ts + dur for _, _, ts, dur in items)
        while True:
            self._check_stop()
            t = self._now() - t0
            for i, g, ts, dur in items:
                if i not in sent and t >= ts:
                    self._wr(i, GOAL, g)
                    sent.add(i)
            self._sleep(poll)
            now = {i: self._rd(i, POSITION) for i, *_ in items}
            self._joints.update(now)
            for i in now:
                self._check_current(i)
            if window is None or self._now() - t_ref >= window - 1e-9:
                if any(abs(now[i] - ref[i]) > 2 for i in now):
                    last_move = self._now()
                ref, t_ref = now, self._now()
            done = len(sent) == len(items)
            if done and all(abs(now[i] - g) <= 12 for i, g, _, _ in items):
                return
            still = self._now() - last_move
            if done and t > end and still > 0.3 and all(abs(now[i] - g) <= tol for i, g, _, _ in items):
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

    def _close_gripper(self, speed=300, give_up_s=None):
        """Close until the jaws stop moving. Returns where they stopped when something is in
        them, else None. grip_bottle raises after 5 s; fetch_bottle (give_up_s=5) just looks."""
        self.trace.append(("close",))
        self._wr(6, ACCEL, 10, 1)
        self._wr(6, SPEED, speed)
        self._wr(6, GOAL, CLOSED)
        t_start = self._now()
        last, t_move = self._rd(6, POSITION), self._now()
        while self._now() - t_move < 0.35:
            self._check_stop()
            if give_up_s is not None and self._now() - t_start >= give_up_s:
                break
            self._sleep(0.03)
            pos = self._rd(6, POSITION)
            if abs(pos - last) > 2:
                last, t_move = pos, self._now()
            if give_up_s is None and self._now() - t_start > 5:
                raise RuntimeError("gripper did not settle")
        pos = self._rd(6, POSITION)
        self._joints[6] = pos
        if pos - CLOSED > MISS_TICKS:
            self._wr(6, GOAL, pos - 70)   # keep a light squeeze, the same force as before
            return pos
        return None

    def _leg_out(self, target, dur, grip=None, settle=True, acc=ACC):
        """Rest side to a pose: the elbow runs the whole time, the shoulder starts after LEAD."""
        items = [(3, target[3], 0, dur), (2, target[2], LEAD * dur, (1 - LEAD) * dur),
                 (1, target[1], 0, dur), (4, target[4], 0, dur)]
        if grip is not None:
            items.append((6, grip, 0, min(S.GRIPPER_OPEN_S, dur)))
        self._blend(items, settle, acc=acc)

    def _gripper_holding(self):
        g_pos, g_goal = self._rd(6, POSITION), self._rd(6, GOAL)
        return g_pos - CLOSED > MISS_TICKS and g_goal < g_pos - 40

    def _leg_back(self, dur, roll=None, acc=ACC):
        """Back to rest: all start together, the shoulder gets back early, the elbow finishes the fold.
        The gripper closes in the last 40 percent, unless it is squeezing something."""
        items = [(3, self.REST[3], 0, dur), (2, self.REST[2], 0, (1 - LEAD) * dur),
                 (1, self.REST[1], 0, dur), (4, self.REST[4], 0, dur)]
        holding = self._gripper_holding()
        if not holding:
            items.append((6, CLOSED, 0.6 * dur, 0.4 * dur))
        if roll is not None:
            items.append((5, roll, 0, dur))
        self._blend(items, settle=False, acc=acc)
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
            left = deadline - self._now()
            if left <= 0:
                return
            if watch and left > 0.08 and self._now() - self._last_cur >= 0.1:
                for i in watch:
                    self._check_current(i)
                self._last_cur = self._now()
                continue
            self._sleep(min(0.01, left))

    def _timeline(self, steps, tail, watch=()):
        """steps: (time s, id, goal, speed, acc), sorted by time."""
        t0 = self._now()
        for ts, i, g, sp, acc in steps:
            self._watch_until(t0 + ts, watch)
            self._go(i, g, sp, acc)
        last = steps[-1][0] if steps else 0.0
        self._watch_until(t0 + last + tail, watch)

    # run
    def _emit(self, name):
        cb = self._emit_cb
        if cb and not self._mute:
            try:
                cb(name)
            except Exception:
                pass   # a broken listener must not break the motion

    def _at_rest_now(self):
        pos = {i: self._rd(i, POSITION) for i in range(1, 5)}
        self._joints.update(pos)
        return all(abs(pos[i] - self.REST[i]) <= REST_TOL for i in range(1, 5))

    def _prepare(self, skill):
        if not self.connected:
            self._connect()
        if not self.connected:
            raise RuntimeError(f"arm not connected: {self.last_error}")
        pos = {i: self._rd(i, POSITION) for i in range(1, 5)}
        self._joints.update(pos)
        if not all(self._rd(i, TORQUE_EN, 1) == 1 for i in range(1, 7)):
            self._torque_on()
        self._torque = True
        if self._base_raised and all(abs(pos[i] - self.REST[i]) <= REST_TOL for i in range(1, 5)):
            self._restore_base_torque()   # a fetch that was stopped at rest left the base at full torque
        if skill not in ("gesture", "fetch_bottle"):   # gestures run in time with the audio; fetch beeps before it moves
            self._beep(1200, 300)

    def _auto_rest(self):
        """The arm was left out of rest (a stop or an error): go to rest first, quietly, then go on."""
        self._emit("auto_rest")
        self._mute = True
        try:
            _, detail = self._skill_rest()
        finally:
            self._mute = False
        if not self._at_rest_now():
            raise RuntimeError("tried to go back to rest first and did not get there")
        return detail

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
                auto = None
                try:
                    self._prepare(skill)
                    if skill != "rest" and not self._at_rest_now():
                        auto = self._auto_rest()
                    res, detail = getattr(self, "_skill_" + skill)(**p)
                    if auto is not None:
                        return S.result(res, f"went back to rest first ({auto}); {detail}", auto_rest=True)
                    return S.result(res, detail)
                except _Stopped:
                    self._hold_here()
                    return S.result("stopped", "stopped; holding where it is", **({"auto_rest": True} if auto else {}))
                except Exception as e:
                    self._hold_here()
                    self._mute = False
                    self._emit("error")
                    return S.result("error", str(e), **({"auto_rest": True} if auto else {}))
                except BaseException:
                    self._hold_here()
                    raise
                finally:
                    self._mute = False
                    self._emit_cb = None
        finally:
            self._lock.release()

    def stop(self):
        """Hold where it is. Only sets a flag; the skill thread does the bus writes."""
        self._stop.set()

    def status(self):
        busy = self._lock.locked()
        holding = None
        if self.connected and not busy and self._bus.acquire(timeout=0.2):
            try:
                for i in range(1, 7):
                    self._joints[i] = self._rd(i, POSITION)
                self._torque = all(self._rd(i, TORQUE_EN, 1) == 1 for i in range(1, 7))
                holding = self._gripper_holding()
            except Exception as e:
                self.last_error = str(e)
            finally:
                self._bus.release()
        j = self._joints
        at_rest = all(i in j and abs(j[i] - self.REST[i]) <= REST_TOL for i in range(1, 5)) if j else None
        return {"connected": self.connected, "busy": busy, "torque": self._torque, "at_rest": at_rest,
                "holding": holding,
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
        t0 = self._now()
        s5 = self._rd(5, POSITION)
        steps = [(ts, 5, s5 + d, S.FLOURISH_SPEED, S.FLOURISH_ACC) for ts, d in S.FLOURISH]
        self._timeline(steps, tail=S.FLOURISH_S - S.FLOURISH[-1][0])
        left = hold_s - (self._now() - t0)
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
        """Back to rest from any pose a skill can leave the arm in.
        Holding the bottle out of rest: stand it on the table where the arm is, let go, lift
        clear and fold back (the recovery in fetch_bottle.py). Jaws open and the arm reaching
        out: lift straight up first so the jaws come off whatever is under them."""
        self._emit("start")
        self._let_gripper_settle()
        pos = {i: self._rd(i, POSITION) for i in range(1, 5)}
        at_rest = all(abs(pos[i] - self.REST[i]) <= REST_TOL for i in range(1, 5))
        detail = "at rest"
        if self._gripper_holding() and not at_rest:
            self._put_down_here(1.0)
            detail = "put the bottle down on the table where the arm was, then went to rest"
            self._emit("returning")
            self._fetch_back_to_rest(1.0)
            holding = False
        else:
            if not at_rest and self._rd(6, POSITION) > CLOSED + 150 and pos[2] > self.REST[2] + 300:
                self._blend([(2, pos[2] - 200, 0, 1.0)], settle=False)
            self._emit("returning")
            holding = self._leg_back(S.REST_LEG_S)
        self._restore_base_torque()
        self._emit("done")
        return "done", "at rest; the gripper is still holding something" if holding else detail

    def _let_gripper_settle(self, limit_s=1.5):
        """A stop during a close leaves the jaws closing (the gripper goal is left alone). Wait
        for them to stop moving before deciding whether they hold the bottle."""
        t0, last = self._now(), self._rd(6, POSITION)
        while self._now() - t0 < limit_s:
            self._wait(0.1)
            pos = self._rd(6, POSITION)
            if abs(pos - last) <= 2:
                return
            last = pos

    def _skill_gesture(self, style, seconds):
        g = S.gesture_plan(style, seconds, self.REST)
        self._emit("start")
        self._leg_out(g["pose"], g["rise_cmd"], grip=g["grip"], settle=False, acc=g["acc"])
        self._emit("at_target")
        base = dict(g["pose"])
        base[5] = self._rd(5, POSITION)
        base[6] = g["grip"]
        steps = S.gesture_steps(style, g["talk_s"], base, g["flex_anchor"], self.REST)
        tail = max(0.0, g["talk_s"] - (steps[-1][0] if steps else 0.0))
        self._timeline(steps, tail=tail, watch=(1, 2, 3, 4))
        self._emit("returning")
        self._leg_back(g["back_cmd"], roll=base[5], acc=g["acc"])
        self._emit("done")
        return "done", f"gesture {style}"

    # fetch_bottle, ported from setup/tools/fetch_bottle.py
    @property
    def vision(self):
        if self._vision is None:
            from backend.vision import Vision
            self._vision = Vision(self.setup_dir)
        return self._vision

    def _fblend(self, items, settle=True):
        """blend() as fetch_bottle.py has it: 110 tick tolerance, quicker arrival checks."""
        self._blend(items, settle, tol=FETCH_TOL, poll=FETCH_POLL, window=MOVE_WINDOW)

    def _three(self, pose, dur):
        return [(2, pose["lift"], 0, dur), (3, pose["elbow"], 0, dur), (4, pose["wrist_flex"], 0, dur)]

    def _fetch_back_to_rest(self, k):
        """back_to_rest() from the script: one move, the gripper closes in the last 40 percent."""
        d = S.FETCH_REST_S * k
        self._fblend([(3, self.REST[3], 0, d), (2, self.REST[2], 0, (1 - LEAD) * d), (1, self.REST[1], 0, d),
                      (4, self.REST[4], 0, d), (6, CLOSED, 0.6 * d, 0.4 * d)], settle=False)

    def _put_down_here(self, k):
        """The script's recovery when the turn cannot finish: hold, pull in to the tuck, stand the
        bottle on the table where the arm is, let go and lift clear. The caller then folds back."""
        for i in (1, 2, 3, 4):
            self._wr(i, GOAL, self._clamp(i, self._rd(i, POSITION)))
        tuck, mid = self.taught.tuck(), self.taught.mid["grasp"]
        table = self.taught.table_lift_tucked()
        self._fblend(self._three(tuck, S.RECOVER_TUCK_S * k), settle=False)
        self._fblend([(3, mid["elbow"], 0, S.RECOVER_ALIGN_S * k), (4, mid["wrist_flex"], 0, S.RECOVER_ALIGN_S * k)], settle=False)
        self._fblend([(2, table, 0, S.RECOVER_TABLE_S * k)])
        self._fblend([(6, FETCH_OPEN, 0, S.FETCH_RELEASE_S)], settle=False)
        self._emit("put_back")
        self._wait(S.FETCH_AFTER_OPEN_S)
        self._fblend([(2, table - 220, 0, S.RECOVER_CLEAR_S * k)], settle=False)

    def _raise_base_torque(self):
        self._wr(1, TORQUE_LIMIT, FETCH_BASE_TORQUE)
        self._base_raised = True
        self._cur_max = dict(FETCH_CUR_MAX)

    def _restore_base_torque(self):
        """Back to the everyday base torque once the arm is home. Torque stays on."""
        if self._base_raised:
            self._wr(1, TORQUE_LIMIT, TORQUE[1])
            self._base_raised = False
        self._cur_max = dict(CUR_MAX)

    def _call_vision(self, fn):
        """Run a camera or Gemini call in a helper thread so stop() still works while it waits."""
        box = {}

        def go():
            try:
                box["v"] = fn()
            except BaseException as e:
                box["e"] = e
        self._check_stop()
        th = threading.Thread(target=go, daemon=True)
        th.start()
        while th.is_alive():
            th.join(0.05)
            self._check_stop()
        self._check_stop()
        if "e" in box:
            raise box["e"]
        return box.get("v")

    def _fetch_find(self, pace):
        """Where the bottle and the person are. Returns (plan, None) or (None, (result, detail, moved))."""
        T, k = self.taught, S.fetch_stretch(pace)
        try:
            seen = self._call_vision(lambda: self.vision.look_overhead(stop=self._stop))
        except _Stopped:
            raise
        except Exception as e:
            return None, ("no_bottle", f"the overhead look failed ({type(e).__name__}: {str(e)[:120]})", False)
        if seen is None:
            if not self.wrist_fallback:
                return None, ("no_bottle", "the overhead camera gives no picture (wrist fallback is off)", False)
            return self._wrist_find(k)
        if "bottle" not in seen:
            return None, ("no_bottle", "I cannot see the bottle on the table", False)
        B = seen["bottle"]
        if not T.in_reach(B["r_cm"]):
            self._emit("seen")
            return None, ("out_of_reach", f"the bottle is {B['r_cm']:.0f} cm from the arm; I can only reach it "
                                          f"between {T.min_r:.0f} and {T.max_r:.0f} cm", False)
        via, grasp, t_mix = T.posture(B["r_cm"])
        lo, hi = self.rng[1]
        pan_b = int(max(lo + 30, min(hi - 30, B["pan"])))
        person = seen["person"]["pan"] if "person" in seen else self.table.pan_for_bearing(1.2)
        pan_p = int(max(PERSON_PAN[0], min(PERSON_PAN[1], person)))
        note = (f"bottle at {B['r_cm']:.1f} cm, pan {pan_b}, blend {t_mix:.2f} between middle and far; held out at pan {pan_p}"
                + ("" if "person" in seen else " (person not seen, went to the right)"))
        self._emit("seen")
        return {"via": via, "grasp": grasp, "pan_b": pan_b, "pan_p": pan_p, "note": note, "mode": "overhead"}, None

    def _wrist_find(self, k):
        """Wrist camera only, for when the overhead phone gives no picture. From rest, tilt the
        wrist up to look across the desk and step the base through a few angles, asking Gemini
        at each for the bottle and the person. Then turn until the bottle is in the middle of
        the picture. Distance cannot be measured with one camera, so the taught middle grip is used."""
        from backend.vision import WRIST, wrist_pan_for
        W, tpr = WRIST, self.table.tpr
        lo, hi = self.rng[1][0] + 30, self.rng[1][1] - 30
        self.trace.append(("wrist_sweep",))
        self._fblend([(4, W["look_flex"], 0, 1.2 * k)], settle=False)
        bottle, person, asked, answered = None, None, 0, 0
        for deg in W["sweep_deg"]:
            pan = int(max(lo, min(hi, self.table.pan_for_bearing(math.radians(deg)))))
            here = self._rd(1, POSITION)
            self._fblend([(1, pan, 0, max(1.0, abs(pan - here) / 300.0) * k)], settle=False)
            self._wait(0.3)       # let the picture settle
            look = self._wrist_ask(pan)
            asked += 1
            if look is None:
                continue
            if not look["picture"]:
                return None, ("no_bottle", "no camera gives a picture (overhead and wrist)", True)
            answered += 1
            if look.get("person"):
                p_pan, xc = wrist_pan_for(look["person"], pan, tpr)
                if person is None or abs(xc - W["centre_x"]) < person[1]:
                    person = (p_pan, abs(xc - W["centre_x"]))
            if look.get("bottle") and bottle is None:
                bottle = (pan, look["bottle"])
            if bottle and person:
                break
        if bottle is None:
            why = "Gemini did not answer" if answered == 0 and asked else "the wrist camera could not find the bottle"
            return None, ("no_bottle", f"{why} (overhead camera gives no picture)", True)
        pan_b, xc = wrist_pan_for(bottle[1], bottle[0], tpr)
        for _ in range(W["centre_tries"]):
            here = self._rd(1, POSITION)
            step = max(-W["max_step_ticks"], min(W["max_step_ticks"], pan_b - here))
            target = int(max(lo, min(hi, here + step)))
            if abs(target - here) > 4:
                self._fblend([(1, target, 0, max(0.8, abs(target - here) / 250.0) * k)], settle=False)
                self._wait(0.3)
            look = self._wrist_ask(target)
            if not look or not look.get("picture") or not look.get("bottle"):
                pan_b = target
                break
            pan_b, xc = wrist_pan_for(look["bottle"], target, tpr)
            if abs(xc - W["centre_x"]) <= W["centred_x"]:
                break               # centred: keep this last, finest estimate
        pan_b = int(max(lo, min(hi, pan_b)))
        if person is not None:
            pan_p, pnote = person[0], "the person seen by the wrist camera"
        else:
            pan_p, pnote = self.table.pan_for_bearing(math.radians(W["person_default_deg"])), "person not seen, 45 degrees right"
        pan_p = int(max(PERSON_PAN[0], min(PERSON_PAN[1], pan_p)))
        try:
            self.vision.close_wrist()
        except Exception:
            pass
        via, grasp, _ = self.taught.middle()
        note = (f"wrist camera only: bottle centred at pan {pan_b}; distance cannot be measured with one camera, "
                f"so it used the taught middle grip; held out at pan {pan_p} ({pnote})")
        self._emit("seen")
        return {"via": via, "grasp": grasp, "pan_b": pan_b, "pan_p": pan_p, "note": note, "mode": "wrist"}, None

    def _wrist_ask(self, pan):
        """One wrist look. None when Gemini did not answer (that stop counts as nothing seen)."""
        try:
            return self._call_vision(lambda: self.vision.wrist_look(pan))
        except _Stopped:
            raise
        except Exception:
            return None

    def _wait_for_take(self, seconds, watcher):
        """'tug' when the bottle is pulled, 'hand' when the wrist camera sees a hand, None when time is up."""
        self.trace.append(("offer", seconds))
        self._wait(S.FETCH_OFFER_SETTLE_S)
        ids = (1, 2, 3, 4, 6)
        ref = {i: self._rd(i, POSITION) for i in ids}
        if watcher is not None:
            watcher.armed = True
        t0 = self._now()
        while self._now() - t0 < seconds:
            self._wait(0.03)
            now = {i: self._rd(i, POSITION) for i in ids}
            if any(abs(now[i] - ref[i]) > (TUG_TICKS if i != 6 else TUG_TICKS_GRIP) for i in ids):
                return "tug"
            if watcher is not None and getattr(watcher, "hand", False):
                return "hand"
        return None

    def _skill_fetch_bottle(self, pace, wait_s, look_first):
        k = S.fetch_stretch(pace)
        T = self.taught
        self._emit("start")
        self._raise_base_torque()
        plan, fail = self._fetch_find(pace)
        if fail:
            res, detail, moved = fail
            if moved:
                self._emit("returning")
                self._fetch_back_to_rest(k)
            self._restore_base_torque()
            self._emit("done")
            return res, detail
        via, grasp, pan_b, pan_p = plan["via"], plan["grasp"], plan["pan_b"], plan["pan_p"]
        far, tuck = T.far_grasp(), T.tuck()
        self._beep(1200, 300)
        # out to just above the cap in one move, then the short final approach
        t_out = S.FETCH_OUT_S * k
        self._fblend([(3, via["elbow"], 0, t_out), (2, via["lift"], LEAD * t_out, (1 - LEAD) * t_out),
                      (1, pan_b, 0, t_out), (4, via["wrist_flex"], 0, t_out),
                      (6, FETCH_OPEN, 0, min(S.FETCH_OPEN_S, t_out))], settle=False)
        if look_first:
            self._wait(S.FETCH_LOOK_FIRST_S)
        self._fblend(self._three(grasp, S.FETCH_APPROACH_S * k))
        self._emit("at_target")
        grip = self._close_gripper(speed=FETCH_CLOSE_SPEED, give_up_s=5.0)
        if grip is None:
            self._emit("missed")
            self._fblend([(6, FETCH_OPEN, 0, S.FETCH_RELEASE_S)], settle=False)
            self._fblend(self._three(via, S.FETCH_VIA_S * k), settle=False)
            self._emit("returning")
            self._fetch_back_to_rest(k)
            self._restore_base_torque()
            self._emit("done")
            return "missed", f"the gripper closed on nothing. {plan['note']}"
        self._emit("gripped")
        # lift and pull the bottle in close in one move: the base joint cannot swing a loaded arm held far out
        self._fblend(self._three(tuck, S.FETCH_TUCK_S * k), settle=False)
        self._emit("lifted")
        turn = S.fetch_turn_s(pan_b, pan_p, pace)
        watcher = None
        try:
            watcher = self.vision.hand_watcher()
        except Exception:
            watcher = None        # no wrist camera: a tug still works
        try:
            self._check_stop()
            self._emit("carrying")
            reach = S.FETCH_REACH_S * k
            try:
                # turn toward the person and start reaching out during the last part of the turn
                self._fblend([(1, pan_p, 0, turn), (2, far["lift"] - 120, 0.5 * turn, reach),
                              (3, far["elbow"], 0.5 * turn, reach), (4, far["wrist_flex"], 0.5 * turn, reach)], settle=False)
            except RuntimeError as e:
                # could not finish the turn (too heavy, or something in the way): stand the bottle down here
                at = self._rd(1, POSITION)
                self._put_down_here(k)
                self._emit("returning")
                self._fetch_back_to_rest(k)
                self._restore_base_torque()
                self._emit("done")
                return "turn_failed", (f"could not finish the turn ({e}) at pan {at}; the bottle is on the table "
                                       f"where the arm stopped. {plan['note']}")
            self._emit("offering")
            self._beep(900, 150)
            self._beep(900, 150)          # ready: pull it out of the jaws
            took = self._wait_for_take(wait_s, watcher)
        finally:
            if watcher is not None:
                watcher.stop()
        checks = getattr(watcher, "checks", 0) if watcher is not None else 0
        if took:
            self._fblend([(6, FETCH_OPEN, 0, S.FETCH_HANDOVER_OPEN_S)], settle=False)
            self._emit("handed_over")
            self._wait(S.FETCH_HANDOVER_WAIT_S)
            self._fblend([(2, far["lift"] - 300, 0, S.FETCH_CLEAR_S * k)], settle=False)   # lift clear of the hand
            res, detail = "handed_over", f"handed over ({took}, {checks} hand checks). {plan['note']}"
        else:
            self._fblend(self._three(tuck, S.FETCH_TUCK_S * k), settle=False)
            self._fblend([(1, pan_b, 0, turn)], settle=False)
            self._fblend([(2, grasp["lift"] - LIFT_UP, 0, S.FETCH_ABOVE_SPOT_S * k),
                          (3, grasp["elbow"], 0, S.FETCH_ABOVE_SPOT_S * k),
                          (4, grasp["wrist_flex"], 0, S.FETCH_ABOVE_SPOT_S * k)], settle=False)
            self._fblend([(2, grasp["lift"], 0, S.FETCH_SET_S * k)])
            self._fblend([(6, FETCH_OPEN, 0, S.FETCH_RELEASE_S)], settle=False)
            self._emit("put_back")
            self._wait(S.FETCH_AFTER_OPEN_S)
            self._fblend(self._three(via, S.FETCH_VIA_S * k), settle=False)
            res, detail = "put_back", f"nobody took it in {wait_s:g} s ({checks} hand checks); put it back. {plan['note']}"
        self._emit("returning")
        self._fetch_back_to_rest(k)
        self._restore_base_torque()
        self._emit("done")
        return res, detail
