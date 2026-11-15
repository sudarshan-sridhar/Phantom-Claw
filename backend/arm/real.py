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

from . import desk
from . import skills as S
from .desk import Refuse

SETUP = Path(__file__).resolve().parents[2] / "setup"

# Feetech STS3215 registers, the same addresses as setup/tools
TORQUE_EN, ACCEL, GOAL, SPEED, TORQUE_LIMIT, POSITION, CURRENT = 40, 41, 42, 46, 48, 56, 69

ACC = 7                                   # gentle ramps for the blended legs (x100 ticks/s^2)
# Smoothness (README "Smooth motion"). Every move writes the acceleration register before its goal;
# 0 would mean no ramp at all, so nothing here ever writes less than 1.
LOAD_ACC_SCALE = 0.7                      # an item in the jaws: speed changes this much gentler, it swings
MIN_LEG_ACC = 3                           # the softest ramp a leg gets (slow pace and loaded)
STOP_ACC = 30                             # a stop: firm, but not the instant 0, and no lurch back
BLEND_WIN = 60                            # blended legs hand over this many ticks (at most) before the goal
APPROACH_WIN = 30                         # into the final approach: a smaller corner over the item
TRACE_HZ = 30.0                           # try_real --trace sample rate
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
TUG_TICKS, TUG_TICKS_GRIP = 8, 6
ROLL_REST = 2016         # the wrist roll at rest: jaws level
TUG_MA = 70.0            # a joint drawing this much more or less current than when it settled is being pulled


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
                 vision=None, wrist_fallback=None, clock=None, quiet=False, wrist_aim=None):
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
        # which tiers pick lines up with the wrist camera before going down: None reads
        # PHANTOM_WRIST_AIM (default "low"); or "all", "off", or a list of tier names
        self.aim_setting = wrist_aim

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
        self._tr = None                       # the motion trace (trace_start), positions at TRACE_HZ
        self._goal_now = {}                   # the last goal and acceleration written to each joint
        self._acc_now = {}
        # what pick left in the jaws: _held (the item, as the planner sees it plus the grip), its tier,
        # the jaw position it closed at, and _carry (the arm is in the tucked carry pose with it)
        self._held, self._held_tier, self._held_grip, self._carry = None, None, None, False
        self._pan_range = (self.rng[1][0] + 30, self.rng[1][1] - 30)

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
                if a == POSITION and self._tr is not None:
                    self._tr["seen"][i] = (self._now(), v)    # free trace sample: the read was needed anyway
                return v
        raise RuntimeError(f"read {i}/{a} failed")

    def _wr(self, i, a, v, n=2):
        h, p = self._h, self._p
        for _ in range(6):
            c, _ = (h.write1ByteTxRx if n == 1 else h.write2ByteTxRx)(p, i, a, v)
            if c == scs.COMM_SUCCESS:
                if a == ACCEL:
                    self._acc_now[i] = int(v)
                elif a == GOAL:
                    self._goal_now[i] = int(v)
                    if self._tr is not None:
                        self._tr_event("goal", i, int(v))
                return
        raise RuntimeError(f"write {i}/{a} failed")

    # motion trace (try_real --trace, backend/arm/trace_report.py)
    def trace_start(self):
        """Record every joint's position at about TRACE_HZ, the phases, every goal written and the
        points where the arm stops on purpose, until trace_stop(). Samples come from the reads the
        motion loops already make; only joints not read in the last 20 ms are read, between moves'
        own bus traffic, in the skill thread, so the motion is not disturbed."""
        self._tr = {"t0": self._now(), "rows": [], "events": [], "last": -1.0, "seen": {}}

    def trace_stop(self):
        tr, self._tr = self._tr, None
        if tr is None:
            return None
        return {"hz": TRACE_HZ, "t0": tr["t0"], "joints": {str(i): self.names.get(i, str(i)) for i in range(1, 7)},
                "rows": tr["rows"], "events": tr["events"]}

    def _tr_event(self, kind, *data):
        tr = self._tr
        if tr is not None:
            tr["events"].append([kind, round(self._now() - tr["t0"], 4), *data])

    def _hold(self, why):
        """Mark a point where the arm stops on purpose (a picture, the jaws closing or opening, the
        offer, a hold). Only for the trace; trace_report counts any other stop as a jitter."""
        self._tr_event("hold", why)

    def _tr_tick(self):
        tr = self._tr
        if tr is None:
            return
        now = self._now()
        if now - tr["last"] < 1.0 / TRACE_HZ - 1e-6:
            return
        tr["last"] = now
        try:
            for i in range(1, 7):
                seen = tr["seen"].get(i)
                if seen is None or now - seen[0] > 0.02:
                    self._rd(i, POSITION)
        except Exception:
            return                        # a missed sample never stops a skill
        tr["rows"].append([round(now - tr["t0"], 4)] + [tr["seen"][i][1] for i in range(1, 7)])

    def _clamp(self, i, g):
        return max(self.rng[i][0], min(self.rng[i][1], int(g)))

    def _torque_on(self):
        start = {i: self._rd(i, POSITION) for i in range(1, 7)}
        bad = [i for i in (1, 2, 3, 4, 6) if not (self.rng[i][0] - 25 <= start[i] <= self.rng[i][1] + 25)]
        if bad:
            raise RuntimeError(f"joints {bad} read out of range: {start}")
        for i in range(1, 7):
            self._wr(i, ACCEL, ACC, 1)    # a power cycle leaves it at its default; never start with no ramp
            self._wr(i, GOAL, self._clamp(i, start[i]))
            self._wr(i, TORQUE_LIMIT, TORQUE[i])
        for i in range(1, 7):
            self._wr(i, TORQUE_EN, 1, 1)

    # stop handling
    def _check_stop(self):
        if self._stop.is_set():
            raise _Stopped()

    def _wait(self, s):
        if self._tr is not None:          # tracing: keep sampling while the arm waits
            end = self._now() + max(0.0, s)
            while True:
                self._check_stop()
                self._tr_tick()
                left = end - self._now()
                if left <= 0:
                    return
                self._wait_plain(min(left, 1.0 / TRACE_HZ))
        self._wait_plain(s)

    def _wait_plain(self, s):
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
        goal is left alone so a held bottle keeps its squeeze. A joint still on its way gets a firm
        ramp (STOP_ACC) and a goal its braking distance on, at most its old goal, so it stops in a
        few ticks instead of running past the spot on the soft leg ramp and swinging back."""
        for i in (1, 2, 3, 4, 5):
            try:
                p = self._rd(i, POSITION)
                try:
                    g, v = self._rd(i, GOAL), self._rd(i, SPEED)
                    ahead = min(abs(g - p), v * v / (2 * STOP_ACC * 100.0)) if v > 0 else 0.0
                    self._wr(i, ACCEL, STOP_ACC, 1)
                except Exception:
                    g, ahead = p, 0.0
                self._wr(i, GOAL, self._clamp(i, p + math.copysign(ahead, g - p)))
            except Exception:
                pass

    # motion, as grip_bottle.py
    def _blend(self, items, settle=True, tol=60, acc=ACC, poll=0.03, window=None, early=None, ramp=None):
        """items: (id, goal, start_s, duration_s). One motion: one goal per joint, the servo's own
        speed and acceleration ramp does the rest (no stepping in Python).
        acc: the ramp of the fastest arm joint (x100 ticks/s^2). The other arm joints get a softer
        one in proportion to their speed, so they all ramp up and down over the same time and the
        joints of a leg finish together; the gripper keeps acc.
        ramp: seconds each arm joint ramps up (and down) instead, acc then only caps it.
        tol: how far short a joint may stop and still count as there (fetch_bottle uses 110).
        window: None tests "has it moved 2 ticks" once a loop, as the scripts do. A number tests it
        over at least that many seconds and at least long enough for the slowest commanded joint to
        cover 4 ticks, so a slow move is not taken for a stall (fetch_bottle).
        early: blend into the next move. Return as soon as every joint is inside its braking window
        (half its braking distance, at most this many ticks) instead of waiting for a dead stop, so
        the next goal arrives while this one is still easing in. Only where no picture or grip comes
        next."""
        items = [(i, self._clamp(i, g), ts, dur) for i, g, ts, dur in items]
        self.trace.append(("blend", tuple((i, g) for i, g, _, _ in items)))
        self._tr_event("leg", [[i, g] for i, g, _, _ in items])
        pos = {i: self._rd(i, POSITION) for i, *_ in items}
        speeds = {i: max(12, round(abs(g - pos[i]) / dur)) for i, g, ts, dur in items}
        arm = [speeds[i] for i, g, _, _ in items if i != 6 and abs(g - pos[i]) > 2]
        tau = ramp if ramp is not None else (max(arm) / (acc * 100.0) if arm else None)
        accs = {}
        for i, g, ts, dur in items:
            # a joint still on its way to an earlier goal (a blended leg, a gesture) may be moving
            # fast: it keeps the full ramp, or the one it is on if firmer, so it can slow down in time
            if abs(self._goal_now.get(i, pos[i]) - pos[i]) > 12:
                accs[i] = max(acc, self._acc_now.get(i, acc))
            else:
                accs[i] = acc if i == 6 or not tau else max(1, min(acc, round(speeds[i] / tau / 100.0)))
            self._wr(i, ACCEL, accs[i], 1)
            self._wr(i, SPEED, speeds[i])
        win = None
        if early is not None:
            win = {i: max(8.0, min(early, 0.25 * speeds[i] ** 2 / (accs[i] * 100.0))) for i in speeds}
        if window is not None:
            moving = [speeds[i] for i, g, _, _ in items if abs(g - pos[i]) > 12]
            window = max(window, 4.0 / min(moving)) if moving else window
        sent, t0, last_move = set(), self._now(), self._now()
        ref, t_ref, prev = dict(pos), self._now(), dict(pos)
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
            self._tr_tick()
            if window is None or self._now() - t_ref >= window - 1e-9:
                if any(abs(now[i] - ref[i]) > 2 for i in now):
                    last_move = self._now()
                ref, t_ref = now, self._now()
            done = len(sent) == len(items)
            if done and all(abs(now[i] - g) <= (12 if win is None else max(12, win[i])) for i, g, _, _ in items):
                return
            # a blended leg whose joints sagged short goes on as soon as they have come most of the way
            # (or had most of the leg's time) and stopped creeping (a tick or less since the last read),
            # not after 0.3 s standing still
            if done and win is not None and all(
                    abs(now[i] - g) <= win[i]
                    or (abs(now[i] - g) <= tol and abs(now[i] - prev[i]) <= 1
                        and (abs(now[i] - g) <= 0.5 * abs(pos[i] - g) or t > 0.7 * end))
                    for i, g, _, _ in items):
                return
            prev = now
            still = self._now() - last_move
            if done and t > end and still > 0.3 and all(abs(now[i] - g) <= tol for i, g, _, _ in items):
                if settle:   # sagged a little short under its own weight: one small push, at low speed
                    self._hold("settle")    # on purpose: only final legs, right before the jaws close or open
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

    def _close_gripper(self, speed=300, give_up_s=None, miss_below=None):
        """Close until the jaws stop moving. Returns where they stopped when something is in
        them, else None. grip_bottle raises after 5 s; fetch_bottle (give_up_s=5) just looks.
        miss_below: for a thin item (pick), jaws stopping at or under this count as a miss instead
        of the usual 100 ticks above fully closed."""
        self.trace.append(("close",))
        self._hold("grip")
        self._tr_event("leg", [[6, CLOSED]])
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
            self._tr_tick()
            if abs(pos - last) > 2:
                last, t_move = pos, self._now()
            if give_up_s is None and self._now() - t_start > 5:
                raise RuntimeError("gripper did not settle")
        pos = self._rd(6, POSITION)
        self._joints[6] = pos
        got = pos - CLOSED > MISS_TICKS if miss_below is None else pos > miss_below
        if got:
            self._wr(6, GOAL, self._clamp(6, pos - 70))   # keep a light squeeze, the same force as before
            return pos
        return None

    def _leg_out(self, target, dur, grip=None, settle=True, acc=ACC, early=None):
        """Rest side to a pose: the elbow runs the whole time, the shoulder starts after LEAD."""
        items = [(3, target[3], 0, dur), (2, target[2], LEAD * dur, (1 - LEAD) * dur),
                 (1, target[1], 0, dur), (4, target[4], 0, dur)]
        if grip is not None:
            items.append((6, grip, 0, min(S.GRIPPER_OPEN_S, dur)))
        self._blend(items, settle, acc=acc, early=early)

    def _gripper_holding(self):
        g_pos, g_goal = self._rd(6, POSITION), self._rd(6, GOAL)
        if self._held_grip is not None:     # a picked item, maybe a thin one: judged against its own grip
            return g_pos > self._held_grip - 45 and g_goal < g_pos - 40
        return g_pos - CLOSED > MISS_TICKS and g_goal < g_pos - 40

    def _leg_acc(self, k=1.0, loaded=False):
        """The ramp for a leg of the general skills: ACC at full pace, softer at a slow pace (k is
        the pace stretch, up to 1.8) so a slow character is the same motion slowed down, and softer
        again with an item in the jaws. Never under MIN_LEG_ACC."""
        a = ACC / max(1.0, k) * (LOAD_ACC_SCALE if loaded else 1.0)
        return max(MIN_LEG_ACC, int(a + 0.5))

    def _level_wrist(self):
        """Turn the wrist back to level if a pick or a handover left it turned. One slow glide."""
        try:
            off = abs(self._rd(5, POSITION) - ROLL_REST)
            if off > 30:
                self._go(5, ROLL_REST, 450, acc=8)
                self._wait(min(1.6, 0.3 + off / 450.0))
        except _Stopped:
            raise
        except Exception:
            pass

    def _level_wrist(self):
        """Turn the wrist back to level if a pick or a handover left it turned. One slow glide."""
        try:
            off = abs(self._rd(5, POSITION) - ROLL_REST)
            if off > 30:
                self._go(5, ROLL_REST, 450, acc=8)
                self._wait(min(1.6, 0.3 + off / 450.0))
        except _Stopped:
            raise
        except Exception:
            pass

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
        if roll is None:
            self._level_wrist()
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
            if left > 0.03:
                self._tr_tick()
            self._sleep(min(0.01, left))

    def _timeline(self, steps, tail, watch=()):
        """steps: (time s, id, goal, speed, acc), sorted by time. Timed single goals per joint with
        the servo's own ramp, as gif_thumbnail.py; never stepped in small increments."""
        self._tr_event("leg", [[i, g] for _, i, g, _, _ in steps])
        t0 = self._now()
        for ts, i, g, sp, acc in steps:
            self._watch_until(t0 + ts, watch)
            self._go(i, g, sp, acc)
        last = steps[-1][0] if steps else 0.0
        self._watch_until(t0 + last + tail, watch)

    # run
    def _emit(self, name):
        self._tr_event("phase", name if not name.startswith("aiming") else name.split(":")[0])
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
        if skill not in ("gesture", "fetch_bottle", "pick", "point_at"):
            # gestures run in time with the audio; fetch, pick and point_at look first and beep before they move
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
                self._tr_event("skill", skill)
                auto = None
                try:
                    early = self._carry_gate(skill)
                    if early is not None:
                        return early
                    if skill != "look":            # look never moves, so it needs no torque and no rest
                        self._prepare(skill)
                        keep = self._carry and skill in ("place", "hand_over")   # carrying: start from the carry pose
                        if skill != "rest" and not keep and not self._at_rest_now():
                            auto = self._auto_rest()
                    out = getattr(self, "_skill_" + skill)(**p)
                    res, detail = out[0], out[1]
                    extra = out[2] if len(out) > 2 else {}
                    if auto is not None:
                        return S.result(res, f"went back to rest first ({auto}); {detail}", auto_rest=True, **extra)
                    return S.result(res, detail, **extra)
                except _Stopped:
                    self._carry = False            # out of the carry pose; the next skill rests first
                    self._hold_here()
                    return S.result("stopped", "stopped; holding where it is", **({"auto_rest": True} if auto else {}))
                except Exception as e:
                    self._carry = False
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
                    self._tr_event("end", skill)
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
                "holding": holding, "held": self._held if self._carry else None,
                "joints": {self.names.get(i, str(i)): v for i, v in sorted(self._joints.items())},
                "error": self.last_error}

    # skills; each returns (result, detail)
    def _skill_grip_bottle(self, pace, hold_s, flourish, look_first):
        t = S.leg_time(pace)
        self._emit("start")
        if look_first:
            self._leg_out(self.VIA, S.LOOK_LEG1 * t, grip=self.OPEN)
            self._hold("look_first")
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
            self._hold("hold")
            if flourish:
                self._flourish(hold_s)
            else:
                self._wait(hold_s)
            self._blend([(2, self.GRASP[2], 0, S.lower_time(pace))])
            self._hold("release")
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
        self._hold("point")
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
        self._hold("point")
        self._wait(S.HALF_HOLD_S)
        self._emit("returning")
        self._leg_back(leg)
        self._emit("done")
        return "done", "reached part of the way"

    def _skill_wave(self):
        self._emit("start")
        # blend straight into the wave: no settle push and no stop at the photo pose. The wave is
        # centred on the pose it was sent to, not on where the joints happen to be as it starts.
        self._leg_out(PHOTO, S.WAVE_LEG_S, grip=PHOTO_GRIP, settle=False, early=BLEND_WIN)
        self._emit("at_target")
        s = {1: PHOTO[1], 3: PHOTO[3], 4: PHOTO[4], 5: self._rd(5, POSITION), 6: PHOTO_GRIP}
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
        # tail: the script waited 1.5 s after the last goals; 0.6 s lets the fold home start as they
        # finish instead of after the arm has stood still for half a second
        self._timeline([(ts, i, g, sp, 12) for ts, i, g, sp in steps], tail=0.6, watch=(1, 3, 4))
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
        if abs(self._rd(5, POSITION) - ROLL_REST) > 30:
            # a pick or a handover left the wrist turned: bring it level again
            self._level_wrist()
        if self._gripper_holding() and not at_rest:
            what = self._held["name"] if self._held else "bottle"
            self._put_down_here(1.0, self._held_tier)
            detail = f"put the {what} down on the table where the arm was, then went to rest"
            self._emit("returning")
            self._fetch_back_to_rest(1.0)
            holding = False
        else:
            if not at_rest and self._rd(6, POSITION) > CLOSED + 150 and pos[2] > self.REST[2] + 300:
                # lift clear, flowing straight into the fold home (no stop between)
                self._blend([(2, pos[2] - 200, 0, 1.0)], settle=False, early=BLEND_WIN)
            self._emit("returning")
            holding = self._leg_back(S.REST_LEG_S)
        self._restore_base_torque()
        self._carry = False
        if not holding:
            self._forget_held()
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
        # the rise flows into the talking moves (they are offsets from the pose it was sent to)
        self._leg_out(g["pose"], g["rise_cmd"], grip=g["grip"], settle=False, acc=g["acc"], early=BLEND_WIN)
        self._emit("at_target")
        base = dict(g["pose"])
        base[5] = self._rd(5, POSITION)
        base[6] = g["grip"]
        steps = S.gesture_steps(style, g["talk_s"], base, g["flex_anchor"], self.REST)
        tail = max(0.0, g["talk_s"] - (steps[-1][0] if steps else 0.0))
        if steps and steps[0][0] > 0.3:
            self._hold("pose")       # this character holds the talking pose before its first move (gowther)
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

    def _fblend(self, items, settle=True, acc=ACC, early=None):
        """blend() as fetch_bottle.py has it: 110 tick tolerance, quicker arrival checks."""
        self._blend(items, settle, tol=FETCH_TOL, acc=acc, poll=FETCH_POLL, window=MOVE_WINDOW, early=early)

    def _three(self, pose, dur):
        return [(2, pose["lift"], 0, dur), (3, pose["elbow"], 0, dur), (4, pose["wrist_flex"], 0, dur)]

    def _fetch_back_to_rest(self, k, acc=ACC):
        """back_to_rest() from the script: one move, the gripper closes in the last 40 percent."""
        d = S.FETCH_REST_S * k
        self._fblend([(3, self.REST[3], 0, d), (2, self.REST[2], 0, (1 - LEAD) * d), (1, self.REST[1], 0, d),
                      (4, self.REST[4], 0, d), (6, CLOSED, 0.6 * d, 0.4 * d)], settle=False, acc=acc)
        self._level_wrist()

    def _put_down_here(self, k, tier=None):
        """The script's recovery when the turn cannot finish: hold, pull in to the tuck, stand the
        bottle on the table where the arm is, let go and lift clear. The caller then folds back.
        tier: an item picked with another tier's grips is set down with that tier's poses.
        Loaded legs use the gentle loaded ramp and flow into each other down to the table."""
        for i in (1, 2, 3, 4):
            self._wr(i, GOAL, self._clamp(i, self._rd(i, POSITION)))
        if tier is None or tier.name == "tall":
            tuck, mid = self.taught.tuck(), self.taught.mid["grasp"]
            table = self.taught.table_lift_tucked()
            opened = FETCH_OPEN
        else:
            tuck, mid, table, opened = tier.tuck(), tier.base_grasp(), tier.table_lift(), tier.open
        held = self._leg_acc(k, loaded=True)
        self._fblend(self._three(tuck, S.RECOVER_TUCK_S * k), settle=False, acc=held, early=BLEND_WIN)
        self._fblend([(3, mid["elbow"], 0, S.RECOVER_ALIGN_S * k), (4, mid["wrist_flex"], 0, S.RECOVER_ALIGN_S * k)],
                     settle=False, acc=held, early=BLEND_WIN)
        self._fblend([(2, table, 0, S.RECOVER_TABLE_S * k)], acc=held)
        self._hold("release")
        self._fblend([(6, opened, 0, S.FETCH_RELEASE_S)], settle=False)
        self._forget_held()
        self._emit("put_back")
        self._wait(S.FETCH_AFTER_OPEN_S)
        self._fblend([(2, table - 220, 0, S.RECOVER_CLEAR_S * k)], settle=False, early=BLEND_WIN)

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
        self._hold("look")
        while th.is_alive():
            th.join(0.05)
            self._check_stop()
            self._tr_tick()
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
            self._hold("picture")
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
                self._hold("picture")
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
        """'tug' when the item is pulled, 'hand' when the wrist camera sees a hand, None when time is up.
        A pull is felt two ways: a joint is moved off where it was holding, or a joint starts
        working harder to stay there (its current changes). Either, twice in a row, counts."""
        self.trace.append(("offer", seconds))
        self._hold("offer")
        self._wait(S.FETCH_OFFER_SETTLE_S)
        ids = (1, 2, 3, 4, 6)
        ref = {i: self._rd(i, POSITION) for i in ids}
        amps = (2, 3, 4, 6)

        def currents():
            return {i: (self._rd(i, CURRENT) & 0x7FFF) * 6.5 for i in amps}

        base = currents()
        for _ in range(4):                      # a short average, so one noisy read is not the baseline
            self._wait(0.03)
            c = currents()
            base = {i: 0.7 * base[i] + 0.3 * c[i] for i in amps}
        if watcher is not None:
            watcher.armed = True
        t0, hits = self._now(), 0
        while self._now() - t0 < seconds:
            self._wait(0.03)
            now = {i: self._rd(i, POSITION) for i in ids}
            moved = any(abs(now[i] - ref[i]) > (TUG_TICKS if i != 6 else TUG_TICKS_GRIP) for i in ids)
            c = currents()
            pulled = any(abs(c[i] - base[i]) > TUG_MA for i in amps)
            hits = hits + 1 if (moved or pulled) else 0
            if hits >= 2:
                return "tug"
            if watcher is not None and getattr(watcher, "hand", False):
                self._wait(0.4)     # give the hand a moment to close on it before the jaws open
                return "hand"
        return None
    def _still_holding(self, grip):
        """True while the bottle is in the jaws. When the grip closes on the cap the gripper is told
        to squeeze 70 ticks further than where it stopped, so if the bottle slips out the jaws
        close in by that much. Seeing them well inside the grip position means it is gone."""
        return self._rd(6, POSITION) > grip - 45

    def _lost_the_bottle(self, k, plan, when, what="the bottle"):
        """The bottle slipped out. Say so and go home; do not carry on and offer an empty hand."""
        self._forget_held()
        self._emit("missed")
        self._emit("returning")
        self._fetch_back_to_rest(k)
        self._restore_base_torque()
        self._emit("done")
        return "missed", f"had {what}, then lost it {when}. {plan['note']}"

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
        held = self._leg_acc(k, loaded=True)     # the bottle in the jaws: gentler speed changes
        t_out = S.FETCH_OUT_S * k
        self._fblend([(3, via["elbow"], 0, t_out), (2, via["lift"], LEAD * t_out, (1 - LEAD) * t_out),
                      (1, pan_b, 0, t_out), (4, via["wrist_flex"], 0, t_out),
                      (6, FETCH_OPEN, 0, min(S.FETCH_OPEN_S, t_out))], settle=False,
                     early=None if look_first else APPROACH_WIN)
        if look_first:
            self._hold("look_first")
            self._wait(S.FETCH_LOOK_FIRST_S)
        self._fblend(self._three(grasp, S.FETCH_APPROACH_S * k))
        self._emit("at_target")
        grip = self._close_gripper(speed=FETCH_CLOSE_SPEED, give_up_s=5.0)
        if grip is None:
            self._emit("missed")
            self._hold("release")
            self._fblend([(6, FETCH_OPEN, 0, S.FETCH_RELEASE_S)], settle=False)
            self._fblend(self._three(via, S.FETCH_VIA_S * k), settle=False, early=BLEND_WIN)
            self._emit("returning")
            self._fetch_back_to_rest(k)
            self._restore_base_torque()
            self._emit("done")
            return "missed", f"the gripper closed on nothing. {plan['note']}"
        self._emit("gripped")
        plan["note"] = f"gripped at {grip}. {plan['note']}"
        # lift and pull the bottle in close in one move: the base joint cannot swing a loaded arm held far out
        self._fblend(self._three(tuck, S.FETCH_TUCK_S * k), settle=False, acc=held)
        if not self._still_holding(grip):
            return self._lost_the_bottle(k, plan, "during the lift")
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
                              (3, far["elbow"], 0.5 * turn, reach), (4, far["wrist_flex"], 0.5 * turn, reach)],
                             settle=False, acc=held)
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
            if not self._still_holding(grip):
                return self._lost_the_bottle(k, plan, "while carrying it")
            self._emit("offering")
            self._beep(900, 150)
            self._beep(900, 150)          # ready: pull it out of the jaws
            took = self._wait_for_take(wait_s, watcher)
        finally:
            if watcher is not None:
                watcher.stop()
        checks = getattr(watcher, "checks", 0) if watcher is not None else 0
        if took:
            self._hold("release")
            self._fblend([(6, FETCH_OPEN, 0, S.FETCH_HANDOVER_OPEN_S)], settle=False)
            self._emit("handed_over")
            self._wait(S.FETCH_HANDOVER_WAIT_S)
            self._fblend([(2, far["lift"] - 300, 0, S.FETCH_CLEAR_S * k)], settle=False,
                         early=BLEND_WIN)   # lift clear of the hand, flowing into the fold home
            res, detail = "handed_over", f"handed over ({took}, {checks} hand checks). {plan['note']}"
        else:
            # tuck, turn back and down above the spot flow into each other; it stops only to stand it down
            self._fblend(self._three(tuck, S.FETCH_TUCK_S * k), settle=False, acc=held, early=BLEND_WIN)
            self._fblend([(1, pan_b, 0, turn)], settle=False, acc=held, early=BLEND_WIN)
            self._fblend([(2, grasp["lift"] - LIFT_UP, 0, S.FETCH_ABOVE_SPOT_S * k),
                          (3, grasp["elbow"], 0, S.FETCH_ABOVE_SPOT_S * k),
                          (4, grasp["wrist_flex"], 0, S.FETCH_ABOVE_SPOT_S * k)], settle=False, acc=held, early=BLEND_WIN)
            self._fblend([(2, grasp["lift"], 0, S.FETCH_SET_S * k)], acc=held)
            self._hold("release")
            self._fblend([(6, FETCH_OPEN, 0, S.FETCH_RELEASE_S)], settle=False)
            self._emit("put_back")
            self._wait(S.FETCH_AFTER_OPEN_S)
            self._fblend(self._three(via, S.FETCH_VIA_S * k), settle=False, early=BLEND_WIN)
            res, detail = "put_back", f"nobody took it in {wait_s:g} s ({checks} hand checks); put it back. {plan['note']}"
        self._emit("returning")
        self._fetch_back_to_rest(k)
        self._restore_base_torque()
        self._emit("done")
        return res, detail

    # The general building blocks: look, point_at, pick, place, hand_over (backend/arm/README.md).
    # They reuse the fetch_bottle legs, numbers and pace stretch; only the poses come from the
    # taught grips of the item's tier (backend/vision/table.py load_tiers).
    CARRY_OK = ("look", "place", "hand_over", "rest", "gesture")

    def _forget_held(self):
        self._held, self._held_tier, self._held_grip, self._carry = None, None, None, False

    def _carry_gate(self, skill):
        """Before anything moves: place and hand_over need something picked up, and while the arm
        carries an item only look, place, hand_over and rest run (a gesture is skipped)."""
        carrying = self._carry and self._held is not None
        if skill in ("place", "hand_over") and not carrying:
            return S.result("nothing_held", "I am not holding anything I picked up")
        if carrying and skill == "gesture":
            return S.result("done", f"gesture skipped while holding the {self._held['name']}", skipped=True)
        if carrying and skill not in self.CARRY_OK:
            return S.result("hands_full", f"I am holding the {self._held['name']}; place it or hand it over first")
        return None

    def _tiers(self):
        from backend.vision.table import load_tiers
        return load_tiers(self.setup_dir)     # read each time, so a grip taught a minute ago counts

    def _scene(self, name=None):
        try:
            s = self._call_vision(lambda: self.vision.scene(name=name, stop=self._stop))
        except _Stopped:
            raise
        except Exception as e:
            raise Refuse("not_found", f"the overhead look failed ({type(e).__name__}: {str(e)[:120]})")
        if s is None:
            s = self._wrist_scene(name)
        return s

    def _wrist_scene(self, name=None):
        """No overhead picture: look for ourselves. Lift the gripper to its look-down pose and
        take a picture straight ahead, then to the left and to the right if what was asked for
        is not found yet. Each picture is placed on the desk with the gripper camera's own map."""
        from backend.vision.table import wrist_map
        wm = wrist_map(self.setup_dir) if hasattr(self.vision, "wrist_scene") else None
        if wm is None:
            raise Refuse("no_picture", "the overhead camera gives no picture and the gripper camera has no desk map")
        pose, tpr = wm["pose"], self.table.tpr
        lo, hi = self.rng[1][0] + 30, self.rng[1][1] - 30
        self._used_wrist_scene = True
        merged = None
        for turn in (0.0, -0.6, 0.6):          # radians from straight ahead
            pan = int(max(lo, min(hi, self.table.pan_zero + turn * tpr)))
            here = self._rd(1, POSITION)
            dur = max(1.6, abs(pan - here) / 350.0)
            self._emit("looking")
            self._blend([(1, pan, 0, dur)] + self._three(pose, dur), settle=True)
            self._hold("picture")
            self._wait(0.35)
            at = self._rd(1, POSITION)
            try:
                s = self._call_vision(lambda: self.vision.wrist_scene(at, name=name))
            except _Stopped:
                raise
            except Exception:
                s = None
            if s is None:
                continue
            if merged is None:
                merged = s
            else:
                for it in s["items"]:
                    near = [m for m in merged["items"]
                            if math.hypot(m["x_cm"] - it["x_cm"], m["y_cm"] - it["y_cm"]) < 6.0]
                    if not near:
                        merged["items"].append(it)
                if s.get("match") and not merged.get("match"):
                    merged["match"] = s["match"]
                if s.get("person") and not merged.get("person"):
                    merged["person"] = s["person"]
            if name and merged.get("match"):
                break
        if merged is None:
            raise Refuse("no_picture", "neither the overhead camera nor the gripper camera gave a usable picture")
        return merged

    def _find_target(self, target):
        """(item, scene) for a spoken name, or for a desk position (the item nearest it)."""
        if not target:
            raise Refuse("not_found", "no target was given")
        if isinstance(target, dict):
            s = self._scene()
            item = desk.nearest(s["items"], target["x_cm"], target["y_cm"])
        else:
            s = self._scene(name=target)
            item = s.get("match")
        if item is None:
            raise Refuse("not_found", f"I cannot see {desk.label(target)} on the desk")
        return item, s

    def _skill_look(self, target):
        self._emit("start")
        try:
            s = self._scene(name=target if isinstance(target, str) else None)
        except Refuse as r:
            self._emit("done")
            return r.result, r.detail, r.extra
        from backend.vision.scene import public
        self._emit("seen")
        if getattr(self, "_used_wrist_scene", False):
            # the gripper went up to look for itself: fold back home before answering
            self._used_wrist_scene = False
            try:
                self._skill_rest()
            except _Stopped:
                raise
            except Exception:
                pass
        self._emit("done")
        items = [public(i) for i in s["items"]]
        names = ", ".join(i["name"] for i in items) or "nothing"
        extra = {"scene": items, "person": s.get("person")}
        detail = f"I can see {len(items)} things: {names}" + ("" if s.get("person") else "; no person seen")
        if isinstance(target, str):
            extra["match"] = public(s.get("match"))
            detail += f"; {target!r} is the {s['match']['name']}" if s.get("match") else f"; nothing fits {target!r}"
        return "done", detail, extra

    def _skill_point_at(self, target, pace):
        from backend.vision.scene import public
        k = S.fetch_stretch(pace)
        self._emit("start")
        try:
            if isinstance(target, dict):
                item = {"name": desk.label(target)[4:], **self.table.point(target["x_cm"], target["y_cm"])}
            else:
                item, _ = self._find_target(target)
            self._emit("seen")
            tiers = [t for t in self._tiers().values() if t.available]
            lo, hi = min(t.min_r for t in tiers), max(t.max_r for t in tiers)
            problem = desk.reach_problem(item, lo, hi, self._pan_range, what=f"the {item['name']}")
            if problem:
                raise Refuse("out_of_reach", problem)
        except Refuse as r:
            self._emit("done")
            return r.result, r.detail, r.extra
        via, _, _ = self._tiers()["tall"].posture(item["dist_cm"])
        pose = {1: int(item["pan"]), 2: via["lift"] - S.POINT_AT_RAISE, 3: via["elbow"], 4: via["wrist_flex"]}
        self._beep(1200, 300)
        t_out = S.FETCH_OUT_S * k
        acc = self._leg_acc(k)
        self._fblend([(3, pose[3], 0, t_out), (2, pose[2], LEAD * t_out, (1 - LEAD) * t_out), (1, pose[1], 0, t_out),
                      (4, pose[4], 0, t_out)], settle=False, acc=acc)
        self._emit("at_target")
        self._hold("point")
        self._wait(S.POINT_AT_HOLD_S)
        self._emit("returning")
        self._fetch_back_to_rest(k, acc)
        self._emit("done")
        return "done", f"pointed at the {item['name']}, {item['dist_cm']:.0f} cm away", {"target": public(item)}

    def _skill_pick(self, target, pace, approach, lift_cm_style):
        from backend.vision.scene import public
        k = S.fetch_stretch(pace)
        self._emit("start")
        try:
            item, s = self._find_target(target)
            self._emit("seen")
            plan = desk.pick_plan(item, self._tiers(), self._pan_range)
        except Refuse as r:
            self._emit("done")
            return r.result, r.detail, r.extra
        tier, via, grasp, pan, roll = plan["tier"], plan["via"], plan["grasp"], plan["pan"], plan["roll"]
        name = item["name"]
        note = (f"the {name} at {plan['dist_cm']:.1f} cm, pan {pan}, {tier.name} grip blended {plan['t']:.2f}"
                + (f", wrist roll {roll} for its angle {item['axis_deg']:.0f} deg" if roll is not None else ""))
        roll_home = self._rd(5, POSITION)
        self._raise_base_torque()
        self._beep(1200, 300)
        t_out = S.FETCH_OUT_S * k
        acc, held = self._leg_acc(k), self._leg_acc(k, loaded=True)
        out = [(3, via["elbow"], 0, t_out), (2, via["lift"], LEAD * t_out, (1 - LEAD) * t_out), (1, pan, 0, t_out),
               (4, via["wrist_flex"], 0, t_out), (6, tier.open, 0, min(S.FETCH_OPEN_S, t_out))]
        if roll is not None:
            out.append((5, roll, 0, t_out))
        aims = tier.name in desk.aim_tiers(self.aim_setting)
        # straight on down when nothing is looked at above the item: the hover flows into the approach
        self._fblend(out, settle=False, acc=acc, early=None if aims or approach == "look_first" else APPROACH_WIN)
        if approach == "look_first":
            self._hold("look_first")
            self._wait(S.FETCH_LOOK_FIRST_S)
        aim = []
        if aims:
            aim = self._aim_with_wrist(name, plan, k)
            via, grasp = plan["via"], plan["grasp"]      # moved in or out for a compact item
        opened = next((r["opened"] for r in aim if r.get("opened")), None)
        rolled = roll is not None or any(r.get("d_roll") for r in aim)
        if aim:
            note += ". " + desk.aim_words(aim)
        aim_extra = {"aim": aim} if aim else {}
        self._fblend(self._three(grasp, S.FETCH_APPROACH_S * k), acc=acc)
        self._emit("at_target")
        self._held_tier = tier
        # the thin item rule (jaws stopping near the taught pen grip count as empty) is only for long
        # thin items; a compact or wide one stops the jaws far out, and only nearly fully closed is a miss
        thin = not opened and not desk.is_compact(name, aim)
        miss_below = None if tier.closed is None or not thin else max(CLOSED + 15, tier.closed - 50)
        grip = self._close_gripper(speed=FETCH_CLOSE_SPEED, give_up_s=5.0, miss_below=miss_below)
        if grip is None:
            self._forget_held()
            self._emit("missed")
            self._hold("release")
            self._fblend([(6, tier.open, 0, S.FETCH_RELEASE_S)], settle=False)
            back = self._three(via, S.FETCH_VIA_S * k) + ([(5, roll_home, 0, S.FETCH_VIA_S * k)] if rolled else [])
            self._fblend(back, settle=False, acc=acc, early=BLEND_WIN)
            self._emit("returning")
            self._fetch_back_to_rest(k, acc)
            self._restore_base_torque()
            self._emit("done")
            return "missed", f"the gripper closed on nothing. {note}", aim_extra
        self._held_grip = grip
        self._emit("gripped")
        carry = tier.carry(lift_cm_style)
        self._fblend(self._three(carry, S.FETCH_TUCK_S * k), settle=False, acc=held)
        if not self._still_holding(grip):
            return self._lost_the_bottle(k, {"note": note}, "during the lift", what=f"the {name}")
        self._emit("lifted")
        self._held = {**public(item), "grip": grip, "rim_offset_cm": item.get("grip_at", {}).get("rim_offset_cm"),
                      "carry": carry, "roll_home": roll_home if rolled else None,
                      "person": s.get("person"), "seen_at": self._now()}
        self._carry = True
        self._emit("done")
        return "holding", f"holding the {name}, gripped at {grip}. {note}", {"held": public(item), **aim_extra}

    def _aim_with_wrist(self, name, plan, k):
        """From the hover pose, line the jaws up with the item using the wrist camera: up to
        WRIST_AIM["rounds"] looks; each one corrects the wrist roll (first), the pan, or for a
        compact item the reach (the hover moves along the taught blend; plan's dist_cm, via and
        grasp are updated), waits for the move, and looks again, until the item sits between the
        jaw tips. Then a wide item makes the jaws open wider (a last entry with "opened"). A picture
        that cannot be read ends the loop and pick carries on with the aim it has. Each look's
        picture is saved as setup/last_wrist_aim_N.jpg (and _N_read.jpg with the reading drawn).
        Returns the rounds, for the result."""
        W = desk.WRIST_AIM
        tier, dist_cm = plan["tier"], plan["dist_cm"]
        width = None
        stretch = 1.0 + 0.25 * (k - 1.0)          # pace slows these small moves only a little
        rounds = []
        for old in self.setup_dir.glob("last_wrist_aim_*.jpg"):
            try:
                old.unlink()
            except OSError:
                pass
        self.trace.append(("wrist_aim",))
        try:
            for n in range(1, W["rounds"] + 1):
                self._settle_for_picture(W["settle_min_s"], W["settle_s"])
                reading, why = self._wrist_aim_look(name, n)
                if reading is None:
                    rounds.append({"round": n, "unread": True, "why": why})
                    self._emit(f"aiming {n}: could not read the wrist picture ({why}); keeping the aim it has")
                    break
                roll_now, pan_now = self._rd(5, POSITION), self._rd(1, POSITION)
                step = desk.aim_step(reading, roll_now, pan_now, dist_cm, pan_range=self._pan_range,
                                     reach_range=(tier.min_r, tier.max_r))
                rec = {"round": n, "source": reading.get("source"), "angle_deg": reading.get("angle_deg"),
                       "offset_px": reading.get("offset_px"), "confidence": reading.get("confidence"),
                       **{key: step[key] for key in ("done", "d_roll", "d_pan", "capped", "why")}}
                if reading.get("compact"):
                    rec.update({key: reading.get(key) for key in ("compact", "centre_y", "width_px", "touches_bottom")},
                               d_reach=step["d_reach"])
                if reading.get("compact") and reading.get("width_px"):     # long items open as before
                    width = reading["width_px"]
                rounds.append(rec)
                if step["done"]:
                    self._emit(f"aiming {n}: {step['why']}")
                    break
                moves, words = {}, []
                if step["roll"] is not None:
                    moves[5] = step["roll"]
                    words.append(f"wrist roll {step['d_roll']:+d} to {step['roll']}")
                if step["pan"] is not None and step["d_pan"]:
                    moves[1] = step["pan"]
                    words.append(f"pan {step['d_pan']:+d} to {step['pan']}")
                if moves:                   # a small turn and the pan from the same picture go in one move
                    self._emit(f"aiming {n}: {step['why']}, " + ", ".join(words) + (" (capped)" if step["capped"] else ""))
                    self._aim_move(moves, desk.aim_move_s(step["d_roll"], step["d_pan"]) * stretch)
                elif step["reach"] is not None:
                    dist_cm = step["reach"]
                    via, grasp, t = tier.posture(dist_cm)
                    plan.update(dist_cm=dist_cm, via=via, grasp=grasp, t=t)
                    self._emit(f"aiming {n}: {step['why']}, reach {step['d_reach']:+.1f} cm to {dist_cm:.1f}"
                               + (" (capped)" if step["capped"] else ""))
                    self._aim_move({2: via["lift"], 3: via["elbow"], 4: via["wrist_flex"]},
                                   desk.aim_move_s(d_reach_cm=step["d_reach"]) * stretch)
                else:
                    rec["why"] += "; cannot turn further"
                    self._emit(f"aiming {n}: {rec['why']}")
                    break
            open_now = self._rd(6, POSITION)
            new_open = desk.jaw_opening(width, open_now)
            if new_open is not None:
                rounds.append({"round": len(rounds) + 1, "opened": new_open, "width_px": width, "done": True,
                               "why": f"{width:.0f} px wide: opening the jaws to {new_open}"})
                self._emit(f"aiming: {width:.0f} px wide, opening the jaws from {open_now} to {new_open}")
                self._fblend([(6, new_open, 0, max(W["min_move_s"], (new_open - open_now) / 800.0))], settle=False)
        finally:
            try:
                self.vision.close_wrist()
            except Exception:
                pass
        return rounds

    def _aim_move(self, goals, move_s):
        """One small correction (wrist roll, pan, or lift, elbow and flex for the reach), all its
        joints in one move over move_s seconds: each ramps up for a third, cruises for a third and
        ramps down for a third, so they start and finish together and a 40 tick pan is a slow, soft
        nudge rather than a twitch at the leg speed."""
        T = max(0.3, move_s)
        self._blend([(i, g, 0, 2 * T / 3) for i, g in goals.items()], False, tol=FETCH_TOL,
                    acc=desk.WRIST_AIM["move_acc_max"], poll=FETCH_POLL, window=MOVE_WINDOW, ramp=T / 3)

    def _settle_for_picture(self, min_s, max_s, joints=(1, 2, 3, 4, 5)):
        """Wait before a wrist picture: at least min_s, then until two reads 40 ms apart agree to a
        tick, never longer than max_s. A soft stop leaves little to wait for."""
        self._hold("aim")
        t0 = self._now()
        self._wait(min_s)
        last = {i: self._rd(i, POSITION) for i in joints}
        while self._now() - t0 < max_s:
            self._wait(0.04)
            now = {i: self._rd(i, POSITION) for i in joints}
            if all(abs(now[i] - last[i]) <= 1 for i in joints):
                return
            last = now

    def _wrist_aim_look(self, name, n):
        """(reading, None) or (None, why not)."""
        path = self.setup_dir / f"last_wrist_aim_{n}.jpg"
        try:
            opening = self._rd(6, POSITION)
            got = self._call_vision(lambda: self.vision.wrist_aim(name, save_to=path, opening=opening))
        except _Stopped:
            raise
        except Exception as e:
            return None, f"{type(e).__name__}: {str(e)[:80]}"
        if not isinstance(got, dict) or not got.get("picture"):
            return None, "no wrist picture"
        if got.get("aim") is None:
            return None, "neither OpenCV nor Gemini was sure"
        return got["aim"], None

    def _slipped(self, k, what):
        """The carried item is no longer in the jaws: go home and say so."""
        self._forget_held()
        self._emit("returning")
        self._fetch_back_to_rest(k)
        self._restore_base_torque()
        self._emit("done")
        return "nothing_held", f"the {what} slipped out of the jaws; went back to rest"

    def _turn_failed(self, k, e, tier, what):
        at = self._rd(1, POSITION)
        self._put_down_here(k, tier)
        self._emit("returning")
        self._fetch_back_to_rest(k)
        self._restore_base_torque()
        self._emit("done")
        return "turn_failed", f"could not finish the turn ({e}) at pan {at}; the {what} is on the table where the arm stopped"

    def _skill_place(self, where, pace, approach):
        k = S.fetch_stretch(pace)
        H, tier = self._held, self._held_tier
        name = H["name"]
        self._emit("start")
        if not self._still_holding(H["grip"]):
            return self._slipped(k, name)
        try:
            def look(n):
                s = self._scene(name=n)
                return s.get("match")
            x, y, words = desk.place_centre(where, H, look)
            spot = desk.jaws_for_centre(x, y, H, self.table.point)
            problem = desk.reach_problem(spot, tier.min_r, tier.max_r, self._pan_range, what="that spot")
            if problem:
                raise Refuse("out_of_reach", f"{problem}; still holding the {name}")
        except Refuse as r:
            self._emit("done")
            return r.result, r.detail, r.extra
        self._emit("seen")
        via, grasp, _ = tier.posture(spot["dist_cm"])
        pan = int(spot["pan"])
        turn = S.fetch_turn_s(self._rd(1, POSITION), pan, pace)
        acc, held = self._leg_acc(k), self._leg_acc(k, loaded=True)
        self._emit("carrying")
        # loaded: gentle ramps, and turn, down above the spot and (direct) down onto it flow into each other
        try:
            # loaded turn in the carry pose, as fetch's put back
            self._fblend([(1, pan, 0, turn)], settle=False, acc=held, early=BLEND_WIN)
        except RuntimeError as e:
            return self._turn_failed(k, e, tier, name)
        self._fblend([(2, grasp["lift"] - LIFT_UP, 0, S.FETCH_ABOVE_SPOT_S * k), (3, grasp["elbow"], 0, S.FETCH_ABOVE_SPOT_S * k),
                      (4, grasp["wrist_flex"], 0, S.FETCH_ABOVE_SPOT_S * k)], settle=False, acc=held,
                     early=None if approach == "look_first" else BLEND_WIN)
        if approach == "look_first":
            self._hold("look_first")
            self._wait(S.FETCH_LOOK_FIRST_S)
        self._fblend([(2, grasp["lift"], 0, S.FETCH_SET_S * k)], acc=held)
        self._emit("at_target")
        self._hold("release")
        self._fblend([(6, tier.open, 0, S.FETCH_RELEASE_S)], settle=False)
        roll_home = H.get("roll_home")
        self._forget_held()
        self._emit("put_back")
        self._wait(S.FETCH_AFTER_OPEN_S)
        clear = self._three(via, S.FETCH_VIA_S * k) + ([(5, roll_home, 0, S.FETCH_VIA_S * k)] if roll_home is not None else [])
        self._fblend(clear, settle=False, acc=acc, early=BLEND_WIN)
        self._emit("returning")
        self._fetch_back_to_rest(k, acc)
        self._restore_base_torque()
        self._emit("done")
        return "placed", f"put the {name} down {words}, {spot['dist_cm']:.0f} cm from the arm", \
            {"placed_at": {"x_cm": round(x, 1), "y_cm": round(y, 1)}}

    def _skill_hand_over(self, wait_s, pace):
        # hold it out long enough for a person to notice and reach, whoever is carrying it
        wait_s = max(float(wait_s or 0), 9.0)
        k = S.fetch_stretch(pace)
        H, tier = self._held, self._held_tier
        name, grip = H["name"], H["grip"]
        self._emit("start")
        if not self._still_holding(grip):
            return self._slipped(k, name)
        person, pnote = H.get("person"), "the person seen when it was picked up"
        if person is None or self._now() - H.get("seen_at", 0) > 20:
            try:
                person, pnote = self._scene().get("person"), "the person seen just now"
            except Refuse:
                person = None
        if person is not None:
            pan_p = person["pan"]
        else:
            pan_p, pnote = self.table.pan_for_bearing(1.2), "person not seen, went to the right"
        pan_p = int(max(PERSON_PAN[0], min(PERSON_PAN[1], pan_p)))
        self._emit("seen")
        far = self.taught.far_grasp()          # held out at full reach: fetch_bottle's offer pose
        turn = S.fetch_turn_s(self._rd(1, POSITION), pan_p, pace)
        acc, held = self._leg_acc(k), self._leg_acc(k, loaded=True)
        try:
            watcher = self.vision.hand_watcher()
        except Exception:
            watcher = None
        try:
            self._check_stop()
            self._emit("carrying")
            reach = S.FETCH_REACH_S * k
            try:
                self._fblend([(1, pan_p, 0, turn), (2, far["lift"] - 120, 0.5 * turn, reach),
                              (3, far["elbow"], 0.5 * turn, reach), (4, far["wrist_flex"], 0.5 * turn, reach)],
                             settle=False, acc=held)
            except RuntimeError as e:
                return self._turn_failed(k, e, tier, name)
            if not self._still_holding(grip):
                return self._lost_the_bottle(k, {"note": f"held out at pan {pan_p}"}, "while carrying it", what=f"the {name}")
            self._emit("offering")
            self._beep(900, 150)
            self._beep(900, 150)
            took = self._wait_for_take(wait_s, watcher)
        finally:
            if watcher is not None:
                watcher.stop()
        checks = getattr(watcher, "checks", 0) if watcher is not None else 0
        if took:
            self._hold("release")
            self._fblend([(6, tier.open, 0, S.FETCH_HANDOVER_OPEN_S)], settle=False)
            self._forget_held()
            self._emit("handed_over")
            self._wait(S.FETCH_HANDOVER_WAIT_S)
            self._fblend([(2, far["lift"] - 300, 0, S.FETCH_CLEAR_S * k)], settle=False, acc=acc, early=BLEND_WIN)
            self._emit("returning")
            self._fetch_back_to_rest(k, acc)
            self._restore_base_torque()
            self._emit("done")
            return "handed_over", f"handed over the {name} ({took}, {checks} hand checks) at pan {pan_p} ({pnote})"
        self._fblend(self._three(H["carry"], S.FETCH_TUCK_S * k), settle=False, acc=held)
        if not self._still_holding(grip):
            return self._slipped(k, name)
        self._carry = True
        self._emit("done")
        return "not_taken", (f"nobody took the {name} in {wait_s:g} s ({checks} hand checks); still holding it "
                             f"in the carry pose at pan {pan_p} ({pnote})")
