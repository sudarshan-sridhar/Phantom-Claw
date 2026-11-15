"""A pretend SO-101 servo bus and pretend cameras, for running RealArm with no hardware.

SimBus stands in for both scservo_sdk's PortHandler and PacketHandler. Each servo moves toward
its goal at the speed and acceleration in its registers, on a pretend clock that only moves when
the arm sleeps, so a 40 second fetch runs in well under a second. It keeps every write, so tests
can check goals and torque. Things that happen in the world:
- grip_ok: the bottle cap is between the jaws, so closing stops at CAP (else they close fully)
- block: a joint cannot pass these limits (a turn that cannot finish)
- at(t, fn): run fn when the pretend clock reaches t (a tug on the bottle)

FakeVision answers like backend.vision.Vision from a pretend desk.
"""
import math

import scservo_sdk as scs

CAP = 1478                 # where the jaws stop on the cap (the taught grips stopped at 1474 to 1483)
CALIBRATED = {1: (734, 3360), 2: (931, 3162), 3: (899, 3194), 4: (905, 3189), 5: (0, 4095), 6: (1305, 2788)}
REST_POS = {1: 2009, 2: 946, 3: 3190, 4: 2632, 5: 2030, 6: 1310}


class VClock:
    def __init__(self, t=1000.0):
        self.t = t

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += max(0.0, s)


class _Servo:
    def __init__(self, pos):
        self.pos = float(pos)
        self.vel = 0.0
        self.goal = int(pos)
        self.speed = 0
        self.acc = 0
        self.torque_en = 0
        self.torque_limit = 1000
        self.block = None
        self.ext = 0.0


class SimBus:
    def __init__(self, clock, start=None, grip_ok=True):
        self.clock = clock
        self.s = {i: _Servo(p) for i, p in (start or REST_POS).items()}
        self.last = clock.time()
        self.writes = []          # (t, id, address, value)
        self.grip_ok = grip_ok
        self.holding = False
        self.events = []
        self.current = 60         # raw current reading (x 6.5 mA)
        self.bus_s = 0.0025       # each read or write takes about this long on the real USB adapter

    # PortHandler
    def openPort(self):
        return True

    def setBaudRate(self, baud):
        return True

    def closePort(self):
        pass

    # world
    def at(self, t, fn):
        self.events.append((t, fn))

    def _advance(self):
        now = self.clock.time()
        while self.last < now:
            dt = min(0.005, now - self.last)
            self.last += dt
            for t, fn in list(self.events):
                if self.last >= t:
                    self.events.remove((t, fn))
                    fn()
            for i, sv in self.s.items():
                if not sv.torque_en:
                    continue
                vmax = sv.speed if sv.speed > 0 else 3000.0
                a = sv.acc * 100.0 if sv.acc > 0 else 1e5
                e = sv.goal - sv.pos
                want = math.copysign(min(vmax, math.sqrt(2 * a * abs(e))), e) if abs(e) > 0.5 else 0.0
                dv = max(-a * dt, min(a * dt, want - sv.vel))
                sv.vel += dv
                new = sv.pos + sv.vel * dt
                if sv.block is not None:
                    lo, hi = sv.block
                    if new < lo or new > hi:
                        new, sv.vel = max(lo, min(hi, new)), 0.0
                if i == 6:
                    # the cap is only between the jaws when the arm is out at the bottle and they close onto it
                    if self.grip_ok and sv.pos >= CAP and new < CAP and self.s[2].pos > 1400:
                        new, sv.vel, self.holding = CAP, 0.0, True
                    if self.holding and new > CAP + 150:        # jaws open: the bottle is out of them
                        self.holding, self.grip_ok = False, False
                        for s2 in self.s.values():
                            s2.ext = 0.0
                sv.pos = new

    # PacketHandler
    def read1ByteTxRx(self, port, i, a):
        self.clock.t += self.bus_s
        self._advance()
        sv = self.s[i]
        return {40: sv.torque_en, 41: sv.acc}.get(a, 0), scs.COMM_SUCCESS, 0

    def read2ByteTxRx(self, port, i, a):
        self.clock.t += self.bus_s
        self._advance()
        sv = self.s[i]
        if a == 56:
            return int(round(sv.pos + sv.ext)), scs.COMM_SUCCESS, 0
        if a == 42:
            return sv.goal, scs.COMM_SUCCESS, 0
        if a == 69:
            return self.current, scs.COMM_SUCCESS, 0
        return {46: sv.speed, 48: sv.torque_limit}.get(a, 0), scs.COMM_SUCCESS, 0

    def write1ByteTxRx(self, port, i, a, v):
        self.clock.t += self.bus_s
        self._advance()
        self.writes.append((self.clock.time(), i, a, v))
        sv = self.s[i]
        if a == 40:
            sv.torque_en = v
        elif a == 41:
            sv.acc = v
        return scs.COMM_SUCCESS, 0

    def write2ByteTxRx(self, port, i, a, v):
        self.clock.t += self.bus_s
        self._advance()
        self.writes.append((self.clock.time(), i, a, v))
        sv = self.s[i]
        if a == 42:
            sv.goal = int(v)
        elif a == 46:
            sv.speed = int(v)
        elif a == 48:
            sv.torque_limit = int(v)
        return scs.COMM_SUCCESS, 0

    # checks
    def goals(self):
        return [(t, i, v) for t, i, a, v in self.writes if a == 42]

    def bad_goals(self):
        return [(i, v) for _, i, v in self.goals() if not (CALIBRATED[i][0] <= v <= CALIBRATED[i][1])]

    def torque_off_writes(self):
        return [w for w in self.writes if w[2] == 40 and w[3] == 0]

    def pos(self, i):
        self._advance()
        return int(round(self.s[i].pos))


class FakeWatcher:
    def __init__(self, clock, hand_after):
        self.clock, self.hand_after = clock, hand_after
        self._armed_at = None
        self.checks = 0
        self.stopped = False

    @property
    def armed(self):
        return self._armed_at is not None

    @armed.setter
    def armed(self, v):
        if v and self._armed_at is None:
            self._armed_at = self.clock.time()

    @property
    def hand(self):
        if self._armed_at is None or self.hand_after is None:
            return False
        self.checks = int((self.clock.time() - self._armed_at) / 0.8) + 1
        return self.clock.time() - self._armed_at >= self.hand_after

    def stop(self):
        self.stopped = True


class FakeVision:
    """A pretend desk. Overhead: the bottle at r_cm and bottle_pan, the person at person_pan.
    Wrist: things are seen when they are inside the camera's field of view from the base's
    present angle, at the column the real geometry would put them."""

    def __init__(self, clock, bus=None, overhead=True, r_cm=30.0, bottle_pan=1990, person_pan=2620,
                 hand_after=None, wrist_picture=True, wrist_bottle_pan=2150, wrist_person_pan=None, tpr=651.9):
        self.clock, self.bus = clock, bus
        self.overhead, self.r_cm, self.bottle_pan, self.person_pan = overhead, r_cm, bottle_pan, person_pan
        self.hand_after, self.wrist_picture = hand_after, wrist_picture
        self.wrist_bottle_pan, self.wrist_person_pan, self.tpr = wrist_bottle_pan, wrist_person_pan, tpr
        self.wrist_asks = []
        self.watchers = []
        self.closed = 0

    def look_overhead(self, stop=None):
        if not self.overhead:
            return None
        out = {"boxes": {}}
        if self.r_cm is not None:
            out["bottle"] = {"r_cm": self.r_cm, "bearing_deg": 0.0, "pan": self.bottle_pan}
        if self.person_pan is not None:
            out["person"] = {"bearing_deg": 0.0, "pan": self.person_pan}
        return out

    def _box(self, target_pan, here):
        from backend.vision import WRIST
        if target_pan is None:
            return None
        off = (target_pan - here) / self.tpr                      # radians the thing is off the camera's axis
        x = WRIST["centre_x"] + WRIST["pan_sign"] * off / math.radians(WRIST["hfov_deg"]) * 1000
        if not 30 < x < 970:
            return None
        return [380.0, x - 40, 720.0, x + 40]

    def wrist_look(self, pan=None):
        here = self.bus.pos(1) if self.bus is not None else pan
        self.wrist_asks.append(here)
        if not self.wrist_picture:
            return {"picture": False, "bottle": None, "person": None}
        return {"picture": True, "bottle": self._box(self.wrist_bottle_pan, here),
                "person": self._box(self.wrist_person_pan, here)}

    def close_wrist(self):
        self.closed += 1

    def hand_watcher(self):
        w = FakeWatcher(self.clock, self.hand_after)
        self.watchers.append(w)
        return w

    # the general skills: a pretend desk of any items, as Vision.scene() would describe it
    scene_items = None       # list of {"name", "x_cm", "y_cm", "tier", "graspable", "axis_deg", "width_cm"}
    scene_asks = None

    def scene(self, name=None, stop=None):
        from backend.vision.scene import grip_point, match_name
        from backend.vision.table import TableMap
        if self.scene_asks is None:
            self.scene_asks = []
        self.scene_asks.append(name)
        if not self.overhead:
            return None
        tm = TableMap()
        items = []
        for e in self.scene_items or []:
            it = {"name": e["name"], "box": None, "tier": e.get("tier", "tall"), "tier_from": "name",
                  "graspable": e.get("graspable", True), "axis_deg": e.get("axis_deg"), "width_cm": e.get("width_cm")}
            it.update(tm.point(e["x_cm"], e["y_cm"]))
            it["grip_at"] = grip_point(it, tm)
            items.append(it)
        person = None
        if self.person_pan is not None:
            person = {"box": None, "x_cm": 40.0, "y_cm": 20.0, "bearing": 60.0, "pan": self.person_pan}
        return {"items": items, "person": person, "match": match_name(items, name) if name else None}


def make_real(grip_ok=True, start=None, quiet=True, setup_dir=None, scene_items=None, **vision):
    """A RealArm on a SimBus with FakeVision. Returns (arm, bus, vision, clock)."""
    from backend.arm.real import RealArm
    clock = VClock()
    bus = SimBus(clock, start=start, grip_ok=grip_ok)
    vis = FakeVision(clock, bus, **vision)
    vis.scene_items = scene_items
    arm = RealArm(port_handler=bus, packet_handler=bus, vision=vis, clock=clock, quiet=quiet, setup_dir=setup_dir)
    return arm, bus, vis, clock
