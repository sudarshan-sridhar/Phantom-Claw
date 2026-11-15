"""Smooth moves between the rest pose and the taught pose. Torque stays on at the end.
Each phase sends one goal per joint and lets the servo run its own speed ramp (no stepping),
with joint speeds scaled so they arrive together.
Usage: python move.py rest pose      (any sequence of: rest, pose)
"""
import json
import sys
import time
import scservo_sdk as scs

D = r"C:\Users\sudar\projects\third-hand\setup"
cal = json.load(open(D + r"\my_follower_calibration_2026-11-14.json"))
taught = json.load(open(D + r"\pose_point_pen.json"))
RNG = {c["id"]: (c["range_min"], c["range_max"]) for c in cal.values()}
SPEED, ACC = 500, 12          # ticks/s at the fastest joint, acceleration (x100 ticks/s^2)
CUR_MAX = {1: 800, 2: 1600, 3: 1400, 4: 800}
POSE = {1: taught["shoulder_pan"], 2: taught["shoulder_lift"] - 45, 3: taught["elbow_flex"], 4: taught["wrist_flex"]}
REST = {1: 2007, 2: RNG[2][0] + 15, 3: RNG[3][1] - 4, 4: 2632}

p, h = scs.PortHandler("COM3"), scs.PacketHandler(0)
p.openPort()
p.setBaudRate(1_000_000)


def rd(i, a, n=2):
    for _ in range(6):
        v, c, _ = (h.read1ByteTxRx if n == 1 else h.read2ByteTxRx)(p, i, a)
        if c == scs.COMM_SUCCESS:
            return v
    raise RuntimeError(f"read {i}/{a} failed")


def wr(i, a, v, n=2):
    for _ in range(6):
        c, _ = (h.write1ByteTxRx if n == 1 else h.write2ByteTxRx)(p, i, a, v)
        if c == scs.COMM_SUCCESS:
            return
    raise RuntimeError(f"write {i}/{a} failed")


peak = {}


def phase(goals):
    pos = {i: rd(i, 56) for i in goals}
    goals = {i: max(RNG[i][0], min(RNG[i][1], g)) for i, g in goals.items()}
    dist = {i: abs(goals[i] - pos[i]) for i in goals}
    far = max(dist.values()) or 1
    for i in goals:
        wr(i, 41, ACC, 1)
        wr(i, 46, max(40, round(SPEED * dist[i] / far)))
    for i in goals:
        wr(i, 42, goals[i])
    t0 = last_move = time.time()
    while True:
        time.sleep(0.05)
        now = {i: rd(i, 56) for i in goals}
        for i in goals:
            ma = (rd(i, 69) & 0x7FFF) * 6.5
            peak[i] = max(peak.get(i, 0), ma)
            if ma > CUR_MAX[i]:
                raise RuntimeError(f"ID {i} current {ma:.0f} mA too high")
        if any(abs(now[i] - pos[i]) > 3 for i in goals):
            last_move = time.time()
        pos = now
        if all(abs(now[i] - goals[i]) <= 14 for i in goals):
            return
        if time.time() - last_move > 0.8:
            raise RuntimeError(f"stalled at {now}, goals {goals}")
        if time.time() - t0 > 15:
            raise RuntimeError("phase took too long")


def sched(items):
    """items: (id, goal, start_s, duration_s). All joints run in one blended motion."""
    items = [(i, max(RNG[i][0], min(RNG[i][1], g)), ts, dur) for i, g, ts, dur in items]
    pos = {i: rd(i, 56) for i, *_ in items}
    for i, g, ts, dur in items:
        wr(i, 41, ACC, 1)
        wr(i, 46, max(12, round(abs(g - pos[i]) / dur)))
    sent = set()
    t0 = last_move = time.time()
    end = max(ts + dur for _, _, ts, dur in items)
    while True:
        t = time.time() - t0
        for i, g, ts, dur in items:
            if i not in sent and t >= ts:
                wr(i, 42, g)
                sent.add(i)
        time.sleep(0.04)
        now = {i: rd(i, 56) for i, *_ in items}
        for i in now:
            ma = (rd(i, 69) & 0x7FFF) * 6.5
            peak[i] = max(peak.get(i, 0), ma)
            if ma > CUR_MAX[i]:
                raise RuntimeError(f"ID {i} current {ma:.0f} mA too high")
        if any(abs(now[i] - pos[i]) > 2 for i in now):
            last_move = time.time()
        pos = now
        if len(sent) == len(items) and all(abs(now[i] - g) <= 14 for i, g, _, _ in items):
            return
        still = time.time() - last_move
        if still > 0.4 and t > end and all(abs(now[i] - g) <= 45 for i, g, _, _ in items):
            # settled a little short under its own weight: push the goal past by the shortfall, once
            for i, g, _, _ in items:
                if abs(g - now[i]) > 6:
                    wr(i, 42, max(RNG[i][0], min(RNG[i][1], g + (g - now[i]))))
            time.sleep(0.5)
            return
        if still > 1.2 and len(sent) == len(items):
            raise RuntimeError(f"stalled at {now}")
        if t > end + 6:
            raise RuntimeError("move took too long")


LEAD = 0.3  # the elbow opens this fraction of the way before the shoulder starts down


def to_pose():
    # elbow runs the whole time; the shoulder joins after the elbow has a head start, both end together
    sched([(3, POSE[3], 0, T), (2, POSE[2], LEAD * T, (1 - LEAD) * T), (1, POSE[1], 0, T), (4, POSE[4], 0, T)])


def to_rest():
    # the same path backwards: both start together, the shoulder gets back early, the elbow finishes the fold
    sched([(3, REST[3], 0, T), (2, REST[2], 0, (1 - LEAD) * T), (1, REST[1], 0, T), (4, REST[4], 0, T)])


try:
    assert all(rd(i, 40, 1) == 1 for i in (1, 2, 3, 4)), "torque is off; start from a held pose"
    T = float(sys.argv[1])  # seconds per leg
    for step in sys.argv[2:]:
        t = time.time()
        if step.startswith("p:"):   # p:pan,lift,elbow,wflex  -> go there along the same blended path
            v = [int(x) for x in step[2:].split(",")]
            POSE.update({1: v[0], 2: v[1], 3: v[2], 4: v[3]})
            step = "pose"
        (to_rest if step == "rest" else to_pose)()
        time.sleep(0.6)
        print(f"{step}: {time.time() - t:.1f} s, at", {i: rd(i, 56) for i in (1, 2, 3, 4)})
    print("peak mA:", {k: round(v) for k, v in peak.items()}, "- HOLDING")
except Exception as e:
    print("STOPPED:", e)
    for i in (1, 2, 3, 4):
        try:
            wr(i, 42, max(RNG[i][0], min(RNG[i][1], rd(i, 56))))
        except Exception:
            pass
    print("holding where it stopped")
finally:
    p.closePort()

