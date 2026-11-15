"""The "grip the bottle" action (also called the bottle lift motion).
From the initial position: go to the bottle, grip it by the cap, lift it, hold it in the air,
put it back in the same place, let go, and return to the initial position.
Usage: python grip_bottle.py [hold_seconds]      (default 5)

One process, one continuous blended move each way (no stop at a midpoint, no pauses between
steps). If the gripper closes on nothing it opens, returns to the initial position and says so.
"""
import json
import sys
import time
import winsound
import scservo_sdk as scs

HOLD = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
D = r"C:\Users\sudar\projects\third-hand\setup"
cal = json.load(open(D + r"\my_follower_calibration_2026-11-14.json"))
P = json.load(open(D + r"\bottle_pick.json"))
RNG = {c["id"]: (c["range_min"], c["range_max"]) for c in cal.values()}
G = P["grasp"]
GRASP = {1: G["pan"], 2: G["lift"], 3: G["elbow"], 4: G["wrist_flex"]}
REST = {1: 2009, 2: RNG[2][0] + 15, 3: RNG[3][1] - 4, 4: 2632}
OPEN, CLOSED, LIFT = P["gripper_open"], 1310, P["lift"]["lift"]
T_OUT, T_BACK, LEAD = 4.2, 4.2, 0.3     # seconds each way; the elbow leads the shoulder by this fraction
ACC = 7                                  # gentle ramps (x100 ticks/s^2)
TORQUE = {1: 500, 2: 1000, 3: 800, 4: 500, 5: 400, 6: 300}
CUR_MAX = {1: 800, 2: 1600, 3: 1400, 4: 800, 6: 700}

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


def clamp(i, g):
    return max(RNG[i][0], min(RNG[i][1], g))


def torque_on():
    start = {i: rd(i, 56) for i in range(1, 7)}
    bad = [i for i in (1, 2, 3, 4, 6) if not (RNG[i][0] - 25 <= start[i] <= RNG[i][1] + 25)]
    if bad:
        raise RuntimeError(f"joints {bad} read out of range: {start}")
    for i in range(1, 7):
        wr(i, 42, clamp(i, start[i]))
        wr(i, 48, TORQUE[i])
    for i in range(1, 7):
        wr(i, 40, 1, 1)


def blend(items, settle=True):
    """items: (id, goal, start_s, duration_s). One motion; each joint runs its own smooth ramp."""
    items = [(i, clamp(i, g), ts, dur) for i, g, ts, dur in items]
    pos = {i: rd(i, 56) for i, *_ in items}
    for i, g, ts, dur in items:
        wr(i, 41, ACC, 1)
        wr(i, 46, max(12, round(abs(g - pos[i]) / dur)))
    sent, t0, last_move = set(), time.time(), time.time()
    end = max(ts + dur for _, _, ts, dur in items)
    while True:
        t = time.time() - t0
        for i, g, ts, dur in items:
            if i not in sent and t >= ts:
                wr(i, 42, g)
                sent.add(i)
        time.sleep(0.03)
        now = {i: rd(i, 56) for i, *_ in items}
        for i in now:
            if (rd(i, 69) & 0x7FFF) * 6.5 > CUR_MAX[i]:
                raise RuntimeError(f"ID {i} drew too much current")
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
                        wr(i, 46, 60)
                        wr(i, 42, clamp(i, g + (g - now[i])))
                time.sleep(0.4)
            return
        if done and still > 1.2:
            raise RuntimeError(f"stalled at {now}")
        if t > end + 6:
            raise RuntimeError("move took too long")


def close_gripper():
    wr(6, 41, 10, 1)
    wr(6, 46, 300)
    wr(6, 42, CLOSED)
    last, t_move = rd(6, 56), time.time()
    while time.time() - t_move < 0.35:
        time.sleep(0.03)
        pos = rd(6, 56)
        if abs(pos - last) > 2:
            last, t_move = pos, time.time()
    pos = rd(6, 56)
    if pos - CLOSED > 100:
        wr(6, 42, pos - 70)   # keep a light squeeze, the same force as before
        return pos
    return None


ok = False
try:
    if not all(rd(i, 40, 1) == 1 for i in (1, 2, 3, 4)):
        torque_on()
    winsound.Beep(1200, 300)
    t0 = time.time()
    blend([(3, GRASP[3], 0, T_OUT), (2, GRASP[2], LEAD * T_OUT, (1 - LEAD) * T_OUT),
           (1, GRASP[1], 0, T_OUT), (4, GRASP[4], 0, T_OUT), (6, OPEN, 0, 1.2)])
    at = {i: rd(i, 56) for i in (1, 2, 3, 4)}
    grip = close_gripper()
    if grip is None:
        blend([(6, OPEN, 0, 0.6)], settle=False)
        print("MISSED: the gripper closed on nothing")
    else:
        print(f"at bottle in {time.time() - t0:.1f} s {at}, gripped at {grip}")
        blend([(2, LIFT, 0, 1.0)])
        print("lifted, holding", HOLD, "s; gripper", rd(6, 56))
        time.sleep(HOLD)
        blend([(2, GRASP[2], 0, 1.2)])
        blend([(6, OPEN, 0, 0.6)], settle=False)
        time.sleep(0.2)
        ok = True
    t1 = time.time()
    blend([(3, REST[3], 0, T_BACK), (2, REST[2], 0, (1 - LEAD) * T_BACK), (1, REST[1], 0, T_BACK),
           (4, REST[4], 0, T_BACK), (6, CLOSED, 0.6 * T_BACK, 0.4 * T_BACK)], settle=False)
    print(f"back at the initial position in {time.time() - t1:.1f} s", {i: rd(i, 56) for i in range(1, 7)})
    print("DONE" if ok else "DONE (no bottle)", f"- total {time.time() - t0:.0f} s, holding")
except Exception as e:
    print("STOPPED:", e)
    for i in (1, 2, 3, 4):
        try:
            wr(i, 42, clamp(i, rd(i, 56)))
        except Exception:
            pass
    print("holding where it stopped")
finally:
    p.closePort()
