"""Step the arm by fixed joint deltas, look for the pen after each step, stop on a condition.
Usage: python servo.py FRAME.jpg N  d1 d2 d3 d4  COND
  d1..d4: ticks added per step to pan, lift, elbow, wrist flex
  COND:   "cy>150", "cx<330", "seen", "none" (python expression on cx, cy, ang, length, width, seen)
Torque must be on. Stops at once if a joint is pushed off its goal (contact) or draws too much current.
"""
import json
import sys
import time
import cv2
import scservo_sdk as scs
from penfind import find_pen

cal = json.load(open(r"C:\Users\sudar\projects\third-hand\setup\my_follower_calibration_2026-11-14.json"))
RNG = {c["id"]: (c["range_min"], c["range_max"]) for c in cal.values()}
frame_path, N = sys.argv[1], int(sys.argv[2])
delta = dict(zip((1, 2, 3, 4), map(int, sys.argv[3:7])))
cond = sys.argv[7]
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


cap = cv2.VideoCapture(2)
cap.set(3, 1280)
cap.set(4, 720)


def look():
    f = None
    for _ in range(6):   # flush old frames
        ok, fr = cap.read()
        if ok:
            f = fr
    f = cv2.rotate(f, cv2.ROTATE_180)
    cv2.imwrite(frame_path, cv2.resize(f, (640, 360)))
    return find_pen(f)


def check(r):
    seen = r is not None
    cx, cy, ang, length, width = r if seen else (0, 0, 0, 0, 0)
    return eval(cond)


try:
    goal = {i: rd(i, 56) for i in (1, 2, 3, 4)}   # where the arm actually is (includes any sag)
    r = look()
    print("start", goal, "pen", None if r is None else [round(v) for v in r])
    for k in range(N):
        if check(r):
            print("condition met")
            break
        for i in (1, 2, 3, 4):
            if delta[i]:
                goal[i] = max(RNG[i][0] + 5, min(RNG[i][1] - 5, goal[i] + delta[i]))
                wr(i, 41, 10, 1)
                wr(i, 46, 150)
                wr(i, 42, goal[i])
        time.sleep(0.7)
        pos = {i: rd(i, 56) for i in (1, 2, 3, 4)}
        ma = {i: round((rd(i, 69) & 0x7FFF) * 6.5) for i in (2, 3)}
        r = look()
        print(k + 1, pos, "mA", ma, "pen", None if r is None else [round(v) for v in r])
        off = [i for i in (1, 2, 3, 4) if delta[i] and abs(pos[i] - goal[i]) > 45]
        if off or ma[2] > 1200 or ma[3] > 1000:
            for i in (1, 2, 3, 4):
                wr(i, 42, pos[i])
            print("STOPPED: contact or load on", off, ma)
            break
        goal = {i: (pos[i] if delta[i] else goal[i]) for i in goal}   # step from where it really is
finally:
    cap.release()
    p.closePort()
