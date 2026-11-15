"""Drive the arm slowly to the pose taught by hand (pointing near the pen) and HOLD it.
Refuses to start if any joint reads outside its calibrated range.
Order: elbow and wrist first while the shoulder stays back (arm high), then pan, then the
shoulder lowers the arm onto the target and stops a little short so it hovers.
Usage: python point_pen.py OUT_DIR
"""
import json
import sys
import time
import cv2
import scservo_sdk as scs

out = sys.argv[1]
cal = json.load(open(r"C:\Users\sudar\projects\third-hand\setup\my_follower_calibration_2026-11-14.json"))
target = json.load(open(r"C:\Users\sudar\projects\third-hand\setup\pose_point_pen.json"))
ID = {n: c["id"] for n, c in cal.items()}
RNG = {c["id"]: (c["range_min"], c["range_max"]) for c in cal.values()}
STEP, LAG_MAX = 5, 150
TORQUE = {1: 500, 2: 1000, 3: 800, 4: 500, 5: 400, 6: 300}
CUR_MAX = {1: 800, 2: 1600, 3: 1400, 4: 800, 5: 600, 6: 600}  # mA
HOVER = 45  # shoulder stops this many ticks short (raw lower = up)

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


def off():
    for _ in range(3):
        for i in range(1, 7):
            h.write1ByteTxRx(p, i, 40, 0)


peak = {}


def go(goals):
    """Move several joints together in small steps, checking each one every step."""
    while any(goal[i] != t for i, t in goals.items()):
        for i, t in goals.items():
            if goal[i] != t:
                goal[i] += max(-STEP, min(STEP, t - goal[i]))
                wr(i, 42, goal[i])
        time.sleep(0.03)
        for i in goals:
            pos = rd(i, 56)
            ma = (rd(i, 69) & 0x7FFF) * 6.5
            peak[i] = max(peak.get(i, 0), ma)
            if abs(pos - goal[i]) > LAG_MAX:
                raise RuntimeError(f"ID {i} fell behind (goal {goal[i]}, at {pos}), blocked")
            if ma > CUR_MAX[i]:
                raise RuntimeError(f"ID {i} current {ma:.0f} mA too high")


started = False
try:
    start = {i: rd(i, 56) for i in range(1, 7)}
    print("start:", start)
    bad = [i for i in (1, 2, 3, 4) if not (RNG[i][0] - 15 <= start[i] <= RNG[i][1] + 15)]
    if bad:
        raise RuntimeError(f"joints {bad} read outside their calibrated range; not powering the arm")
    goal = {i: max(RNG[i][0], min(RNG[i][1], start[i])) for i in range(1, 7)}  # a joint resting a hair past its limit starts at the limit
    for i in range(1, 7):
        wr(i, 42, goal[i])
        assert rd(i, 42) == goal[i], f"goal verify failed on {i}"
        wr(i, 41, 20, 1)
        wr(i, 46, 300)
        wr(i, 48, TORQUE[i])
    for i in range(1, 7):
        wr(i, 40, 1, 1)
    started = True
    time.sleep(0.5)
    clamp = lambda i, v: max(RNG[i][0] + 20, min(RNG[i][1] - 20, v))
    go({3: clamp(3, target["elbow_flex"]), 4: clamp(4, target["wrist_flex"])})
    go({1: clamp(1, target["shoulder_pan"])})
    go({2: clamp(2, target["shoulder_lift"] - HOVER)})
    time.sleep(1.0)
    now = {i: rd(i, 56) for i in range(1, 7)}
    print("now:  ", now)
    print("peak mA:", {k: round(v) for k, v in peak.items()})
    print("HOLDING with torque on")
except Exception as e:
    print("STOPPED:", e)
    if started:
        print("holding where it stopped (torque left on so the arm does not drop)")
finally:
    p.closePort()

cap = cv2.VideoCapture(2)
cap.set(3, 1280)
cap.set(4, 720)
f = None
for _ in range(12):
    ok, fr = cap.read()
    if ok:
        f = fr
cap.release()
if f is not None:
    cv2.imwrite(out + "/after_point.jpg", cv2.resize(cv2.rotate(f, cv2.ROTATE_180), (640, 360)))
