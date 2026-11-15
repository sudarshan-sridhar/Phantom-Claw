"""Move single joints to raw targets with the servo's own speed ramp, one after another, then save
an upright wrist camera frame. Torque must already be on.
Usage: python jog.py FRAME.jpg 6:2300@400 5:3057@250 ...     (id:goal@speed, speed in ticks/s)
A move that stalls or draws too much current is stopped where it is and the rest are skipped.
"""
import json
import sys
import time
import cv2
import scservo_sdk as scs

cal = json.load(open(r"C:\Users\sudar\projects\third-hand\setup\my_follower_calibration_2026-11-14.json"))
RNG = {c["id"]: (c["range_min"], c["range_max"]) for c in cal.values()}
CUR_MAX = {1: 800, 2: 1600, 3: 1400, 4: 800, 5: 500, 6: 700}
frame_path = sys.argv[1]
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


try:
    for arg in sys.argv[2:]:
        i, rest = arg.split(":")
        g, sp = rest.split("@")
        i, g, sp = int(i), int(g), int(sp)
        g = max(RNG[i][0], min(RNG[i][1], g))
        wr(i, 41, 10, 1)
        wr(i, 46, sp)
        wr(i, 42, g)
        last, t_move, peak, t0 = rd(i, 56), time.time(), 0, time.time()
        why = "arrived"
        while True:
            time.sleep(0.04)
            pos = rd(i, 56)
            ma = (rd(i, 69) & 0x7FFF) * 6.5
            peak = max(peak, ma)
            if abs(pos - last) > 2:
                t_move, last = time.time(), pos
            if ma > CUR_MAX[i]:
                wr(i, 42, pos)
                why = f"STOPPED, {ma:.0f} mA"
                break
            if time.time() - t_move > (1.0 if abs(pos - g) > 90 else 0.5):
                if abs(pos - g) > 30 and i == 6 and g < pos:
                    wr(i, 42, max(RNG[6][0], pos - 70))  # closing on something: keep a light squeeze
                    why = f"GRIPPED at {pos} ({pos - RNG[6][0]} ticks from fully closed)"
                elif 8 < abs(pos - g) <= 90 and i in (2, 3):
                    # stopped short under the arm's weight (no integral gain): push past by the shortfall
                    wr(i, 42, max(RNG[i][0], min(RNG[i][1], g + (g - pos))))
                    time.sleep(0.7)
                elif abs(pos - g) > 90:
                    wr(i, 42, pos)
                    why = f"BLOCKED {g - pos:+d} ticks short"
                break
            if time.time() - t0 > 12:
                why = "timeout"
                break
        print(f"ID {i} -> {g}: at {rd(i, 56)}, peak {peak:.0f} mA, {why}")
        if why != "arrived" and not why.startswith("GRIPPED"):
            break
finally:
    print("all:", {i: rd(i, 56) for i in range(1, 7)})
    p.closePort()

cap = cv2.VideoCapture(2)
cap.set(3, 1280)
cap.set(4, 720)
f = None
for _ in range(10):
    ok, fr = cap.read()
    if ok:
        f = fr
cap.release()
if f is not None:
    cv2.imwrite(frame_path, cv2.resize(cv2.rotate(f, cv2.ROTATE_180), (640, 360)))


