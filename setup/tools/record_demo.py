"""Record a hand-guided demonstration: all six joint positions at about 50 Hz with torque off,
plus an upright wrist camera frame twice a second.
Beeps: high = start moving, double = 7 s left, low = stop.
Usage: python record_demo.py NAME SECONDS
"""
import json
import os
import sys
import threading
import time
import winsound
import cv2
import scservo_sdk as scs

name, secs = sys.argv[1], float(sys.argv[2])
D = r"C:\Users\sudar\projects\third-hand\setup"
fdir = os.path.join(D, f"{name}_frames")
os.makedirs(fdir, exist_ok=True)
p, h = scs.PortHandler("COM3"), scs.PacketHandler(0)
p.openPort()
p.setBaudRate(1_000_000)
for _ in range(3):
    for i in range(1, 7):
        h.write1ByteTxRx(p, i, 40, 0)   # torque off on every joint

cap = cv2.VideoCapture(2)
cap.set(3, 1280)
cap.set(4, 720)
cap.read()
stop = False
frames = []


def camera(t0):
    nxt = 0.0
    while not stop:
        ok, f = cap.read()
        t = time.time() - t0
        if ok and t >= nxt:
            fn = f"{t:05.1f}.jpg"
            cv2.imwrite(os.path.join(fdir, fn), cv2.resize(cv2.rotate(f, cv2.ROTATE_180), (640, 360)))
            frames.append([round(t, 2), fn])
            nxt = t + 0.5


def read_all():
    out = []
    for i in range(1, 7):
        v, c, _ = h.read2ByteTxRx(p, i, 56)
        if c != scs.COMM_SUCCESS:
            return None
        out.append(v)
    return out


winsound.Beep(1200, 700)
t0 = time.time()
th = threading.Thread(target=camera, args=(t0,))
th.start()
threading.Timer(secs - 7, lambda: (winsound.Beep(900, 200), winsound.Beep(900, 200))).start()
samples = []
while time.time() - t0 < secs:
    r = read_all()
    if r:
        samples.append([round(time.time() - t0, 3)] + r)
    time.sleep(0.012)
stop = True
th.join()
winsound.Beep(500, 900)
cap.release()
p.closePort()
json.dump({"joints": ["t", "pan", "lift", "elbow", "wrist_flex", "wrist_roll", "gripper"], "samples": samples, "frames": frames},
          open(os.path.join(D, f"{name}.json"), "w"))
lo = [min(s[k] for s in samples) for k in range(1, 7)]
hi = [max(s[k] for s in samples) for k in range(1, 7)]
print(f"{len(samples)} samples ({len(samples) / secs:.0f} Hz), {len(frames)} frames")
print("min", lo)
print("max", hi)
print("end", samples[-1][1:])
