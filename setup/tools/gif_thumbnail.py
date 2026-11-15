"""gif_thumbnail: a lively 5 second move for a gif. Sways on the base, rolls the wrist both ways,
nods the wrist, bobs the elbow and snaps the jaws, then returns to the pose it started from.
Torque must be on. Usage: python gif_thumbnail.py"""
import json
import time
import winsound
import scservo_sdk as scs
cal = json.load(open(r"C:\Users\sudar\projects\third-hand\setup\my_follower_calibration_2026-11-14.json"))
RNG = {c["id"]: (c["range_min"], c["range_max"]) for c in cal.values()}
RNG[5] = (1100, 2960)   # keep the wrist roll inside what the camera cable allows
p, h = scs.PortHandler("COM3"), scs.PacketHandler(0)
p.openPort(); p.setBaudRate(1_000_000)
def rd(i, a):
    for _ in range(6):
        v, c, _ = h.read2ByteTxRx(p, i, a)
        if c == scs.COMM_SUCCESS: return v
    raise RuntimeError("read failed")
def go(i, goal, speed, acc=12):
    goal = max(RNG[i][0] + 10, min(RNG[i][1] - 10, goal))
    for _ in range(6):
        if h.write1ByteTxRx(p, i, 41, acc)[0] == scs.COMM_SUCCESS: break
    for a, v in ((46, speed), (42, goal)):
        for _ in range(6):
            if h.write2ByteTxRx(p, i, a, v)[0] == scs.COMM_SUCCESS: break
s = {i: rd(i, 56) for i in (1, 3, 4, 5, 6)}
SHUT, OPEN = 1400, 1900
steps = [  # time s, joint, goal, speed
    (0.0, 1, s[1] + 250, 520), (0.0, 5, s[5] + 500, 950), (0.0, 4, s[4] - 150, 380), (0.0, 3, s[3] - 120, 320), (0.1, 6, SHUT, 900),
    (0.6, 6, OPEN, 900),
    (1.0, 1, s[1] - 250, 620), (1.0, 5, s[5] - 500, 1150), (1.0, 4, s[4] + 80, 380), (1.0, 3, s[3], 320), (1.3, 6, SHUT, 900),
    (1.8, 6, OPEN, 900),
    (2.0, 1, s[1] + 180, 560), (2.0, 5, s[5] + 400, 1050), (2.0, 4, s[4] - 120, 380), (2.0, 3, s[3] - 100, 320), (2.4, 6, SHUT, 900),
    (2.8, 6, OPEN, 900),
    (3.0, 1, s[1] - 180, 520), (3.0, 5, s[5] - 400, 1050), (3.0, 4, s[4] + 60, 380), (3.0, 3, s[3], 320), (3.4, 6, SHUT, 900),
    (3.8, 6, OPEN, 900),
    (4.0, 1, s[1], 400), (4.0, 5, s[5], 900), (4.0, 4, s[4], 300), (4.0, 3, s[3], 300), (4.5, 6, s[6], 600),
]
winsound.Beep(1200, 300)
t0 = time.time()
for ts, i, g, sp in steps:
    while time.time() - t0 < ts:
        time.sleep(0.01)
    go(i, g, sp)
time.sleep(1.5)
print("back at", {i: rd(i, 56) for i in s}, "started", s)
p.closePort()
