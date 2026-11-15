"""A short wave for a photo or gif: sway left and right, nod the wrist, snap the jaws.
Starts and ends at the current pose. About 5.5 s. Torque must be on."""
import time
import winsound
import scservo_sdk as scs
p, h = scs.PortHandler("COM3"), scs.PacketHandler(0)
p.openPort(); p.setBaudRate(1_000_000)
def rd(i, a):
    for _ in range(6):
        v, c, _ = h.read2ByteTxRx(p, i, a)
        if c == scs.COMM_SUCCESS: return v
    raise RuntimeError("read failed")
def go(i, goal, speed, acc=8):
    for _ in range(6):
        if h.write1ByteTxRx(p, i, 41, acc)[0] == scs.COMM_SUCCESS: break
    for a, v in ((46, speed), (42, goal)):
        for _ in range(6):
            if h.write2ByteTxRx(p, i, a, v)[0] == scs.COMM_SUCCESS: break
s = {i: rd(i, 56) for i in (1, 4, 6)}
steps = [  # time s, joint, goal, speed
    (0.0, 1, s[1] + 170, 260), (0.0, 4, s[4] - 110, 200), (0.2, 6, 1400, 700),
    (0.9, 6, 1900, 700),
    (1.2, 1, s[1] - 170, 300), (1.2, 4, s[4] + 60, 160), (1.6, 6, 1400, 700),
    (2.3, 6, 1900, 700),
    (2.6, 1, s[1] + 110, 300), (2.6, 4, s[4] - 80, 180), (3.0, 6, 1400, 700),
    (3.6, 6, 1900, 700),
    (3.9, 1, s[1], 220), (3.9, 4, s[4], 160), (4.4, 6, s[6], 500),
]
winsound.Beep(1200, 300)
t0 = time.time()
for ts, i, g, sp in steps:
    while time.time() - t0 < ts:
        time.sleep(0.01)
    go(i, g, sp)
time.sleep(1.4)
print("back at", {i: rd(i, 56) for i in (1, 4, 6)}, "started", s)
p.closePort()
