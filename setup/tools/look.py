"""Gentle look-around: move only the listed joints by small offsets, record the wrist camera,
return to the start pose and turn torque off. Other joints are never powered.
Usage: python look.py OUT_DIR NAME 4:-150 1:+200 ...   (id:delta_ticks, run in order)
"""
import sys
import time
import cv2
import scservo_sdk as scs

out, name = sys.argv[1], sys.argv[2]
moves = [(int(a.split(":")[0]), int(a.split(":")[1])) for a in sys.argv[3:]]
ids = sorted({i for i, _ in moves})
STEP, LAG_MAX, CUR_MAX = 6, 120, 700  # ticks per frame, ticks, mA

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


cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
ok, fr = cap.read()
assert ok, "wrist camera gave no frame"
hh, ww = fr.shape[:2]
vw = cv2.VideoWriter(f"{out}/{name}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20, (ww, hh))
shots = 0


def frame(save=False):
    global shots
    ok, f = cap.read()
    if ok:
        f = cv2.rotate(f, cv2.ROTATE_180)  # wrist camera is mounted upside down
        vw.write(f)
        if save:
            cv2.imwrite(f"{out}/{name}_{shots}.jpg", cv2.resize(f, (640, int(640 * hh / ww))))
            shots += 1


def hold(sec, save=False):
    t = time.time()
    while time.time() - t < sec:
        frame()
    if save:
        frame(True)


def go(i, target, peak):
    cur = goal[i]
    while cur != target:
        cur += max(-STEP, min(STEP, target - cur))
        wr(i, 42, cur)
        goal[i] = cur
        frame()
        pos = rd(i, 56)
        ma = (rd(i, 69) & 0x7FFF) * 6.5
        peak[i] = max(peak.get(i, 0), ma)
        if abs(pos - cur) > LAG_MAX:
            raise RuntimeError(f"ID {i} fell behind (goal {cur}, at {pos}), blocked")
        if ma > CUR_MAX:
            raise RuntimeError(f"ID {i} current {ma:.0f} mA too high")


try:
    start = {i: rd(i, 56) for i in range(1, 7)}
    print("start positions:", start)
    lim = {i: (rd(i, 9), rd(i, 11)) for i in ids}
    goal = {}
    for i in ids:
        wr(i, 42, start[i])
        assert rd(i, 42) == start[i], f"goal verify failed on {i}"
        wr(i, 41, 20, 1)   # gentle acceleration
        wr(i, 46, 250)     # speed cap
        wr(i, 48, 450)     # torque limit, under half
        goal[i] = start[i]
    for i in ids:
        wr(i, 40, 1, 1)
    hold(1.0, True)
    peak = {}
    for i, d in moves:
        tgt = max(lim[i][0] + 30, min(lim[i][1] - 30, goal[i] + d))
        print(f"ID {i}: {goal[i]} -> {tgt}")
        go(i, tgt, peak)
        hold(1.2, True)
    for i in reversed(ids):
        go(i, start[i], peak)
    hold(1.0, True)
    print("back at start:", {i: rd(i, 56) for i in ids}, "peak mA:", {k: round(v) for k, v in peak.items()})
except Exception as e:
    print("STOPPED:", e)
finally:
    off()
    print("torque off")
    vw.release()
    cap.release()
    p.closePort()
