"""Camera sweep: smooth moves through a list of (pan, wrist_flex) views, recording upright video
the whole time and saving a still at each view. Returns to the first position at the end.
Torque must be on. Usage: python sweep.py OUT_DIR NAME pan:wflex pan:wflex ...
"""
import json
import sys
import threading
import time
import cv2
import scservo_sdk as scs

out, name = sys.argv[1], sys.argv[2]
views = [tuple(int(x) for x in a.split(":")) for a in sys.argv[3:]]
cal = json.load(open(r"C:\Users\sudar\projects\third-hand\setup\my_follower_calibration_2026-11-14.json"))
RNG = {c["id"]: (c["range_min"], c["range_max"]) for c in cal.values()}
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
ok, f = cap.read()
hh, ww = f.shape[:2]
vw = cv2.VideoWriter(f"{out}/{name}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20, (ww, hh))
latest, stop = [None], False


def camera():
    while not stop:
        ok, fr = cap.read()
        if ok:
            fr = cv2.rotate(fr, cv2.ROTATE_180)
            vw.write(fr)
            latest[0] = fr


th = threading.Thread(target=camera)
th.start()


def goto(pan, wf, speed=220):
    goals = {1: max(RNG[1][0], min(RNG[1][1], pan)), 4: max(RNG[4][0], min(RNG[4][1], wf))}
    for i, g in goals.items():
        wr(i, 41, 8, 1)
        wr(i, 46, speed)
        wr(i, 42, g)
    t0, last, t_move = time.time(), {i: rd(i, 56) for i in goals}, time.time()
    while time.time() - t0 < 12:
        time.sleep(0.05)
        now = {i: rd(i, 56) for i in goals}
        if any(abs(now[i] - last[i]) > 2 for i in goals):
            t_move, last = time.time(), now
        if all(abs(now[i] - goals[i]) <= 12 for i in goals) or time.time() - t_move > 0.8:
            break
    time.sleep(0.6)
    return {i: rd(i, 56) for i in goals}


try:
    home = (rd(1, 56), rd(4, 56))
    for k, (pan, wf) in enumerate(views):
        at = goto(pan, wf)
        cv2.imwrite(f"{out}/{name}_{k}.jpg", cv2.resize(latest[0], (640, 360)))
        print(f"view {k}: pan {at[1]} wrist {at[4]}")
    goto(*home)
    print("back at", {1: rd(1, 56), 4: rd(4, 56)})
finally:
    stop = True
    th.join()
    vw.release()
    cap.release()
    p.closePort()
