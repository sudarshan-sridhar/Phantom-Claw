"""Fetch the bottle from wherever it stands and bring it toward the person.

1. Overhead camera: find the bottle and the person (Gemini gives boxes), turn them into table
   positions with the marker mapping.
2. Turn to face the bottle, pick the reach by blending the taught middle and far grips for that
   distance, grip the cap, lift.
3. Carry it round to face the person and hold it out at full taught reach.
4. Let go when the person tugs the bottle. If nobody takes it, put it back where it was.
5. Return to the initial position.

Usage: python fetch_bottle.py [--look]     (--look only prints what the camera found, no motion)
"""
import base64
import json
import sys
import threading
import time
import urllib.request
import winsound
import cv2
import numpy as np
import scservo_sdk as scs

LOOK_ONLY = "--look" in sys.argv
D = r"C:\Users\sudar\projects\third-hand\setup"
cal = json.load(open(D + r"\my_follower_calibration_2026-11-14.json"))
tm = json.load(open(D + r"\table_map.json"))
bt = json.load(open(D + r"\bottle_teach.json"))
RNG = {c["id"]: (c["range_min"], c["range_max"]) for c in cal.values()}
H = np.array(tm["homography_px_to_cm"])
BASE = np.array(tm["base_cm"])
PAN_ZERO, TPR = tm["pan_zero"], tm["ticks_per_rad"]
JAW_OFFSET = 85          # the camera bearing reads this many ticks right of the pan that grips (from the 3 teachings)
REST = {1: 2009, 2: RNG[2][0] + 15, 3: RNG[3][1] - 4, 4: 2632}
OPEN, CLOSED = 1790, 1310
LIFT_UP = 170            # shoulder ticks to raise the bottle off the table
ACC, LEAD = 7, 0.3
TORQUE = {1: 1000, 2: 1000, 3: 800, 4: 500, 5: 400, 6: 300}   # base raised from 500: it has to swing a loaded arm
CUR_MAX = {1: 1300, 2: 1600, 3: 1400, 4: 800, 6: 700}


def env_key(name):
    for line in open(r"C:\Users\sudar\projects\third-hand\.env", encoding="utf-8-sig"):
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(name + " missing")


def overhead():
    for _ in range(4):
        cap = cv2.VideoCapture(3, cv2.CAP_DSHOW)
        cap.set(3, 1920); cap.set(4, 1080)
        f = None
        for _ in range(10):
            ok, fr = cap.read()
            if ok:
                f = fr
        cap.release()
        if f is not None and f.mean() > 30:
            return f
        time.sleep(1)
    raise RuntimeError("overhead camera gives no picture")


def ask_gemini(img):
    ok, jpg = cv2.imencode(".jpg", cv2.resize(img, (960, 540)), [cv2.IMWRITE_JPEG_QUALITY, 85])
    prompt = ("This is a photo of a desk from above. Find (1) the clear plastic water bottle standing on the desk and "
              "(2) the person sitting at the edge of the picture (body, arm, hand or lap; ignore the robot arm). "
              'Reply with JSON only: {"bottle": [ymin, xmin, ymax, xmax] or null, "person": [ymin, xmin, ymax, xmax] or null} '
              "with each box in 0 to 1000 units of the image height and width.")
    body = {"contents": [{"parts": [{"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(jpg.tobytes()).decode()}},
                                    {"text": prompt}]}], "generationConfig": {"responseMimeType": "application/json"}}
    last = None
    for model in ("gemini-3.5-flash-lite", "gemini-flash-lite-latest", "gemini-3.5-flash-lite"):
        try:
            req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                                         headers={"x-goog-api-key": env_key("GEMINI_API_KEY"), "Content-Type": "application/json"},
                                         data=json.dumps(body).encode())
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.loads(json.loads(r.read())["candidates"][0]["content"]["parts"][0]["text"])
        except Exception as e:
            last = e
    raise RuntimeError(f"Gemini did not answer: {last}")


def to_table(x_px, y_px):
    return cv2.perspectiveTransform(np.array([[[x_px, y_px]]], dtype=np.float64), H)[0, 0]


def _box(v):
    """Gemini sometimes wraps the box in another list or a dict. Return four numbers or None."""
    while isinstance(v, dict):
        v = v.get("box_2d") or v.get("box") or next(iter(v.values()), None)
    while isinstance(v, list) and len(v) == 1 and isinstance(v[0], (list, dict)):
        v = v[0]
        while isinstance(v, dict):
            v = v.get("box_2d") or v.get("box") or next(iter(v.values()), None)
    if isinstance(v, list) and len(v) == 4 and all(isinstance(x, (int, float)) for x in v):
        return [float(x) for x in v]
    return None


def locate(img):
    h, w = img.shape[:2]
    sx, sy = tm["image_size"][0] / 1000.0, tm["image_size"][1] / 1000.0
    boxes = {}
    for _ in range(3):          # ask again if the answer is not two usable boxes
        raw = ask_gemini(img)
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            raw = raw[0]
        if isinstance(raw, dict):
            boxes = {"bottle": _box(raw.get("bottle")), "person": _box(raw.get("person"))}
            if boxes["bottle"]:
                break
    out = {"boxes": boxes}
    b = boxes.get("bottle")
    if b:
        ymin, xmin, ymax, xmax = b
        base_px = ((xmin + xmax) / 2 * sx, (ymax - 0.07 * (ymax - ymin)) * sy)   # where the bottle stands
        t = to_table(*base_px) - BASE
        out["bottle"] = {"r_cm": float(np.hypot(*t)), "bearing_deg": float(np.degrees(np.arctan2(t[0], t[1]))),
                         "pan": float(PAN_ZERO + TPR * np.arctan2(t[0], t[1]) - JAW_OFFSET)}
    p = boxes.get("person")
    if p:
        ymin, xmin, ymax, xmax = p
        t = to_table((xmin + xmax) / 2 * sx, (ymin + ymax) / 2 * sy) - BASE
        out["person"] = {"bearing_deg": float(np.degrees(np.arctan2(t[0], t[1]))),
                         "pan": float(PAN_ZERO + TPR * np.arctan2(t[0], t[1]))}
    return out


def posture(r):
    """Blend the taught middle and far grips for this distance."""
    m, f = bt["mid"], bt["far"]
    t = float(np.clip((r - m["r_cm"]) / (f["r_cm"] - m["r_cm"]), 0.0, 1.0))
    mix = lambda a, b: {k: round(a[k] + t * (b[k] - a[k])) for k in ("lift", "elbow", "wrist_flex")}
    return mix(m["via"], f["via"]), mix(m["grasp"], f["grasp"]), t


_img = [a for a in sys.argv[1:] if a.lower().endswith(".jpg")]     # a saved photo, for checking the camera maths
seen = locate(cv2.imread(_img[0]) if _img else overhead())
print("camera:", json.dumps({k: v for k, v in seen.items() if k != "boxes"}))
if "bottle" not in seen:
    raise SystemExit("I cannot see the bottle on the table.")
B = seen["bottle"]
m_r, f_r = bt["mid"]["r_cm"], bt["far"]["r_cm"]
if not (m_r - 4 <= B["r_cm"] <= f_r + 2):
    raise SystemExit(f"The bottle is {B['r_cm']:.0f} cm from the arm. I can only reach it between {m_r - 4:.0f} and {f_r + 2:.0f} cm.")
via, grasp, t_mix = posture(B["r_cm"])
far = bt["far"]["grasp"]
pan_b = int(np.clip(B["pan"], RNG[1][0] + 30, RNG[1][1] - 30))
pan_p = int(np.clip(seen["person"]["pan"] if "person" in seen else PAN_ZERO + TPR * 1.2, 950, 3250))
print(f"plan: bottle at {B['r_cm']:.1f} cm, turn to pan {pan_b}, blend {t_mix:.2f} between middle and far, grasp {grasp}; then hold it out at pan {pan_p}"
      + ("" if "person" in seen else " (person not seen, going to the right)"))
if LOOK_ONLY:
    raise SystemExit(0)

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
    return max(RNG[i][0], min(RNG[i][1], int(g)))


def torque_on():
    start = {i: rd(i, 56) for i in range(1, 7)}
    bad = [i for i in (1, 2, 3, 4, 6) if not (RNG[i][0] - 25 <= start[i] <= RNG[i][1] + 25)]
    if bad:
        raise RuntimeError(f"joints {bad} read out of range: {start}")
    off = [i for i in (1, 2, 3, 4) if abs(start[i] - REST[i]) > 200]
    if off:
        raise RuntimeError(f"the arm is not in the initial position (joints {off}); fold it back first")
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
            if i in CUR_MAX and (rd(i, 69) & 0x7FFF) * 6.5 > CUR_MAX[i]:
                raise RuntimeError(f"ID {i} drew too much current")
        if any(abs(now[i] - pos[i]) > 2 for i in now):
            last_move = time.time()
        pos = now
        done = len(sent) == len(items)
        if done and all(abs(now[i] - g) <= 12 for i, g, _, _ in items):
            return
        still = time.time() - last_move
        if done and t > end and still > 0.3 and all(abs(now[i] - g) <= 110 for i, g, _, _ in items):   # carrying a bottle, the elbow settles further short
            if settle:
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
    wr(6, 46, 500)
    wr(6, 42, CLOSED)
    last, t_move, t0 = rd(6, 56), time.time(), time.time()
    while time.time() - t_move < 0.35 and time.time() - t0 < 5:
        time.sleep(0.03)
        pos = rd(6, 56)
        if abs(pos - last) > 2:
            last, t_move = pos, time.time()
    pos = rd(6, 56)
    if pos - CLOSED > 100:
        wr(6, 42, pos - 70)
        return pos
    return None


def back_to_rest(dur=4.5, close=True):
    items = [(3, REST[3], 0, dur), (2, REST[2], 0, (1 - LEAD) * dur), (1, REST[1], 0, dur), (4, REST[4], 0, dur)]
    if close:
        items.append((6, CLOSED, 0.6 * dur, 0.4 * dur))
    blend(items, settle=False)


def wait_for_tug(seconds):
    """True when someone pulls on the bottle: the arm or the jaws get moved off where they were holding."""
    time.sleep(0.6)
    ref = {i: rd(i, 56) for i in (1, 2, 3, 4, 6)}
    t0 = time.time()
    while time.time() - t0 < seconds:
        time.sleep(0.04)
        now = {i: rd(i, 56) for i in ref}
        if any(abs(now[i] - ref[i]) > (14 if i != 6 else 9) for i in ref):
            return True
    return False


def watch_for_hand(state):
    """Runs beside the arm: once armed, asks Gemini whether a hand is in the wrist camera's view."""
    cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    cap.set(3, 1280); cap.set(4, 720)
    key = env_key("GEMINI_API_KEY")
    prompt = ('This is the view from a camera on a robot gripper that is holding a bottle out to a person. '
              'Is a human hand or fingers visible, reaching for the bottle or close to it? Reply with JSON only: {"hand": true} or {"hand": false}.')
    while not state["done"]:
        f = None
        for _ in range(2):
            ok, fr = cap.read()
            if ok:
                f = fr
        if f is None or not state["armed"]:
            time.sleep(0.02)
            continue
        f = cv2.rotate(f, cv2.ROTATE_180)
        ok, jpg = cv2.imencode(".jpg", cv2.resize(f, (640, 360)), [cv2.IMWRITE_JPEG_QUALITY, 80])
        cv2.imwrite(D + r"\last_handover_view.jpg", cv2.resize(f, (640, 360)))
        body = {"contents": [{"parts": [{"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(jpg.tobytes()).decode()}},
                                        {"text": prompt}]}], "generationConfig": {"responseMimeType": "application/json"}}
        try:
            req = urllib.request.Request("https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent",
                                         headers={"x-goog-api-key": key, "Content-Type": "application/json"}, data=json.dumps(body).encode())
            with urllib.request.urlopen(req, timeout=4) as r:
                ans = json.loads(json.loads(r.read())["candidates"][0]["content"]["parts"][0]["text"])
            state["checks"] += 1
            if isinstance(ans, dict) and ans.get("hand") is True:
                state["hand"] = True
                break
        except Exception:
            pass
    cap.release()


def wait_for_take(seconds, state):
    """Returns 'tug' when the bottle is pulled, 'hand' when the wrist camera sees a hand, or None when time is up."""
    time.sleep(0.35)
    ref = {i: rd(i, 56) for i in (1, 2, 3, 4, 6)}
    state["armed"] = True
    t0 = time.time()
    while time.time() - t0 < seconds:
        time.sleep(0.03)
        now = {i: rd(i, 56) for i in ref}
        if any(abs(now[i] - ref[i]) > (14 if i != 6 else 9) for i in ref):
            return "tug"
        if state["hand"]:
            return "hand"
    return None


class Film(threading.Thread):
    """Records the overhead camera to a video file while the arm works."""
    def __init__(self, path):
        super().__init__(daemon=True)
        self.path, self.stop_now = path, False

    def run(self):
        cap = cv2.VideoCapture(3, cv2.CAP_DSHOW)
        cap.set(3, 1920); cap.set(4, 1080)
        vw = None
        while not self.stop_now:
            ok, f = cap.read()
            if not ok:
                continue
            if vw is None:
                vw = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (f.shape[1], f.shape[0]))
            vw.write(f)
        cap.release()
        if vw is not None:
            vw.release()


def put_down_and_go_home(lift_for_table, note):
    """Lower the bottle onto the table where the arm is, let go, lift clear, fold back."""
    print(note)
    blend([(2, lift_for_table, 0, 1.6)])
    blend([(6, OPEN, 0, 0.6)], settle=False)
    time.sleep(0.2)
    blend([(2, lift_for_table - 220, 0, 1.2)], settle=False)
    back_to_rest()


film = Film(D + r"\fetch_run.mp4")
try:
    torque_on()
    film.start()
    time.sleep(1.0)
    winsound.Beep(1200, 300)
    t0 = time.time()
    T = 3.3
    # out to just above the cap in one move, then the short final approach
    blend([(3, via["elbow"], 0, T), (2, via["lift"], LEAD * T, (1 - LEAD) * T), (1, pan_b, 0, T),
           (4, via["wrist_flex"], 0, T), (6, OPEN, 0, 1.2)], settle=False)
    blend([(2, grasp["lift"], 0, 1.0), (3, grasp["elbow"], 0, 1.0), (4, grasp["wrist_flex"], 0, 1.0)])
    grip = close_gripper()
    if grip is None:
        blend([(6, OPEN, 0, 0.6)], settle=False)
        blend([(2, via["lift"], 0, 1.2), (3, via["elbow"], 0, 1.2), (4, via["wrist_flex"], 0, 1.2)], settle=False)
        back_to_rest()
        raise SystemExit("MISSED: the gripper closed on nothing. Back at the initial position.")
    print(f"gripped at {grip} after {time.time() - t0:.1f} s")
    # lift and pull the bottle in close in one move: the base joint cannot swing a loaded arm held far out
    tuck = {"lift": bt["mid"]["grasp"]["lift"] - 230, "elbow": bt["mid"]["grasp"]["elbow"], "wrist_flex": bt["mid"]["grasp"]["wrist_flex"]}   # the taught middle grip, raised: known to clear the table
    blend([(2, tuck["lift"], 0, 1.9), (3, tuck["elbow"], 0, 1.9), (4, tuck["wrist_flex"], 0, 1.9)], settle=False)
    table_lift_tucked = bt["mid"]["grasp"]["lift"]        # shoulder value that stands the bottle on the table at this tuck
    turn = max(1.8, abs(pan_p - pan_b) / 400)
    hand_state = {"done": False, "armed": False, "hand": False, "checks": 0}
    threading.Thread(target=watch_for_hand, args=(hand_state,), daemon=True).start()
    try:
        # turn toward the person and start reaching out during the last part of the turn
        blend([(1, pan_p, 0, turn), (2, far["lift"] - 120, 0.5 * turn, 1.5), (3, far["elbow"], 0.5 * turn, 1.5),
               (4, far["wrist_flex"], 0.5 * turn, 1.5)], settle=False)
    except RuntimeError as e:
        # could not finish the turn (too heavy, or something in the way): stand the bottle down here
        for i in (1, 2, 3, 4):
            wr(i, 42, clamp(i, rd(i, 56)))
        blend([(2, tuck["lift"], 0, 1.5), (3, tuck["elbow"], 0, 1.5), (4, tuck["wrist_flex"], 0, 1.5)], settle=False)
        blend([(3, bt["mid"]["grasp"]["elbow"], 0, 1.0), (4, bt["mid"]["grasp"]["wrist_flex"], 0, 1.0)], settle=False)
        put_down_and_go_home(table_lift_tucked, f"could not finish the turn ({e}) at pan {rd(1, 56)}; putting the bottle down here")
        raise SystemExit("TURN FAILED: the bottle is on the table where the arm stopped. Back at the initial position.")
    winsound.Beep(900, 150); winsound.Beep(900, 150)                 # ready: pull it out of the jaws
    took = wait_for_take(3.0, hand_state)
    hand_state["done"] = True
    if took:
        blend([(6, OPEN, 0, 0.4)], settle=False)
        print(f"handed over ({took}); hand checks made: {hand_state['checks']}")
        time.sleep(0.8)
        blend([(2, far["lift"] - 300, 0, 1.0)], settle=False)        # lift clear of the hand
        back_to_rest()
    else:
        print(f"nobody took it in 3 s (hand checks made: {hand_state['checks']}); putting it back")
        blend([(2, tuck["lift"], 0, 1.9), (3, tuck["elbow"], 0, 1.9), (4, tuck["wrist_flex"], 0, 1.9)], settle=False)
        blend([(1, pan_b, 0, turn)], settle=False)
        blend([(2, grasp["lift"] - LIFT_UP, 0, 1.6), (3, grasp["elbow"], 0, 1.6), (4, grasp["wrist_flex"], 0, 1.6)], settle=False)
        blend([(2, grasp["lift"], 0, 1.2)])
        blend([(6, OPEN, 0, 0.6)], settle=False)
        time.sleep(0.2)
        blend([(2, via["lift"], 0, 1.2), (3, via["elbow"], 0, 1.2), (4, via["wrist_flex"], 0, 1.2)], settle=False)
        back_to_rest()
    print(f"DONE in {time.time() - t0:.0f} s, holding at the initial position")
except SystemExit:
    raise
except Exception as e:
    print("STOPPED:", e)
    for i in (1, 2, 3, 4):
        try:
            wr(i, 42, clamp(i, rd(i, 56)))
        except Exception:
            pass
    print("holding where it stopped")
finally:
    film.stop_now = True
    if film.is_alive():
        film.join(timeout=5)
    p.closePort()



