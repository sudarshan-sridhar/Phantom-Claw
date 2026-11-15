"""The two cameras: the overhead phone (index 3) and the wrist camera (index 2).

Both may be missing. The overhead phone runs out of battery, so every grab returns None when
there is no picture instead of raising.
"""
import threading
import time
from pathlib import Path

from . import gemini

SETUP = Path(__file__).resolve().parents[2] / "setup"
OVERHEAD_INDEX, WRIST_INDEX = 3, 2
BLACK_MEAN = 30          # a frame darker than this on average is the phone's black screen
WHITE_SHARE = 0.6        # more than this share of near-white pixels is iVCam's "Waiting for connection" card
                         # (real desk frames have none: mean about 105, white share 0.00)


def _open(index, w, h):
    import cv2
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    return cap


def is_placeholder(f):
    """True for the white card the iVCam driver shows on index 3 while the phone is off or asleep."""
    import cv2
    g = cv2.cvtColor(cv2.resize(f, (320, 180)), cv2.COLOR_BGR2GRAY)
    return float((g > 235).mean()) > WHITE_SHARE


def grab_overhead(index=OVERHEAD_INDEX, tries=4, pause=1.0, stop=None):
    """One good overhead frame, or None. The phone sometimes sends black frames for a moment
    after the stream opens, so read ten and try again (up to four times) if the last is black.
    With the phone off, the iVCam driver still answers on index 3 with a white "waiting" card;
    that counts as no picture too."""
    for n in range(tries):
        # 1280x720: what the phone really sends (the table map and every saved photo are this size).
        # Asking for 1920x1080 as the script did makes each read take a full second on the iVCam driver.
        cap = _open(index, 1280, 720)
        f = None
        try:
            if cap.isOpened():
                for _ in range(10):
                    ok, fr = cap.read()
                    if ok:
                        f = fr
        finally:
            cap.release()
        if f is not None and f.mean() > BLACK_MEAN:
            if is_placeholder(f):
                return None       # the phone is off: asking again only wastes seconds
            return f
        if stop is not None and stop.is_set():
            return None
        if n < tries - 1:
            time.sleep(pause)
    return None


class WristCamera:
    """The camera on the gripper. It is mounted upside down, so frames are turned 180 degrees."""

    def __init__(self, index=WRIST_INDEX):
        self.index = index
        self.cap = None

    def frame(self, flush=4):
        """One upright frame, or None. Reads a few first so the picture is current, not buffered."""
        import cv2
        if self.cap is None:
            self.cap = _open(self.index, 1280, 720)
        if not self.cap.isOpened():
            return None
        f = None
        for _ in range(flush):
            ok, fr = self.cap.read()
            if ok:
                f = fr
        if f is None or f.mean() < 5:
            return None
        return cv2.rotate(f, cv2.ROTATE_180)

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


HAND_PROMPT = ('This is the view from a camera on a robot gripper that is holding a bottle out to a person. '
               'Is a human hand or fingers visible, reaching for the bottle or close to it? Reply with JSON only: '
               '{"hand": true} or {"hand": false}.')


# Seeing a hand without asking anyone. The arm stands still while it holds the item out, so the
# wrist picture only changes when something comes into it. A hand shows up as skin-coloured
# pixels that were not there a moment ago, or, when it is right at the lens, as most of the
# picture changing at once.
HAND = {
    "baseline_frames": 5,       # pictures averaged, right after the arm stops, as "nothing there"
    "diff": 28,                 # grey level change that counts as a changed pixel
    "skin_share": 0.03,         # this share of the picture newly skin-coloured means a hand
    "big_share": 0.30,          # or this share changed at all (a hand right at the lens)
    "frames": 3,                # for this many pictures in a row
}


def hand_shares(base_gray, frame):
    """(share of the picture that changed and is skin-coloured, share that changed at all)."""
    import cv2
    import numpy as np
    small = cv2.resize(frame, (160, 90))
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    changed = cv2.absdiff(gray, base_gray) > HAND["diff"]
    ycc = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycc, (40, 135, 85), (255, 180, 135)) > 0
    n = float(changed.size)
    return float(np.count_nonzero(changed & skin)) / n, float(np.count_nonzero(changed)) / n


class HandWatcher(threading.Thread):
    """Runs beside the arm while it holds an item out: once armed, watches the wrist camera for a
    hand coming in, on the laptop itself, many times a second. .hand turns True when it sees one."""

    def __init__(self, index=WRIST_INDEX, save=True):
        super().__init__(daemon=True)
        self.index = index
        self.save = save
        self.armed = False
        self.hand = False
        self.checks = 0
        self.done = False
        self.seen = None        # the shares that set it off, for the result line

    def run(self):
        import cv2
        import numpy as np
        cap = _open(self.index, 1280, 720)
        base, stack, streak = None, [], 0
        try:
            while not self.done:
                f = None
                for _ in range(2):
                    ok, fr = cap.read()
                    if ok:
                        f = fr
                if f is None or not self.armed:
                    base, stack, streak = None, [], 0
                    time.sleep(0.02)
                    continue
                f = cv2.rotate(f, cv2.ROTATE_180)
                if base is None:
                    small = cv2.resize(f, (160, 90))
                    stack.append(cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0))
                    if len(stack) >= HAND["baseline_frames"]:
                        base = np.median(np.stack(stack), axis=0).astype(np.uint8)
                    continue
                skin, big = hand_shares(base, f)
                self.checks += 1
                streak = streak + 1 if (skin >= HAND["skin_share"] or big >= HAND["big_share"]) else 0
                if streak >= HAND["frames"]:
                    self.hand = True
                    self.seen = {"skin_share": round(skin, 3), "changed_share": round(big, 3)}
                    if self.save:
                        try:
                            cv2.imwrite(str(SETUP / "last_handover_view.jpg"), cv2.resize(f, (640, 360)))
                        except Exception:
                            pass
                    break
        finally:
            cap.release()

    def stop(self):
        self.done = True
