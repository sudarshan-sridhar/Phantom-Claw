"""Cameras and Gemini for the arm: where the bottle and the person are.

    from backend.vision import Vision
    v = Vision()
    seen = v.look_overhead()          # None when the overhead phone gives no picture
    look = v.wrist_look(pan=2009)     # one upright wrist frame asked for boxes
    w = v.hand_watcher()              # thread that watches for a hand while offering

The arm only uses these four calls, so tests can pass any object with the same methods.
"""
import math
from pathlib import Path

from . import cameras, gemini
from .gemini import GeminiError, parse_box, parse_boxes
from .table import JAW_OFFSET, TableMap, Taught

__all__ = ["Vision", "TableMap", "Taught", "parse_box", "parse_boxes", "GeminiError", "JAW_OFFSET",
           "wrist_pan_for", "WRIST"]

OVERHEAD_PROMPT = (
    "This is a photo of a desk from above. Find (1) the clear plastic water bottle standing on the desk and "
    "(2) the person sitting at the edge of the picture (body, arm, hand or lap; ignore the robot arm). "
    'Reply with JSON only: {"bottle": [ymin, xmin, ymax, xmax] or null, "person": [ymin, xmin, ymax, xmax] or null} '
    "with each box in 0 to 1000 units of the image height and width.")

WRIST_PROMPT = (
    "This photo is from a small camera on a robot arm's gripper, looking across a desk. Find (1) the clear plastic "
    "water bottle standing on the desk and (2) a person (face, body, arm or hand; ignore the robot arm itself). "
    'Reply with JSON only: {"bottle": [ymin, xmin, ymax, xmax] or null, "person": [ymin, xmin, ymax, xmax] or null} '
    "with each box in 0 to 1000 units of the image height and width.")

# Wrist camera only (no overhead picture). These are first guesses to check at the arm:
# the field of view of the wrist camera, the image column the bottle sits in when the jaws are
# lined up with it, and which way the base must turn for something on the right of the picture.
WRIST = {
    "look_flex": 2150,          # wrist flex that looks across the desk from rest
    "sweep_deg": (-40, -20, 0, 20, 40),   # base angles to look from, left to right of straight ahead
    "hfov_deg": 65.0,
    "centre_x": 500.0,          # 0 to 1000 across the upright picture
    "pan_sign": 1,              # +1: something right of centre needs a larger pan value
    "centred_x": 40.0,          # within this of centre_x counts as centred
    "centre_tries": 3,
    "max_step_ticks": 250,      # never turn more than this in one centring step
    "person_default_deg": 45.0,  # not found: hold it out 45 degrees to the right
}


def wrist_pan_for(box, pan_here, tpr, w=WRIST):
    """The pan that puts this box in the middle of the wrist picture, seen from pan_here."""
    ymin, xmin, ymax, xmax = box
    xc = (xmin + xmax) / 2.0
    off = (xc - w["centre_x"]) / 1000.0 * math.radians(w["hfov_deg"])
    return pan_here + w["pan_sign"] * off * tpr, xc


class Vision:
    def __init__(self, setup_dir=None, overhead_index=cameras.OVERHEAD_INDEX, wrist_index=cameras.WRIST_INDEX):
        d = Path(setup_dir) if setup_dir else cameras.SETUP
        self.table = TableMap(d / "table_map.json")
        self.overhead_index, self.wrist_index = overhead_index, wrist_index
        self._wrist = None

    def locate(self, img, tries=3):
        """Ask Gemini for the bottle and the person in an overhead picture and place them on the desk.
        Asks again (up to three times) if the answer has no usable bottle box."""
        boxes = {"bottle": None, "person": None}
        for _ in range(tries):
            boxes = parse_boxes(gemini.ask_json(img, OVERHEAD_PROMPT, size=(960, 540)))
            if boxes["bottle"]:
                break
        return self.table.locate(boxes)

    def look_overhead(self, stop=None):
        """{"boxes", "bottle", "person"} from the overhead camera, or None when it gives no picture."""
        img = cameras.grab_overhead(self.overhead_index, stop=stop)
        if img is None:
            return None
        return self.locate(img)

    def wrist_look(self, pan=None):
        """Boxes for the bottle and the person in one upright wrist frame:
        {"picture": bool, "bottle": box or None, "person": box or None}."""
        if self._wrist is None:
            self._wrist = cameras.WristCamera(self.wrist_index)
        f = self._wrist.frame()
        if f is None:
            return {"picture": False, "bottle": None, "person": None}
        try:
            import cv2
            cv2.imwrite(str(cameras.SETUP / "last_wrist_look.jpg"), cv2.resize(f, (640, 360)))
        except Exception:
            pass
        boxes = parse_boxes(gemini.ask_json(f, WRIST_PROMPT, size=(960, 540), timeout=10))
        return {"picture": True, **boxes}

    def close_wrist(self):
        if self._wrist is not None:
            self._wrist.close()
            self._wrist = None

    def hand_watcher(self):
        """A started HandWatcher. Set .armed = True when the bottle is held out; read .hand; call .stop()."""
        self.close_wrist()      # one user of the wrist camera at a time
        w = cameras.HandWatcher(self.wrist_index)
        w.start()
        return w
