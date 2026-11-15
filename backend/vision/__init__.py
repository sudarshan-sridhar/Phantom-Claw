"""Cameras and Gemini for the arm: where the bottle and the person are.

    from backend.vision import Vision
    v = Vision()
    seen = v.look_overhead()          # None when the overhead phone gives no picture
    look = v.wrist_look(pan=2009)     # one upright wrist frame asked for boxes
    w = v.hand_watcher()              # thread that watches for a hand while offering
    s = v.scene()                     # everything on the desk, one picture, one Gemini call
    it = v.find("something to drink") # the one item that best fits a spoken name

The arm only uses these calls, so tests can pass any object with the same methods.
"""
import math
from pathlib import Path

from . import cameras, gemini
from .gemini import GeminiError, parse_box, parse_boxes
from .scene import match_name, parse_scene, scene_prompt, tier_for
from .table import JAW_OFFSET, GripTier, TableMap, Taught, load_tiers

__all__ = ["Vision", "TableMap", "Taught", "GripTier", "load_tiers", "parse_box", "parse_boxes", "GeminiError",
           "JAW_OFFSET", "wrist_pan_for", "WRIST", "parse_scene", "scene_prompt", "match_name", "tier_for"]

OVERHEAD_PROMPT = (
    "This is a photo of a desk from above. Find (1) the clear plastic water bottle standing on the desk and "
    "(2) the owner: the person sitting next to the laptop computer (if there are several people, only that one; never anyone else) (body, arm, hand or lap; ignore the robot arm). "
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


def in_grid(scene_out, table, margin=3.0):
    """Keep only the things lying inside the rectangle the four paper markers mark out. The
    cups, phones and people's things around the edge of the table are not the arm's business."""
    cm = getattr(table, "grid_cm", None) or ((-4.0, -4.0), (54.6, 37.0))
    (x0, y0), (x1, y1) = cm
    bx, by = table.base

    def inside(it):
        tx, ty = it["x_cm"] + bx, it["y_cm"] + by
        return x0 - margin <= tx <= x1 + margin and y0 - margin <= ty <= y1 + margin

    kept = [it for it in scene_out["items"] if inside(it)]
    scene_out["outside"] = len(scene_out["items"]) - len(kept)
    scene_out["items"] = kept
    if scene_out.get("match") is not None and not any(scene_out["match"] is it for it in kept):
        scene_out["match"] = None
    return scene_out


PIN_MODEL = "claude-opus-5-5"


def pin_down(img, item, table, asked=None):
    """Place the one item the job is about exactly, with the stronger model. A box from the
    quick scene call puts a standing thing anywhere between its base and its top (the glue stick
    read 36 cm one look and 43 the next). Asked for the single point where it touches the desk,
    Claude gives the same answer every time, in about a second. For a lying thing it gives the
    two ends, which also fix its angle. Changes the item in place; on any failure leaves it alone."""
    name = asked or item.get("name") or "object"
    lying = item.get("tier") == "low"
    if lying:
        q = (f"This is a photo of a desk from above and slightly to one side. Find the {name} lying on the desk "
             f"(the scene list calls it '{item.get('name')}'). Give the two ends of its long side, and for scissors "
             'also the middle of the handles. Reply with JSON only: {"ends": [[y1, x1], [y2, x2]], "handle": [y, x] or null} '
             "in 0 to 1000 units of the image height and width.")
    else:
        q = (f"This is a photo of a desk from above and slightly to one side. Find the {name} standing on the desk "
             f"(the scene list calls it '{item.get('name')}'). Find the exact point where its bottom touches the desk "
             '(its base, not its top or middle). Reply with JSON only: {"base": [y, x]} in 0 to 1000 units of the '
             "image height and width.")
    old = gemini.CLAUDE_VISION
    try:
        gemini.CLAUDE_VISION = PIN_MODEL
        raw = gemini.ask_claude(img, q, timeout=8.0)
    except Exception:
        return item
    finally:
        gemini.CLAUDE_VISION = old
    sx, sy = table.image_size[0] / 1000.0, table.image_size[1] / 1000.0
    try:
        if lying:
            (y1, x1), (y2, x2) = raw["ends"]
            h = raw.get("handle")
            py, px = (h if (isinstance(h, list) and len(h) == 2) else ((y1 + y2) / 2.0, (x1 + x2) / 2.0))
            item["axis_deg"] = round(table.axis_of(item.get("box"), ends=((y1, x1), (y2, x2))), 1)
        else:
            py, px = raw["base"]
        x, y = table._arm_xy(float(px) * sx, float(py) * sy)
        item.update(table.point(x, y))
        item["pinned"] = PIN_MODEL
    except Exception:
        pass
    return item


class Vision:
    def __init__(self, setup_dir=None, overhead_index=cameras.OVERHEAD_INDEX, wrist_index=cameras.WRIST_INDEX):
        d = Path(setup_dir) if setup_dir else cameras.SETUP
        self.setup_dir = d
        self.table = TableMap(d / "table_map.json")
        self.overhead_index, self.wrist_index = overhead_index, wrist_index
        self._wrist = None

    def locate(self, img, tries=3):
        """Ask Gemini for the bottle and the person in an overhead picture and place them on the desk.
        Asks again (up to three times) if the answer has no usable bottle box."""
        # The phone may have been moved since the map was made: line the map up again from the markers.
        from . import markers
        self.markers_used = markers.refresh(self.table, img)
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

    def scene(self, name=None, stop=None, img=None):
        """Everything on the desk from one overhead picture and one Gemini call:
        {"items": [...], "person": {...} or None, "match": item or None, "markers_used": n}, or None
        when the overhead camera gives no picture. With a name, Gemini also picks the item that
        best fits it ("match"). See backend/vision/scene.py for the fields and the desk frame."""
        if img is None:
            img = cameras.grab_overhead(self.overhead_index, stop=stop)
            if img is None:
                return None
        from . import markers
        self.markers_used = markers.refresh(self.table, img)
        raw = gemini.ask_json(img, scene_prompt(name), size=(960, 540))
        out = in_grid(parse_scene(raw, self.table, want=name), self.table)
        if name and out.get("match") is not None:
            pin_down(img, out["match"], self.table, asked=name)
        out["markers_used"] = self.markers_used
        self.last_scene = out
        return out

    def wrist_scene(self, pan, name=None):
        """The same answer as scene(), from one gripper camera picture taken at the look-down
        pose with the base at this pan. None when the camera gives no picture or the map is missing."""
        from .table import WristTable, wrist_map
        if wrist_map(self.setup_dir) is None:
            return None
        if self._wrist is None:
            self._wrist = cameras.WristCamera(self.wrist_index)
        f = self._wrist.frame()
        if f is None:
            return None
        try:
            import cv2
            cv2.imwrite(str(cameras.SETUP / "last_wrist_scene.jpg"), cv2.resize(f, (640, 360)))
        except Exception:
            pass
        prompt = scene_prompt(name).replace(
            "This is a photo of a desk from above.",
            "This photo is from a camera on a robot gripper looking down at a desk. The two grey jaws of the "
            "gripper are at the bottom edge of the picture; ignore them.")
        raw = gemini.ask_json(f, prompt, size=(960, 540))
        wt = WristTable(pan, self.setup_dir)
        out = in_grid(parse_scene(raw, wt, want=name), wt)
        out["markers_used"] = 0
        out["camera"] = "wrist"
        self.last_scene = out
        return out

    def find(self, name, stop=None, img=None):
        """The one item that best matches a spoken name ("the mug", "something to drink"), or None."""
        s = self.scene(name=name, stop=stop, img=img)
        return None if s is None else s["match"]

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

    def wrist_aim(self, name=None, shape="auto", save_to=None, use_gemini=True, opening=None):
        """Read one upright wrist picture for pick's last aim (backend/vision/wrist_aim.py).
        {"picture": bool, "aim": reading or None, "tried": what OpenCV and Gemini said}.
        save_to: path for the picture (640 x 360); a copy with the reading drawn on it goes
        next to it with _read added to the name. opening: the gripper's ticks (where the jaw tips are)."""
        from . import wrist_aim
        if self._wrist is None:
            self._wrist = cameras.WristCamera(self.wrist_index)
        f = self._wrist.frame(flush=6)        # a few more reads: the arm has just moved
        if f is None:
            return {"picture": False, "aim": None, "tried": {}}
        tried = {}
        aim = wrist_aim.read_aim(f, name=name, shape=shape, use_gemini=use_gemini, log=tried, opening=opening)
        if save_to:
            try:
                import cv2
                p = Path(save_to)
                cv2.imwrite(str(p), cv2.resize(f, wrist_aim.AIM["size"]))
                shown = aim or tried.get("gemini") or tried.get("opencv")
                cv2.imwrite(str(p.with_name(p.stem + "_read" + p.suffix)), wrist_aim.draw(f, shown))
            except Exception:
                pass
        return {"picture": True, "aim": aim, "tried": tried}

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
