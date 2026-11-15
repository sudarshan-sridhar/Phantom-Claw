"""Everything on the desk from one overhead picture and one Gemini call.

Pure code: the prompt, the parser for Gemini's answer and the name matching. The camera and the
network live in backend.vision.Vision.scene(), so tests can feed a canned answer straight in.

Desk frame for every position here (the arm frame): centimetres from the arm base, x to the right
in the overhead picture, y straight away from the arm. dist_cm is the distance from the base,
bearing the angle in degrees from straight ahead (positive to the right), pan the base joint value
that lines the jaws up with the item (the same maths fetch_bottle uses for the bottle).

Long axis of a lying item (axis_deg): degrees in the same frame, 0 = the item's long side points
straight away from the arm base (along +y), positive turns it toward +x (to the right). An axis
has no head or tail, so it is kept between -90 and 90.
"""
import json
import math
import re

from .gemini import parse_box

TIERS = ("tall", "can", "low")

# Known names pick the tier; Gemini's height guess is used only for names not listed here.
# Checked in this order, so "juice bottle" is tall and "juice can" is can.
_LOW_WORDS = ("sharpie", "marker", "pen", "pencil", "scissors", "scissor", "highlighter", "screwdriver", "spoon",
              "fork", "knife", "chopstick", "brush", "earbud", "earphone", "airpod")
_CAN_WORDS = ("glue", "can", "red bull", "redbull", "cup", "mug", "glass", "tin")
_TALL_WORDS = ("bottle", "flask", "thermos")
_CAN_LATE = ("juice", "soda", "coke", "pepsi", "sprite", "drink")
CUP_WORDS = ("cup", "mug", "glass")       # gripped by pinching the wall nearest the arm, not the centre

_IGNORE = ("robot", "aruco", "fiducial", "qr code", "paper marker", "marker sheet", "printed marker", "arm base",
           "gripper", "servo", "usb cable", "cable")

_STOP = {"the", "a", "an", "my", "me", "that", "this", "those", "these", "some", "something", "please", "of", "to",
         "for", "it", "one", "thing", "over", "there", "here", "on", "desk", "table", "get", "grab", "pick", "up",
         "little", "small", "big", "your", "our", "can you"}
_SYNONYMS = {
    "drink": ("bottle", "can", "cup", "mug", "glass", "juice", "red bull", "soda"),
    "water": ("bottle",),
    "phone": ("phone", "mobile", "smartphone", "iphone"),
    "pen": ("pen", "sharpie", "marker", "pencil"),
    "marker": ("sharpie", "marker", "pen"),
    "sharpie": ("sharpie", "marker"),
    "energy": ("red bull",),
    "redbull": ("red bull",),
    "coffee": ("mug", "cup"),
    "cutter": ("scissors",),
}

SCENE_PROMPT = (
    "This is a photo of a desk from above. List every separate object on the desk. Ignore the robot arm "
    "(its links, gripper and cable) and the square black-and-white paper markers taped to the desk. "
    "Also find the person at the edge of the picture (body, arm, hand or lap).\n"
    "For each object give:\n"
    '- "name": a short plain label of one to three words, such as "water bottle", "red bull can", "cup", '
    '"sharpie", "scissors";\n'
    '- "box": [ymin, xmin, ymax, xmax] in 0 to 1000 units of the image height and width;\n'
    '- "height": "tall" if it stands about as high as a water bottle (15 cm or more), "medium" if it stands '
    'about as high as a drink can or a cup, "low" if it lies flat on the desk;\n'
    '- "graspable": true if a small two-finger gripper that opens about 6 cm could close around it or around '
    "one of its walls;\n"
    '- "ends": only for "low" objects, the two ends of its long side as [[y1, x1], [y2, x2]] in the same '
    "0 to 1000 units.\n"
    'Reply with JSON only: {"items": [{"name": ..., "box": [...], "height": ..., "graspable": ..., "ends": ...}], '
    '"person": [ymin, xmin, ymax, xmax] or null}')


def scene_prompt(name=None):
    """The prompt, with the matching request added when a spoken name is given."""
    if not name:
        return SCENE_PROMPT
    want = json.dumps(str(name)[:60])
    return (SCENE_PROMPT + f"\nThe person asked for {want}. Also add \"match\": the index in items (0 based) of the "
            "one object that best fits what they asked for, or null if nothing on the desk fits.")


def _words(s):
    return re.findall(r"[a-z0-9]+", str(s).lower())


def _has(name, words):
    padded = " " + " ".join(_words(name)) + " "
    for w in words:
        if (" " + w + " ") in padded or (" " + w + "s ") in padded:
            return True
    return False


def tier_for(name, height_guess=None):
    """(tier, where it came from): the known-name table first, then Gemini's height guess."""
    for words, tier in ((_LOW_WORDS, "low"), (_CAN_WORDS, "can"), (_TALL_WORDS, "tall"), (_CAN_LATE, "can")):
        if _has(name, words):
            return tier, "name"
    g = str(height_guess or "").strip().lower()
    if g in ("tall", "high"):
        return "tall", "gemini"
    if g in ("low", "flat", "lying", "short"):
        return "low", "gemini"
    return "can", "gemini"        # "medium", or no guess: the middle tier


def known_name(name):
    return tier_for(name)[1] == "name"


def is_cup(name):
    return _has(name, CUP_WORDS)


def _flag(v, default=False):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "1"):
            return True
        if s in ("false", "no", "0"):
            return False
    if isinstance(v, (int, float)):
        return v != 0
    return default


def _label(v):
    if not isinstance(v, str):
        return ""
    s = " ".join(v.strip().lower().split())
    return s[:40]


def _ignored(name):
    return any(w in name for w in _IGNORE)


def _ends(v):
    """Two points [[y, x], [y, x]] from Gemini, or None."""
    if isinstance(v, dict):
        v = [v.get("a") or v.get("start") or v.get("p1"), v.get("b") or v.get("end") or v.get("p2")]
    if not (isinstance(v, list) and len(v) == 2):
        return None
    out = []
    for p in v:
        if isinstance(p, dict):
            p = [p.get("y"), p.get("x")]
        if not (isinstance(p, list) and len(p) == 2 and all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in p)):
            return None
        out.append([float(p[0]), float(p[1])])
    if out[0] == out[1]:
        return None
    return out


def axis_angle(dx, dy):
    """Desk-frame direction (dx, dy) to the long-axis angle: 0 along +y, positive toward +x, in (-90, 90]."""
    a = math.degrees(math.atan2(dx, dy))
    while a > 90:
        a -= 180
    while a <= -90:
        a += 180
    return a


def parse_scene(raw, table, want=None, max_items=15):
    """Gemini's scene answer to {"items": [...], "person": {...} or None, "match": item or None}.
    table: a TableMap (gives desk positions). want: the spoken name, for "match"."""
    if isinstance(raw, list):
        raw = {"items": raw}
    if not isinstance(raw, dict):
        raw = {}
    got = raw.get("items") or raw.get("objects") or []
    if isinstance(got, dict):
        got = [got]
    if not isinstance(got, list):
        got = []
    items, kept = [], {}
    for i, it in enumerate(got):
        if len(items) >= max_items:
            break
        if not isinstance(it, dict):
            continue
        name = _label(it.get("name") or it.get("label"))
        box = parse_box(it.get("box") if it.get("box") is not None else it.get("box_2d"))
        if not name or box is None or _ignored(name):
            continue
        tier, source = tier_for(name, it.get("height"))
        item = {"name": name, "box": box, "tier": tier, "tier_from": source}
        item.update(table.place_box(box, tier))
        item["graspable"] = True if source == "name" else _flag(it.get("graspable"), False)
        item["axis_deg"] = None
        if tier == "low":
            ends = _ends(it.get("ends"))
            item["axis_deg"] = round(table.axis_of(box, ends), 1)
        item["grip_at"] = grip_point(item, table)
        kept[i] = len(items)
        items.append(item)
    pbox = parse_box(raw.get("person"))
    person = table.place_person(pbox) if pbox else None
    match = None
    if want:
        m = raw.get("match")
        if isinstance(m, int) and not isinstance(m, bool) and m in kept:
            match = items[kept[m]]
        elif isinstance(m, str):
            match = match_name(items, m)
        if match is None:
            match = match_name(items, want)
        # The camera may call it something else ("white marker" for a glue stick). What the
        # person asked for decides how it is gripped when that name is a known one.
        if match is not None:
            asked_tier, source = tier_for(want)
            if source == "name" and asked_tier != match["tier"]:
                match["tier"] = asked_tier
                if asked_tier != "low":
                    match["axis_deg"] = None
    return {"items": items, "person": person, "match": match}


def grip_point(item, table, default_radius=4.0):
    """Where the jaws go. The centre for most things; for a cup, the point of its rim nearest the
    arm (its centre moved toward the base by its radius), since a cup is gripped by pinching its wall."""
    pos = {k: item[k] for k in ("x_cm", "y_cm", "dist_cm", "bearing", "pan")}
    if not is_cup(item.get("name", "")):
        return pos
    r = item.get("width_cm")
    r = default_radius if not r else max(2.0, min(6.0, r / 2.0))
    return rim_point(item["x_cm"], item["y_cm"], r, table)


def rim_point(x, y, radius, table):
    d = math.hypot(x, y)
    if d < 1e-6:
        return table.point(x, y)
    s = max(0.0, d - radius) / d
    out = table.point(x * s, y * s)
    out["rim_offset_cm"] = round(radius, 1)
    return out


def match_name(items, name):
    """The item whose name best fits a spoken name, by shared words (and a few synonyms).
    Ties go to the nearest one. None when nothing shares a word."""
    want = [w for w in _words(name) if w not in _STOP]
    if not want or not items:
        return None
    joined = " ".join(want)
    best, best_score = None, 0
    for it in items:
        have = set(_words(it.get("name", "")))
        have |= {w[:-1] for w in have if w.endswith("s")}
        score = 0
        for w in want:
            ws = w[:-1] if w.endswith("s") and len(w) > 3 else w
            if w in have or ws in have:
                score += 2
            elif any(_has(it.get("name", ""), (s,)) for s in _SYNONYMS.get(ws, ())):
                score += 1
        if joined and joined in " ".join(_words(it.get("name", ""))):
            score += 1
        if score > best_score or (score == best_score and score > 0 and best is not None
                                  and it.get("dist_cm", 1e9) < best.get("dist_cm", 1e9)):
            best, best_score = it, score
    return best if best_score > 0 else None


def public(item):
    """An item as the planner sees it (no internal keys)."""
    if item is None:
        return None
    keep = ("name", "box", "x_cm", "y_cm", "dist_cm", "bearing", "tier", "graspable", "axis_deg")
    return {k: item.get(k) for k in keep if k in item}
