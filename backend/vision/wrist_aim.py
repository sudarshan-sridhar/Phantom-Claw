"""Read one upright wrist picture taken from the hover pose above an item, for pick's last aim.

From the hover ("via") pose with the jaws open, the wrist camera looks down and ahead at the item
and the two grey jaw tips sit at the bottom edge. The jaws grip well when the item's long side
runs straight up the picture and passes between the jaw tips. A reading says how far off that is:

    {"angle_deg": clockwise angle of the long side from vertical (-90 to 90), or None for a
                  round or square item,
     "x_at_jaws": x (in a 640 wide picture) where the item's centre line crosses the jaw row,
     "offset_px": x_at_jaws minus the jaw gap centre (negative: the item is to the left),
     "between_jaws": True when that crossing lies between the jaw tips at the present opening,
     "confidence": 0 to 1,
     "source": "opencv" or "gemini",
     "line": [[x1, y1], [x2, y2]] the centre line found (for drawing), or None}

A compact item (round, square, a case: no long side crossing the jaw row) also gets where it sits,
so pick can move in or out and open wider: "compact": True, "centre_y", "top_y", "bottom_y",
"left_x", "right_x", "width_px" (side to side, the direction the jaws close) and "touches_bottom"
(cut off by the bottom edge of the picture: the item is nearer the arm than the jaws).

read_aim() tries plain OpenCV first (instant) and asks Gemini only when OpenCV is not sure.
Everything is measured on the picture scaled to 640 x 360, whatever size it came in.
The numbers below were measured on the arm on 2026-11-15 (setup/wrist_samples).
"""
import math
import time

from . import gemini

AIM = {
    "size": (640, 360),        # every picture is scaled to this before reading
    "jaw_x": 452.0,            # x of the jaw gap centre: 0.706 of the width, not the middle
    # Jaw tips at the jaw row. The left jaw is the fixed one and stays at about x 300; the right one
    # moves out as the gripper opens. At 1750 ticks the tips were at about x 300 and 520 (the
    # 2026-11-15 earbuds case miss), and by 2400 the right tip had left the picture, so the gap is
    # taken to grow about 0.5 px per tick (it also gives about 0 px near fully closed, 1310).
    "jaw_open_ref": 1750,      # gripper ticks the tips below were seen at
    "jaw_tips_ref": (300.0, 520.0),
    "jaw_px_per_tick": 0.5,    # the right tip moves this many px per tick of opening
    "jaw_row": 290.0,          # y where the centre line is read, just above the jaw tips
    "bottom_edge_px": 6,       # an item reaching this close to the bottom edge is cut off by it
    "compact_cv_conf": 0.45,   # OpenCV's blob for a compact item is never trusted alone (see read_cv)
    "compact_max_width_px": 330.0,   # a compact box wider than this (1.5 jaw gaps) is most likely the jaws
    "item_rows": 240,          # the item is looked for above this row (upper two thirds); the jaws are below
    "min_width": 20.0,         # a long item is between this and max_width pixels from side to side
    "max_width": 200.0,
    "max_gap": 80.0,           # edge lines of one item lie closer than this to each other across it
    "round_ratio": 1.8,        # a blob less elongated than this is round or square: no angle
    "cv_min_conf": 0.6,        # OpenCV's reading is used when at least this sure
    "gemini_min_conf": 0.5,    # and Gemini's when at least this sure
    "gemini_timeout_s": 4.0,  # one Gemini call, all models together
}

# Items with no long side: only the centre matters
ROUND_WORDS = ("earbud", "earphone", "airpod", "case", "cap", "lid", "coin", "ball", "puck", "button", "tape")
LONG_WORDS = ("sharpie", "marker", "pen", "pencil", "scissors", "scissor", "highlighter", "screwdriver", "spoon",
              "fork", "knife", "chopstick", "brush", "glue", "stick", "ruler", "cable")


def shape_for(name):
    """"round", "long" or None (not known: decided from the picture)."""
    s = " " + " ".join(str(name or "").lower().replace("-", " ").split()) + " "
    if any(" " + w in s for w in ROUND_WORDS):
        return "round"
    if any(" " + w in s for w in LONG_WORDS):
        return "long"
    return None


def wrap90(a):
    while a > 90:
        a -= 180
    while a <= -90:
        a += 180
    return a


def _upright(img, cfg=AIM):
    import cv2
    w, h = cfg["size"]
    if img.shape[1] != w or img.shape[0] != h:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img


def jaw_tips(opening=None, cfg=AIM):
    """(left x, right x) of the jaw tips in the picture with the gripper at this many ticks."""
    left, right = cfg["jaw_tips_ref"]
    if opening is None:
        return left, right
    return left, right + cfg["jaw_px_per_tick"] * (opening - cfg["jaw_open_ref"])


def jaw_gap(opening=None, cfg=AIM):
    left, right = jaw_tips(opening, cfg)
    return max(0.0, right - left)


def extents(reading, left, top, right, bottom, cfg=AIM):
    """Add where a compact item sits (pixels in the 640 x 360 picture) to its reading."""
    h = cfg["size"][1]
    reading.update(compact=True, centre_y=round((top + bottom) / 2.0, 1), top_y=round(top, 1),
                   bottom_y=round(bottom, 1), left_x=round(left, 1), right_x=round(right, 1),
                   width_px=round(right - left, 1), touches_bottom=bottom >= h - cfg["bottom_edge_px"])
    return reading


def line_reading(p_top, p_bot, source, confidence, cfg=AIM, opening=None):
    """A reading from two points on the item's centre line (pixels in the 640 x 360 picture)."""
    (x1, y1), (x2, y2) = p_top, p_bot
    if y1 > y2:
        (x1, y1), (x2, y2) = (x2, y2), (x1, y1)
    angle = wrap90(math.degrees(math.atan2(x1 - x2, y2 - y1)))
    if abs(y2 - y1) < 1e-6 or abs(angle) > 80:
        x = (x1 + x2) / 2.0           # lying across the picture: the crossing is ill defined, use the middle
    else:
        x = x1 + (cfg["jaw_row"] - y1) * (x2 - x1) / (y2 - y1)
    return _reading(angle, x, source, confidence, [[x1, y1], [x2, y2]], cfg, opening)


def point_reading(x, source, confidence, cfg=AIM, line=None, opening=None):
    return _reading(None, x, source, confidence, line, cfg, opening)


def _reading(angle, x, source, confidence, line, cfg, opening=None):
    off = x - cfg["jaw_x"]
    left, right = jaw_tips(opening, cfg)
    return {"angle_deg": None if angle is None else round(angle, 1), "x_at_jaws": round(x, 1),
            "offset_px": round(off, 1), "between_jaws": left < x < right,
            "confidence": round(max(0.0, min(1.0, confidence)), 2), "source": source,
            "line": None if line is None else [[round(a, 1) for a in p] for p in line]}


# OpenCV: the item's two long edges
def _lines(gray, cfg):
    import cv2
    import numpy as np
    b = cv2.GaussianBlur(gray, (5, 5), 0)
    e = cv2.Canny(b, 40, 120)
    e[cfg["item_rows"]:] = 0
    raw = cv2.HoughLinesP(e, 1, np.pi / 180, 50, minLineLength=60, maxLineGap=10)
    out = []
    for x1, y1, x2, y2 in ([] if raw is None else raw[:, 0]):
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        a = wrap90(math.degrees(math.atan2(x1 - x2, y2 - y1)))
        out.append({"a": a, "len": math.hypot(x2 - x1, y2 - y1), "p": ((float(x1), float(y1)), (float(x2), float(y2)))})
    return out


def _frame(a):
    """Up direction d and the normal n for a line at angle a (clockwise from vertical, y down)."""
    t = math.radians(a)
    return (math.sin(t), -math.cos(t)), (math.cos(t), math.sin(t))


def _mean_angle(angles, weights):
    s = sum(w * math.sin(math.radians(2 * a)) for a, w in zip(angles, weights))
    c = sum(w * math.cos(math.radians(2 * a)) for a, w in zip(angles, weights))
    return wrap90(math.degrees(math.atan2(s, c)) / 2.0)


def _strip_mean(lab, a, c0, c1, cfg):
    """Mean Lab colour of the pixels whose offset along the normal lies in [c0, c1], upper rows only."""
    import numpy as np
    h, w = cfg["item_rows"], lab.shape[1]
    _, n = _frame(a)
    ys, xs = np.mgrid[0:h:2, 0:w:2]
    off = xs * n[0] + ys * n[1]
    m = (off >= min(c0, c1)) & (off <= max(c0, c1))
    if m.sum() < 30:
        return None
    return lab[0:h:2, 0:w:2][m].reshape(-1, 3).mean(0)


def read_cv(img, shape=None, cfg=AIM, opening=None):
    """OpenCV reading, or None. A long item is found by its two long parallel edges; a round one
    (or a long one whose edges cannot be paired) by the blob that differs most from the desk.
    A compact blob is never sure enough alone (at most compact_cv_conf): on the 2026-11-15 miss
    it took desk glare above the jaws for an earbuds case that sat cut off between the jaw tips,
    said 0.9 and lined up on nothing. Gemini decides those."""
    import cv2
    import numpy as np
    img = _upright(img, cfg)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    if shape != "round":
        band = _best_band(_lines(gray, cfg), lab, cfg, opening)
        if band is not None:
            return band
    return _blob(lab, shape, cfg, opening)


def _best_band(lines, lab, cfg, opening=None):
    """The item's two sides: the strongest direction among the edge lines, then the group of lines
    of that direction lying close together across it (a gap wider than max_gap starts a new group).
    The outermost lines of the group are the sides. Lines inside (printing, a highlight) do not
    move the centre; one side missing makes the reading unsure."""
    import numpy as np
    if not lines:
        return None
    # strongest direction: the angle with the most line length within 6 degrees of it
    best_a, best_w = None, 0.0
    for L in lines:
        w = sum(M["len"] for M in lines if abs(wrap90(M["a"] - L["a"])) <= 6)
        if w > best_w:
            best_a, best_w = L["a"], w
    same = [L for L in lines if abs(wrap90(L["a"] - best_a)) <= 6]
    a = _mean_angle([L["a"] for L in same], [L["len"] for L in same])
    d, n = _frame(a)
    for L in same:
        (x1, y1), (x2, y2) = L["p"]
        L["o"] = ((x1 + x2) / 2) * n[0] + ((y1 + y2) / 2) * n[1]
    same.sort(key=lambda L: L["o"])
    groups, cur = [], [same[0]]
    for L in same[1:]:
        if L["o"] - cur[-1]["o"] > cfg["max_gap"]:
            groups.append(cur)
            cur = []
        cur.append(L)
    groups.append(cur)
    # the group with the most length, nearer the jaws on a tie
    y_mid = cfg["item_rows"] / 2.0

    def x_of(c):
        return (c - y_mid * n[1]) / n[0] if abs(n[0]) > 1e-3 else cfg["size"][0] / 2.0

    def weight(g):
        c = (g[0]["o"] + g[-1]["o"]) / 2.0
        return sum(L["len"] for L in g) / (1.0 + abs(x_of(c) - cfg["jaw_x"]) / 400.0)
    groups.sort(key=lambda g: -weight(g))
    g = groups[0]
    lo, hi = g[0]["o"], g[-1]["o"]
    width = hi - lo
    if width > cfg["max_width"]:
        return None
    side_lo = sum(L["len"] for L in g if L["o"] - lo <= 8)
    side_hi = sum(L["len"] for L in g if hi - L["o"] <= 8)
    one_side = width < cfg["min_width"]
    if one_side:
        side_hi = 0.0
    inside = _strip_mean(lab, a, lo + 3, hi - 3, cfg) if not one_side else None
    outs = [o for o in (_strip_mean(lab, a, lo - 25, lo - 4, cfg), _strip_mean(lab, a, hi + 4, hi + 25, cfg))
            if o is not None]
    contrast = float(np.mean([np.abs(inside - o).sum() for o in outs])) if inside is not None and outs else 0.0
    conf_len = min(1.0, min(side_lo, side_hi) / 100.0)
    conf_con = min(1.0, contrast / 25.0)
    conf = math.sqrt(conf_len * conf_con)
    if one_side:
        conf = 0.3                      # one edge only: where the middle is cannot be told
    if len(groups) > 1:                 # another group of edges nearly as strong: which is the item?
        conf *= 1.0 - 0.6 * min(1.0, weight(groups[1]) / weight(g))
    c = (lo + hi) / 2.0
    x_mid = x_of(c)
    top = (x_mid + 100 * d[0], y_mid + 100 * d[1])
    bot = (x_mid - 100 * d[0], y_mid - 100 * d[1])
    r = line_reading(top, bot, "opencv", conf, cfg, opening)
    r["width_px"] = round(width, 1)
    return r


def _blob(lab, shape, cfg, opening=None):
    """The biggest region in the upper rows that differs from the desk colour."""
    import cv2
    import numpy as np
    h = cfg["item_rows"]
    up = lab[:h]
    desk = np.median(up.reshape(-1, 3), axis=0)
    dist = np.abs(up - desk).sum(axis=2)
    thr = max(30.0, float(np.percentile(dist, 70)))
    m = (dist > thr).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lbl, stats, cent = cv2.connectedComponentsWithStats(m)
    best, best_score = None, 0.0
    for k in range(1, n):
        area = stats[k, cv2.CC_STAT_AREA]
        if area < 400:
            continue
        cx = cent[k][0]
        score = area / (1.0 + abs(cx - cfg["jaw_x"]) / 200.0)
        if score > best_score:
            best, best_score = k, score
    if best is None:
        return None
    ys, xs = np.nonzero(lbl == best)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    (cx, cy), (rw, rh), _ = cv2.minAreaRect(pts)
    area = len(xs)
    contrast = float(dist[lbl == best].mean())
    conf = min(1.0, contrast / 60.0) * min(1.0, area / 3000.0)
    if area > 0.6 * up.shape[0] * up.shape[1]:
        conf *= 0.3                     # most of the picture: the desk itself, not an item
    ratio = max(rw, rh) / max(1.0, min(rw, rh))
    if shape == "round" or (shape is None and ratio < cfg["round_ratio"]):
        r = point_reading(float(cx), "opencv", min(conf * 0.9, cfg["compact_cv_conf"]), cfg, opening=opening)
        return extents(r, float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()), cfg)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    conf *= 0.7                         # one blob is less sure than two clean edges
    return line_reading((x0 + 100 * vx, y0 + 100 * vy), (x0 - 100 * vx, y0 - 100 * vy), "opencv", conf, cfg, opening)


# Gemini: one call, the two ends of the long side and the box
def gemini_prompt(name, shape=None):
    what = f"the {name}" if name else "the object nearest the middle of the picture"
    if shape == "round":
        # a compact item is often close: cut off by the bottom edge, between the jaw tips. Told to
        # ignore everything at the bottom edge, Gemini would miss it.
        return (
            "This photo is from a small camera on a robot arm's gripper, held just above a desk and looking down and "
            "ahead. The two dark grey plastic jaw tips of the gripper stick up from the bottom edge of the picture; they "
            f"are not the object. Find {what} lying on the desk. It may be close to the camera, partly cut off by the "
            "bottom edge of the picture, or between or behind the jaw tips. Give the box of the part of it you can see, "
            "not including the jaws. If you see only the jaws and the desk, answer found false. "
            'Reply with JSON only: {"found": true or false, "box": [ymin, xmin, ymax, xmax], "ends": null, '
            '"long": false, "confidence": 0 to 1} with every number in 0 to 1000 units of the image height and width.')
    return (
        "This photo is from a small camera on a robot arm's gripper, held just above a desk and looking down and "
        "ahead. The two grey plastic jaw tips of the gripper are at the bottom edge of the picture: ignore them. "
        f"Find {what} lying on the desk. Give its box, and if it is longer than it is wide, the two ends of its long "
        "side, as points on the middle line of the object (where it ends, or where it leaves the picture). "
        'Reply with JSON only: {"found": true or false, "box": [ymin, xmin, ymax, xmax], '
        '"ends": [[y1, x1], [y2, x2]] or null, "long": true or false, "confidence": 0 to 1} '
        "with every number in 0 to 1000 units of the image height and width.")


def parse_gemini(raw, shape=None, cfg=AIM, opening=None):
    """A reading from Gemini's answer, or None."""
    from .gemini import parse_box
    from .scene import _ends
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        raw = raw[0]
    if not isinstance(raw, dict) or raw.get("found") is False:
        return None
    w, h = cfg["size"]
    box = parse_box(raw.get("box") if raw.get("box") is not None else raw.get("box_2d"))
    ends = raw.get("ends")
    if isinstance(ends, list) and len(ends) == 2 and all(isinstance(p, list) and len(p) == 4 for p in ends):
        # each end answered as a small box: its middle, unless a "box" is most of the item (a wrong answer)
        small = all(isinstance(c, (int, float)) for p in ends for c in p) and             all(abs(p[2] - p[0]) <= 150 and abs(p[3] - p[1]) <= 150 for p in ends)
        ends = [[(p[0] + p[2]) / 2, (p[1] + p[3]) / 2] for p in ends] if small else None
    ends = _ends(ends)
    conf = raw.get("confidence")
    conf = float(conf) if isinstance(conf, (int, float)) and not isinstance(conf, bool) else 0.7
    if conf > 1.0:
        conf /= 100.0                   # answered as a percentage
    is_long = raw.get("long")
    if shape == "round":
        is_long = False
    elif shape == "long":
        is_long = True
    if ends is not None and is_long is not False:
        (y1, x1), (y2, x2) = ends
        p1, p2 = (x1 / 1000 * w, y1 / 1000 * h), (x2 / 1000 * w, y2 / 1000 * h)
        if math.hypot(p1[0] - p2[0], p1[1] - p2[1]) < 15:
            ends = None
        else:
            if box is not None:
                ymin, xmin, ymax, xmax = box
                pad = 80
                inside = all(xmin - pad <= x <= xmax + pad and ymin - pad <= y <= ymax + pad for y, x in ends)
                if not inside:
                    conf *= 0.5         # the ends do not sit in the box: one of them is wrong
            return line_reading(p1, p2, "gemini", conf, cfg, opening)
    if box is None:
        return None
    if shape == "long":
        conf *= 0.4                     # a long item with no ends: its angle is unknown, do not aim by it
    ymin, xmin, ymax, xmax = box
    if shape != "long" and (xmax - xmin) / 1000 * w > cfg["compact_max_width_px"]:
        conf *= 0.4                     # wider than the jaws can take: on an empty desk it boxed the jaws
    r = point_reading((xmin + xmax) / 2000 * w, "gemini", conf, cfg, opening=opening)
    if shape == "long":
        return r
    return extents(r, xmin / 1000 * w, ymin / 1000 * h, xmax / 1000 * w, ymax / 1000 * h, cfg)


def read_gemini(img, name=None, shape=None, timeout=None, cfg=AIM, opening=None):
    """One Gemini question about the picture; the models are tried in turn inside one time limit.
    Returns a reading or None. Raises GeminiError when no model answers."""
    img = _upright(img, cfg)
    limit = cfg["gemini_timeout_s"] if timeout is None else timeout
    t0, last = time.time(), None
    models = list(dict.fromkeys(gemini.MODELS))      # each model once
    for k, model in enumerate(models):
        left = limit - (time.time() - t0)
        if left < 1.0:
            break
        # the first model gets most of the time but leaves some for the next (a busy model often hangs)
        slice_s = left if k == len(models) - 1 else max(1.0, 0.65 * left)
        try:
            raw = gemini.ask_json(img, gemini_prompt(name, shape), size=cfg["size"], timeout=slice_s, models=(model,))
            return parse_gemini(raw, shape, cfg, opening)
        except gemini.GeminiError as e:
            last = e
    raise last or gemini.GeminiError("no time left to ask Gemini")


def read_aim(img, name=None, shape="auto", use_gemini=True, cfg=AIM, ask=None, log=None, opening=None):
    """The reading to aim by: OpenCV when it is sure, else Gemini when it is sure, else None.
    The returned reading also carries "tried": what each reader said. log: a dict that gets the
    same, also when nothing was sure. ask: stands in for read_gemini in tests. opening: the
    gripper's ticks when the picture was taken (where the jaw tips are)."""
    if shape == "auto":
        shape = shape_for(name)
    tried = log if log is not None else {}
    cv = None
    try:
        cv = read_cv(img, shape, cfg, opening)
    except Exception as e:             # a broken picture must not stop the pick
        tried["opencv_error"] = f"{type(e).__name__}: {str(e)[:80]}"
    tried["opencv"] = cv
    # A reading far from the jaws is some other thing in the picture (a pen lying next to the
    # scissors), not what the arm is hovering over. Never aim by it.
    def near(r):
        return r is not None and abs(r.get("offset_px") or 0.0) <= cfg.get("max_offset_px", 260)

    if cv is not None and not near(cv):
        tried["opencv_far"] = cv.get("offset_px")
        cv = None
    if cv is not None and cv["confidence"] >= cfg["cv_min_conf"]:
        return {**cv, "tried": dict(tried)}
    if use_gemini:
        try:
            g = (ask or read_gemini)(img, name=name, shape=shape, cfg=cfg, **({"opening": opening} if opening is not None else {}))
        except Exception as e:
            g = None
            tried["gemini_error"] = f"{type(e).__name__}: {str(e)[:120]}"
        tried["gemini"] = g
        if g is not None and near(g) and g["confidence"] >= cfg["gemini_min_conf"]:
            return {**g, "tried": dict(tried)}
    return None


def draw(img, reading, cfg=AIM):
    """A copy of the picture (640 x 360) with the jaw centre, the jaw row and the line found."""
    import cv2
    out = _upright(img, cfg).copy()
    jx, jr = int(cfg["jaw_x"]), int(cfg["jaw_row"])
    cv2.line(out, (jx, 0), (jx, out.shape[0]), (0, 200, 0), 1)
    cv2.line(out, (0, jr), (out.shape[1], jr), (0, 200, 0), 1)
    cv2.line(out, (0, cfg["item_rows"]), (out.shape[1], cfg["item_rows"]), (0, 120, 0), 1)
    text = "not read"
    if reading and not reading.get("unread"):
        if reading.get("line"):
            (x1, y1), (x2, y2) = reading["line"]
            dx, dy = x2 - x1, y2 - y1
            k = 1000.0 / max(1e-6, math.hypot(dx, dy))
            cv2.line(out, (int(x1 - dx * k), int(y1 - dy * k)), (int(x1 + dx * k), int(y1 + dy * k)), (0, 0, 255), 2)
        cv2.circle(out, (int(reading["x_at_jaws"]), jr), 6, (0, 0, 255), 2)
        if reading.get("compact") and reading.get("left_x") is not None:
            cv2.rectangle(out, (int(reading["left_x"]), int(reading["top_y"])),
                          (int(reading["right_x"]), int(reading["bottom_y"])), (255, 0, 255), 2)
        ang = "round" if reading["angle_deg"] is None else f"{reading['angle_deg']:+.0f} deg"
        text = f"{reading['source']} {ang} off {reading['offset_px']:+.0f}px conf {reading['confidence']:.2f}"
        if reading.get("compact") and reading.get("width_px") is not None:
            text += f" w {reading['width_px']:.0f} y {reading['centre_y']:.0f}" + (" cut" if reading.get("touches_bottom") else "")
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return out


def main(argv=None):
    """Read saved wrist pictures and print what each reader says:
        python -m backend.vision.wrist_aim setup/wrist_samples/*.jpg --name sharpie [--gemini]
    With --draw, a copy with the reading drawn on it is written next to each as NAME_read.jpg."""
    import argparse
    import glob
    from pathlib import Path
    import cv2
    ap = argparse.ArgumentParser(description="Read wrist pictures for pick's last aim.")
    ap.add_argument("pictures", nargs="+")
    ap.add_argument("--name", default="sharpie")
    ap.add_argument("--gemini", action="store_true", help="also ask Gemini (one call per picture)")
    ap.add_argument("--timeout", type=float, default=AIM["gemini_timeout_s"])
    ap.add_argument("--draw", action="store_true", help="write NAME_read.jpg with the reading drawn on it")
    ap.add_argument("--opening", type=int, default=None, help="gripper ticks when the picture was taken")
    a = ap.parse_args(argv)
    shape = shape_for(a.name)
    keep = ("angle_deg", "x_at_jaws", "offset_px", "between_jaws", "confidence", "width_px", "centre_y",
            "touches_bottom")
    for pat in a.pictures:
        for f in sorted(glob.glob(pat)) or [pat]:
            img = cv2.imread(f)
            if img is None:
                print(f, "cannot open")
                continue
            cv = read_cv(img, shape, opening=a.opening)
            print(f, "opencv:", None if cv is None else {k: cv[k] for k in keep if k in cv})
            shown = cv
            if a.gemini:
                t0 = time.time()
                try:
                    g = read_gemini(img, a.name, shape, timeout=a.timeout, opening=a.opening)
                    print(f, f"gemini ({time.time() - t0:.1f} s):", None if g is None else {k: g[k] for k in keep if k in g})
                    shown = g if (cv is None or cv["confidence"] < AIM["cv_min_conf"]) and g else shown
                except Exception as e:
                    print(f, "gemini failed:", type(e).__name__, str(e)[:100])
            if a.draw and not Path(f).stem.endswith("_read"):
                p = Path(f)
                cv2.imwrite(str(p.with_name(p.stem + "_read.jpg")), draw(img, shown))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
