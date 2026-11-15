"""Table mapping from the four ArUco markers and the touch round.
- Builds a flat table frame in centimetres from the overhead picture (origin at marker 0's centre).
- Finds where the arm's base axis is on the table and which pan reading points where,
  from the pan values recorded while the gripper tip was held on each marker.
Saves setup/table_map.json. Usage: python table_map.py [image]"""
import json, sys
import cv2, numpy as np

D = r"C:\Users\sudar\projects\third-hand\setup"
SIDE = 8.0   # marker side, cm
TICKS_PER_RAD = 4096 / (2 * np.pi)

def grab():
    cap = cv2.VideoCapture(3, cv2.CAP_DSHOW); cap.set(3, 1920); cap.set(4, 1080)
    f = None
    for _ in range(12):
        ok, fr = cap.read()
        if ok: f = fr
    cap.release()
    return f

det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters())
# The arm covers one marker in any single picture, so read the markers from several pictures taken
# with the camera in the same place (the four touch photos, or one given on the command line) and average.
imgs = [cv2.imread(a) for a in sys.argv[1:]] or [cv2.imread(rf"{D}\marker_touch_{m}.jpg") for m in range(4)]
img = imgs[0]
seen = {}
for im in imgs:
    corners, ids, _ = det.detectMarkers(im)
    if ids is None:
        continue
    for c, i in zip(corners, ids.flatten()):
        seen.setdefault(int(i), []).append(c[0].astype(np.float64))
px = {m: np.mean(v, axis=0) for m, v in seen.items()}
print("marker sightings:", {m: len(v) for m, v in sorted(seen.items())})
assert len(px) >= 3, f"need at least three markers, saw {sorted(px)}"

# first guess: a metric frame from marker 0 alone, then refine with all four markers as rigid 8 cm squares
sq = np.array([[-SIDE / 2, SIDE / 2], [SIDE / 2, SIDE / 2], [SIDE / 2, -SIDE / 2], [-SIDE / 2, -SIDE / 2]])
H, _ = cv2.findHomography(px[0], sq)
def to_table(H, pts):
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
for _ in range(4):
    src, dst = [], []
    for m in sorted(px):
        t = to_table(H, px[m])
        c = t.mean(0)
        # best rigid 8 cm square through these four points (rotation from the corner directions)
        ang = np.mean([np.arctan2(*(t[k] - c)[::-1]) - np.arctan2(*sq[k][::-1]) for k in range(4)] if m else [0.0])
        if m:
            d = [(np.arctan2(*(t[k] - c)[::-1]) - np.arctan2(*sq[k][::-1]) + np.pi) % (2 * np.pi) - np.pi for k in range(4)]
            ang = float(np.mean(d))
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        fit = (sq @ R.T) + (c if m else 0)
        src.append(px[m]); dst.append(fit)
    H, _ = cv2.findHomography(np.vstack(src), np.vstack(dst))
centre = {m: to_table(H, px[m]).mean(0) for m in px}
sides = {m: float(np.mean([np.linalg.norm(to_table(H, px[m])[k] - to_table(H, px[m])[(k + 1) % 4]) for k in range(4)])) for m in px}
# a marker hidden under the arm: use its centre from the earlier clean picture (pixels at 1280x720)
EARLIER_CENTRE_PX = {3: (276.0, 129.0)}
for m, p in EARLIER_CENTRE_PX.items():
    if m not in centre:
        centre[m] = to_table(H, np.array([p]))[0]
        sides[m] = float('nan')
print("marker centres on the table, cm (x to the right in the picture, y away from the arm):")
for m in (0, 1, 2, 3):
    print(f"  {m}: x {centre[m][0]:6.1f}  y {centre[m][1]:6.1f}   measured side {sides[m]:.2f} cm (true 8.00)")
print(f"  0 to 1: {np.linalg.norm(centre[1] - centre[0]):.1f} cm   3 to 2: {np.linalg.norm(centre[2] - centre[3]):.1f} cm   0 to 3: {np.linalg.norm(centre[3] - centre[0]):.1f} cm   1 to 2: {np.linalg.norm(centre[2] - centre[1]):.1f} cm")

touch = json.load(open(D + r"\marker_touch.json"))
pan = {int(k): v["pan"] for k, v in touch.items()}

def solve(points):   # points: {id: table xy}. Unknowns: base x, base y, pan reading that points along +y
    ids_ = sorted(points)
    best = None
    for bx in np.arange(-10, 60, 0.5):
        for by in np.arange(-45, 5, 0.5):
            ang = np.array([np.arctan2(points[m][0] - bx, points[m][1] - by) for m in ids_])
            p0 = np.mean([pan[m] - TICKS_PER_RAD * a for m, a in zip(ids_, ang)])
            err = np.array([pan[m] - (p0 + TICKS_PER_RAD * a) for m, a in zip(ids_, ang)])
            s = float((err ** 2).sum())
            if best is None or s < best[0]:
                best = (s, bx, by, p0, err)
    return best

pts = {m: centre[m] for m in (0, 1, 2)}
s, bx, by, p0, err = solve(pts)
base = np.array([bx, by])
# marker 3: only its nearest part could be reached, so use its corner closest to the base
pts[3] = centre[3]   # second touch round reached the centre of marker 3
s, bx, by, p0, err = solve(pts)
base = np.array([bx, by])
print(f"arm base axis on the table: x {bx:.1f}  y {by:.1f} cm;  pan {p0:.0f} points straight away from the arm")
for m in (0, 1, 2, 3):
    r = np.linalg.norm(pts[m] - base)
    print(f"  touch {m}: reach {r:5.1f} cm, pan error {err[sorted(pts).index(m)] * 360 / 4096:+.1f} deg, joints lift {touch[str(m)]['lift']} elbow {touch[str(m)]['elbow']} wrist {touch[str(m)]['wrist_flex']}")
json.dump({"homography_px_to_cm": H.tolist(), "image_size": [img.shape[1], img.shape[0]],
           "marker_centres_cm": {str(m): centre[m].tolist() for m in centre}, "base_cm": [bx, by], "pan_zero": float(p0),
           "ticks_per_rad": TICKS_PER_RAD, "touch_points_cm": {str(m): pts[m].tolist() for m in pts}, "touch_joints": touch},
          open(D + r"\table_map.json", "w"), indent=1)

