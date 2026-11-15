"""Find a pen-like object in an upright wrist camera frame.
Returns (cx, cy, angle_deg, length, width) in a 640x360 frame, or None.
The pen is near white or lavender; this camera sees the wood as grey, a little darker and less blue.
Glare patches pass the colour test, so the shape decides: a pen is long and thin."""
import cv2
import numpy as np


def find_pen(img):
    img = cv2.resize(img, (640, 360))
    b, g, r = [c.astype(np.int16) for c in cv2.split(img)]
    bright = (b + g + r) // 3
    mask = ((bright > 240) | (((b - r) > 35) & (bright > 190))).astype(np.uint8) * 255
    mask[250:, 180:460] = 0  # the gripper jaws
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    best = None
    for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        if cv2.contourArea(c) < 700:
            continue
        blob = np.zeros_like(mask)
        cv2.drawContours(blob, [c], -1, 255, -1)
        ys, xs = np.nonzero(blob)
        pts = np.stack([xs, ys], 1).astype(np.float64)
        mean = pts.mean(0)
        vals, vecs = np.linalg.eigh(np.cov((pts - mean).T))   # spread along the short and long axes
        length, width = 3.46 * np.sqrt(vals[1]), 3.46 * np.sqrt(max(vals[0], 1e-6))
        if length / width < 2.8 or length < 90:
            continue
        vx, vy = vecs[:, 1]
        ang = (np.degrees(np.arctan2(vy, vx)) + 90) % 180 - 90   # long axis, 0 = along image x
        if best is None or length > best[3]:
            best = (mean[0], mean[1], ang, length, width)
    return best


if __name__ == "__main__":
    import sys
    for f in sys.argv[1:]:
        r = find_pen(cv2.imread(f))
        print(f.split("\\")[-1], "none" if r is None else "cx %.0f cy %.0f angle %.0f len %.0f width %.0f" % r)
