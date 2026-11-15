"""Fit the arm's reach maths from taught poses.
Model: three links in the vertical plane through the base axis. Each joint angle is a straight line
in its raw reading (4096 ticks per turn); the unknowns are the three link lengths, each joint's zero,
the shoulder's height above the table and its distance in front of the base axis.
Data: the four marker touches (tip on the table) and the taught bottle grips (jaws at cap height).
Saves setup/arm_model.json and prints how well each taught pose is reproduced."""
import json, sys
import numpy as np
from scipy.optimize import least_squares

D = r"C:\Users\sudar\projects\third-hand\setup"
K = 2 * np.pi / 4096
tm = json.load(open(D + r"\table_map.json"))
base = np.array(tm["base_cm"])
rows = []   # (name, lift, elbow, wflex, r_cm, z_cm, is_grip, weight)
for m, j in tm["touch_joints"].items():
    p = np.array(tm["touch_points_cm"][m])
    rows.append((f"touch {m}", j["lift"], j["elbow"], j["wrist_flex"], float(np.linalg.norm(p - base)), 0.0, 0, 1.0))
CAP_Z = 19.0   # height of the jaws on the bottle cap, cm (500 ml bottle is about 20 cm tall)
bt = json.load(open(D + r"\bottle_teach.json"))
for name in ("near", "mid", "far"):
    if name in bt and isinstance(bt[name], dict) and bt[name].get("r_cm"):
        g = bt[name]["grasp"]
        rows.append((f"grip {name}", g["lift"], g["elbow"], g["wrist_flex"], bt[name]["r_cm"], CAP_Z, 1, 0.6))

def fk(p, lift, elbow, wf, grip):
    a, h, L1, L2, L3, o2, o3, o4, d = p
    f1 = K * (lift - o2); f2 = f1 + K * (elbow - o3); f3 = f2 + K * (wf - o4)
    L3e = L3 - (d if grip else 0.0)
    r = a + L1 * np.sin(f1) + L2 * np.sin(f2) + L3e * np.sin(f3)
    z = h + L1 * np.cos(f1) + L2 * np.cos(f2) + L3e * np.cos(f3)
    return r, z, np.degrees(f3)

PRIOR = np.array([3.0, 11.0, 11.6, 13.5, 16.0, 2046, 2046, 2046, 3.0])
SCALE = np.array([3.0, 3.0, 1.0, 1.0, 2.5, 500, 500, 500, 2.0])
def resid(p):
    out = []
    for name, lift, elbow, wf, r, z, grip, w in rows:
        rr, zz, _ = fk(p, lift, elbow, wf, grip)
        out += [w * (rr - r), w * (zz - z)]
    out += list(0.15 * (p - PRIOR) / SCALE)   # gentle pull toward plausible sizes
    return out

best = None
rng = np.random.default_rng(0)
for trial in range(60):
    x0 = PRIOR + rng.normal(0, 1, 9) * SCALE * (0 if trial == 0 else 1)
    s = least_squares(resid, x0, loss="soft_l1", f_scale=1.0)
    if best is None or s.cost < best.cost:
        best = s
p = best.x
print("fit: base offset %.1f, shoulder height %.1f, links %.1f %.1f %.1f cm, zeros %.0f %.0f %.0f, grip point %.1f cm back from the tip" % tuple(p))
worst = 0
for name, lift, elbow, wf, r, z, grip, w in rows:
    rr, zz, pitch = fk(p, lift, elbow, wf, grip)
    e = float(np.hypot(rr - r, zz - z)); worst = max(worst, e)
    print(f"  {name:10} wanted r {r:5.1f} z {z:5.1f}   model r {rr:5.1f} z {zz:5.1f}   off by {e:4.1f} cm   gripper angle {pitch:5.0f} deg from straight up")
print(f"worst error {worst:.1f} cm over {len(rows)} poses")
json.dump({"params": dict(zip(["a", "h", "L1", "L2", "L3", "o2", "o3", "o4", "grip_back"], map(float, p))), "worst_cm": worst,
           "poses_used": [r[0] for r in rows]}, open(D + r"\arm_model.json", "w"), indent=1)
