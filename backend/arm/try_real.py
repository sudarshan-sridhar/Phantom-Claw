"""Run one skill on the real SO-101, by hand, standing next to the arm.

From the project root (C:\\Users\\sudar\\projects\\third-hand):
    C:\\Users\\sudar\\.conda\\envs\\lerobot\\python.exe -m backend.arm.try_real rest
    ... -m backend.arm.try_real half_reach --pace 0
    ... -m backend.arm.try_real point --pace 0
    ... -m backend.arm.try_real grip_bottle --pace 0.5 --hold 5 [--flourish] [--look-first]
    ... -m backend.arm.try_real fetch_bottle --pace 0 --wait 6 [--look-first] [--no-wrist]
    ... -m backend.arm.try_real fetch_bottle --pace 0 --plan --look   (cameras and Gemini only, no port)
    ... -m backend.arm.try_real wave
    ... -m backend.arm.try_real gesture --style king --seconds 4
    ... -m backend.arm.try_real gesture --style escanor --seconds 4 --plan   (print the plan, do not move)

The general skills (several in a row share what the arm is holding):
    ... -m backend.arm.try_real look --plan --look                  (cameras and Gemini only, no port)
    ... -m backend.arm.try_real pick --target "red bull" --plan --look
    ... -m backend.arm.try_real point_at --target "the cup" --pace 0
    ... -m backend.arm.try_real pick place --target "water bottle" --where back --pace 0
    ... -m backend.arm.try_real pick place --target bottle --near cup --side right --approach look_first
    ... -m backend.arm.try_real pick hand_over --target sharpie --wait 5 --lift-style high
    --target and --where also take a desk position "x,y" in cm from the arm base (x right, y away).

Any pose is fine to start from: if the arm is not at rest it goes to rest first (and puts a held
bottle down on the table). Press Ctrl+C to stop: the arm holds where it is.
"""
import argparse
import math
import sys
import threading
import time
from pathlib import Path

from . import skills as S


def show_plan(skill, params):
    print(f"skill {skill} {params}, about {S.nominal_duration(skill, params):.1f} s")
    if skill == "gesture":
        g = S.gesture_plan(params["style"], params["seconds"])
        print(f"  rise {g['rise_s']:.2f} s (asked {g['rise_cmd']:.2f}) to f={g['f']:.2f} pose {g['pose']} gripper {g['grip']}, "
              f"talk {g['talk_s']:.2f} s, back {g['back_s']:.2f} s (asked {g['back_cmd']:.2f}), "
              f"whole gesture {g['rise_s'] + g['talk_s'] + g['back_s']:.2f} s for a {params['seconds']:.2f} s line")
        base = {**g["pose"], 5: 2030, 6: g["grip"]}
        for st in S.gesture_steps(params["style"], g["talk_s"], base, g["flex_anchor"]):
            print("  t=%5.2f  joint %d -> %4d  speed %4d  acc %d" % st, "(wrist roll shown from 2030)" if st[1] == 5 else "")
    elif skill == "fetch_bottle":
        k = S.fetch_stretch(params["pace"])
        print(f"  every leg x{k:.2f} of the script's time: out {S.FETCH_OUT_S * k:.2f} s, final approach "
              f"{S.FETCH_APPROACH_S * k:.2f} s, tuck {S.FETCH_TUCK_S * k:.2f} s, reach {S.FETCH_REACH_S * k:.2f} s, "
              f"back to rest {S.FETCH_REST_S * k:.2f} s; offers it for {params['wait_s']:g} s")
        for out in ("handed_over", "put_back", "missed", "turn_failed"):
            print(f"  {out:<12} about {S.fetch_nominal(params['pace'], params['wait_s'], params['look_first'], out):.0f} s")
    elif skill in S.GENERAL:
        show_general(skill, params)
    elif "pace" in params:
        print(f"  leg time {S.leg_time(params['pace']):.2f} s at pace {params['pace']}")


OUTCOMES = {"look": ["done"], "point_at": ["done", "not_found"], "pick": ["holding", "missed", "not_taught"],
            "place": ["placed", "out_of_reach"], "hand_over": ["handed_over", "not_taken"]}


def show_general(skill, params):
    """What a general skill would do, from the files alone (no port, no camera)."""
    from backend.vision.table import load_tiers
    k = S.fetch_stretch(params.get("pace", 0.5))
    print(f"  legs x{k:.2f} of the fetch_bottle times; params {params}")
    for out in OUTCOMES[skill]:
        print(f"  {out:<12} about {S.general_nominal(skill, params, out):.0f} s")
    for name, t in load_tiers().items():
        ring = t.ring()
        print(f"  tier {name:<5} " + (f"reaches {ring['min_cm']:.1f} to {ring['max_cm']:.1f} cm from spots {ring['spots']}, "
                                      f"opens to {t.open}" if ring else "not taught (setup/tools/teach_grip.py)"))
    steps = {
        "look": "one overhead picture and one Gemini call; lists every item with its desk position and tier; no motion",
        "point_at": "look for the target, turn the base to it and reach out above it (the tall pose over it, shoulder "
                    f"{S.POINT_AT_RAISE} ticks higher), hold {S.POINT_AT_HOLD_S:g} s, back to rest",
        "pick": "look for the target, choose the tier from its name or height, refuse if untaught, too big or out of "
                "reach; out above it with the jaws open (lying items: wrist roll set across them)"
                + (", pause 1 s above it" if params.get("approach") == "look_first" else "")
                + ", line up with the wrist camera (tiers " + (", ".join(desk_aim_tiers()) or "none")
                + f": up to {aim_rounds()} looks, roll first, then pan)"
                + f", down onto it, close, lift into the carry pose ({params.get('lift_cm_style')}), check it is still held, "
                  "stay holding",
        "place": f"turn to {params.get('where')!r} with the item tucked in, down to just above the spot"
                 + (", pause 1 s" if params.get("approach") == "look_first" else "")
                 + ", stand it on the desk, open, lift clear, back to rest",
        "hand_over": f"turn to the person and hold it out, wait up to {params.get('wait_s', 3):g} s for a tug or a hand, "
                     "let go and go home; nobody takes it: tuck back into the carry pose, still holding",
    }
    print("  steps:", steps[skill])


def desk_aim_tiers():
    from . import desk
    return desk.aim_tiers(_AIM_SETTING[0])


def aim_rounds():
    from . import desk
    return desk.WRIST_AIM["rounds"]


_AIM_SETTING = [None]


def look_general(skill, params):
    """With --plan --look: what pick or point_at would do with what the overhead camera sees now."""
    import json
    from backend.vision import Vision
    from backend.vision.scene import public
    from backend.vision.table import load_tiers
    from . import desk
    target = params.get("target")
    v = Vision()
    s = v.scene(name=target if isinstance(target, str) else None)
    if s is None:
        print("overhead camera gives no picture: result would be no_picture")
        return
    for it in s["items"]:
        print("  item", json.dumps(public(it)))
    print("  person", s["person"])
    if skill not in ("pick", "point_at") or not target:
        return
    item = s["match"] if isinstance(target, str) else desk.nearest(s["items"], target["x_cm"], target["y_cm"])
    if item is None:
        print(f"result would be not_found: nothing fits {target!r}")
        return
    with open(Path(__file__).resolve().parents[2] / "setup" / "my_follower_calibration_2026-11-14.json") as f:
        cal = json.load(f)
    pan = next(c for c in cal.values() if c["id"] == 1)
    pan_range = (pan["range_min"] + 30, pan["range_max"] - 30)
    try:
        plan = desk.pick_plan(item, load_tiers(), pan_range)
    except desk.Refuse as r:
        print(f"result would be {r.result}: {r.detail}")
        return
    print(f"plan: the {item['name']} ({item['tier']}) at {plan['dist_cm']:.1f} cm, pan {plan['pan']}, via {plan['via']}, "
          f"grasp {plan['grasp']}, roll {plan['roll']}, blend {plan['t']:.2f}")


def look_only():
    """What the fetch would do, from the cameras alone. Never opens the serial port."""
    from backend.vision import Vision, Taught
    from backend.vision.table import TableMap
    v, T, tm = Vision(), Taught(), TableMap()
    print("looking with the overhead camera (index 3)...")
    seen = v.look_overhead()
    if seen is None:
        print("overhead camera gives no picture: the arm would use the wrist-camera sweep "
              "(that needs the arm to move, so it is not shown here)")
        return
    print("boxes:", seen["boxes"])
    if "bottle" not in seen:
        print("result would be no_bottle: the bottle is not seen")
        return
    B = seen["bottle"]
    if not T.in_reach(B["r_cm"]):
        print(f"result would be out_of_reach: {B['r_cm']:.1f} cm (reach {T.min_r:.0f} to {T.max_r:.0f} cm)")
        return
    via, grasp, t = T.posture(B["r_cm"])
    pan_b = int(max(734 + 30, min(3360 - 30, B["pan"])))
    pp = seen["person"]["pan"] if "person" in seen else tm.pan_for_bearing(1.2)
    pan_p = int(max(950, min(3250, pp)))
    print(f"plan: bottle at {B['r_cm']:.1f} cm (bearing {B['bearing_deg']:.1f} deg), turn to pan {pan_b}, "
          f"blend {t:.2f} between middle and far, via {via}, grasp {grasp}; then hold it out at pan {pan_p}"
          + ("" if "person" in seen else " (person not seen, going to the right)")
          + f", a turn of {abs(pan_p - pan_b) / tm.tpr * 180 / math.pi:.0f} degrees")


def _xy_or_name(s):
    if s is None:
        return None
    parts = s.split(",")
    if len(parts) == 2:
        try:
            return {"x_cm": float(parts[0]), "y_cm": float(parts[1])}
        except ValueError:
            pass
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run one arm skill (or several in a row) on the real arm.")
    ap.add_argument("skills", nargs="+", choices=list(S.SKILLS), metavar="skill",
                    help="one or more of: " + ", ".join(S.SKILLS))
    ap.add_argument("--target", help="pick, point_at, look: an item name, or x,y in cm")
    ap.add_argument("--where", default="back", help="place: back, or x,y in cm")
    ap.add_argument("--near", help="place: next to this item")
    ap.add_argument("--side", default="front", choices=list(S.SIDES), help="place --near: which side")
    ap.add_argument("--approach", default="direct", choices=list(S.CHOICES["approach"]))
    ap.add_argument("--lift-style", default="normal", choices=list(S.CHOICES["lift_cm_style"]))
    ap.add_argument("--pace", type=float, default=0.5)
    ap.add_argument("--hold", type=float, default=5.0)
    ap.add_argument("--wait", type=float, default=3.0, help="fetch_bottle: seconds it offers the bottle")
    ap.add_argument("--flourish", action="store_true")
    ap.add_argument("--look-first", action="store_true")
    ap.add_argument("--no-wrist", action="store_true", help="fetch_bottle: no wrist-camera sweep when the overhead camera is off")
    ap.add_argument("--style", default="gowther", choices=list(S.STYLES))
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--plan", action="store_true", help="print the plan only; do not open the port")
    ap.add_argument("--look", action="store_true", help="with --plan and fetch_bottle: also ask the overhead camera")
    ap.add_argument("--yes", action="store_true", help="skip the 'area clear' question")
    ap.add_argument("--aim", choices=["low", "all", "off"],
                    help="pick: which tiers line up with the wrist camera before going down (default low, "
                         "or PHANTOM_WRIST_AIM)")
    a = ap.parse_args(argv)
    _AIM_SETTING[0] = a.aim

    where = {"near": a.near, "side": a.side} if a.near else _xy_or_name(a.where)
    raw = {"pace": a.pace, "hold_s": a.hold, "wait_s": a.wait, "flourish": a.flourish, "look_first": a.look_first,
           "style": a.style, "seconds": a.seconds, "target": _xy_or_name(a.target), "where": where,
           "approach": "look_first" if a.look_first else a.approach, "lift_cm_style": a.lift_style}
    plan = [(sk, S.validate(sk, raw)) for sk in a.skills]
    for sk, params in plan:
        show_plan(sk, params)
    if a.plan:
        if a.look:
            for sk, params in plan:
                if sk == "fetch_bottle":
                    look_only()
                elif sk in ("look", "pick", "point_at"):
                    look_general(sk, params)
        return 0

    from .real import RealArm
    arm = RealArm(port=a.port, wrist_fallback=False if a.no_wrist else None, wrist_aim=a.aim)
    st = arm.status()
    print("status:", st)
    if not st["connected"]:
        print("not connected, nothing moved")
        return 1
    if not a.yes and input("Area clear and hand near the power switch? Type go and press Enter: ").strip() != "go":
        print("cancelled, nothing moved")
        arm.close()
        return 1

    t0 = time.time()
    out = {}

    def phase(name):
        print(f"{time.time() - t0:6.1f} s  {name}", flush=True)

    for sk, params in plan:
        out = {}
        print(f"--- {sk}", flush=True)
        th = threading.Thread(target=lambda: out.update(arm.run(sk, params, phase)), daemon=True)
        th.start()
        try:
            while th.is_alive():
                th.join(0.1)
        except KeyboardInterrupt:
            print("Ctrl+C: stopping, the arm holds where it is", flush=True)
            arm.stop()
            th.join(10)
            break
        print("result:", out)
        if not out.get("ok") and out.get("result") != "not_taken":
            break           # the next skill would not make sense; it can be run by hand
    st = arm.status()
    print("joints:", st["joints"], "holding:", st.get("held"))
    arm.close()
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
