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

Any pose is fine to start from: if the arm is not at rest it goes to rest first (and puts a held
bottle down on the table). Press Ctrl+C to stop: the arm holds where it is.
"""
import argparse
import math
import sys
import threading
import time

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
    elif "pace" in params:
        print(f"  leg time {S.leg_time(params['pace']):.2f} s at pace {params['pace']}")


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


def main():
    ap = argparse.ArgumentParser(description="Run one arm skill on the real arm.")
    ap.add_argument("skill", choices=list(S.SKILLS))
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
    a = ap.parse_args()

    params = S.validate(a.skill, {"pace": a.pace, "hold_s": a.hold, "wait_s": a.wait, "flourish": a.flourish,
                                  "look_first": a.look_first, "style": a.style, "seconds": a.seconds})
    show_plan(a.skill, params)
    if a.plan:
        if a.look and a.skill == "fetch_bottle":
            look_only()
        return 0

    from .real import RealArm
    arm = RealArm(port=a.port, wrist_fallback=False if a.no_wrist else None)
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

    th = threading.Thread(target=lambda: out.update(arm.run(a.skill, params, phase)), daemon=True)
    th.start()
    try:
        while th.is_alive():
            th.join(0.1)
    except KeyboardInterrupt:
        print("Ctrl+C: stopping, the arm holds where it is", flush=True)
        arm.stop()
        th.join(10)
    print("result:", out)
    print("joints:", arm.status()["joints"])
    arm.close()
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
