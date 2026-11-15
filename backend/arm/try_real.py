"""Run one skill on the real SO-101, by hand, standing next to the arm.

From the project root (C:\\Users\\sudar\\projects\\third-hand):
    C:\\Users\\sudar\\.conda\\envs\\lerobot\\python.exe -m backend.arm.try_real rest
    ... -m backend.arm.try_real half_reach --pace 0
    ... -m backend.arm.try_real point --pace 0
    ... -m backend.arm.try_real grip_bottle --pace 0.5 --hold 5 [--flourish] [--look-first]
    ... -m backend.arm.try_real wave
    ... -m backend.arm.try_real gesture --style king --seconds 4
    ... -m backend.arm.try_real gesture --style escanor --seconds 4 --plan   (print the plan, do not move)

Every skill except rest must start from rest. Press Ctrl+C to stop: the arm holds where it is.
"""
import argparse
import sys
import threading
import time

from . import skills as S


def show_plan(skill, params):
    print(f"skill {skill} {params}, about {S.nominal_duration(skill, params):.1f} s")
    if skill == "gesture":
        g = S.gesture_plan(params["style"], params["seconds"])
        print(f"  rise {g['rise_s']:.2f} s to f={g['f']:.2f} pose {g['pose']} gripper {g['grip']}, "
              f"talk {g['talk_s']:.2f} s, back {g['back_s']:.2f} s")
        base = {**g["pose"], 5: 2030, 6: g["grip"]}
        for st in S.gesture_steps(params["style"], g["talk_s"], base, g["flex_anchor"]):
            print("  t=%5.2f  joint %d -> %4d  speed %4d  acc %d" % st, "(wrist roll shown from 2030)" if st[1] == 5 else "")
    elif "pace" in params:
        print(f"  leg time {S.leg_time(params['pace']):.2f} s at pace {params['pace']}")


def main():
    ap = argparse.ArgumentParser(description="Run one arm skill on the real arm.")
    ap.add_argument("skill", choices=list(S.SKILLS))
    ap.add_argument("--pace", type=float, default=0.5)
    ap.add_argument("--hold", type=float, default=5.0)
    ap.add_argument("--flourish", action="store_true")
    ap.add_argument("--look-first", action="store_true")
    ap.add_argument("--style", default="gowther", choices=list(S.STYLES))
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--plan", action="store_true", help="print the plan only; do not open the port")
    ap.add_argument("--yes", action="store_true", help="skip the 'area clear' question")
    a = ap.parse_args()

    params = S.validate(a.skill, {"pace": a.pace, "hold_s": a.hold, "flourish": a.flourish,
                                  "look_first": a.look_first, "style": a.style, "seconds": a.seconds})
    show_plan(a.skill, params)
    if a.plan:
        return 0

    from .real import RealArm
    arm = RealArm(port=a.port)
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
