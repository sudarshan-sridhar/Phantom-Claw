"""Teach one grip by hand for the general pick and place skills.

Usage (from the project root, standing next to the arm, one hand on it):
    C:\\Users\\sudar\\.conda\\envs\\lerobot\\python.exe setup\\tools\\teach_grip.py TIER SPOT [--object NAME]
        [--r-cm R] [--wait S]
    TIER: tall | can | low        SPOT: near | middle | far | farthest

What happens, said out loud first each time:
1. Hold the arm: torque goes off on every joint after the beep, so the arm goes limp.
2. Open the jaws around the object where they should grip it (a lying item: across its short
   side; a cup: on the wall nearest the arm) and hold still. Beep, the pose is read.
3. Squeeze the jaws shut on it by hand and hold. Beep, the gripper is read.
4. Keep the jaws shut on it and lift it straight up off the desk, about five centimetres (a hand's
   width), without folding the arm back home. Beep, the pose above it is read, and the overhead
   camera takes a picture so Gemini can say how far the object is from the base. If the shoulder
   moved more than MAX_LIFT_TICKS from the grasp (the arm was folded home), it says so and asks
   again once; still too far, the pose above is made from the grasp instead (shoulder VIA_LIFT up).
Saved into setup/grip_teach.json under tiers TIER / SPOT. The arm is left with torque off:
put it back to rest by hand, then run any skill (it switches torque on itself).

The object's distance is what the blend uses, so it must come from the overhead picture, the
same way it is measured when the skill runs. If there is no picture, pass --r-cm by hand or the
spot is saved without a distance and does not count yet. With --r-cm the photo is still saved,
but its bearing and axis are not used (the hand is in that picture): the spot keeps the bearing
and axis it had before, if any.

This script was written, not run: check it on the arm with the area clear.
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SETUP = ROOT / "setup"
OUT = SETUP / "grip_teach.json"
NAMES = ["pan", "lift", "elbow", "wrist_flex", "wrist_roll", "gripper"]
TIERS = ("tall", "can", "stick", "low")
SPOTS = ("near", "middle", "far", "farthest")
DEFAULT_OBJECT = {"tall": "water bottle", "can": "sanitizer", "stick": "glue stick", "low": "sharpie"}
GRIP_RANGE = (1305, 2788)
MAX_LIFT_TICKS = 300      # the lifted pose may differ from the grasp by at most this much shoulder lift
VIA_LIFT = 150            # otherwise the pose above is the grasp with the shoulder this much higher
LIFT_SAY = ("Keep the jaws shut on it and lift it straight up off the desk, about five centimetres, "
            "a hand's width. Do not fold the arm back home. Hold it there, still, for the photo.")


def say(text):
    print(text, flush=True)
    t = text.replace("'", "")
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{t}')"])


def beep(freq, ms):
    try:
        import winsound
        winsound.Beep(freq, ms)
    except Exception:
        pass


class Bus:
    def __init__(self, port="COM3"):
        import scservo_sdk as scs
        self.scs = scs
        self.p, self.h = scs.PortHandler(port), scs.PacketHandler(0)
        if not self.p.openPort() or not self.p.setBaudRate(1_000_000):
            raise SystemExit(f"could not open {port}")

    def read_all(self):
        out = []
        for i in range(1, 7):
            for _ in range(6):
                v, c, _ = self.h.read2ByteTxRx(self.p, i, 56)
                if c == self.scs.COMM_SUCCESS:
                    out.append(v)
                    break
            else:
                raise RuntimeError(f"joint {i} did not answer")
        return out

    def torque_off(self):
        for _ in range(3):
            for i in range(1, 7):
                self.h.write1ByteTxRx(self.p, i, 40, 0)

    def close(self):
        self.p.closePort()


def hold_and_read(bus, text, wait):
    """Say what to do, wait, beep, then average one second of readings. Returns ({joint: value}, shake)."""
    say(text)
    time.sleep(wait)
    beep(1200, 400)
    samples, t = [], time.time()
    while time.time() - t < 1.0:
        samples.append(bus.read_all())
        time.sleep(0.03)
    mean = [round(sum(s[k] for s in samples) / len(samples)) for k in range(6)]
    shake = max(max(s[k] for s in samples) - min(s[k] for s in samples) for k in range(4))
    beep(700, 200)
    return dict(zip(NAMES, mean)), shake


def lifted_ok(grasp, above, limit=MAX_LIFT_TICKS):
    """True when the lifted pose is a short straight lift from the grasp, not the arm folded home."""
    return abs(above["lift"] - grasp["lift"]) <= limit


def via_from_grasp(grasp, up=VIA_LIFT):
    return {"lift": grasp["lift"] - up, "elbow": grasp["elbow"], "wrist_flex": grasp["wrist_flex"]}


def look_overhead(obj, tier, spot, ask=True):
    """(distance cm, bearing deg, axis deg or None, note) of the object from the overhead picture, or Nones.
    ask=False only saves the picture (nothing is read from it)."""
    try:
        import cv2
        from backend.vision import Vision, cameras
        from backend.vision.scene import grip_point
        img = cameras.grab_overhead(cameras.OVERHEAD_INDEX)
        if img is None:
            return None, None, None, "the overhead camera gives no picture"
        cv2.imwrite(str(SETUP / f"teach_grip_{tier}_{spot}_overhead.jpg"), img)
        if not ask:
            return None, None, None, "photo saved; nothing read from it"
        item = Vision().find(obj, img=img)
        if item is None:
            return None, None, None, f"Gemini did not find the {obj} in the picture"
        at = item.get("grip_at") or item       # a cup: the rim nearest the arm, as pick will aim
        return at["dist_cm"], at["bearing"], item.get("axis_deg"), f"seen as {item['name']!r}, tier {item['tier']}"
    except Exception as e:
        return None, None, None, f"the overhead look failed ({type(e).__name__}: {str(e)[:100]})"


def pose3(p):
    return {"lift": p["lift"], "elbow": p["elbow"], "wrist_flex": p["wrist_flex"]}


def main():
    ap = argparse.ArgumentParser(description="Teach one grip by hand for pick and place.")
    ap.add_argument("tier", choices=TIERS)
    ap.add_argument("spot", choices=SPOTS)
    ap.add_argument("--object", help="what is being gripped, as Gemini should find it (default per tier)")
    ap.add_argument("--r-cm", type=float, help="the object's distance from the base, if there is no overhead picture")
    ap.add_argument("--wait", type=float, default=12.0, help="seconds to get each pose right")
    ap.add_argument("--port", default="COM3")
    a = ap.parse_args()
    obj = a.object or DEFAULT_OBJECT[a.tier]

    bus = Bus(a.port)
    bus.read_all()
    say(f"{obj}, {a.spot}. Hold the arm.")
    time.sleep(1)
    beep(1000, 300)
    bus.torque_off()
    try:
        # Two holds only: jaws on it, then squeezed shut. The pose above is worked out from the grip.
        grasp, s1 = hold_and_read(bus, "Jaws on it.", a.wait)
        shut, s2 = hold_and_read(bus, "Squeeze.", a.wait * 0.6)
        above, s3 = {**grasp, **via_from_grasp(grasp)}, 0
    finally:
        bus.close()
    via_taught = None
    if not lifted_ok(grasp, above):
        via_taught = pose3(above)
        say("Still too far. I will make the pose above from the grip instead.")
        print(f"lifted pose rejected: shoulder {above['lift']} vs grasp {grasp['lift']} (more than {MAX_LIFT_TICKS} apart)")
        above = {**above, **via_from_grasp(grasp)}
    hand_r = a.r_cm is not None
    # with a distance given by hand the hand is in the photo: save it, but read nothing from it
    r, bearing, axis, note = look_overhead(obj, a.tier, a.spot, ask=not hand_r)
    if hand_r:
        r, note = a.r_cm, f"{note}; distance given by hand"
    print(note)

    data = json.load(open(OUT)) if OUT.exists() else {}
    data.setdefault("note", "Grips taught by hand with setup/tools/teach_grip.py for the general pick and place skills. "
                            "Raw ticks. r_cm is the object's distance from the arm base as Gemini reads it from the "
                            "overhead picture (for a cup, its rim nearest the arm). tall spots here add to the ones in "
                            "bottle_teach.json.")
    tier = data.setdefault(a.tier, {})
    before = tier.get(a.spot) if isinstance(tier.get(a.spot), dict) else {}
    spot = {"r_cm": None if r is None else round(r, 1), "pan": grasp["pan"], "via": pose3(above), "grasp": pose3(grasp),
            "roll": grasp["wrist_roll"], "gripper_open": grasp["gripper"], "gripped_at": shut["gripper"],
            "object": obj, "shake_ticks": max(s1, s2, s3), "taught": datetime.now().isoformat(timespec="seconds"),
            "photo": f"teach_grip_{a.tier}_{a.spot}_overhead.jpg", "look": note}
    if via_taught is not None:
        spot["via_taught"] = via_taught          # what the hand gave, kept for a look; not used
    if hand_r:
        # keep the bearing and axis from before: the photo with the hand in it is not trusted for them
        for k in ("bearing_deg", "axis_deg", "rel_axis_deg"):
            if k in before:
                spot[k] = before[k]
        bearing = axis = None
    if bearing is not None:
        spot["bearing_deg"] = round(bearing, 1)
    if axis is not None and bearing is not None:
        # the item's angle relative to the arm's line of sight when it was taught; pick works the
        # wrist roll zero out from this and the roll the hand chose (backend/arm/desk.py jaw_roll)
        rel = axis - bearing
        while rel > 90:
            rel -= 180
        while rel <= -90:
            rel += 180
        spot["axis_deg"] = round(axis, 1)
        spot["rel_axis_deg"] = round(rel, 1)
    tier[a.spot] = spot
    # tier-wide gripper values: open a little wider than the hand did, closed where the hand squeezed
    opens = [v["gripper_open"] for k, v in tier.items() if k in SPOTS and isinstance(v, dict) and "gripper_open" in v]
    shuts = [v["gripped_at"] for k, v in tier.items() if k in SPOTS and isinstance(v, dict) and "gripped_at" in v]
    if a.tier != "tall" or "open" in tier:
        tier["open"] = int(min(GRIP_RANGE[1] - 10, round(sum(opens) / len(opens)) + 30))
    tier["closed"] = int(round(sum(shuts) / len(shuts)))
    json.dump(data, open(OUT, "w"), indent=1)
    print(json.dumps({a.tier: {a.spot: spot}}, indent=1))
    if r is None:
        say("Saved, but without a distance. This spot does not count until it has one. You can let go of the arm.")
    else:
        say(f"Saved. The {obj} is {r:.0f} centimetres from the base. Put the arm back to rest by hand.")
    taught = sorted(v["r_cm"] for k, v in tier.items() if k in SPOTS and isinstance(v, dict) and v.get("r_cm"))
    if taught:
        print(f"{a.tier} tier now reaches {taught[0] - 4:.1f} to {taught[-1] + 2:.1f} cm"
              + (" (with bottle_teach.json mid and far added)" if a.tier == "tall" else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
