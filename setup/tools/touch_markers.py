"""Record where the four table markers are in the arm's own joints.
Torque is off. A voice calls each marker; bring the closed gripper tip to its centre and hold.
After the wait a beep sounds and the joints are saved, along with an overhead photo.
Usage: python touch_markers.py [seconds_per_marker]"""
import json, subprocess, sys, time, winsound
import cv2
import scservo_sdk as scs

WAIT = float(sys.argv[1]) if len(sys.argv) > 1 else 12
D = r"C:\Users\sudar\projects\third-hand\setup"
NAMES = ["pan", "lift", "elbow", "wrist_flex", "wrist_roll", "gripper"]
WHERE = {0: "near left", 1: "near right", 2: "far right", 3: "far left"}

def say(text):
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')"])

p, h = scs.PortHandler("COM3"), scs.PacketHandler(0)
p.openPort(); p.setBaudRate(1_000_000)
def read_all():
    out = []
    for i in range(1, 7):
        for _ in range(6):
            v, c, _ = h.read2ByteTxRx(p, i, 56)
            if c == scs.COMM_SUCCESS:
                out.append(v); break
        else:
            raise RuntimeError(f"joint {i} did not answer")
    return out
read_all()
for _ in range(3):
    for i in range(1, 7):
        h.write1ByteTxRx(p, i, 40, 0)
cap = cv2.VideoCapture(3, cv2.CAP_DSHOW)
cap.set(3, 1920); cap.set(4, 1080)
import os
ONLY = [int(a) for a in sys.argv[2:]] or [0, 1, 2, 3]
result = json.load(open(rf"{D}\marker_touch.json")) if os.path.exists(rf"{D}\marker_touch.json") and len(ONLY) < 4 else {}
say("Torque is off. Touch round starting.")
for m in ONLY:
    say(f"Marker {m}, {WHERE[m]}. Bring the gripper tip to its centre and hold still.")
    time.sleep(WAIT)
    winsound.Beep(1200, 400)
    samples = []
    t = time.time()
    while time.time() - t < 1.0:
        samples.append(read_all()); time.sleep(0.03)
    mean = [round(sum(s[k] for s in samples) / len(samples)) for k in range(6)]
    spread = max(max(s[k] for s in samples) - min(s[k] for s in samples) for k in range(4))
    f = None
    for _ in range(6):
        ok, fr = cap.read()
        if ok: f = fr
    if f is not None:
        cv2.imwrite(rf"{D}\marker_touch_{m}.jpg", f)
    result[str(m)] = dict(zip(NAMES, mean)) | {"shake_ticks": spread}
    print(m, result[str(m)])
    winsound.Beep(700, 200)
cap.release(); p.closePort()
json.dump(result, open(rf"{D}\marker_touch.json", "w"), indent=1)
say("Saved. You can let go of the arm.")

