"""Checks before anything moves, in order: cameras, motors, the app, the model services.

    python setup\\tools\\preflight.py

Run it with the app already started. It prints one line per check and says the verdict out loud.
Nothing moves.
"""
import os
import subprocess
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SERVER = "http://127.0.0.1:8765"
REST_ROLL = 2016


def say(text):
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    "Add-Type -AssemblyName System.Speech; "
                    f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')"], capture_output=True)


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    bad = []

    def line(group, name, ok, note=""):
        print(f"{group:8} {name:22} {'ok' if ok else 'FAIL':5} {note}")
        if not ok:
            bad.append(name)

    # 1. cameras
    from backend.vision import cameras
    img = cameras.grab_overhead(cameras.OVERHEAD_INDEX, tries=2)
    seen = 0
    if img is not None:
        from backend.vision import markers
        seen = len(markers.find(img))
    line("camera", "overhead", img is not None, f"{seen} of 4 markers seen" if img is not None else "no picture: wake the phone, open iVCam")
    status = None
    try:
        status = httpx.get(SERVER + "/api/status", timeout=4).json()
    except Exception:
        pass
    # the app holds the wrist camera only while it is aiming, so it can be opened here
    try:
        cam = cameras.WristCamera()
        f = cam.frame()
        cam.close()
    except Exception:
        f = None
    line("camera", "gripper", f is not None)

    # 2. motors
    arm = (status or {}).get("arm") or {}
    j = arm.get("joints") or {}
    line("motors", "connected", bool(arm.get("connected")), arm.get("error") or "")
    line("motors", "torque on", bool(arm.get("torque")))
    line("motors", "at rest", bool(arm.get("at_rest")))
    roll = j.get("wrist_roll")
    line("motors", "wrist level", roll is not None and abs(roll - REST_ROLL) < 60, f"roll {roll}")

    # 3. the app
    line("app", "server", status is not None)
    try:
        ok = httpx.get("http://127.0.0.1:8766/api/panel", timeout=4).status_code == 200
    except Exception:
        ok = False
    line("app", "record page", ok)
    try:
        listen = httpx.get(SERVER + "/api/listen", timeout=4).json()
        line("app", "listener", True, "awake" if listen.get("awake") else "asleep, say Phantom Claw start")
    except Exception:
        line("app", "listener", False)

    # 4. model services (a tiny call each; keys are never printed)
    def check(name, fn):
        try:
            r = fn()
            line("api", name, r.status_code == 200, "" if r.status_code == 200 else f"http {r.status_code}")
        except Exception as e:
            line("api", name, False, type(e).__name__)

    k = os.environ.get
    check("Claude", lambda: httpx.post("https://api.anthropic.com/v1/messages", timeout=10,
          headers={"x-api-key": k("ANTHROPIC_API_KEY", ""), "anthropic-version": "2023-06-01"},
          json={"model": "claude-haiku-5-5", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]}))
    check("OpenAI", lambda: httpx.get("https://api.openai.com/v1/models/gpt-4o-mini", timeout=10,
          headers={"Authorization": "Bearer " + k("OPENAI_API_KEY", "")}))
    check("Gemini", lambda: httpx.get("https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite",
          timeout=10, headers={"x-goog-api-key": k("GEMINI_API_KEY", "")}))
    check("ElevenLabs", lambda: httpx.get("https://api.elevenlabs.io/v1/user", timeout=10,
          headers={"xi-api-key": k("ELEVENLABS_API_KEY", "")}))

    if bad:
        print("NOT READY:", ", ".join(bad))
        say("Not ready. " + ", ".join(bad[:3]) + " failed.")
        return 1
    print("READY")
    say("All checks passed. Ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
