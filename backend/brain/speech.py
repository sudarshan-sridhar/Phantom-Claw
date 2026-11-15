"""Spoken prompts and beeps, so the person wearing the puck never has to look at the screen.
Speech goes through Windows' own System.Speech in a PowerShell child, beeps through winsound."""
import subprocess
import threading

try:
    import winsound
except ImportError:  # not on Windows: beeps become no-ops
    winsound = None

MUTED = False


def say(text: str, block: bool = True) -> None:
    print(f"  (says) {text}", flush=True)
    if MUTED:
        return
    quoted = text.replace("'", "''")
    cmd = ("Add-Type -AssemblyName System.Speech; "
           "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Rate = 0; "
           f"$s.Speak('{quoted}')")
    args = ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd]
    try:
        if block:
            subprocess.run(args, timeout=30, capture_output=True)
        else:
            subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        pass  # no speech is annoying but not a reason to stop


def beep(freq: int = 880, ms: int = 150, times: int = 1, block: bool = True) -> None:
    if MUTED or winsound is None:
        return

    def _play():
        for i in range(times):
            winsound.Beep(freq, ms)
            if i < times - 1:
                threading.Event().wait(0.12)

    if block:
        _play()
    else:
        threading.Thread(target=_play, daemon=True).start()
