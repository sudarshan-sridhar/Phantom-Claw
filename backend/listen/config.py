"""Settings for the listener. Each one can be changed with an environment variable."""
from __future__ import annotations

import os


def _flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or not v.strip():
        return default
    return v.strip().lower() not in ("0", "false", "no", "off")


def _num(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


SERVER = os.environ.get("PHANTOM_SERVER", "http://127.0.0.1:8765").rstrip("/")

# By default an utterance doesn't need a name: an unnamed one goes to the server with agent null
# and Merlin picks who takes it by the kind of job. Chatter that isn't a request is dropped first
# (see looks_like_request). In a noisy room set LISTEN_REQUIRE_NAME=1 (or --require-name) so only
# utterances that name somebody go anywhere.
REQUIRE_NAME = _flag("LISTEN_REQUIRE_NAME", False)

# Speech model. Empty means small.en on the GPU and base.en on the CPU.
MODEL = os.environ.get("LISTEN_MODEL", "").strip()
# auto tries the GPU, then the CPU, then the server. cuda, cpu or server forces one.
DEVICE = os.environ.get("LISTEN_DEVICE", "auto").strip().lower()

RATE = 16000
FRAME_MS = 30
VAD_LEVEL = int(_num("LISTEN_VAD", 2))           # webrtcvad aggressiveness, 0 to 3
ENERGY_MIN = _num("LISTEN_ENERGY_MIN", 0)        # extra loudness gate (int16 RMS); 0 means off
SILENCE_MS = int(_num("LISTEN_SILENCE_MS", 450))  # this much quiet ends an utterance
MIN_SPEECH_MS = int(_num("LISTEN_MIN_MS", 300))  # shorter clips are thrown away
MAX_UTTERANCE_S = _num("LISTEN_MAX_S", 12)       # a loud room may never go quiet, so cut here
PREROLL_MS = 300

MUTE_TAIL_S = _num("LISTEN_MUTE_TAIL", 0.4)      # keep the mic shut this long after a line ends
CHORD_MS = int(_num("LISTEN_CHORD_MS", 600))     # Ctrl+Shift must be released within this

# Words that help the speech model spell the names right.
PROMPT = "Phantom Claw start. Escanor, Elizabeth, Ban, King, Gowther, Merlin. ArUco marker."
