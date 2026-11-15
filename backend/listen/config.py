"""Settings for the listener. Each one can be changed with an environment variable."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# settings such as LISTEN_MIC may sit in the project's .env file
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)


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

# Which speech to text does the work.
#   auto   the local model when a big one (large-v3-turbo and so on) runs on the GPU, which beat
#          ElevenLabs in the offline check; else ElevenLabs when ELEVENLABS_API_KEY is set; else local
#   eleven ElevenLabs scribe, with the local model as the fallback on an error or a slow reply
#   local  faster-whisper on this laptop only
ENGINE = os.environ.get("LISTEN_ENGINE", "auto").strip().lower() or "auto"
ELEVEN_MODEL = os.environ.get("LISTEN_ELEVEN_MODEL", "scribe_v1").strip() or "scribe_v1"
ELEVEN_TIMEOUT_S = _num("LISTEN_ELEVEN_TIMEOUT", 2.5)  # after this the local result is used

# Local speech model. Empty means: on the GPU the best model already in the download cache
# (large-v3-turbo, distil-large-v3, medium.en, in that order), else small.en; base.en on the CPU.
# Nothing is downloaded at start unless you name a model here.
MODEL = os.environ.get("LISTEN_MODEL", "").strip()
GPU_MODELS = ("large-v3-turbo", "distil-large-v3", "medium.en")
# auto tries the GPU, then the CPU, then the server. cuda, cpu or server forces one.
DEVICE = os.environ.get("LISTEN_DEVICE", "auto").strip().lower()
BEAM = int(_num("LISTEN_BEAM", 5))

# Microphone: an index, or part of the device name ("headset", "usb"). Empty means the
# laptop's built-in Microphone Array, else the system default. See --list-mics.
MIC = os.environ.get("LISTEN_MIC", "").strip()

RATE = 16000
FRAME_MS = 30
VAD_LEVEL = int(_num("LISTEN_VAD", 2))           # webrtcvad aggressiveness, 0 to 3
ENERGY_MIN = _num("LISTEN_ENERGY_MIN", 0)        # extra loudness gate (int16 RMS, after any VAD gain); 0 means off
SILENCE_MS = int(_num("LISTEN_SILENCE_MS", 700))  # this much quiet ends an utterance
MIN_SPEECH_MS = int(_num("LISTEN_MIN_MS", 300))  # shorter clips are thrown away
MAX_UTTERANCE_S = _num("LISTEN_MAX_S", 9)        # a loud room may never go quiet, so cut here
END_DB = _num("LISTEN_END_DB", 18)               # quieter than this below the utterance's loudest part counts as quiet
PREROLL_MS = int(_num("LISTEN_PREROLL_MS", 500))  # audio kept from before speech was detected

# Automatic gain. Each finished utterance is normalised so its peak sits near AGC_PEAK before it
# goes to speech to text. LISTEN_AGC=0 turns that off.
# LISTEN_AGC_VAD=1 also boosts each frame before the VAD (noise floor up to AGC_NOISE_TARGET, at
# most AGC_MAX_GAIN). It is off by default: in the offline check it made far away voices count
# as speech, so utterances ran into each other, while quiet speech at the measured mic level was
# already found without it. Turn it on if --mic-test shows voiced frames only with the gain.
AGC = _flag("LISTEN_AGC", True)
AGC_VAD = _flag("LISTEN_AGC_VAD", False)  # the gain in front of the VAD (the clip normalising is separate)
AGC_MAX_GAIN = _num("LISTEN_AGC_MAX_GAIN", 16)
AGC_NOISE_TARGET = _num("LISTEN_AGC_NOISE", 120)   # int16 RMS
AGC_SPEECH_TARGET = _num("LISTEN_AGC_SPEECH", 2500)  # int16 RMS of loud voiced frames
AGC_PEAK = _num("LISTEN_AGC_PEAK", 16000)          # about half of full scale

# Fix near misses of the cast names, items and "Phantom Claw" after speech to text.
FIX = _flag("LISTEN_FIX", True)

MUTE_TAIL_S = _num("LISTEN_MUTE_TAIL", 0.4)      # keep the mic shut this long after a line ends
CHORD_MS = int(_num("LISTEN_CHORD_MS", 600))     # Ctrl+Shift must be released within this

NAMES = ["Escanor", "Elizabeth", "Ban", "King", "Gowther", "Merlin"]
ITEMS = ["bottle", "Sharpie", "scissors", "glue stick", "earbuds"]
VERBS = ["give me", "bring me", "hand me", "point at", "put", "pick up", "open", "play", "search", "stop"]

# Words that help the local speech model spell the names right. Written as sentences, which
# whisper follows better than a bare list.
PROMPT = (
    "Phantom Claw, start. Escanor, give me the bottle. Elizabeth, bring me the Sharpie. "
    "Ban, hand me the scissors. King, point at the glue stick. Gowther, pick up the earbuds. "
    "Merlin, open, play, search, put it down, stop. ArUco marker."
)

# Sent to ElevenLabs as keyterms so it leans towards these spellings.
KEYTERMS = ["Phantom Claw"] + NAMES + ["Sharpie", "glue stick", "earbuds", "scissors", "ArUco"]
