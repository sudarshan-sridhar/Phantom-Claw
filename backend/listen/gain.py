"""Level handling for a quiet microphone.

Two parts:
- LevelTracker follows the room's noise floor and the level of recent speech, and gives the
  gain each frame should get before the VAD and the energy gate look at it (only with
  LISTEN_AGC_VAD=1; off by default, see config.py). The gain is held
  back so the noise floor never ends up louder than AGC_NOISE_TARGET, which keeps an empty
  room from being boosted into "speech".
- normalize() brings a finished utterance to a sane peak before speech to text, unless the clip
  has nothing in it that stands clear of its own background.

Software cannot raise the Windows input level, so mic_report() also says when the owner should.
"""
from __future__ import annotations

import numpy as np

from . import config


def frame_rms(frame: np.ndarray) -> float:
    x = frame.astype(np.float32)
    return float(np.sqrt(np.mean(x * x))) if len(x) else 0.0


class LevelTracker:
    def __init__(
        self,
        max_gain: float = config.AGC_MAX_GAIN,
        noise_target: float = config.AGC_NOISE_TARGET,
        speech_target: float = config.AGC_SPEECH_TARGET,
    ):
        self.max_gain = max_gain
        self.noise_target = noise_target
        self.speech_target = speech_target
        self.floor: float | None = None   # noise floor, int16 RMS
        self.speech: float | None = None  # level of recent loud frames, int16 RMS

    def update(self, rms: float) -> None:
        rms = max(rms, 1.0)
        if self.floor is None:
            self.floor = rms
        elif rms < self.floor:
            self.floor += 0.2 * (rms - self.floor)  # falls quickly
        else:
            self.floor *= 1.002                     # rises slowly, about 7 % a second
            self.floor = min(self.floor, rms)
        if rms > 3 * self.floor:
            if self.speech is None or rms > self.speech:
                self.speech = rms if self.speech is None else self.speech + 0.3 * (rms - self.speech)
            else:
                self.speech += 0.01 * (rms - self.speech)

    def gain(self) -> float:
        if self.floor is None:
            return 1.0
        g = self.noise_target / self.floor  # the most the noise floor allows
        if self.speech is not None:
            g = min(g, max(1.0, self.speech_target / self.speech))
        return float(max(1.0, min(self.max_gain, g)))

    def apply(self, frame: bytes) -> bytes:
        """Update from this frame and return it with the gain applied (for the VAD only)."""
        x = np.frombuffer(frame, dtype=np.int16)
        self.update(frame_rms(x))
        g = self.gain()
        if g <= 1.0:
            return frame
        return np.clip(x.astype(np.float32) * g, -32767, 32767).astype(np.int16).tobytes()


def clip_levels(audio: np.ndarray, frame: int = 480) -> dict:
    """Peak, RMS, noise floor and speech level of a clip, all in int16 units."""
    x = audio.astype(np.float32)
    if len(x) == 0:
        return {"peak": 0, "rms": 0.0, "floor": 0.0, "speech": 0.0}
    n = max(1, len(x) // frame)
    rms = np.sqrt(np.mean(x[: n * frame].reshape(n, -1) ** 2, axis=1)) if len(x) >= frame else np.array([frame_rms(x)])
    return {
        "peak": int(np.abs(x).max()),
        "rms": float(np.sqrt(np.mean(x * x))),
        "floor": float(np.percentile(rms, 10)),
        "speech": float(np.percentile(rms, 95)),
    }


def normalize(audio: np.ndarray, target_peak: float = config.AGC_PEAK, max_gain: float = config.AGC_MAX_GAIN * 2) -> tuple[np.ndarray, float]:
    """Scale a clip so its peak (ignoring the odd click) sits near target_peak.

    Returns (audio, gain). A clip with no part well above its own background, or one that is
    already loud, is returned as it is.
    """
    if len(audio) == 0:
        return audio, 1.0
    x = audio.astype(np.float32)
    lv = clip_levels(audio)
    peak = float(np.percentile(np.abs(x), 99.9))
    if peak < 1 or lv["speech"] < 2.5 * max(lv["floor"], 1.0):
        return audio, 1.0
    g = min(max_gain, target_peak / peak)
    if g <= 1.05:
        return audio, 1.0
    return np.clip(x * g, -32767, 32767).astype(np.int16), float(g)


def mic_report(levels: dict, voiced_raw: int, voiced_gain: int, frames: int) -> tuple[str, list[str]]:
    """A verdict and some advice from a mic test where the owner was asked to talk."""
    peak, floor, speech = levels["peak"], levels["floor"], levels["speech"]
    tips = []
    if peak < 3000 or speech < 600:
        verdict = "TOO QUIET"
        tips.append("Raise the microphone level in Windows: Settings, System, Sound, Input, pick the "
                    "microphone, set Input volume to about 90 (100 is fine). If the Realtek panel has "
                    "Microphone Boost, set it to +10 or +20 dB. Then run --mic-test again; aim for a peak "
                    "above 5000 while you talk.")
        tips.append("The listener normalises each clip itself, so it still works, but a higher Windows "
                    "level gives the VAD and the speech model a cleaner signal.")
        if voiced_gain > 2 * max(voiced_raw, 1):
            tips.append("Your voice was mostly found only with the extra gain: if you cannot raise the "
                        "Windows level, set LISTEN_AGC_VAD=1.")
    elif peak >= 32000:
        verdict = "TOO LOUD"
        tips.append("The signal clips. Lower the Windows input volume to about 70 or sit further back.")
    else:
        verdict = "OK"
    if peak < 50:
        tips.append("Almost nothing came in (peak %d). The mic may be muted in Windows or by a key on the "
                    "laptop, or another app may hold it exclusively." % peak)
    elif floor > 400 or (speech > 100 and speech / max(floor, 1) < 4):
        tips.append("The room is loud compared with your voice. A headset or USB mic helps most "
                    "(--list-mics, then LISTEN_MIC=<part of its name>), and LISTEN_REQUIRE_NAME=1 stops "
                    "background chatter going to the agents.")
    if frames and voiced_raw == 0 and voiced_gain == 0:
        tips.append("No speech was detected at all. Check that this is the right microphone and that it is not muted.")
    return verdict, tips
