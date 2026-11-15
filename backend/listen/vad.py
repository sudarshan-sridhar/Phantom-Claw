"""Cutting a stream of 30 ms frames into utterances.

Speech starts when most of the last few frames sound like a voice. It ends after about
450 ms of quiet. Anything with less than about 300 ms of actual voice is dropped, which
gets rid of coughs, clicks and chair noise.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import config


@dataclass
class Utterance:
    audio: np.ndarray      # int16 mono at 16 kHz
    t_start: float         # monotonic time speech started
    t_end: float           # monotonic time the last voiced frame ended
    t_cut: float           # monotonic time we decided it was over
    voiced_ms: int

    @property
    def seconds(self) -> float:
        return len(self.audio) / config.RATE


def make_vad(level: int = config.VAD_LEVEL, energy_min: float = config.ENERGY_MIN) -> Callable[[bytes], bool]:
    """A frame test using webrtcvad, with an optional loudness floor on top."""
    import webrtcvad

    vad = webrtcvad.Vad(max(0, min(3, level)))

    def is_speech(frame: bytes) -> bool:
        if energy_min > 0:
            x = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
            if float(np.sqrt(np.mean(x * x))) < energy_min:
                return False
        return vad.is_speech(frame, config.RATE)

    return is_speech


class Segmenter:
    def __init__(
        self,
        is_speech: Callable[[bytes], bool],
        frame_ms: int = config.FRAME_MS,
        silence_ms: int = config.SILENCE_MS,
        min_speech_ms: int = config.MIN_SPEECH_MS,
        max_s: float = config.MAX_UTTERANCE_S,
        preroll_ms: int = config.PREROLL_MS,
    ):
        self.is_speech = is_speech
        self.frame_ms = frame_ms
        self.end_frames = max(1, silence_ms // frame_ms)
        self.min_voiced = max(1, min_speech_ms // frame_ms)
        self.max_frames = int(max_s * 1000 / frame_ms)
        self.pre = deque(maxlen=max(1, preroll_ms // frame_ms))
        self.recent = deque(maxlen=5)
        self.reset()

    def reset(self) -> None:
        """Forget anything half heard (used when the mic is muted)."""
        self.active = False
        self.frames: list[bytes] = []
        self.voiced = 0
        self.quiet = 0
        self.t_start = 0.0
        self.t_last_voice = 0.0
        self.pre.clear()
        self.recent.clear()

    def feed(self, frame: bytes, t: float) -> Utterance | None:
        """Give one frame that ended at time t. Returns an utterance when one has just finished."""
        voiced = self.is_speech(frame)
        if not self.active:
            self.pre.append(frame)
            self.recent.append(voiced)
            # Three voiced frames out of the last five starts an utterance.
            if sum(self.recent) >= 3:
                self.active = True
                self.frames = list(self.pre)
                self.voiced = sum(self.recent)
                self.quiet = 0
                self.t_start = t - len(self.frames) * self.frame_ms / 1000
                self.t_last_voice = t
            return None

        self.frames.append(frame)
        if voiced:
            self.voiced += 1
            self.quiet = 0
            self.t_last_voice = t
        else:
            self.quiet += 1
        if self.quiet >= self.end_frames or len(self.frames) >= self.max_frames:
            return self._finish(t)
        return None

    def flush(self, t: float) -> Utterance | None:
        """End whatever is in progress (end of a file)."""
        return self._finish(t) if self.active else None

    def _finish(self, t: float) -> Utterance | None:
        # Keep about 150 ms of the trailing quiet, the model likes a little room.
        keep_tail = max(0, self.quiet - 5)
        frames = self.frames[: len(self.frames) - keep_tail] if keep_tail else self.frames
        voiced, t_start, t_end = self.voiced, self.t_start, self.t_last_voice
        self.reset()
        if voiced < self.min_voiced:
            return None
        audio = np.frombuffer(b"".join(frames), dtype=np.int16).copy()
        return Utterance(audio=audio, t_start=t_start, t_end=t_end, t_cut=t, voiced_ms=voiced * self.frame_ms)
