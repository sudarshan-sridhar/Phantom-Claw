"""Speech to text, on this laptop when possible.

Order of preference:
1. faster-whisper on the GPU (small.en, float16). Needs the cuBLAS and cuDNN DLLs, which come
   from the nvidia-cublas-cu12 and nvidia-cudnn-cu12 pip packages.
2. faster-whisper on the CPU (base.en, int8) if the CUDA libraries are missing.
3. The server's POST /api/stt (ElevenLabs) if faster-whisper will not load at all.
"""
from __future__ import annotations

import glob
import io
import os
import site
import sys
import time
import wave

import numpy as np

from . import config

_HALLUCINATIONS = {"", "you", "thank you", "thanks for watching", "thank you for watching", "bye"}


def _add_cuda_dlls() -> None:
    """Make the pip installed NVIDIA DLLs findable. ctranslate2 loads cuBLAS on first use."""
    if sys.platform != "win32":
        return
    for base in site.getsitepackages():
        for d in glob.glob(os.path.join(base, "nvidia", "*", "bin")):
            try:
                os.add_dll_directory(d)
            except OSError:
                pass
            if d not in os.environ.get("PATH", ""):
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def to_wav_bytes(audio: np.ndarray, rate: int = config.RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(audio.astype(np.int16).tobytes())
    return buf.getvalue()


class Transcriber:
    def __init__(self, device: str = config.DEVICE, model: str = config.MODEL, server: str = config.SERVER):
        self.device = device
        self.model_name = model
        self.server = server
        self.model = None
        self.path = "not loaded"
        self.kind = None  # "gpu", "cpu" or "server"

    def load(self) -> str:
        """Load the best path that works and warm it up. Returns a line describing it."""
        tried = []
        if self.device in ("auto", "cuda", "gpu"):
            try:
                _add_cuda_dlls()
                self._load_whisper(self.model_name or "small.en", "cuda", "float16")
                self.kind = "gpu"
            except Exception as e:  # noqa: BLE001
                tried.append(f"GPU: {type(e).__name__}: {str(e)[:120]}")
                self.model = None
        if self.model is None and self.device in ("auto", "cpu"):
            try:
                self._load_whisper(self.model_name or "base.en", "cpu", "int8")
                self.kind = "cpu"
            except Exception as e:  # noqa: BLE001
                tried.append(f"CPU: {type(e).__name__}: {str(e)[:120]}")
                self.model = None
        if self.model is None:
            self.kind = "server"
            self.path = f"server speech to text ({self.server}/api/stt, ElevenLabs)"
        for t in tried:
            print(f"[listen] speech path not available, {t}", flush=True)
        return self.path

    def _load_whisper(self, name: str, device: str, compute: str) -> None:
        from faster_whisper import WhisperModel

        t0 = time.monotonic()
        self.model = WhisperModel(name, device=device, compute_type=compute)
        # A dummy run so the first real utterance does not pay for loading kernels.
        # CUDA problems (a missing cuBLAS DLL) only show up here, not at load time.
        noise = (np.random.default_rng(0).standard_normal(config.RATE) * 300).astype(np.int16)
        self._whisper(noise)
        self.path = f"faster-whisper {name} on the {'GPU' if device == 'cuda' else 'CPU'} ({compute}), ready in {time.monotonic() - t0:.1f} s"

    def _whisper(self, audio: np.ndarray) -> str:
        x = audio.astype(np.float32) / 32768.0
        segments, _info = self.model.transcribe(
            x,
            language="en",
            beam_size=1,
            without_timestamps=True,
            condition_on_previous_text=False,
            initial_prompt=config.PROMPT,
            vad_filter=False,
        )
        parts = []
        for s in segments:
            # Skip what the model itself thinks is not speech.
            if s.no_speech_prob > 0.6 and s.avg_logprob < -0.8:
                continue
            parts.append(s.text.strip())
        return " ".join(parts).strip()

    def _server(self, audio: np.ndarray) -> str:
        import httpx

        files = {"file": ("speech.wav", to_wav_bytes(audio), "audio/wav")}
        r = httpx.post(self.server + "/api/stt", files=files, timeout=20)
        r.raise_for_status()
        return str(r.json().get("text") or "").strip()

    def transcribe(self, audio: np.ndarray) -> str:
        text = self._whisper(audio) if self.model is not None else self._server(audio)
        plain = " ".join(text.lower().replace(".", " ").replace(",", " ").replace("!", " ").split())
        if plain in _HALLUCINATIONS or plain == " ".join(config.PROMPT.lower().replace(".", " ").replace(",", " ").split()):
            return ""
        return text
