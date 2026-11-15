"""Speech to text.

Two engines (LISTEN_ENGINE or --engine):
- eleven: ElevenLabs scribe (the same endpoint and key as backend/agents/voice.py), sent the
  cut utterance as a WAV, English, with the cast names and items as keyterms. The local model
  runs on the same clip at the same time, and its answer is used if ElevenLabs fails or takes
  longer than LISTEN_ELEVEN_TIMEOUT (2.5 s). So a dead network costs at most 2.5 s, once per clip.
- local: faster-whisper on this laptop. On the GPU it uses the best model already downloaded
  (large-v3-turbo, distil-large-v3, medium.en, else small.en), float16, beam 5, temperature 0,
  English, with the names, items and verbs in the prompt. base.en int8 on the CPU if CUDA will not
  load, and the server's POST /api/stt if faster-whisper will not load at all.
auto (the default) means local when a big model (large-v3-turbo, distil-large-v3, medium.en) is
loaded on the GPU, which did best in the offline check (evalcheck.py); else eleven when
ELEVENLABS_API_KEY is set; else local with whatever model loaded.

Each clip is normalised first (gain.normalize) so a quiet mic still gives a full scale signal.
"""
from __future__ import annotations

import glob
import io
import os
import site
import sys
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

import numpy as np

from . import config
from .gain import normalize

_HALLUCINATIONS = {"", "you", "thank you", "thanks for watching", "thank you for watching", "bye"}


def _plain(text: str) -> str:
    return " ".join(text.lower().replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ").split())


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


def cached_model(names=config.GPU_MODELS) -> str | None:
    """The first of these faster-whisper models that is already in the download cache."""
    try:
        from faster_whisper.utils import download_model
    except Exception:  # noqa: BLE001
        return None
    for n in names:
        try:
            download_model(n, local_files_only=True)
            return n
        except Exception:  # noqa: BLE001
            continue
    return None


def eleven_key() -> str:
    try:
        from backend.agents.config import key  # loads .env

        return key("ELEVENLABS_API_KEY")
    except Exception:  # noqa: BLE001
        return (os.environ.get("ELEVENLABS_API_KEY") or "").strip()


def is_big(model_name: str | None) -> bool:
    n = (model_name or "").lower()
    return n.startswith(("large", "distil-large", "medium"))


def resolve_engine(engine: str, has_key: bool, big_local: bool = False) -> str:
    engine = (engine or "auto").lower()
    if engine in ("eleven", "elevenlabs", "scribe"):
        return "eleven" if has_key else "local"
    if engine == "local":
        return "local"
    if big_local:
        return "local"
    return "eleven" if has_key else "local"


class ElevenSTT:
    """ElevenLabs speech to text, called straight from the listener (no trip through the server)."""

    URL = "https://api.elevenlabs.io/v1/speech-to-text"

    def __init__(self, model: str = config.ELEVEN_MODEL, keyterms=config.KEYTERMS, timeout: float = 10.0):
        import httpx

        self.model = model
        self.keyterms = list(keyterms)
        self.extras = True  # language, no audio tags, keyterms; dropped if the API ever refuses them
        self.http = httpx.Client(timeout=httpx.Timeout(timeout, connect=3.0))

    def __call__(self, audio: np.ndarray) -> str:
        data = {"model_id": self.model}
        if self.extras:
            data.update({"language_code": "en", "tag_audio_events": "false", "keyterms": self.keyterms})
        files = {"file": ("speech.wav", to_wav_bytes(audio), "audio/wav")}
        headers = {"xi-api-key": eleven_key()}
        r = self.http.post(self.URL, headers=headers, data=data, files=files)
        if r.status_code in (400, 422) and self.extras:
            self.extras = False
            r = self.http.post(self.URL, headers=headers, data={"model_id": self.model}, files=files)
        if r.status_code != 200:
            raise RuntimeError(f"ElevenLabs stt http {r.status_code}")
        return str(r.json().get("text") or "").strip()


class Transcriber:
    def __init__(
        self,
        device: str = config.DEVICE,
        model: str = config.MODEL,
        server: str = config.SERVER,
        engine: str = config.ENGINE,
        eleven_timeout: float = config.ELEVEN_TIMEOUT_S,
        agc: bool = config.AGC,
        beam: int = config.BEAM,
        prompt: str = config.PROMPT,
        eleven=None,
    ):
        self.device = device
        self.model_name = model
        self.server = server
        self.engine_req = engine
        self.engine = None           # "eleven" or "local", set by load()
        self.eleven_timeout = eleven_timeout
        self.agc = agc
        self.beam = beam
        self.prompt = prompt
        self.eleven = eleven         # a callable audio -> text; made in load() if None
        self.model = None
        self.path = "not loaded"
        self.kind = None  # "gpu", "cpu" or "server"
        self.last: dict = {}
        self.pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="stt")

    # ------------------------------------------------------------ loading

    def load(self) -> str:
        """Load the engine and the local model (used alone, or as the fallback), warm them up."""
        has_key = bool(eleven_key()) or self.eleven is not None
        self.load_local()
        self.engine = resolve_engine(self.engine_req, has_key, self.kind == "gpu" and is_big(self.model_name))
        if self.engine_req in ("eleven", "elevenlabs") and not has_key:
            print("[listen] LISTEN_ENGINE=eleven but ELEVENLABS_API_KEY is not set, using the local model", flush=True)
        if self.engine == "eleven":
            if self.eleven is None:
                self.eleven = ElevenSTT()
            return f"ElevenLabs {getattr(self.eleven, 'model', 'scribe')} (keyterms, {self.eleven_timeout:.1f} s limit), fallback {self.path}"
        return self.path

    def load_local(self) -> str:
        tried = []
        if self.device in ("auto", "cuda", "gpu"):
            try:
                _add_cuda_dlls()
                name = self.model_name or cached_model() or "small.en"
                self._load_whisper(name, "cuda", "float16")
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
        self.model_name = name
        self.path = f"faster-whisper {name} on the {'GPU' if device == 'cuda' else 'CPU'} ({compute}, beam {self.beam}), ready in {time.monotonic() - t0:.1f} s"

    # ------------------------------------------------------------ engines

    def _whisper(self, audio: np.ndarray) -> str:
        x = audio.astype(np.float32) / 32768.0
        segments, _info = self.model.transcribe(
            x,
            language="en",
            beam_size=self.beam,
            temperature=0.0,
            without_timestamps=True,
            condition_on_previous_text=False,
            initial_prompt=self.prompt,
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

    def _local(self, audio: np.ndarray) -> tuple[str, float]:
        t0 = time.monotonic()
        text = self._whisper(audio) if self.model is not None else self._server(audio)
        return text, (time.monotonic() - t0) * 1000

    def _eleven(self, audio: np.ndarray) -> tuple[str, float]:
        t0 = time.monotonic()
        text = self.eleven(audio)
        return text, (time.monotonic() - t0) * 1000

    # ------------------------------------------------------------ one clip

    def transcribe(self, audio: np.ndarray) -> str:
        """The raw text for one utterance. Details (engine, timings, gain) go to self.last."""
        if self.engine is None:
            self.engine = resolve_engine(self.engine_req, self.eleven is not None)
        t0 = time.monotonic()
        g = 1.0
        if self.agc:
            audio, g = normalize(audio)
        last = {"gain": round(g, 1), "engine": self.engine}
        if self.engine == "eleven" and self.eleven is not None:
            fe = self.pool.submit(self._eleven, audio)
            # The local model works on the same clip meanwhile, so a failure costs no extra wait.
            fl = self.pool.submit(self._local, audio) if self.model is not None else None
            try:
                text, last["eleven_ms"] = fe.result(timeout=self.eleven_timeout)
            except FutureTimeout:
                last["fallback"] = f"ElevenLabs slower than {self.eleven_timeout:.1f} s"
                text = None
            except Exception as e:  # noqa: BLE001
                last["fallback"] = f"ElevenLabs failed ({type(e).__name__}: {str(e)[:80]})"
                text = None
            if text is None:
                last["engine"] = "local"
                if fl is None:
                    text = ""
                else:
                    try:
                        text, last["local_ms"] = fl.result(timeout=30)
                    except Exception as e:  # noqa: BLE001
                        last["fallback"] += f", local failed too ({type(e).__name__})"
                        text = ""
        else:
            text, last["local_ms"] = self._local(audio)
        last["ms"] = round((time.monotonic() - t0) * 1000)
        plain = _plain(text)
        if plain in _HALLUCINATIONS or plain == _plain(self.prompt):
            text = ""
        last["raw"] = text
        self.last = last
        return text
