"""ElevenLabs text to speech (cached on disk) and speech to text."""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

import httpx

from .config import CAST_VOICES_FILE, MODELS, PUBLIC_BASE, TTS_CACHE_DIR, key

TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=mp3_44100_128"
STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
CHARS_PER_S = 14.0

_tts_model = {"name": MODELS["tts"]}
_locks: dict[str, asyncio.Lock] = {}


class VoiceError(Exception):
    pass


@lru_cache(maxsize=1)
def voices() -> dict:
    return json.loads(Path(CAST_VOICES_FILE).read_text(encoding="utf-8"))


def voice_id(agent: str) -> str:
    return voices()[agent]["voice_id"]


def clip_name(vid: str, text: str) -> str:
    h = hashlib.sha1(f"{vid}\n{text}".encode("utf-8")).hexdigest()[:20]
    return f"{vid}_{h}.mp3"


def audio_url(name: str) -> str:
    return f"{PUBLIC_BASE}/audio/{name}"


def safe_cache_path(name: str) -> Path | None:
    """Resolve a requested audio file name inside the cache, refusing anything else."""
    if not re.fullmatch(r"[A-Za-z0-9_]+\.mp3", name or ""):
        return None
    p = TTS_CACHE_DIR / name
    return p if p.is_file() else None


def estimate_seconds(text: str) -> float:
    return round(max(1.0, len(text) / CHARS_PER_S), 2)


def mp3_seconds(path: Path, text: str = "") -> float:
    try:
        from mutagen.mp3 import MP3

        return round(float(MP3(str(path)).info.length), 2)
    except Exception:
        try:
            return round(path.stat().st_size * 8 / 128000, 2)  # 128 kbps constant bitrate
        except OSError:
            return estimate_seconds(text)


def _meta_path(mp3: Path) -> Path:
    return mp3.with_suffix(".json")


def clip_duration(name: str, text: str = "") -> float:
    p = TTS_CACHE_DIR / name
    meta = _meta_path(p)
    try:
        return float(json.loads(meta.read_text(encoding="utf-8"))["duration_s"])
    except Exception:
        return mp3_seconds(p, text) if p.exists() else estimate_seconds(text)


async def _post_tts(client: httpx.AsyncClient, vid: str, text: str, model: str) -> httpx.Response:
    return await client.post(
        TTS_URL.format(voice_id=vid),
        headers={"xi-api-key": key("ELEVENLABS_API_KEY"), "accept": "audio/mpeg"},
        json={"text": text, "model_id": model},
    )


async def tts(agent: str, text: str) -> dict:
    """Return {"file", "audio_url", "duration_s", "cached"} for this agent's line."""
    text = text.strip()
    vid = voice_id(agent)
    name = clip_name(vid, text)
    path = TTS_CACHE_DIR / name
    lock = _locks.setdefault(name, asyncio.Lock())
    async with lock:
        if path.exists() and path.stat().st_size > 0:
            return {"file": name, "audio_url": audio_url(name), "duration_s": clip_duration(name, text), "cached": True}
        if not key("ELEVENLABS_API_KEY"):
            raise VoiceError("ELEVENLABS_API_KEY missing")
        TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        async with httpx.AsyncClient(timeout=20) as client:
            model = _tts_model["name"]
            r = await _post_tts(client, vid, text, model)
            if r.status_code not in (200, 401, 429) and model != MODELS["tts_fallback"]:
                model = MODELS["tts_fallback"]
                r = await _post_tts(client, vid, text, model)
                if r.status_code == 200:
                    _tts_model["name"] = model
            if r.status_code != 200:
                raise VoiceError(f"tts http {r.status_code}: {r.text[:200]}")
        tmp = path.with_suffix(".part")
        tmp.write_bytes(r.content)
        tmp.replace(path)
        dur = mp3_seconds(path, text)
        _meta_path(path).write_text(
            json.dumps({"voice_id": vid, "agent": agent, "text": text, "model": model, "duration_s": dur}),
            encoding="utf-8",
        )
        return {"file": name, "audio_url": audio_url(name), "duration_s": dur, "cached": False}


async def stt(data: bytes, filename: str = "audio.webm", content_type: str = "application/octet-stream") -> str:
    if not key("ELEVENLABS_API_KEY"):
        raise VoiceError("ELEVENLABS_API_KEY missing")
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            STT_URL,
            headers={"xi-api-key": key("ELEVENLABS_API_KEY")},
            data={"model_id": MODELS["stt"]},
            files={"file": (filename, data, content_type)},
        )
    if r.status_code != 200:
        raise VoiceError(f"stt http {r.status_code}: {r.text[:200]}")
    return (r.json().get("text") or "").strip()
