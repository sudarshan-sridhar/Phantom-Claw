"""Shared settings for the agent layer. Model names live here and only here."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent

load_dotenv(PROJECT_DIR / ".env", override=False)

# One place to change models.
MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-3.5-flash-lite",
    "anthropic": "claude-haiku-5-5",
    "tts": "eleven_flash_v2_5",
    "tts_fallback": "eleven_multilingual_v2",
    "stt": "scribe_v1",
}

PROVIDER_TIMEOUT_S = 8.0
FALLBACK = {"openai": "gemini", "gemini": "openai", "anthropic": "openai"}

HOST = "127.0.0.1"
PORT = 8765
PUBLIC_BASE = f"http://{HOST}:{PORT}"

TTS_CACHE_DIR = BACKEND_DIR / "cache" / "tts"
NOTES_FILE = BACKEND_DIR / "notes.md"
CAST_VOICES_FILE = PROJECT_DIR / "setup" / "cast_voices.json"


def key(name: str) -> str:
    """Read a key from the environment. Never print the result."""
    return (os.environ.get(name) or "").strip()
