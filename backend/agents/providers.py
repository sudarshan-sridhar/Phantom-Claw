"""Get one JSON object from an LLM, with an 8 second timeout and fall through to the other provider."""
from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

import httpx

from .config import FALLBACK, MODELS, PROVIDER_TIMEOUT_S, key

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# What answered last, for /api/status.
LAST: dict[str, Any] = {"by_agent": {}, "last": None}


class ProviderError(Exception):
    pass


def _parse_json(text: str) -> dict:
    text = (text or "").strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise ProviderError("reply was not JSON")
        obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ProviderError("reply was not a JSON object")
    return obj


async def _openai(client: httpx.AsyncClient, system: str, user: str, model: str) -> dict:
    api_key = key("OPENAI_API_KEY")
    if not api_key:
        raise ProviderError("OPENAI_API_KEY missing")
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "response_format": {"type": "json_object"},
        "temperature": 0.9,
        "max_tokens": 300,
    }
    r = await client.post(OPENAI_URL, json=body, headers={"Authorization": f"Bearer {api_key}"})
    if r.status_code != 200:
        raise ProviderError(f"openai http {r.status_code}: {r.text[:200]}")
    data = r.json()
    return _parse_json(data["choices"][0]["message"]["content"])


async def _gemini(client: httpx.AsyncClient, system: str, user: str, model: str) -> dict:
    api_key = key("GEMINI_API_KEY")
    if not api_key:
        raise ProviderError("GEMINI_API_KEY missing")
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.9, "maxOutputTokens": 400},
    }
    # Key goes in a header, never in the URL.
    r = await client.post(GEMINI_URL.format(model=model), json=body, headers={"x-goog-api-key": api_key})
    if r.status_code != 200:
        raise ProviderError(f"gemini http {r.status_code}: {r.text[:200]}")
    data = r.json()
    try:
        parts = data["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError):
        raise ProviderError("gemini returned no candidates")
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    return _parse_json(text)


_CALLERS = {"openai": _openai, "gemini": _gemini}


async def _call_once(provider: str, system: str, user: str, model: str) -> dict:
    async with httpx.AsyncClient(timeout=PROVIDER_TIMEOUT_S) as client:
        return await asyncio.wait_for(_CALLERS[provider](client, system, user, model), PROVIDER_TIMEOUT_S)


async def get_json(
    provider: str,
    system: str,
    user: str,
    agent: str | None = None,
    models: dict | None = None,
) -> tuple[dict, dict]:
    """Return (json_object, info). info has provider, model, fallback, ms, errors.

    Tries the agent's provider first; on error or timeout tries the other one.
    Raises ProviderError if both fail.
    """
    models = {**MODELS, **(models or {})}
    order = [provider, FALLBACK[provider]]
    errors: list[str] = []
    for i, prov in enumerate(order):
        t0 = time.perf_counter()
        try:
            obj = await _call_once(prov, system, user, models[prov])
            info = {
                "provider": prov,
                "model": models[prov],
                "fallback": i > 0,
                "ms": int((time.perf_counter() - t0) * 1000),
                "errors": errors,
                "at": time.time(),
            }
            LAST["last"] = {"agent": agent, **info}
            if agent:
                LAST["by_agent"][agent] = info
            return obj, info
        except asyncio.TimeoutError:
            errors.append(f"{prov}: timed out after {PROVIDER_TIMEOUT_S:.0f} s")
        except (httpx.HTTPError, ProviderError, ValueError, KeyError) as e:
            errors.append(f"{prov}: {type(e).__name__}: {str(e)[:200]}")
    LAST["last"] = {"agent": agent, "provider": None, "errors": errors, "at": time.time()}
    raise ProviderError("; ".join(errors))
