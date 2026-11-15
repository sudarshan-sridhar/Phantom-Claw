"""Live fall through test. Costs two tiny LLM calls. Run with PHANTOM_LIVE=1."""
import asyncio
import os

import pytest

from backend.agents import providers

live = pytest.mark.skipif(os.environ.get("PHANTOM_LIVE") != "1", reason="set PHANTOM_LIVE=1 to call the real APIs")
SYSTEM = 'Reply with JSON {"say": "<three words>"}'


@live
def test_gemini_bad_model_falls_through_to_openai():
    obj, info = asyncio.run(providers.get_json("gemini", SYSTEM, "hi", models={"gemini": "no-such-model-x"}))
    assert info["provider"] == "openai" and info["fallback"] and "say" in obj
    assert info["errors"] and info["errors"][0].startswith("gemini")


@live
def test_openai_bad_model_falls_through_to_gemini():
    obj, info = asyncio.run(providers.get_json("openai", SYSTEM, "hi", models={"openai": "no-such-model-x"}))
    assert info["provider"] == "gemini" and info["fallback"] and "say" in obj
