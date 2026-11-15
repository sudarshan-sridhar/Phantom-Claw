"""Pipeline tests with a fake LLM, no speech calls and a fast stand-in arm (no hardware, no network)."""
import asyncio
import random

import pytest

from backend.agents import cast, merlin, pipeline, providers
from backend.agents._standin_arm import StandinArm


@pytest.fixture(autouse=True)
def fresh_memory():
    cast.reset_memory()
    yield
    cast.reset_memory()


def _fake_llm(arm=None, laptop=None):
    async def fake(provider, system, user, agent=None, models=None):
        return {"agent": agent, "say": "Fine.", "arm": arm, "laptop": laptop or []}, {
            "provider": provider, "model": "fake", "fallback": False, "ms": 1, "errors": []}
    return fake


def _run(p, text, agent=None, force_argue=None):
    return asyncio.run(p.handle(text, agent, force_argue=force_argue))


def _pipe(events, arm):
    async def emit(ev):
        events.append(ev)
    return pipeline.Pipeline(arm, emit, speak=False, run_tools=False, pace_audio=False, rng=random.Random(3))


def test_line_without_arm_skill_gets_gesture_in_speaker_style(monkeypatch):
    monkeypatch.setattr(providers, "get_json", _fake_llm(arm=None))
    arm = StandinArm(speed=50)
    events = []
    res = _run(_pipe(events, arm), "Gowther what time is it", "gowther")
    assert arm.log == [("gesture", {"style": "gowther", "seconds": arm.log[0][1]["seconds"]})]
    assert 1.5 <= arm.log[0][1]["seconds"] <= 10
    say = [e for e in events if e["type"] == "say"][0]
    assert "duration_s" in say
    arm_ev = [e for e in events if e["type"] == "arm"]
    assert arm_ev and all(e["skill"] == "gesture" and e["agent"] == "gowther" for e in arm_ev)
    # the gesture starts right after the say event
    assert events.index(say) < events.index(arm_ev[0])
    assert res["arm_result"]["skill"] == "gesture"


def test_line_with_arm_skill_has_no_gesture(monkeypatch):
    monkeypatch.setattr(providers, "get_json", _fake_llm(arm={"skill": "grip_bottle"}))
    arm = StandinArm(speed=50)
    events = []
    _run(_pipe(events, arm), "Ban lift the bottle", None)
    assert [s for s, _ in arm.log] == ["grip_bottle"]


def test_argument_gets_one_gesture_per_line_in_order(monkeypatch):
    monkeypatch.setattr(providers, "get_json", _fake_llm(arm=None))
    arm = StandinArm(speed=50)
    events = []
    res = _run(_pipe(events, arm), "say something", None, force_argue=True)
    m = [e for e in events if e["type"] == "merlin"][0]
    assert len(m["argued"]) == 2 and "duration_s" in m
    styles = [p["style"] for s, p in arm.log if s == "gesture"]
    assert styles == [m["argued"][0]["agent"], m["argued"][1]["agent"], "merlin", res["agent"]]
    # contract order: request, merlin, say
    types = [e["type"] for e in events]
    assert types.index("request") < types.index("merlin") < types.index("say")
    assert events[-1]["type"] == "agent_state" and events[-1]["state"] == "idle"


def test_busy_arm_skips_gesture(monkeypatch):
    monkeypatch.setattr(providers, "get_json", _fake_llm(arm=None))

    class BusyArm(StandinArm):
        def status(self):
            return {**super().status(), "busy": True}

    arm = BusyArm(speed=50)
    events = []
    res = _run(_pipe(events, arm), "Gowther hello", "gowther")
    assert arm.log == []
    assert res["arm_result"]["result"] == "busy"
    assert any(e["type"] == "say" for e in events)


def test_provider_failure_gives_safe_plan(monkeypatch):
    async def broken(*a, **k):
        raise providers.ProviderError("both down")

    monkeypatch.setattr(providers, "get_json", broken)
    events = []
    res = _run(_pipe(events, StandinArm(speed=50)), "King hand me the bottle", "king")
    assert res["plan"]["arm"] is None and res["say"] == cast.SAFE_LINES["king"]
    assert any(e["type"] == "error" and e["where"] == "provider" for e in events)
