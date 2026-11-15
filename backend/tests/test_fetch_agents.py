"""The characters and the pipeline with fetch_bottle: request kinds, ego mapping, the plan
check, the auto-rest, and the spoken-line polish. Fake LLM, FakeArm, no network."""
import asyncio
import random
from datetime import datetime

import pytest

from backend.agents import cast, merlin, pipeline, providers
from backend.agents._standin_arm import StandinArm
from backend.arm import FakeArm

NOON = datetime(2026, 11, 15, 12, 0)
NIGHT = datetime(2026, 11, 15, 23, 0)


@pytest.fixture(autouse=True)
def fresh_memory():
    cast.reset_memory()
    yield
    cast.reset_memory()


@pytest.mark.parametrize("text,kind", [
    ("get me the bottle", "fetch"), ("hand me the bottle, please", "fetch"), ("pass it here", "fetch"),
    ("bring it over", "fetch"), ("give me the water", "fetch"), ("can I have the bottle", "fetch"),
    ("grab me that", "fetch"), ("the bottle, please", "fetch"),
    ("pick it up", "grip"), ("pick up the bottle", "grip"), ("lift the bottle", "grip"), ("hold it up", "grip"),
    ("Ban grab it", "grip"),
    ("what's in the bottle?", None), ("give me a minute", None), ("get it done", None), ("set a timer", None),
    ("hold that thought", None), ("wave at me", None),
])
def test_request_kinds(text, kind):
    assert cast.bottle_kind(text) == kind


@pytest.mark.parametrize("agent,text,clock,want", [
    ("gowther", "get me the bottle", NOON, {"skill": "fetch_bottle", "pace": 0.6, "wait_s": 3, "look_first": False}),
    ("ban", "hand me the bottle", NOON, {"skill": "fetch_bottle", "pace": 1.0, "wait_s": 2, "look_first": False}),
    ("elizabeth", "pass it here", NOON, {"skill": "fetch_bottle", "pace": 0.3, "wait_s": 5, "look_first": True}),
    ("escanor", "hand me the bottle", NOON, {"skill": "point", "pace": 0.6}),
    ("escanor", "hand me the bottle, please", NOON, {"skill": "fetch_bottle", "pace": 0.8, "wait_s": 3, "look_first": False}),
    ("escanor", "hand me the bottle", NIGHT, {"skill": "fetch_bottle", "pace": 0.8, "wait_s": 3, "look_first": False}),
    ("king", "hand me the bottle", NOON, {"skill": "half_reach", "pace": 0.3}),
    ("gowther", "lift the bottle", NOON, {"skill": "grip_bottle", "pace": 0.5, "hold_s": 5, "flourish": False, "look_first": False}),
])
def test_ego_mapping_for_fetch(agent, text, clock, want):
    assert cast.decide(agent, text, clock)["arm"] == want


def test_asked_again_escanor_and_king_fetch():
    for agent, want in (("escanor", {"skill": "fetch_bottle", "pace": 0.8, "wait_s": 3, "look_first": False}),
                        ("king", {"skill": "fetch_bottle", "pace": 0.4, "wait_s": 2, "look_first": False})):
        d = cast.decide(agent, "get me the bottle", NOON)
        cast.remember(agent, "get me the bottle", cast.outcome_of(d))
        assert cast.decide(agent, "get me the bottle", NOON)["arm"] == want


def test_plan_check_clamps_fetch_and_keeps_arm_for_physical_only():
    raw = {"agent": "gowther", "say": "Fetching.", "arm": {"skill": "fetch_bottle", "pace": 7, "wait_s": 60, "speed": 2}}
    plan, issues, replaced = merlin.check_plan(raw, "gowther")
    assert not replaced
    # not a bottle job by the text, so the agent's own fetch mapping is applied
    assert plan["arm"] == cast.ARM_MAP["gowther"]["fetch"]
    a, _ = merlin._check_arm({"skill": "fetch_bottle", "pace": 7, "wait_s": 60, "look_first": 1, "speed": 2}, "ban", issues := [])
    assert a == {"skill": "fetch_bottle", "pace": 1.0, "wait_s": 6, "look_first": True}
    assert any("dropped arm param speed" in i for i in issues) and any("clamped wait_s" in i for i in issues)
    # a bottle job gets the decided arm even if the LLM forgot it
    d = cast.decide("ban", "hand me the bottle", NOON)
    plan, _, _ = merlin.check_plan({"agent": "ban", "say": "On it.", "arm": None}, "ban", d)
    assert plan["arm"]["skill"] == "fetch_bottle"
    # a question about the bottle is not physical: the arm stays off
    d = cast.decide("ban", "what's in the bottle?", NOON)
    plan, _, _ = merlin.check_plan({"agent": "ban", "say": "Water, I think.", "arm": None}, "ban", d)
    assert plan["arm"] is None


def test_prompts_mention_fetch():
    assert "fetch_bottle" in cast.system_prompt("ban") and "given, handed, passed or brought" in cast.system_prompt("ban")
    up = cast.user_prompt("ban", "get me the bottle", cast.decide("ban", "get me the bottle", NOON))
    assert '"skill": "fetch_bottle"' in up and "hold it out" in up


# spoken lines
@pytest.mark.parametrize("raw,want", [
    ("I am taking it now.", "I'm taking it now."),
    ("We are doing this. It is right there.", "We're doing this. It's right there."),
    ("I will not do that.", "I won't do that."),
    ("Do not worry, I have got it.", "Don't worry, I've got it."),
    ("Cannot reach it.", "Can't reach it."),
    ("Here is your bottle.", "Here's your bottle."),
    ("Yes, I am.", "Yes, I am."),
    ("Here it is.", "Here it is."),
    ("That is not mine.", "That isn't mine."),
])
def test_contractions(raw, want):
    assert cast.contract(raw) == want


def test_elizabeth_offers_only_now_and_then():
    line = "Here you go, careful now. Is there anything else you need?"
    out = []
    for _ in range(6):
        d = cast.decide("elizabeth", "get me the bottle", NOON)
        plan, _, _ = merlin.check_plan({"agent": "elizabeth", "say": line, "arm": None}, "elizabeth", d)
        out.append(plan["say"])
    offers = [s for s in out if "anything else" in s]
    assert len(offers) == 2 and out[2] == line and out[5] == line      # lines 3 and 6 only
    for raw, want in (("I'll lift it up for you, and do you need anything else?", "I'll lift it up for you."),
                      ("I'll pop that on your list, sweetie. Need anything else at all?", "I'll pop that on your list, sweetie."),
                      ("Saved. Let me know if you need anything else!", "Saved.")):
        cast.reset_memory()
        assert cast.polish_line("elizabeth", raw) == want
    assert all(s == "Here you go, careful now." for s in out if s not in offers)
    # other characters keep their lines (apart from contractions)
    plan, _, _ = merlin.check_plan({"agent": "ban", "say": "I am fast. Anything else?", "arm": None}, "ban")
    assert plan["say"] == "I'm fast. Anything else?"


def test_elizabeth_prompt_says_when_she_may_offer():
    d = cast.decide("elizabeth", "make a note", NOON)
    assert "Do not ask" in cast.user_prompt("elizabeth", "make a note", d)
    cast.polish_line("elizabeth", "Saved it.")
    cast.polish_line("elizabeth", "Done.")
    d = cast.decide("elizabeth", "make a note", NOON)
    assert "may end by asking" in cast.user_prompt("elizabeth", "make a note", d)
    cast.polish_line("elizabeth", "Saved it. Anything else?")
    d = cast.decide("elizabeth", "make a note", NOON)
    assert "Do not ask if they need anything else" in cast.user_prompt("elizabeth", "make a note", d)


# pipeline auto-rest
def _pipe(events, arm):
    async def emit(ev):
        events.append(ev)
    return pipeline.Pipeline(arm, emit, speak=False, run_tools=False, pace_audio=False, rng=random.Random(3))


def _fake_llm(arm=None):
    async def fake(provider, system, user, agent=None, models=None):
        return {"agent": agent, "say": "Fine.", "arm": arm, "laptop": []}, {
            "provider": provider, "model": "fake", "fallback": False, "ms": 1, "errors": []}
    return fake


def test_pipeline_runs_fetch_for_hand_me(monkeypatch):
    monkeypatch.setattr(providers, "get_json", _fake_llm(arm=None))
    arm = FakeArm(time_scale=0.001)
    events = []
    res = asyncio.run(_pipe(events, arm).handle("Gowther, get me the bottle", None))
    assert res["plan"]["arm"] == {"skill": "fetch_bottle", "pace": 0.6, "wait_s": 3, "look_first": False}
    assert res["arm_result"]["result"] == "handed_over" and res["arm_result"]["ok"]
    phases = [e["phase"] for e in events if e["type"] == "arm"]
    assert phases == ["start", "seen", "at_target", "gripped", "lifted", "carrying", "offering", "handed_over",
                      "returning", "done"]


def test_pipeline_auto_rest_after_stop(monkeypatch):
    monkeypatch.setattr(providers, "get_json", _fake_llm(arm={"skill": "wave"}))
    arm = FakeArm(time_scale=0.001)
    arm.run("fetch_bottle", {}, lambda p: arm.stop() if p == "carrying" else None)   # left holding the bottle
    assert arm.status()["at_rest"] is False and arm.status()["holding"] is True
    events = []
    res = asyncio.run(_pipe(events, arm).handle("Ban, wave at me", None))
    ar = res["arm_result"]
    assert ar["skill"] == "wave" and ar["result"] == "done"
    assert ar["auto_rest"]["result"] == "done" and "put the bottle down" in ar["auto_rest"]["detail"]
    arm_ev = [(e["skill"], e["phase"], e.get("auto", False)) for e in events if e["type"] == "arm"]
    assert arm_ev[0] == ("rest", "auto_rest", True)
    assert [p for s, p, a in arm_ev if a] == ["auto_rest", "start", "put_back", "returning", "done"]
    assert [p for s, p, a in arm_ev if not a] == ["start", "at_target", "returning", "done"]
    assert arm.status()["holding"] is False


def test_pipeline_gesture_after_stop_is_replaced_by_rest(monkeypatch):
    monkeypatch.setattr(providers, "get_json", _fake_llm(arm=None))
    arm = FakeArm(time_scale=0.001)
    arm.run("wave", {}, lambda p: arm.stop() if p == "at_target" else None)
    events = []
    res = asyncio.run(_pipe(events, arm).handle("Gowther, what time is it", "gowther"))
    assert res["arm_result"]["result"] == "skipped" and res["arm_result"]["auto_rest"]["ok"]
    assert arm.status()["at_rest"] is True


def test_pipeline_flag_for_arms_without_at_rest(monkeypatch):
    """The stand-in arm reports no at_rest: the pipeline remembers a stop itself."""
    monkeypatch.setattr(providers, "get_json", _fake_llm(arm={"skill": "point"}))
    arm = StandinArm(speed=200)
    events = []
    p = _pipe(events, arm)
    p._left_out = True
    asyncio.run(p.handle("Ban, point at it", None))
    assert [s for s, _ in arm.log] == ["rest", "point"]
    asyncio.run(p.handle("Ban, point at it", None))
    assert [s for s, _ in arm.log] == ["rest", "point", "point"]
