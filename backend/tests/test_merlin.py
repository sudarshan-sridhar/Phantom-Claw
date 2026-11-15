"""Offline tests for name matching, Escanor and King rules, and the plan check."""
from datetime import datetime

import pytest

from backend.agents import cast, merlin


@pytest.fixture(autouse=True)
def fresh_memory():
    cast.reset_memory()
    yield
    cast.reset_memory()


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Gauther, what is ohm's law", "gowther"),
        ("Escanoor hand me the bottle", "escanor"),
        ("hey elisabeth make a note", "elizabeth"),
        ("Ban grab it", "ban"),
        ("king, set a timer", "king"),
        ("merlin who should do this", "merlin"),
        ("hand me the bottle", None),
        ("can you make a banana note", None),
        ("thinking about the bottle", None),
    ],
)
def test_names(text, expected):
    assert merlin.find_named_agent(text) == expected


def test_role_choice():
    # research and questions go to Escanor now, timers to Elizabeth (the routing by kind of job)
    assert merlin.choose_by_role("what is a servo?") == "escanor"
    assert merlin.choose_by_role("remind me to buy solder") == "elizabeth"
    assert merlin.choose_by_role("set a timer for 5 minutes") == "elizabeth"


def test_escanor_clock_and_politeness():
    noon = datetime(2026, 11, 15, 12, 0)
    night = datetime(2026, 11, 15, 23, 0)
    assert cast.decide("escanor", "hand me the bottle", noon)["arm"]["skill"] == "point"
    assert cast.decide("escanor", "please hand me the bottle", noon)["arm"] == \
        {"skill": "fetch_bottle", "pace": 0.8, "wait_s": 3, "look_first": False}
    assert cast.decide("escanor", "hand me the bottle", night)["arm"]["skill"] == "fetch_bottle"
    assert cast.decide("escanor", "please lift the bottle", noon)["arm"]["skill"] == "grip_bottle"
    assert cast.decide("escanor", "pick it up", noon)["arm"]["skill"] == "point"
    # asked twice
    d = cast.decide("escanor", "hand me the bottle", noon)
    cast.remember("escanor", "hand me the bottle", cast.outcome_of(d))
    d2 = cast.decide("escanor", "hand me the bottle now", noon)
    assert d2["again"] and d2["arm"]["skill"] == "fetch_bottle"


def test_king_half_then_grip():
    d = cast.decide("king", "hand me the bottle")
    assert d["arm"]["skill"] == "half_reach"
    cast.remember("king", "hand me the bottle", cast.outcome_of(d))
    d2 = cast.decide("king", "give me the bottle")
    assert d2["arm"] == {"skill": "fetch_bottle", "pace": 0.4, "wait_s": 2, "look_first": False}
    cast.reset_memory()
    assert cast.decide("king", "lift the bottle")["arm"]["skill"] == "half_reach"
    cast.remember("king", "lift the bottle", "half")
    assert cast.decide("king", "lift the bottle up")["arm"] == \
        {"skill": "grip_bottle", "pace": 0.4, "hold_s": 1, "flourish": False, "look_first": False}
    cast.remember("king", "hand me the bottle", "half")
    # Merlin hands a repeated request back to whoever dodged it
    assert merlin.choose_by_role("hand me the bottle") == "king"


def test_bad_plan_unknown_skill_replaced():
    bad = {"agent": "ban", "say": "Yeet", "arm": {"skill": "throw_bottle", "pace": 9}, "laptop": []}
    plan, issues, replaced = merlin.check_plan(bad, "ban")
    assert replaced and plan["arm"] is None and plan["say"] == cast.SAFE_LINES["ban"]


def test_gesture_from_llm_rejected():
    bad = {"agent": "gowther", "say": "Talking.", "arm": {"skill": "gesture", "style": "gowther", "seconds": 3}}
    plan, issues, replaced = merlin.check_plan(bad, "gowther")
    assert replaced and plan["arm"] is None


def test_pace_clamped_and_tools_checked():
    raw = {
        "agent": "elizabeth",
        "say": "x" * 300,
        "arm": {"skill": "point", "pace": 9, "laser": True},
        "laptop": [
            {"tool": "note", "text": "buy solder"},
            {"tool": "open_url", "url": "file:///C:/Windows"},
            {"tool": "send_email", "to": "x"},
            {"tool": "timer", "seconds": 99999, "label": "tea"},
        ],
    }
    plan, issues, replaced = merlin.check_plan(raw, "elizabeth")
    assert not replaced
    assert plan["arm"] == {"skill": "point", "pace": 1.0}
    assert len(plan["say"]) <= 140
    assert [c["tool"] for c in plan["laptop"]] == ["note", "timer"]
    assert plan["laptop"][1]["seconds"] == 3600


def test_bottle_job_forces_ego_mapping():
    raw = {"agent": "ban", "say": "Mine!", "arm": {"skill": "grip_bottle", "pace": 0.1, "hold_s": 8}, "laptop": []}
    d = cast.decide("ban", "lift the bottle")
    plan, _, _ = merlin.check_plan(raw, "ban", d)
    assert plan["arm"] == {"skill": "grip_bottle", "pace": 1.0, "hold_s": 3, "flourish": True, "look_first": False}
    # a hand-over request gets fetch_bottle even when the LLM picks the wrong skill
    d = cast.decide("ban", "hand me the bottle")
    plan, _, _ = merlin.check_plan(raw, "ban", d)
    assert plan["arm"] == {"skill": "fetch_bottle", "pace": 1.0, "wait_s": 2, "look_first": False}


def test_not_json_object_replaced():
    plan, _, replaced = merlin.check_plan("hello", "king")
    assert replaced and plan["arm"] is None
