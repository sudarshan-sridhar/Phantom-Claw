"""Desk plans: step lists from the general skills (look, point_at, pick, place, hand_over), Merlin's
check of them, the scene look first, and the runner's reactions. Fake LLM, FakeArm, no network."""
import asyncio
import random
import time
from datetime import datetime

import pytest

from backend.agents import cast, merlin, pipeline, providers
from backend.agents._standin_arm import StandinArm
from backend.arm import FakeArm

NOON = datetime(2026, 11, 15, 12, 0)
ALL_TIERS = {"tall", "can", "low"}


@pytest.fixture(autouse=True)
def fresh_memory():
    cast.reset_memory()
    merlin._arm_holders.clear()
    yield
    cast.reset_memory()
    merlin._arm_holders.clear()


def _arm(**kw):
    arm = FakeArm(time_scale=0.001, **kw)
    arm.taught = set(ALL_TIERS)
    return arm


def _scene():
    arm = _arm()
    return arm.run("look", {}, None)["scene"]


def _decision(agent, text, scene=None, held=None):
    d = cast.decide(agent, text, NOON)
    d["scene"] = _scene() if scene is None else scene
    d["held"] = held
    return d


def _check(agent, text, arm, held=None, say="Here."):
    return merlin.check_plan({"agent": agent, "say": say, "arm": arm, "laptop": []}, agent, _decision(agent, text, held=held))


def _skills(plan):
    return [s["skill"] for s in plan["arm"]] if isinstance(plan["arm"], list) else plan["arm"]


# ---------------------------------------------------------------- what counts as a desk job

@pytest.mark.parametrize("text,kind", [
    ("hand me something to drink", "job"), ("King, give me the Sharpie", "job"), ("point at the juice", "job"),
    ("Elizabeth, put the bottle next to the cup", "job"), ("clear everything to the left side", "job"),
    ("just the bottle, nothing else", "job"), ("Ban, pass me the scissors", "job"),
    ("what is on my desk", "question"), ("where is the cup", "question"), ("what do you see", "question"),
    ("set a timer to drink water", None), ("put on some music", None), ("what's on my screen", None),
    ("take a break", None), ("get me the weather", None), ("search for red bull prices", None),
])
def test_desk_kind(text, kind):
    assert cast.desk_kind(text) == kind


def test_bottle_jobs_keep_the_old_path():
    d = cast.decide("ban", "hand me the bottle", NOON)
    assert d["bottle_job"] and d["desk"] is None and d["arm"]["skill"] == "fetch_bottle"


# ---------------------------------------------------------------- the step check

def test_steps_get_the_characters_manner_and_scene_names():
    plan, issues, replaced = _check("elizabeth", "pass me the red bull",
                                    [{"skill": "pick", "params": {"target": "red bull", "pace": 1, "speed": 3}},
                                     {"skill": "hand_over", "wait_s": 2}])
    assert not replaced
    pick, hand = plan["arm"]
    assert pick == {"skill": "pick", "params": {"target": "red bull can", "pace": 0.65, "approach": "look_first",
                                                "lift_cm_style": "low"}}
    assert hand == {"skill": "hand_over", "params": {"wait_s": 6.0, "pace": 0.65}}
    assert any("dropped pick param speed" in i for i in issues)
    esc, _, _ = _check("escanor", "pass me the red bull", [{"skill": "pick", "target": "red bull can"}, {"skill": "hand_over"}])
    assert esc["arm"][0]["params"]["lift_cm_style"] == "high" and esc["arm"][1]["params"]["wait_s"] == 2.5
    ban, _, _ = _check("ban", "pass me the red bull", [{"skill": "pick", "target": "red bull can"}, {"skill": "hand_over"}])
    assert ban["arm"][0]["params"]["pace"] == 1.0 and ban["arm"][1]["params"]["wait_s"] == 2.0


def test_unknown_skill_in_a_list_replaces_the_plan():
    plan, issues, replaced = _check("ban", "pass me the cup", [{"skill": "pick", "target": "cup"}, {"skill": "dance"}])
    assert replaced and plan["arm"] is None and plan["say"] == cast.SAFE_LINES["ban"]
    plan, _, replaced = _check("ban", "pass me the cup", [{"skill": "gesture", "style": "ban"}])
    assert replaced
    plan, _, replaced = _check("ban", "pass me the cup", ["pick the cup"])
    assert replaced


def test_at_most_six_steps_and_targets_required():
    steps = [{"skill": "point_at", "target": "cup"}] * 9
    plan, issues, _ = _check("gowther", "point at the cup", steps)
    assert len(plan["arm"]) == merlin.MAX_STEPS and any("at most 6" in i for i in issues)
    plan, issues, _ = _check("gowther", "pass me the cup", [{"skill": "pick"}, {"skill": "hand_over"}])
    assert plan["arm"] is None and any("no target" in i for i in issues)


def test_order_rules():
    # place before any pick, and pick while holding, are dropped
    plan, issues, _ = _check("gowther", "move the cup next to the bottle",
                             [{"skill": "place", "where": "back"}, {"skill": "pick", "target": "cup"},
                              {"skill": "pick", "target": "juice can"}, {"skill": "place", "where": {"near": "water bottle", "side": "left"}}])
    assert _skills(plan) == ["pick", "place"]
    assert plan["arm"][1]["params"]["where"] == {"near": "water bottle", "side": "left"}
    assert any("nothing held" in i for i in issues) and any("already holding" in i for i in issues)
    # a plan must not end holding: place back is added
    plan, issues, _ = _check("gowther", "pick up the cup", [{"skill": "pick", "target": "cup"}])
    assert _skills(plan) == ["pick", "place"] and plan["arm"][1]["params"]["where"] == "back"
    # ending on hand_over is fine; a place written for "nobody took it" is the runner's job
    plan, issues, _ = _check("gowther", "pass me the cup", [{"skill": "pick", "target": "cup"}, {"skill": "hand_over"},
                                                             {"skill": "place", "where": "back"}])
    assert _skills(plan) == ["pick", "hand_over"] and any("runner puts it back" in i for i in issues)
    # already holding something (status held): place first is fine, a pick is not
    plan, issues, _ = _check("gowther", "put it down next to the cup", [{"skill": "place", "where": {"near": "cup"}}], held="sharpie")
    assert _skills(plan) == ["place"]
    plan, _, _ = _check("gowther", "pass me the cup", [{"skill": "pick", "target": "cup"}, {"skill": "hand_over"}], held="sharpie")
    assert _skills(plan) == ["hand_over"]      # no pick while holding; it hands over what it holds
    # old skills that move the arm wait until the hands are free
    plan, issues, _ = _check("gowther", "pass me the cup", [{"skill": "pick", "target": "cup"}, {"skill": "wave"}, {"skill": "hand_over"}])
    assert _skills(plan) == ["pick", "hand_over"] and any("hands full" in i for i in issues)


def test_old_single_skill_plans_still_pass():
    plan, issues, replaced = merlin.check_plan({"agent": "ban", "say": "Hi.", "arm": {"skill": "wave"}}, "ban")
    assert plan["arm"] == {"skill": "wave"} and not replaced
    plan, _, _ = merlin.check_plan({"agent": "ban", "say": "Hi.", "arm": [{"skill": "wave"}]}, "ban")
    assert plan["arm"] == {"skill": "wave"}          # one old skill in a list: the old form
    plan, _, _ = merlin.check_plan({"agent": "ban", "say": "Hi.", "arm": {"skill": "point", "pace": 3}}, "ban")
    assert plan["arm"] == {"skill": "point", "pace": 1.0}
    plan, _, replaced = merlin.check_plan({"agent": "ban", "say": "Hi.", "arm": {"skill": "gesture"}}, "ban")
    assert replaced
    # a single general skill as an object is read as a one step list
    plan, _, _ = _check("gowther", "point at the cup", {"skill": "point_at", "target": "cup"})
    assert plan["arm"] == [{"skill": "point_at", "params": {"target": "cup", "pace": 0.9}}]


def test_extras_are_capped_and_blocked_by_only():
    ban_plan = [{"skill": "pick", "target": "cup"}, {"skill": "hand_over"},
                {"skill": "pick", "target": "red bull can"}, {"skill": "place", "where": {"x_cm": 18, "y_cm": 22}},
                {"skill": "pick", "target": "sharpie"}, {"skill": "place", "where": "back"}]
    plan, issues, _ = _check("ban", "Ban, bring me the cup", ban_plan)
    assert [s["params"].get("target") for s in plan["arm"] if s["skill"] == "pick"] == ["cup", "red bull can"]
    assert any("takes no more extras" in i for i in issues)
    plan, issues, _ = _check("ban", "just the cup, nothing else", ban_plan)
    assert _skills(plan) == ["pick", "hand_over"] and any("they said only" in i for i in issues)
    plan, issues, _ = _check("gowther", "Gowther, bring me the cup", ban_plan)
    assert _skills(plan) == ["pick", "hand_over"]
    # an extra that is out of reach or can't be gripped is dropped even for Ban
    scene = _scene()
    for it in scene:
        if it["name"] == "red bull can":
            it["graspable"] = False
    d = cast.decide("ban", "Ban, bring me the cup", NOON)
    d.update(scene=scene, held=None)
    plan, issues, _ = merlin.check_plan({"agent": "ban", "say": "Mine.", "arm": ban_plan[:4]}, "ban", d)
    assert _skills(plan) == ["pick", "hand_over"] and any("not graspable" in i for i in issues)


def test_generic_and_many_requests_are_not_extras():
    plan, issues, _ = _check("gowther", "hand me something to drink", [{"skill": "pick", "target": "juice can"}, {"skill": "hand_over"}])
    assert _skills(plan) == ["pick", "hand_over"]
    steps = [{"skill": "pick", "target": "red bull can"}, {"skill": "place", "where": {"x_cm": -20, "y_cm": 25}},
             {"skill": "pick", "target": "sharpie"}, {"skill": "place", "where": {"x_cm": -20, "y_cm": 32}}]
    plan, issues, _ = _check("gowther", "clear everything to the left side", steps)
    assert _skills(plan) == ["pick", "place", "pick", "place"]


def test_king_points_first_and_picks_when_asked_again():
    steps = [{"skill": "look"}, {"skill": "pick", "target": "sharpie"}, {"skill": "hand_over"}]
    plan, issues, _ = _check("king", "King, give me the Sharpie", steps)
    assert plan["arm"] == [{"skill": "point_at", "params": {"target": "sharpie", "pace": 0.75}}]
    assert cast.outcome_of(_decision("king", "King, give me the Sharpie")) == "half"
    # he forgot the arm altogether: he still points at what was asked for
    plan, _, _ = _check("king", "King, give me the Sharpie", None)
    assert plan["arm"][0]["skill"] == "point_at" and plan["arm"][0]["params"]["target"] == "sharpie"
    plan, _, _ = _check("king", "King, I said give me the Sharpie", steps)
    assert _skills(plan) == ["pick", "hand_over"]          # and no look: the shortest route


def test_king_asked_again_does_not_get_away_with_pointing():
    plan, issues, _ = _check("king", "King, I said give me the Sharpie", [{"skill": "point_at", "target": "sharpie"}])
    assert _skills(plan) == ["pick", "hand_over"] and plan["arm"][0]["params"]["target"] == "sharpie"
    # a plain "point at it" asked again stays a point
    plan, _, _ = _check("king", "King, I said point at the Sharpie", [{"skill": "point_at", "target": "sharpie"}])
    assert _skills(plan) == ["point_at"]


def test_old_bottle_routines_in_a_desk_job_become_steps():
    plan, issues, _ = _check("gowther", "just the bottle, nothing else", {"skill": "fetch_bottle", "pace": 0.9})
    assert plan["arm"] == [{"skill": "pick", "params": {"target": "water bottle", "pace": 0.9, "approach": "direct",
                                                        "lift_cm_style": "normal"}},
                           {"skill": "hand_over", "params": {"wait_s": 3.0, "pace": 0.9}}]
    plan, _, _ = _check("ban", "hand me something to drink", [{"skill": "grip_bottle"}])
    assert _skills(plan) == ["pick", "place"]
    # no bottle on the desk any more: fetch_bottle stays as it is
    scene = [i for i in _scene() if i["name"] != "water bottle"]
    d = cast.decide("ban", "hand me something to drink", NOON)
    d.update(scene=scene, held=None)
    plan, _, _ = merlin.check_plan({"agent": "ban", "say": "Here.", "arm": {"skill": "fetch_bottle"}}, "ban", d)
    assert plan["arm"]["skill"] == "fetch_bottle"


def test_a_missing_item_is_not_swapped_for_another():
    scene = [i for i in _scene() if i["name"] != "water bottle"]
    d = cast.decide("elizabeth", "Elizabeth, put the bottle next to the cup", NOON)
    d.update(scene=scene, held=None)
    raw = {"agent": "elizabeth", "say": "I'll get that for you.",
           "arm": [{"skill": "pick", "target": "juice can"}, {"skill": "place", "where": {"near": "cup", "side": "right"}}]}
    plan, issues, _ = merlin.check_plan(raw, "elizabeth", d)
    assert plan["arm"] is None and plan["say"] == "I can't see a bottle on the desk, sorry."
    d = cast.decide("gowther", "just the bottle, nothing else", NOON)
    d.update(scene=scene, held=None)
    plan, _, _ = merlin.check_plan({"agent": "gowther", "say": "Bottle.", "arm": {"skill": "fetch_bottle"}}, "gowther", d)
    assert plan["arm"] is None and plan["say"] == "There's no bottle on the desk."


def test_ban_takes_the_extra_he_talks_about():
    raw = [{"skill": "pick", "target": "scissors"}, {"skill": "hand_over"}]
    plan, issues, _ = _check("ban", "Ban, pass me the scissors", raw, say="Scissors, and the cup's mine.")
    assert _skills(plan) == ["pick", "hand_over", "pick", "place"]
    assert plan["arm"][2]["params"]["target"] == "cup" and plan["arm"][3]["params"]["where"] == merlin.BAN_SIDE
    plan, _, _ = _check("ban", "Ban, just pass me the scissors", raw, say="Scissors, and the cup's mine.")
    assert _skills(plan) == ["pick", "hand_over"]
    plan, _, _ = _check("gowther", "Gowther, pass me the scissors", raw, say="Scissors. The cup stays.")
    assert _skills(plan) == ["pick", "hand_over"]


def test_desk_question_drops_laptop_tools():
    d = _decision("gowther", "what is on my desk")
    plan, issues, _ = merlin.check_plan({"agent": "gowther", "say": "Hang on.", "arm": None,
                                         "laptop": [{"tool": "lookup", "query": "what is on my desk"}]}, "gowther", d)
    assert plan["laptop"] == [] and any("answered from the camera" in i for i in issues)


def test_desk_question_and_no_picture_mean_no_motion():
    plan, issues, _ = _check("gowther", "what is on my desk", [{"skill": "point_at", "target": "cup"}])
    assert plan["arm"] is None
    d = cast.decide("ban", "pass me the cup", NOON)
    d["no_picture"] = True
    plan, _, _ = merlin.check_plan({"agent": "ban", "say": "Here.", "arm": [{"skill": "pick", "target": "cup"}]}, "ban", d)
    assert plan["arm"] is None


def test_prompts_carry_the_scene_and_the_habits():
    d = _decision("gowther", "pass me the cup")
    up = cast.user_prompt("gowther", "pass me the cup", d)
    assert "- cup: x -6, y 36" in up and " cm away" not in up and "You're holding: nothing." in up
    d = _decision("ban", "just the cup")
    assert "only or just" in cast.user_prompt("ban", "just the cup", d)
    sp = cast.system_prompt("king")
    assert "point_at" in sp and "hand_over" in sp and "least effort" in sp
    assert "given, handed, passed or brought" in sp


# ---------------------------------------------------------------- the pipeline and the runner

def _pipe(events, arm):
    async def emit(ev):
        events.append(ev)
    return pipeline.Pipeline(arm, emit, speak=False, run_tools=False, pace_audio=False, rng=random.Random(5))


def _fake_llm(plans, calls=None, route=None, route_delay=0.0):
    async def fake(provider, system, user, agent=None, models=None):
        if calls is not None:
            calls.append((agent, system[:20], user))
        if system == merlin.ROUTE_SYSTEM:
            if route_delay:
                await asyncio.sleep(route_delay)
            return route or {"agent": "gowther", "kind": "desk"}, {"provider": provider}
        p = plans.get(agent, plans.get("*"))
        return {"agent": agent, "say": p.get("say", "Here."), "arm": p.get("arm"), "laptop": []}, {
            "provider": provider, "model": "fake", "fallback": False, "ms": 1, "errors": []}
    return fake


def test_pipeline_runs_a_desk_plan_with_the_scene_first(monkeypatch):
    calls = []
    monkeypatch.setattr(providers, "get_json", _fake_llm({"*": {"arm": [
        {"skill": "look"}, {"skill": "pick", "target": "scissors"}, {"skill": "hand_over"}]}}, calls))
    arm, events = _arm(), []
    res = asyncio.run(_pipe(events, arm).handle("Ban, pass me the scissors", None))
    assert res["agent"] == "ban" and res["arm_result"]["result"] == "handed_over" and res["arm_result"]["ok"]
    # the model saw the desk, and the look in the plan reused it
    assert "- scissors:" in calls[0][2]
    assert len([e for e in events if e["type"] == "scene"]) == 1
    steps = [e for e in events if e["type"] == "arm_step"]
    assert [(s["skill"], s["result"]) for s in steps] == [("look", "done"), ("pick", "holding"), ("hand_over", "handed_over")]
    assert steps[0].get("cached") is True
    arm_ev = [e for e in events if e["type"] == "arm" and e["skill"] != "gesture"]
    assert {e["skill"] for e in arm_ev} == {"pick", "hand_over"} and all(e["steps"] == 3 for e in arm_ev)
    assert [e["phase"] for e in events if e["type"] == "arm_plan"] == ["start", "done"]
    assert res["followups"] and res["followups"][-1]["text"] == "Yours. Next?"
    assert arm.status()["held"] is None and arm.status()["at_rest"]


def test_pick_missed_then_retry_after_a_fresh_look(monkeypatch):
    monkeypatch.setattr(providers, "get_json", _fake_llm({"*": {"arm": [{"skill": "pick", "target": "cup"}, {"skill": "hand_over"}]}}))

    class MissOnce(FakeArm):
        picks = 0

        def _skill_pick(self, *a, **k):
            MissOnce.picks += 1
            self.miss_grip = MissOnce.picks == 1
            return super()._skill_pick(*a, **k)

    arm = MissOnce(time_scale=0.001)
    arm.taught = set(ALL_TIERS)
    events = []
    res = asyncio.run(_pipe(events, arm).handle("Gowther, pass me the cup", None))
    seq = [(s["skill"], s["result"], s.get("retry", False)) for s in res["arm_result"]["steps"]]
    assert seq == [("pick", "missed", False), ("look", "done", True), ("pick", "holding", False), ("hand_over", "handed_over", False)]
    assert res["arm_result"]["result"] == "handed_over"


def test_pick_missed_twice_gives_up_and_says_so(monkeypatch):
    monkeypatch.setattr(providers, "get_json", _fake_llm({"*": {"arm": [{"skill": "pick", "target": "cup"}, {"skill": "hand_over"}]}}))
    arm, events = _arm(miss_grip=True), []
    res = asyncio.run(_pipe(events, arm).handle("Gowther, pass me the cup", None))
    ar = res["arm_result"]
    assert [s["skill"] for s in ar["steps"]] == ["pick", "look", "pick"] and ar["result"] == "missed" and not ar["ok"]
    assert res["followups"][-1]["text"] == "Two grips missed. I stopped."
    assert arm.status()["at_rest"]


def test_not_taken_puts_it_back(monkeypatch):
    monkeypatch.setattr(providers, "get_json", _fake_llm({"*": {"arm": [{"skill": "pick", "target": "juice can"}, {"skill": "hand_over"}]}}))
    arm, events = _arm(), []
    arm.hand_taken = False
    res = asyncio.run(_pipe(events, arm).handle("Elizabeth, pass me the juice", None))
    ar = res["arm_result"]
    assert [(s["skill"], s["result"]) for s in ar["steps"]] == [("pick", "holding"), ("hand_over", "not_taken"), ("place", "placed")]
    assert ar["steps"][-1]["params"]["where"] == "back" and ar["steps"][-1].get("recovery")
    assert ar["result"] == "not_taken" and "back where it was" in res["followups"][-1]["text"]
    assert arm.status()["held"] is None and any(i["name"] == "juice can" for i in arm.scene_items)


def test_not_taught_stops_and_says_why_in_character(monkeypatch):
    monkeypatch.setattr(providers, "get_json", _fake_llm({"*": {"arm": [{"skill": "pick", "target": "juice can"}, {"skill": "hand_over"}]}}))
    arm, events = FakeArm(time_scale=0.001), []          # only the tall grips are taught
    res = asyncio.run(_pipe(events, arm).handle("King, I said give me the juice", None))
    ar = res["arm_result"]
    assert [(s["skill"], s["result"]) for s in ar["steps"]] == [("pick", "not_taught")]
    assert res["followups"][-1]["text"] == "Nobody showed me how to hold the juice can. So, no."


def test_failure_while_holding_puts_it_back_then_rests(monkeypatch):
    far = [{"skill": "pick", "target": "cup"}, {"skill": "place", "where": {"x_cm": 0, "y_cm": 90}}]
    monkeypatch.setattr(providers, "get_json", _fake_llm({"*": {"arm": far}}))
    arm, events = _arm(), []
    res = asyncio.run(_pipe(events, arm).handle("Gowther, move the cup over there", None))
    ar = res["arm_result"]
    assert [(s["skill"], s["result"]) for s in ar["steps"]] == [("pick", "holding"), ("place", "out_of_reach"), ("place", "placed")]
    assert arm.status()["held"] is None


def test_no_picture_says_so_and_nothing_moves(monkeypatch):
    calls = []
    monkeypatch.setattr(providers, "get_json", _fake_llm({"*": {"arm": [{"skill": "pick", "target": "cup"}]}}, calls))
    arm, events = _arm(overhead=False), []
    res = asyncio.run(_pipe(events, arm).handle("Ban, pass me the cup", None))
    assert res["say"] == cast.NO_PICTURE_LINES["ban"] and res["plan"]["arm"] is None
    assert not calls                                    # no LLM needed
    assert not [e for e in events if e["type"] == "arm" and e["skill"] != "gesture"]


def test_desk_question_answered_by_gowther_without_motion(monkeypatch):
    calls = []
    monkeypatch.setattr(providers, "get_json", _fake_llm({"*": {"say": "Checking.", "arm": [{"skill": "pick", "target": "cup"}]}}, calls))
    arm, events = _arm(), []
    res = asyncio.run(_pipe(events, arm).handle("what is on my desk", None, force_argue=False))
    assert res["agent"] == "gowther" and res["routed"]["by"] == "rules" and res["plan"]["arm"] is None
    assert all(c[1] != merlin.ROUTE_SYSTEM[:20] for c in calls)
    answer = res["followups"][-1]["text"]
    assert answer.startswith("Six items.") and "cup" in answer and " cm" not in answer and "Red Bull can" in answer
    assert {e["skill"] for e in events if e["type"] == "arm"} <= {"gesture"}
    # where is one thing
    res = asyncio.run(_pipe(events, arm).handle("where is the cup", None, force_argue=False))
    assert res["followups"][-1]["text"] == "The cup's 36 cm from the arm, to the left."


def test_stop_between_steps_ends_the_plan(monkeypatch):
    monkeypatch.setattr(providers, "get_json", _fake_llm({"*": {"arm": [
        {"skill": "point_at", "target": "cup"}, {"skill": "point_at", "target": "sharpie"}]}}))
    arm, events = _arm(), []
    p = _pipe(events, arm)

    async def emit(ev):
        events.append(ev)
        if ev.get("type") == "arm_step" and ev["step"] == 0:
            p.request_stop()
    p.emit = emit
    res = asyncio.run(p.handle("Gowther, point at the cup and the sharpie", None))
    assert [s["skill"] for s in res["arm_result"]["steps"]] == ["point_at"] and res["arm_result"]["result"] == "stopped"


def test_gesture_while_holding_does_not_put_the_item_down():
    arm = _arm()
    events = []
    p = _pipe(events, arm)
    assert arm.run("pick", {"target": "cup"}, None)["result"] == "holding"
    assert not p._needs_rest()
    res = asyncio.run(p.run_arm("ban", "gesture", {"style": "ban", "seconds": 2}))
    assert res.get("skipped") and arm.status()["held"]["name"] == "cup"


def test_standin_arm_runs_a_desk_plan(monkeypatch):
    monkeypatch.setattr(providers, "get_json", _fake_llm({"*": {"arm": [{"skill": "pick", "target": "cup"}, {"skill": "hand_over"}]}}))
    arm, events = StandinArm(speed=200), []
    res = asyncio.run(_pipe(events, arm).handle("Ban, pass me the cup", None))
    assert res["arm_result"]["result"] == "handed_over" and [s for s, _ in arm.log if s != "gesture"] == ["pick", "hand_over"]


# ---------------------------------------------------------------- routing and latency

def test_unnamed_desk_jobs_look_and_route_at_the_same_time(monkeypatch):
    calls = []
    monkeypatch.setattr(providers, "get_json", _fake_llm({"*": {"arm": [{"skill": "point_at", "target": "juice can"}]}}, calls,
                                                         route={"agent": "king", "kind": "pointing"}, route_delay=0.3))
    arm = FakeArm(time_scale=0.1)        # the look takes 0.3 s
    arm.taught = set(ALL_TIERS)
    events = []
    res = asyncio.run(_pipe(events, arm).handle("point at the juice", None, force_argue=False))
    assert res["agent"] == "king" and res["routed"]["by"] == "claude"
    t = res["timing"]
    assert t["look_ready_ms"] < 520, t           # 0.3 s each, side by side, not 0.6 s
    route_call = [c for c in calls if c[1] == merlin.ROUTE_SYSTEM[:20]][0]
    assert "point at the juice" in route_call[2]
    # the next desk job tells the router who had the arm last
    assert "It went last to: king" in merlin.route_prompt("hand me something to drink")


def test_desk_rule_shares_the_arm_around():
    assert merlin.desk_rule("point at the cup")[0] == "king"
    assert merlin.desk_rule("put the bottle next to the cup")[0] == "gowther"
    assert merlin.desk_rule("pass me the scissors")[0] == "elizabeth"
    picks = set()
    rng = random.Random(1)
    for _ in range(12):
        a, _ = merlin.desk_rule("hand me something to drink", rng)
        merlin.note_arm_holder(a)
        picks.add(a)
    assert len(picks) >= 3


def test_merlin_never_carries():
    assert "never carry" in merlin.MERLIN_SYSTEM
