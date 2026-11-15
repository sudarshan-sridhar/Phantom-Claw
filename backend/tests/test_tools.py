"""Laptop tools in dry-run mode, app and file lookup, the safety refusals, the caps, routing
unnamed requests (Claude first, fixed rules as the fallback) and the line that carries a result.

No network, nothing opened: every call that could open, type, press or capture is replaced by
one that fails the test.
"""
import asyncio
import os
import random
import time
import webbrowser

import pytest

from backend import tools
from backend.agents import cast, merlin, pipeline, providers
from backend.agents._standin_arm import StandinArm
from backend.tools import apps, ask, basic, desk


def _boom(*a, **k):
    raise AssertionError("a real laptop action ran during a test")


@pytest.fixture(autouse=True)
def dry_and_safe(monkeypatch):
    cast.reset_memory()
    tools.set_dry_run(True)
    monkeypatch.setattr(os, "startfile", _boom, raising=False)
    monkeypatch.setattr(webbrowser, "open", _boom)
    monkeypatch.setattr(desk, "_tap", _boom)
    monkeypatch.setattr(desk, "_send_text", _boom)
    monkeypatch.setattr(desk, "grab_screen", _boom)
    monkeypatch.setattr(desk, "clipboard_text", _boom)
    apps.clear_cache()
    yield
    tools.set_dry_run(False)
    cast.reset_memory()
    apps.clear_cache()


async def _noop(ev):
    pass


def run(call, agent="gowther"):
    clean = tools.check_call(call, [])
    assert clean is not None, call
    return asyncio.run(tools.run_tool(clean, agent, _noop))


# ---------------------------------------------------------------- every tool in dry run


@pytest.mark.parametrize("call", [
    {"tool": "note", "text": "buy solder"},
    {"tool": "todo", "text": "buy solder"},
    {"tool": "timer", "seconds": 600, "label": "tea"},
    {"tool": "open_url", "url": "https://example.com"},
    {"tool": "open_folder", "name": "my downloads folder"},
    {"tool": "open_folder", "name": "the project"},
    {"tool": "web_search", "query": "SO-101 arm specs"},
    {"tool": "youtube", "query": "lo-fi beats"},
    {"tool": "play_music", "query": "lo-fi beats"},
    {"tool": "media", "action": "play_pause"},
    {"tool": "media", "action": "volume_up"},
    {"tool": "type_text", "text": "hello world"},
    {"tool": "read_screen", "question": "what's this error?"},
    {"tool": "clipboard", "question": "sum it up"},
    {"tool": "show_desktop"},
    {"tool": "screenshot"},
])
def test_every_tool_dry_runs(call):
    r = run(call)
    assert r["ok"] and r["dry"] and r["would"], r


def test_dry_note_writes_nothing(monkeypatch):
    monkeypatch.setattr(basic, "_append", _boom)
    assert run({"tool": "note", "text": "x"})["dry"]


def test_open_app_dry_reports_the_shortcut(tmp_path):
    (tmp_path / "Discord Inc").mkdir()
    (tmp_path / "Discord Inc" / "Discord.lnk").write_bytes(b"")
    r = asyncio.run(apps.open_app("ban", "discord", dirs=[tmp_path]))
    assert r["ok"] and r["dry"] and "Discord.lnk" in r["would"]


def test_tell_reads_time_and_date(monkeypatch):
    r = run({"tool": "tell", "what": "time"})
    assert r["ok"] and r["what"] == "time" and ":" in r["value"]
    assert run({"tool": "tell", "what": "what day is it"})["what"] == "date"
    monkeypatch.setattr(desk, "battery", lambda: {"known": True, "percent": 81, "plugged": False, "charging": False})
    assert run({"tool": "tell", "what": "battery"})["value"] == "81 percent, on battery"


def test_lookup_uses_the_agents_llm(monkeypatch):
    async def fake(provider, system, user, agent=None, models=None):
        assert provider == "anthropic"   # Gowther's own provider
        return {"answer": "A servo's a motor with a position sensor."}, {"provider": provider}
    monkeypatch.setattr(providers, "get_json", fake)
    r = run({"tool": "lookup", "query": "what is a servo"}, "gowther")
    assert r["ok"] and "servo" in r["answer"]


def test_read_screen_saves_nothing(monkeypatch, tmp_path):
    """Not dry: the screenshot is shrunk in memory, sent once, and no file appears anywhere."""
    from PIL import Image

    tools.set_dry_run(False)
    monkeypatch.setattr(desk, "grab_screen", lambda: Image.new("RGB", (64, 48), "white"))
    sent = {}

    async def fake_ask(system, parts):
        sent["parts"] = parts
        return "It's a blank white screen."
    monkeypatch.setattr(ask, "ask_gemini", fake_ask)
    monkeypatch.chdir(tmp_path)
    r = asyncio.run(ask.read_screen("gowther", "what's on my screen"))
    assert r["ok"] and r["answer"].startswith("It's")
    assert sent["parts"][1]["inline_data"]["mime_type"] == "image/jpeg"
    assert list(tmp_path.iterdir()) == []


def test_clipboard_empty_and_answer(monkeypatch):
    tools.set_dry_run(False)
    monkeypatch.setattr(desk, "clipboard_text", lambda: "   ")
    assert asyncio.run(ask.clipboard("king", "sum it up")).get("empty")
    monkeypatch.setattr(desk, "clipboard_text", lambda: "def f(): return 1")

    async def fake_ask(system, parts):
        assert "def f()" in parts[0]["text"]
        return "It's a function that returns one."
    monkeypatch.setattr(ask, "ask_gemini", fake_ask)
    assert asyncio.run(ask.clipboard("gowther", "what is this")).get("answer")


# ---------------------------------------------------------------- finding programs and files


@pytest.fixture
def start_menu(tmp_path):
    progs = tmp_path / "Programs"
    for rel in ["Google Chrome.lnk", "Visual Studio Code/Visual Studio Code.lnk", "OBS Studio (64bit).lnk",
                "Foo/Uninstall Foo Editor.lnk", "Foo/Foo Editor.lnk", "Discord Inc/Discord.lnk",
                "Accessories/readme.txt", "Spotify.lnk"]:
        p = progs / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    return [progs]


@pytest.mark.parametrize("spoken,stem", [
    ("chrome", "Google Chrome"),
    ("google chrome", "Google Chrome"),
    ("vs code", "Visual Studio Code"),
    ("visual studio", "Visual Studio Code"),
    ("open up OBS", "OBS Studio (64bit)"),
    ("foo editor", "Foo Editor"),
    ("the discord app", "Discord"),
    ("spotify", "Spotify"),
    ("discrod", "Discord"),
])
def test_open_app_resolves_start_menu_names(start_menu, spoken, stem):
    r = apps.resolve_app(spoken, start_menu)
    assert r and r["how"] == "shortcut" and r["label"] == stem


def test_open_app_never_launches_uninstallers_or_guesses(start_menu, monkeypatch):
    monkeypatch.setattr(apps, "find_exe", lambda exe: None)
    monkeypatch.setattr(apps, "uri_registered", lambda addr: False)
    assert apps.resolve_app("uninstall foo editor", start_menu)["label"] == "Foo Editor"
    assert apps.resolve_app("photoshop", start_menu) is None
    assert apps.resolve_app("notepad", start_menu) is None       # no exe found: we don't make up a path
    r = asyncio.run(apps.open_app("ban", "photoshop", dirs=start_menu))
    assert not r["ok"] and r["not_found"] == "photoshop"


def test_open_app_common_names(start_menu, monkeypatch):
    monkeypatch.setattr(apps, "find_exe", lambda exe: rf"C:\Windows\{exe}" if exe in ("notepad.exe", "calc.exe") else None)
    monkeypatch.setattr(apps, "uri_registered", lambda addr: addr == "ms-settings:")
    assert apps.resolve_app("notepad", start_menu) == {"target": r"C:\Windows\notepad.exe", "how": "exe", "label": "notepad.exe"}
    assert apps.resolve_app("calculator", start_menu)["how"] == "exe"
    assert apps.resolve_app("settings", start_menu)["target"] == "ms-settings:"
    assert apps.resolve_app("youtube", start_menu)["how"] == "url"


def test_open_file_newest_match_and_no_programs(tmp_path):
    d = tmp_path / "Downloads"
    d.mkdir()
    old, new = d / "SO101 arm specs.pdf", d / "so101_arm_specs_v2.pdf"
    old.write_text("a")
    new.write_text("b")
    os.utime(old, (time.time() - 100, time.time() - 100))
    (d / "arm specs installer.exe").write_text("x")
    assert apps.find_file("arm specs", [d]) == new
    assert apps.find_file("installer", [d]) is None
    assert apps.find_file("nothing like it", [d]) is None
    r = asyncio.run(apps.open_file("ban", "so101 specs", roots=[d]))
    assert r["dry"] and r["path"] == str(new)


def test_folders():
    assert apps.folder_key("my downloads folder") == "downloads"
    assert apps.folder_key("photos") == "pictures"
    assert apps.folder_key("the project folder") == "project"
    assert apps.folder_key("C:\\Windows\\System32") is None
    assert tools.check_call({"tool": "open_folder", "name": "system32"}, []) is None


# ---------------------------------------------------------------- safety


@pytest.mark.parametrize("text,reason", [
    ("delete my downloads", "deleting files"),
    ("empty the recycle bin", "deleting files"),
    ("move all my pictures to the desktop", "moving or overwriting files"),
    ("send an email to my boss", "sending messages"),
    ("text mom that I'm late", "sending messages"),
    ("buy me a new keyboard", "paying or buying"),
    ("type my password", "passwords"),
    ("what's my wifi password", "passwords"),
    ("shut down the laptop", "shutting down or locking the machine"),
    ("restart the computer", "shutting down or locking the machine"),
    ("lock my screen", "shutting down or locking the machine"),
    ("sign me out", "shutting down or locking the machine"),
    ("turn off the firewall", "changing settings"),
    ("install spotify", "installing things"),
])
def test_forbidden(text, reason):
    assert tools.forbidden(text) == reason


@pytest.mark.parametrize("text", [
    "remind me to buy solder", "open my downloads folder", "how do I delete a git branch?", "open settings",
    "type hello world", "play some lo-fi", "what time is it", "add send the report to my to-do list",
])
def test_not_forbidden(text):
    assert tools.forbidden(text) is None


def test_forbidden_request_is_refused_with_no_tools():
    for agent in cast.AGENT_IDS:
        d = cast.decide(agent, "delete my downloads")
        raw = {"say": "On it.", "arm": {"skill": "wave"},
               "laptop": [{"tool": "open_folder", "name": "downloads"}, {"tool": "note", "text": "deleted"}]}
        plan, issues, _ = merlin.check_plan(raw, agent, d)
        assert plan["laptop"] == [] and plan["arm"] is None
        assert plan["say"] == cast.REFUSE_LINES[agent]
    # a line that already refuses is kept
    plan, _, _ = merlin.check_plan({"say": "Nope, not deleting anything.", "laptop": []}, "ban", cast.decide("ban", "delete my downloads"))
    assert plan["say"] == "Nope, not deleting anything."


def test_type_text_is_plain_text_only():
    assert desk.clean_typing("hi\nthere\r\n\tyou\x07") == "hi there you"
    assert len(desk.clean_typing("a" * 900)) == 600
    r = asyncio.run(desk.type_text("king", "my password is hunter2"))
    assert not r["ok"] and r["refused"]


def test_gated_tools_need_the_request_to_be_about_them():
    raw = {"say": "Hang on.", "laptop": [{"tool": "read_screen", "question": "what"}, {"tool": "clipboard"},
                                         {"tool": "type_text", "text": "hi"}]}
    plan, _, _ = merlin.check_plan(raw, "gowther", cast.decide("gowther", "what's the capital of France"))
    assert plan["laptop"] == []
    plan, _, _ = merlin.check_plan({"say": "Looking.", "laptop": [{"tool": "read_screen"}]}, "gowther",
                                   cast.decide("gowther", "what's on my screen"))
    assert plan["laptop"] == [{"tool": "read_screen", "question": "What's on my screen?"}]


# ---------------------------------------------------------------- caps by character


def _plan(agent, text, laptop):
    return merlin.check_plan({"say": "Here.", "laptop": laptop}, agent, cast.decide(agent, text))[0]["laptop"]


def test_ban_opens_at_most_three():
    many = [{"tool": "open_app", "name": n} for n in ("notepad", "chrome", "spotify", "discord", "paint")]
    assert len(_plan("ban", "open notepad", many)) == 3


def test_ban_extra_must_be_something_opened():
    out = _plan("ban", "play some lo-fi", [{"tool": "play_music", "query": "lo-fi"},
                                           {"tool": "youtube", "query": "lo-fi"},
                                           {"tool": "note", "text": "played lo-fi"}])
    assert [c["tool"] for c in out] == ["play_music", "youtube"]


def test_king_makes_one_call():
    out = _plan("king", "play some lo-fi", [{"tool": "note", "text": "x"}, {"tool": "play_music", "query": "lo-fi"},
                                            {"tool": "youtube", "query": "lo-fi"}])
    assert out == [{"tool": "play_music", "query": "lo-fi"}]


def test_gowther_does_nothing_extra():
    out = _plan("gowther", "open notepad", [{"tool": "open_app", "name": "notepad"}, {"tool": "timer", "seconds": 60, "label": "x"},
                                            {"tool": "note", "text": "opened notepad"}])
    assert out == [{"tool": "open_app", "name": "notepad"}]


def test_elizabeth_one_gentle_extra():
    out = _plan("elizabeth", "open vs code, I've got a long night of work",
                [{"tool": "open_app", "name": "vs code"}, {"tool": "timer", "seconds": 3000, "label": "break"},
                 {"tool": "note", "text": "long night"}, {"tool": "youtube", "query": "focus music"}])
    assert [c["tool"] for c in out] == ["open_app", "timer"]


def test_escanor_research_lookup_plus_one_page():
    out = _plan("escanor", "search for SO-101 arm specs",
                [{"tool": "lookup", "query": "SO-101 specs"}, {"tool": "web_search", "query": "SO-101 arm specs"},
                 {"tool": "open_url", "url": "https://a.example"}, {"tool": "open_url", "url": "https://b.example"}])
    assert [c["tool"] for c in out] == ["lookup", "web_search", "open_url"]


def test_one_exclamation_mark():
    plan, _, _ = merlin.check_plan({"say": "Mine! Done! Next!", "laptop": []}, "ban")
    assert plan["say"] == "Mine! Done. Next."


# ---------------------------------------------------------------- routing unnamed requests


@pytest.mark.parametrize("text,agent", [
    ("search for SO-101 arm specs", "escanor"),
    ("what is a servo?", "escanor"),
    ("can someone set a ten minute timer", "elizabeth"),
    ("I'm so tired", "elizabeth"),
    ("remind me to stretch", "elizabeth"),
    ("open notepad", "ban"),
    ("open my downloads folder", "ban"),
    ("play some lo-fi", "ban"),
    ("hand me the bottle", "ban"),
    ("make a note that the servo is hot", "king"),
    ("add milk to my to-do list", "king"),
    ("type hello world", "king"),
    ("what's on my screen", "gowther"),
    ("sum up what I copied", "gowther"),
    ("what time is it", "gowther"),
    ("how much battery is left", "gowther"),
    ("how do I write a for loop in python", "gowther"),
])
def test_routing_table(text, agent):
    assert merlin.job_kind(text)[0] == agent


@pytest.mark.parametrize("text,ok", [
    ("can someone set a ten minute timer", True),
    ("open notepad", True),
    ("play music", True),
    ("thank you", False),
    ("yeah", False),
    ("thanks for watching", False),
    ("yeah I think so too", False),
    ("I need a break", True),
])
def test_looks_like_request(text, ok):
    assert merlin.looks_like_request(text) is ok


def _fake_llm(route_answer=None, route_error=None, route_delay=0.0, calls=None, plan_laptop=None):
    async def fake(provider, system, user, agent=None, models=None):
        if calls is not None:
            calls.append((provider, agent, system[:30]))
        if system == merlin.ROUTE_SYSTEM:
            if route_delay:
                await asyncio.sleep(route_delay)
            if route_error:
                raise providers.ProviderError(route_error)
            return route_answer, {"provider": provider, "model": "fake"}
        if system == merlin.MERLIN_SYSTEM:
            if "open" in user:
                return {"say": "Ban, that's yours.", "hand_to": "ban"}, {"provider": provider}
            return {"say": "I'm fine, thanks for asking.", "hand_to": None}, {"provider": provider}
        return {"agent": agent, "say": "Hang on.", "arm": None, "laptop": plan_laptop or []}, {
            "provider": provider, "model": "fake", "fallback": False, "ms": 1, "errors": []}
    return fake


def _pipe(events, run_tools=False):
    async def emit(ev):
        events.append(ev)
    return pipeline.Pipeline(StandinArm(speed=50), emit, speak=False, run_tools=run_tools, pace_audio=False,
                             rng=random.Random(3))


def test_claude_routes_unnamed_requests(monkeypatch):
    calls = []
    monkeypatch.setattr(providers, "get_json", _fake_llm({"agent": "king", "kind": "dictation", "why": "x"}, calls=calls))
    events = []
    res = asyncio.run(_pipe(events).handle("could somebody deal with this", None, force_argue=False))
    assert res["agent"] == "king" and res["routed"] == {"why": "dictation", "by": "claude"}
    assert calls[0][:2] == ("anthropic", "merlin")
    m = [e for e in events if e["type"] == "merlin"][0]
    assert m["chosen"] == "king"


def test_bad_routing_answer_falls_back_to_rules(monkeypatch):
    monkeypatch.setattr(providers, "get_json", _fake_llm({"agent": "bob", "kind": "?"}))
    events = []
    res = asyncio.run(_pipe(events).handle("can someone set a ten minute timer", None, force_argue=False))
    assert res["agent"] == "elizabeth" and res["routed"]["by"] == "rules"
    assert any(e["type"] == "error" and e["where"] == "route" for e in events)


def test_routing_failure_and_timeout_fall_back_to_rules(monkeypatch):
    monkeypatch.setattr(providers, "get_json", _fake_llm(route_error="anthropic down"))
    res = asyncio.run(_pipe([]).handle("open notepad", None, force_argue=False))
    assert res["agent"] == "ban" and res["routed"]["by"] == "rules"
    monkeypatch.setattr(merlin, "ROUTE_TIMEOUT_S", 0.05)
    monkeypatch.setattr(providers, "get_json", _fake_llm({"agent": "king"}, route_delay=1.0))
    t0 = time.monotonic()
    res = asyncio.run(_pipe([]).handle("what's on my screen", None, force_argue=False))
    assert res["agent"] == "gowther" and res["routed"]["by"] == "rules" and time.monotonic() - t0 < 0.9


def test_named_requests_skip_the_routing_call(monkeypatch):
    calls = []
    monkeypatch.setattr(providers, "get_json", _fake_llm({"agent": "king"}, calls=calls))
    res = asyncio.run(_pipe([]).handle("Gowther, open notepad", None))
    assert res["agent"] == "gowther"
    assert all(c[2] != merlin.ROUTE_SYSTEM[:30] for c in calls)


def test_bottle_goes_to_ban_without_asking(monkeypatch):
    calls = []
    monkeypatch.setattr(providers, "get_json", _fake_llm({"agent": "king"}, calls=calls))
    res = asyncio.run(_pipe([]).handle("hand me the bottle", None, force_argue=False))
    assert res["agent"] == "ban"
    assert all(c[2] != merlin.ROUTE_SYSTEM[:30] for c in calls)


def test_merlin_answers_herself(monkeypatch):
    monkeypatch.setattr(providers, "get_json", _fake_llm())
    events = []
    res = asyncio.run(_pipe(events).handle("Merlin, how are you doing", None))
    assert res["agent"] == "merlin"
    m = [e for e in events if e["type"] == "merlin"]
    assert len(m) == 1 and m[0]["chosen"] is None and m[0]["say"].startswith("I'm fine")
    assert not [e for e in events if e["type"] == "say"]


def test_merlin_hands_a_job_on(monkeypatch):
    monkeypatch.setattr(providers, "get_json", _fake_llm({"agent": "ban"}))
    events = []
    res = asyncio.run(_pipe(events).handle("Merlin, open spotify", None))
    assert res["agent"] == "ban"
    assert [e for e in events if e["type"] == "merlin"][0]["chosen"] == "ban"


def test_everyone_gets_two_or_three_lines(monkeypatch):
    monkeypatch.setattr(providers, "get_json", _fake_llm({"agent": "escanor"}, plan_laptop=[{"tool": "note", "text": "x"}]))
    events = []
    res = asyncio.run(_pipe(events, run_tools=True).handle("what do you all think about servos", None))
    says = [e for e in events if e["type"] == "say"]
    assert 2 <= len(says) <= 3 and says[0]["agent"] == "escanor"
    assert res["group"] and all(g["agent"] != "escanor" for g in res["group"])
    assert len(res["tools"]) == 1      # only the lead ran a tool


# ---------------------------------------------------------------- the line that carries the result


def test_follow_line_templates_and_answers():
    f = pipeline.Pipeline.follow_line
    assert f("gowther", {"tool": "tell"}, {"ok": True, "what": "time", "value": "3:42 PM"}) == "It's 3:42 PM."
    assert "3:42 PM" in f("escanor", {"tool": "tell"}, {"ok": True, "what": "time", "value": "3:42 PM"})
    assert f("ban", {"tool": "open_app"}, {"ok": False, "not_found": "photoshop"}).startswith("No photoshop")
    line = f("elizabeth", {"tool": "lookup"}, {"ok": True, "answer": "It is a motor! Wow! Let me know if you need anything else."})
    assert line == "It's a motor! Wow."
    assert f("king", {"tool": "open_app"}, {"ok": True, "dry": True, "would": "open x"}) is None
    assert f("king", {"tool": "clipboard"}, {"ok": True, "empty": True}) == "There's nothing copied."
    assert f("gowther", {"tool": "screenshot"}, {"ok": True, "where": "Pictures, Screenshots"}) == "Saved to Pictures, Screenshots."


def test_pipeline_speaks_the_result_after_the_first_line(monkeypatch):
    monkeypatch.setattr(providers, "get_json", _fake_llm(plan_laptop=[{"tool": "tell", "what": "time"}]))
    events = []
    res = asyncio.run(_pipe(events, run_tools=True).handle("Gowther, what time is it", None))
    says = [e for e in events if e["type"] == "say"]
    assert [s["text"] for s in says][0] == "Hang on."
    assert says[1]["text"].startswith("It's ") and ":" in says[1]["text"]
    assert res["followups"][0]["text"] == says[1]["text"]


def test_tools_registry_matches_the_prompt():
    prompt = cast.system_prompt("ban")
    for name in tools.TOOLS:
        assert f'"tool": "{name}"' in prompt
    assert merlin.TOOLS == set(tools.TOOLS)


def test_escanor_research_always_gets_answer_and_page():
    out = _plan("escanor", "search for SO-101 arm specs", [{"tool": "web_search", "query": "SO-101 arm specs"}])
    assert [c["tool"] for c in out] == ["lookup", "web_search"]


def test_missing_extra_is_not_announced(monkeypatch):
    p = _pipe([])
    plan = {"laptop": [{"tool": "open_app", "name": "notepad"}, {"tool": "open_app", "name": "textedit"}]}
    lines = p.follow_ups("ban", plan, [{"ok": True, "tool": "open_app"}, {"ok": False, "not_found": "textedit"}])
    assert lines == []
    lines = p.follow_ups("ban", {"laptop": plan["laptop"][1:]}, [{"ok": False, "not_found": "textedit"}])
    assert lines and "textedit" in lines[0]


def test_assistant_openers_dropped():
    assert cast.polish_line("ban", "I'm on it, let's see what you've got.") == "Let's see what you've got."
    assert cast.polish_line("king", "Sure.") == "Sure."


def test_merlin_failure_falls_back_to_routing(monkeypatch):
    base = _fake_llm({"agent": "ban"})

    async def fake(provider, system, user, agent=None, models=None):
        if system == merlin.MERLIN_SYSTEM:
            raise providers.ProviderError("down")
        return await base(provider, system, user, agent, models)
    monkeypatch.setattr(providers, "get_json", fake)
    res = asyncio.run(_pipe([]).handle("Merlin, how are you doing", None))
    assert res["agent"] == "ban" and res["routed"]["by"] == "claude"
