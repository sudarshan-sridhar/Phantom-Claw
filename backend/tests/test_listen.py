"""Offline tests for the voice listener: phrases, names, cutting audio, the shortcut, the mute gate."""
from __future__ import annotations

import numpy as np
import pytest

from backend.listen.hotkey import ChordDetector
from backend.listen.listener import Gate, Listener
from backend.listen.names import called_text, find_agent, find_wake, name_alone
from backend.listen.vad import Segmenter


# ---------------------------------------------------------------- wake and stop phrases

@pytest.mark.parametrize("text", [
    "Phantom Claw, start.", "phantom claw start", "Fantom claw start", "Phantom clock start.",
    "Phantom cloud", "phantom claw", "Phantom Claws, start!", "PhantomClaw start", "phantom claw please start",
])
def test_wake_variants(text):
    assert find_wake(text)[0] == "start"


@pytest.mark.parametrize("text", [
    "Phantom Claw, stop.", "phantom claw sleep", "Fantom clock stop", "Phantom Claw, go to sleep.", "phantom cloud stop",
])
def test_stop_variants(text):
    assert find_wake(text)[0] == "stop"


@pytest.mark.parametrize("text", ["Ban, how is it going", "what time is it", "the claw is open", "phantom of the opera"])
def test_not_wake(text):
    assert find_wake(text)[0] is None


def test_start_then_other_stop_word_is_still_start():
    assert find_wake("Phantom Claw start, Ban stop that")[0] == "start"


# ---------------------------------------------------------------- names

@pytest.mark.parametrize("text,agent", [
    ("Ban, how is it going?", "ban"), ("Band, how's it going", "ban"), ("Van, grab the bottle", "ban"),
    ("Bann!", "ban"), ("Ben, how is it going", "ban"), ("Escanor", "escanor"), ("Escanoor, lift it", "escanor"), ("scanner, wave", "escanor"),
    ("Gowther, what is an ArUco marker?", "gowther"), ("Gother what is that", "gowther"), ("Gauther?", "gowther"),
    ("Merlyn, who should do it", "merlin"), ("Elisabeth, take a note", "elizabeth"), ("King, set a timer", "king"),
    ("Hey King's turn", "king"),
])
def test_names(text, agent):
    assert find_agent(text)[0] == agent


@pytest.mark.parametrize("text", ["what time is it", "can someone lift the bottle", "banana bread", "the kingdom"])
def test_no_name(text):
    assert find_agent(text)[0] is None


def test_name_alone():
    assert name_alone("Ban.", "ban")
    assert name_alone("Hey, Ban!", "ban")
    assert name_alone("Escanor?", "escanor")
    assert not name_alone("Ban, how is it going", "ban")
    assert called_text("ban") == "Ban, are you there?"


# ---------------------------------------------------------------- cutting audio into utterances

FRAME = b"\x00\x00" * 480


def run_segmenter(pattern, **kw):
    """pattern is a string of 1 (voice) and 0 (quiet), one character per 30 ms frame."""
    flags = iter(pattern)
    seg = Segmenter(lambda _f: next(flags) == "1", **kw)
    out = []
    for i in range(len(pattern)):
        u = seg.feed(FRAME, (i + 1) * 0.03)
        if u:
            out.append(u)
    return out


def test_segment_ends_after_silence():
    utts = run_segmenter("0" * 10 + "1" * 30 + "0" * 30)
    assert len(utts) == 1
    u = utts[0]
    assert abs(u.t_end - 40 * 0.03) < 1e-6           # last voiced frame
    assert abs(u.t_cut - u.t_end - 23 * 0.03) < 1e-6  # 700 ms (23 frames) of quiet ends it


def test_short_clip_dropped():
    assert run_segmenter("0" * 5 + "1" * 6 + "0" * 30) == []  # 180 ms of voice


def test_short_pause_does_not_split():
    utts = run_segmenter("1" * 20 + "0" * 20 + "1" * 20 + "0" * 30)  # a 600 ms pause in the middle
    assert len(utts) == 1


def test_long_speech_is_cut():
    utts = run_segmenter("1" * 500, max_s=3)
    assert len(utts) >= 4


# ---------------------------------------------------------------- the shortcut

class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_chord_toggles_on_clean_press():
    hits, clk = [], Clock()
    d = ChordDetector(lambda: hits.append(1), clock=clk)
    d.event("ctrl", True); d.event("shift", True); clk.t = 0.2
    d.event("shift", False); d.event("ctrl", False)
    assert hits == [1]


def test_chord_ignores_other_keys_and_slow_presses():
    hits, clk = [], Clock()
    d = ChordDetector(lambda: hits.append(1), clock=clk)
    d.event("left ctrl", True); d.event("shift", True); d.event("space", True)
    d.event("space", False); d.event("shift", False); d.event("left ctrl", False)
    assert hits == []
    d.event("ctrl", True); d.event("right shift", True); clk.t = 1.5
    d.event("ctrl", False); d.event("right shift", False)
    assert hits == []
    d.event("ctrl", True); d.event("ctrl", False)  # ctrl alone
    assert hits == []


# ---------------------------------------------------------------- the mute gate and the two states

def test_gate_queues_lines():
    clk = Clock()
    g = Gate(tail_s=0.4, clock=clk)
    g.speak(2.0)
    assert g.muted() and not g.idle()
    g.speak(1.0)  # starts after the first line
    clk.t = 3.7  # 2.0 + 0.4, then 1.0 + 0.4
    assert g.muted()
    clk.t = 3.9
    assert not g.muted()


class FakeSTT:
    def __init__(self):
        self.texts = []

    def transcribe(self, audio):
        return self.texts.pop(0)


class FakeServer:
    def __init__(self):
        self.calls = []

    def event(self, ev):
        self.calls.append(("event", ev))

    def set_listen(self, awake, source, say=True):
        self.calls.append(("listen", awake))

    def get_listen(self):
        return False

    def stop(self):
        self.calls.append(("stop",))
        return {"ok": True}

    def request(self, text, agent):
        self.calls.append(("request", text, agent))
        return {"agent": agent, "say": "ok"}


def make_listener(require_name=True):
    stt, srv = FakeSTT(), FakeServer()
    lis = Listener(stt, Segmenter(lambda f: False), srv, require_name=require_name)
    return lis, stt, srv


def say(lis, stt, text):
    from backend.listen.vad import Utterance

    stt.texts.append(text)
    rec = lis.process(Utterance(np.zeros(4800, dtype=np.int16), 0.0, 1.0, 1.45, 600))
    lis.pool.shutdown(wait=True)
    from concurrent.futures import ThreadPoolExecutor

    lis.pool = ThreadPoolExecutor(max_workers=4)
    return rec


def test_states_and_routing():
    lis, stt, srv = make_listener()
    assert say(lis, stt, "Ban, how is it going")["action"] == "ignored, asleep"
    assert say(lis, stt, "Phantom Claw, start.")["action"] == "wake"
    assert lis.awake and ("listen", True) in srv.calls
    r = say(lis, stt, "Ban, how is it going")
    assert r["action"] == "request" and ("request", "Ban, how is it going", "ban") in srv.calls
    assert ("event", {"type": "hear", "agent": "ban", "text": "Ban, how is it going"}) in srv.calls
    say(lis, stt, "Escanor")
    assert ("request", "Escanor, are you there?", "escanor") in srv.calls
    assert say(lis, stt, "what time is it")["action"] == "ignored, no name"
    say(lis, stt, "Merlin, who is free")
    assert ("request", "Merlin, who is free", None) in srv.calls
    assert say(lis, stt, "Phantom Claw, stop.")["action"] == "stop, sleep"
    assert not lis.awake and ("stop",) in srv.calls
    assert srv.calls.index(("stop",)) < srv.calls.index(("listen", False))  # the arm stops first


def test_unnamed_goes_to_merlin_when_names_not_required():
    lis, stt, srv = make_listener(require_name=False)
    lis.awake = True
    assert say(lis, stt, "what time is it")["action"] == "request"
    assert ("request", "what time is it", None) in srv.calls
    assert ("event", {"type": "hear", "agent": "merlin", "text": "what time is it"}) in srv.calls


def test_muted_speech_is_cut_and_marked_muted():
    pattern = "1" * 40 + "0" * 40
    flags = iter(pattern * 3)
    lis = Listener(FakeSTT(), Segmenter(lambda f: next(flags) == "1"), None)
    lis.gate.until = 2.0  # muted for the first 2 s, open after
    lis.feed_frames((FRAME, 0.03 * (i + 1)) for i in range(len(pattern) * 3))
    got = []
    while not lis.utts.empty():
        got.append(lis.utts.get()[2].muted)
    assert got[0] is True and got[-1] is False


def test_only_a_stop_counts_while_muted():
    lis, stt, srv = make_listener(require_name=False)
    lis.awake = True
    from backend.listen.vad import Utterance

    def heard(text):
        stt.texts.append(text)
        u = Utterance(np.zeros(4800, dtype=np.int16), 0.0, 1.0, 1.45, 600, muted=True)
        return lis.process(u)

    assert heard("Escanor, give me the bottle")["action"] == "ignored, muted"
    assert heard("I will stop when I am done")["action"] == "ignored, muted"
    assert not any(c[0] == "request" for c in srv.calls)
    assert heard("stop!")["action"] == "stop, sleep" and ("stop",) in srv.calls


def test_busy_reply_is_not_retried():
    lis, stt, srv = make_listener(require_name=False)
    srv.request = lambda text, agent: srv.calls.append(("request", text, agent)) or {"ignored": "busy"}
    lis.awake = True
    r = say(lis, stt, "Ban, hand me the scissors")
    assert r.get("ignored") == "busy" and sum(c[0] == "request" for c in srv.calls) == 1


def test_names_not_required_by_default_but_chatter_is_dropped():
    from backend.listen import config

    assert config.REQUIRE_NAME is False or __import__("os").environ.get("LISTEN_REQUIRE_NAME")
    lis, stt, srv = make_listener(require_name=False)
    lis.awake = True
    assert say(lis, stt, "thanks for watching")["action"] == "ignored, not a request"
    assert say(lis, stt, "yeah")["action"] == "ignored, not a request"
    assert say(lis, stt, "can someone set a ten minute timer")["action"] == "request"
    assert ("request", "can someone set a ten minute timer", None) in srv.calls
    assert say(lis, stt, "open notepad")["action"] == "request"
