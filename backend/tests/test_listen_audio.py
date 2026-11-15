"""Offline tests for the listener's gain, the transcript fixer, the wake variants and the
ElevenLabs fallback. No network, no microphone, no speech model."""
from __future__ import annotations

import time

import numpy as np
import pytest

from backend.listen import config
from backend.listen.fix import repair
from backend.listen.gain import LevelTracker, clip_levels, mic_report, normalize
from backend.listen.names import find_agent, find_wake
from backend.listen.stt import Transcriber, resolve_engine

from backend.tests.listen_phrases import NOT_WAKE, SHORT_STOP, START, STOP

RNG = np.random.default_rng(1)


def tone(seconds, amp, f=220.0):
    t = np.arange(int(seconds * config.RATE)) / config.RATE
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.int16)


# ---------------------------------------------------------------- gain

def test_quiet_speech_is_normalised():
    quiet = np.concatenate([np.zeros(4000, np.int16) + (RNG.standard_normal(4000) * 4).astype(np.int16), tone(1.0, 400)])
    out, g = normalize(quiet)
    assert g > 10
    assert 12000 < np.abs(out).max() <= 32767


def test_silence_is_not_blown_up():
    hiss = (RNG.standard_normal(16000) * 5).astype(np.int16)
    out, g = normalize(hiss)
    assert g == 1.0 and np.array_equal(out, hiss)


def test_loud_clip_left_alone():
    loud = tone(0.5, 20000)
    out, g = normalize(loud)
    assert g == 1.0 and np.array_equal(out, loud)


def test_tracker_gain_follows_noise_floor_and_is_capped():
    tr = LevelTracker(max_gain=16, noise_target=120, speech_target=2500)
    for _ in range(100):
        tr.update(10.0)          # a very quiet room
    assert tr.gain() == pytest.approx(12.0, rel=0.05)
    for _ in range(100):
        tr.update(1.0)           # digital silence: capped, not infinite
    assert tr.gain() == 16
    tr2 = LevelTracker()
    for _ in range(100):
        tr2.update(300.0)        # a loud room: no boost
    assert tr2.gain() == 1.0


def test_tracker_holds_back_when_speech_is_already_loud():
    tr = LevelTracker(max_gain=16, noise_target=120, speech_target=2500)
    for _ in range(50):
        tr.update(10.0)
    for _ in range(20):
        tr.update(5000.0)        # loud speech
    assert tr.gain() == 1.0


def test_vad_hears_quiet_speech_with_gain():
    """A low level voice-like signal: the gained VAD should find at least as many frames."""
    pytest.importorskip("webrtcvad")
    from backend.listen.vad import make_vad

    t = np.arange(int(1.5 * config.RATE)) / config.RATE
    # A buzzy, voice-like signal: a pitch with harmonics and a syllable rate envelope.
    sig = sum(np.sin(2 * np.pi * 140 * k * t) / k for k in range(1, 12)) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)) ** 2
    lead = (RNG.standard_normal(16000) * 4)
    x = np.concatenate([lead, sig / np.abs(sig).max() * 300 + RNG.standard_normal(len(t)) * 4]).astype(np.int16)
    counts = []
    for agc in (False, True):
        v = make_vad(agc=agc)
        counts.append(sum(v(x[i:i + 480].tobytes()) for i in range(0, len(x) - 479, 480)))
    assert counts[1] >= counts[0]


def test_mic_report_verdicts():
    quiet = clip_levels(np.concatenate([tone(1, 400), (RNG.standard_normal(8000) * 5).astype(np.int16)]))
    assert mic_report(quiet, 10, 20, 50)[0] == "TOO QUIET"
    assert "about 90" in " ".join(mic_report(quiet, 10, 20, 50)[1])
    ok = clip_levels(np.concatenate([tone(1, 12000), (RNG.standard_normal(8000) * 50).astype(np.int16)]))
    assert mic_report(ok, 30, 30, 50)[0] == "OK"
    assert mic_report(clip_levels(tone(1, 32767)), 30, 30, 50)[0] == "TOO LOUD"


# ---------------------------------------------------------------- the fixer

@pytest.mark.parametrize("raw,fixed", [
    ("S kanor, give me the bottle.", "Escanor, give me the bottle."),
    ("Escanoor give me the bottle", "Escanor give me the bottle"),
    ("Go there, what is an ArUco marker?", "Gowther, what is an ArUco marker?"),
    ("Gauther, hand me the scissors", "Gowther, hand me the scissors"),
    ("Marlin, who is free?", "Merlin, who is free?"),
    ("Elizabeth, bring me the sharp e.", "Elizabeth, bring me the Sharpie."),
    ("Phantom clock start.", "Phantom Claw start."),
    ("Fan tom claw, start", "Phantom Claw, start"),
    ("hey phantom claws start", "hey Phantom Claw start"),
    ("King, point at the ear buds", "King, point at the earbuds"),
    ("Ban, put the bottel down", "Ban, put the bottle down"),
    ("Band, play some music", "Ban, play some music"),
    ("Bring me the Sharpie, Elisabeth", "Bring me the Sharpie, Elizabeth"),
    ("Phantom Claw start, go there, open notepad", "Phantom Claw start, Gowther, open notepad"),
    ("Okay, eskanor, stop", "Okay, Escanor, stop"),
])
def test_fixer_repairs_near_misses(raw, fixed):
    assert repair(raw)[0] == fixed


@pytest.mark.parametrize("text", [
    "go there and pick it up", "I want to gather my things", "Escape the room", "the other one please",
    "Mother, open notepad", "what time is it", "can you give me the bottle", "Kingdom of heaven",
    "open the phantom of the opera", "Banana bread recipe", "together we can", "set a timer for ten minutes",
    "the battle is over", "merely a scratch", "search for water bottles",
])
def test_fixer_leaves_ordinary_words(text):
    assert repair(text) == (text, [])


@pytest.mark.parametrize("raw,fixed", [
    ("Gowther pointed the bottle", "Gowther, point at the bottle"),
    ("Hey King points at the glue stick", "Hey King, point at the glue stick"),
    ("Escanor picked up the bottle", "Escanor, pick up the bottle"),
    ("The goals are pointed at the bottom", "The goals are pointed at the bottom"),
    ("Ban, he pointed the way", "Ban, he pointed the way"),
])
def test_fixer_verb_after_name(raw, fixed):
    assert repair(raw)[0] == fixed


class StopServer:
    def __init__(self):
        self.calls = []

    def stop(self):
        self.calls.append("stop")
        return {"ok": True}

    def set_listen(self, awake, source, say=True):
        self.calls.append(("listen", awake))

    def event(self, ev):
        pass

    def request(self, text, agent):
        self.calls.append(("request", text))
        return {}


def stop_listener(text, awake):
    from backend.listen.listener import Listener
    from backend.listen.vad import Segmenter, Utterance

    srv = StopServer()
    lis = Listener(make_tr(lambda a: text), Segmenter(lambda f: False), srv)
    lis.awake = awake
    rec = lis.process(Utterance(AUDIO, 0.0, 1.0, 1.7, 600))
    lis.pool.shutdown(wait=True)
    return rec, srv, lis


@pytest.mark.parametrize("text", STOP)
def test_stop_phrase_stops_the_arm_awake_or_asleep(text):
    for awake in (True, False):
        rec, srv, lis = stop_listener(text, awake)
        assert srv.calls and srv.calls[0] == "stop", text
        assert not lis.awake


@pytest.mark.parametrize("text", SHORT_STOP)
def test_short_stop_when_awake(text):
    rec, srv, lis = stop_listener(text, True)
    assert srv.calls[0] == "stop" and rec["action"] == "stop, sleep"
    rec, srv, lis = stop_listener(text, False)  # asleep, a bare "stop" is not ours to act on
    assert srv.calls == []


@pytest.mark.parametrize("text", START)
def test_start_phrase_wakes(text):
    rec, srv, lis = stop_listener(text, False)
    assert rec["action"] == "wake" and lis.awake and "stop" not in srv.calls, text


@pytest.mark.parametrize("text", NOT_WAKE)
def test_ordinary_speech_triggers_nothing(text):
    assert find_wake(text)[0] is None and find_wake(repair(text)[0])[0] is None
    rec, srv, lis = stop_listener(text, False)
    assert srv.calls == [] and not lis.awake, text
    rec, srv, lis = stop_listener(text, True)
    assert "stop" not in srv.calls, text


def test_phrase_lists_are_big_enough():
    assert len(START) >= 30 and len(STOP) >= 30 and len(NOT_WAKE) >= 30


def test_fixer_reports_changes():
    out, changes = repair("Marlin, open the glue stik")
    assert out == "Merlin, open the glue stick"
    assert ("Marlin", "Merlin") in changes and ("glue stik", "glue stick") in changes


# ---------------------------------------------------------------- wake variants

@pytest.mark.parametrize("text", [
    "phantom claw start", "Phantom claws start", "phantom clock start", "hey phantom claw", "Hey Phantom Claw, start.",
    "phantom call start", "fan tom claw start", "Fantom Claw. Start.",
])
def test_wake_variants_after_fix(text):
    assert find_wake(repair(text)[0])[0] == "start"


def test_names_found_after_fix():
    assert find_agent(repair("S kanor, give me the bottle")[0])[0] == "escanor"
    assert find_agent(repair("Go there, point at the Sharpie")[0])[0] == "gowther"


# ---------------------------------------------------------------- engine choice and fallback

class FakeModel:
    """Stands in for faster-whisper: returns a fixed text, after a delay."""

    def __init__(self, text="local text", delay=0.0):
        self.text, self.delay, self.calls = text, delay, 0


def make_tr(eleven, local_text="local text", timeout=0.3):
    tr = Transcriber(engine="eleven", eleven=eleven, eleven_timeout=timeout)
    tr.engine = "eleven"
    tr.model = FakeModel(local_text)
    tr._whisper = lambda audio: (tr.model.__setattr__("calls", tr.model.calls + 1), tr.model.text)[1]
    return tr


AUDIO = np.concatenate([np.zeros(3000, np.int16), tone(0.8, 400)])


def test_resolve_engine():
    assert resolve_engine("auto", True) == "eleven"
    assert resolve_engine("auto", True, big_local=True) == "local"
    assert resolve_engine("eleven", True, big_local=True) == "eleven"
    assert resolve_engine("auto", False) == "local"
    assert resolve_engine("eleven", False) == "local"
    assert resolve_engine("local", True) == "local"


def test_eleven_used_when_it_answers():
    tr = make_tr(lambda a: "Escanor, give me the bottle")
    assert tr.transcribe(AUDIO) == "Escanor, give me the bottle"
    assert tr.last["engine"] == "eleven" and "fallback" not in tr.last
    assert tr.last["gain"] > 1  # the quiet clip was normalised first


def test_falls_back_to_local_on_error():
    def boom(_a):
        raise RuntimeError("ElevenLabs stt http 500")

    tr = make_tr(boom)
    assert tr.transcribe(AUDIO) == "local text"
    assert tr.last["engine"] == "local" and "failed" in tr.last["fallback"]


def test_falls_back_to_local_when_slow():
    def slow(_a):
        time.sleep(1.0)
        return "too late"

    tr = make_tr(slow, timeout=0.2)
    t0 = time.monotonic()
    assert tr.transcribe(AUDIO) == "local text"
    assert time.monotonic() - t0 < 0.8
    assert "slower" in tr.last["fallback"]


def test_local_engine_never_calls_eleven():
    called = []
    tr = make_tr(lambda a: called.append(1) or "x")
    tr.engine = "local"
    assert tr.transcribe(AUDIO) == "local text" and not called


def test_hallucination_dropped():
    tr = make_tr(lambda a: "Thank you.")
    assert tr.transcribe(AUDIO) == ""


def test_listener_logs_raw_and_fixed():
    from backend.listen.listener import Listener
    from backend.listen.vad import Segmenter, Utterance

    tr = make_tr(lambda a: "Marlin, open notepad")
    lis = Listener(tr, Segmenter(lambda f: False), None, require_name=True)
    lis.awake = True
    rec = lis.process(Utterance(AUDIO, 0.0, 1.0, 1.7, 600))
    assert rec["raw"] == "Marlin, open notepad" and rec["text"] == "Merlin, open notepad"
    assert rec["action"] == "request" and rec["agent"] == "merlin" and rec["engine"] == "eleven"


def test_mic_name_fragment():
    from backend.listen.__main__ import match_device

    devs = [(1, "Microphone (XWF-1080P)", "MME", 44100), (3, "Microphone Array (Realtek(R) Au", "MME", 44100),
            (9, "Microphone Array (Realtek(R) Audio)", "Windows DirectSound", 44100),
            (17, "Headset (OnePlus Nord Buds 2)", "Windows WDM-KS", 16000)]
    assert match_device(devs, "array") == 3
    assert match_device(devs, "HEADSET") is None       # WDM-KS only: may lock the mic, never picked by name
    assert match_device(devs, "17") == 17                # asked for by number: used, with a warning
    devs.append((26, "Microphone Array (Realtek HD Audio Mic Array input)", "Windows WDM-KS", 44100))
    assert match_device(devs, "26") == 3                 # swapped for the same mic under MME
    assert match_device(devs, "9") == 9
    assert match_device(devs, "nothing like it") is None
    assert match_device(devs, "") is None


def test_defaults():
    assert config.SILENCE_MS == 700 or __import__("os").environ.get("LISTEN_SILENCE_MS")
    assert config.PREROLL_MS >= 450


def test_utterance_ends_while_far_voices_carry_on():
    """The VAD keeps saying voice (people talking further away), but 20 dB down from the speaker."""
    from backend.listen.vad import Segmenter

    loud = (np.ones(480) * 3000).astype(np.int16).tobytes()
    far = (np.ones(480) * 300).astype(np.int16).tobytes()
    seg = Segmenter(lambda f: True, end_db=18)
    out = []
    frames = [far] * 10 + [loud] * 40 + [far] * 60
    for i, f in enumerate(frames):
        u = seg.feed(f, (i + 1) * 0.03)
        if u:
            out.append(u)
    assert len(out) == 1 and out[0].seconds < 2.6
    # With the gate off the same stream never ends on its own.
    seg = Segmenter(lambda f: True, end_db=0)
    assert not any(seg.feed(f, (i + 1) * 0.03) for i, f in enumerate(frames))
