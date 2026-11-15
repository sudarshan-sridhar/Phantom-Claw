"""Jaw-clench stop: detector, hold and release, things that must not fire, calibration and
stop selection. All on the simulator; no board."""
import numpy as np
import pytest

from backend.brain import clench as C
from backend.brain.rules import StopRule
from backend.brain.ssvep import Decider, Thresholds
from backend.brain.ssvep_session import Gate, parse_script
from backend.brain.ssvep_sim import SimSignal

FS = 237.0
TICK = 0.25


class Bench:
    """A simulator feeding a detector tick by tick, the way the reader does."""

    def __init__(self, seed=3, fs=FS, cal=None, **kw):
        self.sig = SimSignal(fs=fs, seed=seed, **kw)
        self.fs = fs
        self.buf = self.sig.take(2.0)
        self.det = C.detector_from(cal) if cal else C.ClenchDetector()
        self.t = 0.0

    def ticks(self, seconds):
        out = []
        for _ in range(int(round(seconds / TICK))):
            self.buf = np.vstack([self.buf, self.sig.take(TICK)])[-int(2 * self.fs):]
            self.t += TICK
            m = self.det.measure(self.buf, self.fs)
            out.append((self.t, m))
        return out


def protocol_rows(seed=5, clench_uv=30.0, talk_uv=8.0, n=5):
    """The calibration steps run offline on the simulator, labelled as the real one labels."""
    b = Bench(seed=seed, clench_uv=clench_uv, talk_uv=talk_uv)
    rows = []

    def rec(label, seconds, event=None, skip=0.0):
        start = b.t
        for t, _ in b.ticks(seconds):
            if t - start <= skip:
                continue
            p = C.pin_powers(b.buf[-int(round(C.WINDOW_S * FS)):], FS)
            rows.append({"t": t, "label": label, "event": event, "hf": p["hf"], "lf": p["lf"], "usable": p["usable"]})

    b.ticks(2.0)
    rec("rest", 10.0, skip=1.5)
    for i in range(n):
        b.sig.schedule("clench", True, delay_s=0.35)
        rec("clench", 2.0, i)
        b.sig.schedule("clench", False, delay_s=0.3)
        rec("clench", 0.9, i)
        rec("gap", 2.1, i)
    b.sig.schedule("talk", True, delay_s=0.4)
    rec("talk", 10.0, skip=0.5)
    b.sig.schedule("talk", False)
    for i in range(n):
        b.sig.schedule("blink", delay_s=0.3)
        rec("blink", 2.0, i)
    return rows


@pytest.fixture(scope="module")
def cal():
    return C.fit_clench(protocol_rows())


def drive(bench, rule, seconds):
    events = []
    for t, m in bench.ticks(seconds):
        ev = rule.update(t, m["score"], m["tilt"], m["ok"])
        if ev:
            events.append((round(t, 2), ev))
    return events


def warm(cal, seed=11, **kw):
    b = Bench(seed=seed, cal=cal, **kw)
    rule = C.ClenchRule.from_calibration(cal)
    assert drive(b, rule, 8.0) == []
    return b, rule


def test_calibration_on_the_simulator_is_good(cal):
    assert cal["verdict"] == "good"
    assert cal["caught"] == cal["n_clenches"] == 5
    assert not cal["would_fire"]
    assert cal["margin"] >= C.GOOD_MARGIN
    assert max(cal["others"].values()) < cal["threshold"] < min(cal["clench_levels"])


def test_clench_fires_and_release_clears(cal):
    b, rule = warm(cal)
    b.sig.schedule("clench", True)
    ev = drive(b, rule, 2.0)
    assert [e for _, e in ev] == ["stop"]
    assert ev[0][0] - 8.0 <= 0.9          # fired within about 0.4 s hold plus the window
    b.sig.schedule("clench", False)
    ev = drive(b, rule, 3.0)
    assert [e for _, e in ev] == ["clear"]
    assert 1.0 <= ev[0][0] - 10.0 <= 2.0   # a second below the release level, plus the window


def test_a_clench_shorter_than_the_hold_does_not_fire(cal):
    b, rule = warm(cal)
    b.sig.schedule("clench", True)
    b.sig.schedule("clench", False, delay_s=0.1)
    assert drive(b, rule, 3.0) == []


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_blinks_do_not_fire(cal, seed):
    b, rule = warm(cal, seed=seed)
    for _ in range(6):
        b.sig.schedule("blink")
        assert drive(b, rule, 1.5) == []


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_talking_does_not_fire(cal, seed):
    b, rule = warm(cal, seed=seed)
    b.sig.schedule("talk", True)
    assert drive(b, rule, 15.0) == []


def test_head_turn_does_not_fire(cal):
    b, rule = warm(cal)
    for _ in range(4):
        b.sig.schedule("turn")
        assert drive(b, rule, 2.0) == []


@pytest.mark.parametrize("hz", [6.55, 14.4, 16.0, 20.57, 24.0])
def test_flicker_response_does_not_fire(cal, hz):
    b, rule = warm(cal, ssvep_uv=6.0)    # three times the simulator's usual response
    b.sig.gaze_hz = hz
    assert drive(b, rule, 8.0) == []


def test_eyes_closed_alpha_does_not_fire(cal):
    b, rule = warm(cal)
    b.sig.eyes_closed = True
    assert drive(b, rule, 8.0) == []


def test_tilt_guard_blocks_a_slow_swing_even_when_large():
    # a big low frequency swing with an edge: lots of power everywhere, but far more low than high
    rule = C.ClenchRule(threshold=3.0, release=1.7, tilt_min=2.0)
    assert not rule.hot(score=8.0, tilt=0.6)
    assert rule.hot(score=8.0, tilt=5.0)


def test_too_few_pins_never_fires(cal):
    b, rule = warm(cal)
    b.sig.schedule("clench", True)
    fired = []
    for t, m in b.ticks(2.0):
        # 14 pins floating: what the detector would see from the "few" test source
        raw = b.buf.copy()
        raw[:, 2:] = np.random.default_rng(0).normal(0, 0.3, raw[:, 2:].shape)
        m2 = C.ClenchDetector(cal["base_hf"], cal["base_lf"]).measure(raw, FS)
        assert not m2["ok"]
        fired.append(rule.update(t, m2["score"], m2["tilt"], m2["ok"]))
    assert all(e is None for e in fired)


def test_a_loud_pin_does_not_fire_it_alone(cal):
    det = C.ClenchDetector(cal["base_hf"], cal["base_lf"])
    sig = SimSignal(fs=FS, seed=4)
    raw = sig.take(2.0)
    raw[:, 3] += 80 * np.random.default_rng(1).standard_normal(len(raw))   # one pin full of noise
    m = det.measure(raw, FS)
    assert m["score"] < cal["threshold"]


def test_rule_keeps_its_latch_on_bad_contact_and_resets_the_hold():
    rule = C.ClenchRule(threshold=5.0, release=2.0, tilt_min=2.0, hold_s=0.4, release_s=1.0)
    assert rule.update(0.0, 10, 10, True) is None
    assert rule.update(0.25, 10, 10, False) is None     # bad contact forgets the hold
    assert rule.update(0.5, 10, 10, True) is None
    assert rule.update(0.75, 10, 10, True) is None
    assert rule.update(1.0, 10, 10, True) == "stop"
    assert rule.update(1.25, 1, 1, False) is None and rule.active
    for t in (1.5, 1.75, 2.0, 2.25):
        assert rule.update(t, 1, 1, True) is None
    assert rule.update(2.5, 1, 1, True) == "clear"


def test_release_needs_a_full_second_below():
    rule = C.ClenchRule(threshold=5.0, release=2.0, hold_s=0.0)
    assert rule.update(0.0, 10, 10, True) == "stop"
    assert rule.update(0.5, 1, 1, True) is None
    assert rule.update(1.0, 3, 3, True) is None          # back above release: the timer restarts
    assert rule.update(1.5, 1, 1, True) is None
    assert rule.update(2.25, 1, 1, True) is None
    assert rule.update(2.5, 1, 1, True) == "clear"


def test_weak_clench_is_reported_not_good():
    fit = C.fit_clench(protocol_rows(seed=6, clench_uv=2.5, n=3))
    assert fit["verdict"] in ("not good", "usable")
    assert fit["verdict"] != "good"


def test_overlap_puts_the_threshold_above_everything_else():
    fit = C.fit_clench(protocol_rows(seed=7, clench_uv=6.0, talk_uv=12.0, n=3))
    assert fit["threshold"] > max(fit["others"].values())
    if fit["margin"] <= 1.0:
        assert fit["verdict"] == "not good"


def test_calibration_round_trip(tmp_path, cal):
    path = tmp_path / "brain_clench.json"
    C.save_clench(cal, path)
    back = C.load_clench(path)
    assert back["threshold"] == pytest.approx(cal["threshold"])
    assert back["base_hf"] == cal["base_hf"]
    assert "created" in back
    r = C.ClenchRule.from_calibration(back)
    assert r.threshold == pytest.approx(cal["threshold"]) and r.release_at <= r.threshold
    assert len(C.detector_from(back).measure(SimSignal(fs=FS).take(1.0), FS)) > 0


def test_missing_or_broken_calibration_says_what_to_run(tmp_path):
    assert C.load_clench(tmp_path / "none.json", required=False) is None
    with pytest.raises(SystemExit, match="clench-calibrate"):
        C.load_clench(tmp_path / "none.json")
    bad = tmp_path / "bad.json"
    bad.write_text('{"threshold": 3}', encoding="utf-8")
    with pytest.raises(SystemExit, match="release"):
        C.load_clench(bad)


def test_held_level():
    t = np.arange(10) * 0.25
    v = np.array([1, 9, 9, 9, 1, 5, 5, 1, 1, 1], dtype=float)
    assert C.held_level(t, v, 0.4) == 9     # three ticks span 0.5 s
    assert C.held_level(t, v, 0.6) == 1
    assert C.held_level(t[:1], v[:1], 0.4) == 0.0


# ---------------------------------------------------------------- which stop

@pytest.mark.parametrize("req,alpha,reliable,clench,want", [
    (None, True, True, True, {"alpha", "clench"}),
    (None, True, False, True, {"clench"}),          # unreliable alpha is left out of the default
    (None, True, False, False, {"alpha"}),
    (None, False, False, True, {"clench"}),
    (None, True, True, False, {"alpha"}),
    ("alpha", True, True, True, {"alpha"}),
    ("clench", True, True, True, {"clench"}),
    ("both", True, False, True, {"alpha", "clench"}),
])
def test_pick_stops(req, alpha, reliable, clench, want):
    stops, notes = C.pick_stops(req, alpha, reliable, clench)
    assert stops == want
    if "alpha" in stops and not reliable:
        assert any("WARNING" in n for n in notes)


@pytest.mark.parametrize("req,alpha,clench", [(None, False, False), ("alpha", False, True),
                                               ("clench", True, False), ("both", True, False)])
def test_pick_stops_refuses_what_is_not_calibrated(req, alpha, clench):
    with pytest.raises(SystemExit):
        C.pick_stops(req, alpha, True, clench)


def test_alpha_reliable_reads_both_calibration_kinds():
    from backend.brain.run import alpha_reliable
    assert alpha_reliable({"stop_alpha": 1})
    assert not alpha_reliable(None)
    assert not alpha_reliable({"warning": "eyes closed did not clearly raise alpha; the stop will be unreliable"})
    assert not alpha_reliable({"warnings": ["weak responses for: cup",
                                            "eyes closed did not clearly raise alpha; the stop will be unreliable"]})


# ---------------------------------------------------------------- with the chooser

def fake_scores(idx, n=5, hi=0.9, lo=0.1):
    s = np.full(n, lo)
    s[idx] = hi
    return s


def test_clench_blocks_choosing_like_the_alpha_stop():
    th = Thresholds(threshold=0.5, margin=0.2, hold=4, baseline=[0.0] * 5, scale=[1.0] * 5)
    crule = C.ClenchRule(threshold=5.0, release=2.0, tilt_min=2.0)
    gate = Gate(None, Decider(th, refractory_s=5.0), crule)
    hot = {"score": 20.0, "tilt": 20.0, "ok": True}
    calm = {"score": 1.0, "tilt": 1.0, "ok": True}
    t = 0.0
    # above threshold, not fired yet: already blocks a choice in progress
    for _ in range(3):
        _, choice, _, near = gate.step(t, 0.0, True, fake_scores(1), True, calm)
        t += 0.25
    _, choice, _, near = gate.step(t, 0.0, True, fake_scores(1), True, hot)
    assert near and choice is None
    fired = []
    for _ in range(4):
        t += 0.25
        gate.step(t, 0.0, True, fake_scores(1), True, hot)
        fired.append(gate.clench_event)
    assert "stop" in fired
    # release, then the refractory time keeps choosing off for 5 s after the last block
    cleared = False
    while not cleared:
        t += 0.25
        _, choice, _, _ = gate.step(t, 0.0, True, fake_scores(1), True, calm)
        cleared = gate.clench_event == "clear"
        assert choice is None
    t_clear = t
    chosen_at = None
    while chosen_at is None and t < t_clear + 10:
        t += 0.25
        _, choice, _, _ = gate.step(t, 0.0, True, fake_scores(1), True, calm)
        if choice is not None:
            chosen_at = t
    assert chosen_at is not None and chosen_at - t_clear >= 4.75


def test_gate_with_both_stops_still_takes_the_alpha_stop():
    th = Thresholds(threshold=0.5, margin=0.2, hold=4, baseline=[0.0] * 5, scale=[1.0] * 5)
    gate = Gate(StopRule(10.0, 5.0), Decider(th), C.ClenchRule(5.0, 2.0))
    calm = {"score": 1.0, "tilt": 1.0, "ok": True}
    evs = [gate.step(i * 0.25, 50.0, True, fake_scores(0), True, calm)[0] for i in range(8)]
    assert "stop" in evs and gate.clench_event is None


def test_sim_script_knows_the_muscle_actions():
    got = parse_script("1:clench,2:release,3:talk,4:quiet,5:blink,6:turn,7:closed", ["cup"])
    assert got == [(1.0, "clench", True), (2.0, "clench", False), (3.0, "talk", True), (4.0, "talk", False),
                   (5.0, "blink", None), (6.0, "turn", None), (7.0, "eyes", True)]


def test_muscle_actions_leave_the_brain_part_unchanged():
    a = SimSignal(fs=250.0, seed=2).take(3.0)
    s = SimSignal(fs=250.0, seed=2)
    s.schedule("blink", delay_s=10.0)   # scheduled but not yet due
    b = s.take(3.0)
    assert np.allclose(a, b)
