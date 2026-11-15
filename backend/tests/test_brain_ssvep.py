"""Offline tests for choosing by looking (SSVEP). No LSL stream, no board, no server."""
import numpy as np
import pytest

from backend.brain import ssvep
from backend.brain.chooser import ChooserState
from backend.brain.rules import StopRule
from backend.brain.ssvep_session import Gate, load_ssvep_calibration, parse_script, save_calibration
from backend.brain.ssvep_sim import SimSignal

FS = 250.0
STEP = 0.25
RATES_144 = [r.hz for r in ssvep.pick_rates(144.0, 5)]


def scores_stream(sig, segments, rates, step=STEP):
    """Feed the simulator through sliding windows like run mode does.
    segments: (seconds, gaze index or None, eyes_closed). Yields (t, gaze, closed, scores, raw)."""
    hist = sig.take(6.0)
    t = 0.0
    for secs, gaze, closed in segments:
        sig.gaze_hz = None if gaze is None else rates[gaze]
        sig.eyes_closed = closed
        for _ in range(int(round(secs / step))):
            hist = np.vstack([hist, sig.next(int(step * FS))])[-int(6 * FS):]
            t += step
            yield t, gaze, closed, ssvep.window_scores(hist, FS, rates), hist


def calibrate_sim(rates, amp, seed, hold=6):
    """What ssvep-calibrate does, on the simulator: 9 s per step, windows from 4 s on."""
    sig = SimSignal(fs=FS, ssvep_uv=amp, seed=seed)
    segs = [(9.0, None, False)] + [(9.0, i, False) for i in range(len(rates))]
    by = {}
    rows = list(scores_stream(sig, segs, rates))
    per = int(9.0 / STEP)
    for k, (_, g, _) in enumerate(segs):
        part = rows[k * per:(k + 1) * per]
        by[g] = np.array([r[3] for j, r in enumerate(part) if (j + 1) * STEP >= 4.0])
    rest = by.pop(None)
    return by, rest, ssvep.fit_thresholds(by, rest, hold)


# ---------------------------------------------------------------- CCA

@pytest.mark.parametrize("amp,min_acc", [(4.0, 0.95), (2.5, 0.85), (1.5, 0.6)])
def test_cca_picks_the_right_rate_at_several_noise_levels(amp, min_acc):
    hits = total = 0
    for i, f in enumerate(RATES_144):
        sig = SimSignal(fs=FS, ssvep_uv=amp, seed=100 + i)
        sig.gaze_hz = f
        x = sig.take(10.0)
        for end in np.arange(4.0, 10.01, 0.5):
            s = ssvep.window_scores(x[:int(end * FS)], FS, RATES_144)
            hits += int(np.argmax(s) == i)
            total += 1
    assert hits / total >= min_acc


def test_cca_on_a_clean_sine_is_near_one_and_rejects_other_rates():
    t = np.arange(int(3 * FS)) / FS
    x = np.sin(2 * np.pi * 15.0 * t)[:, None] + 0.05 * np.random.default_rng(0).standard_normal((len(t), 3))
    s = ssvep.cca_scores(x, FS, [12.0, 15.0, 20.0])
    assert s[1] > 0.95 and s[0] < 0.3 and s[2] < 0.3


def test_one_channel_is_enough():
    sig = SimSignal(fs=FS, n_ch=1, ssvep_uv=4.0, seed=3)
    sig.gaze_hz = RATES_144[0]
    s = ssvep.window_scores(sig.take(5.0), FS, RATES_144)
    assert int(np.argmax(s)) == 0


# ---------------------------------------------------------------- deciding

def test_no_choice_on_rest_data():
    by, rest, th = calibrate_sim(RATES_144, amp=3.0, seed=7)
    d = ssvep.Decider(th, refractory_s=5.0)
    sig = SimSignal(fs=FS, ssvep_uv=3.0, seed=777)
    fired = [d.update(t, s) for t, _, _, s, _ in scores_stream(sig, [(45.0, None, False)], RATES_144)]
    assert [c for c in fired if c is not None] == []


def test_looking_at_a_tile_chooses_it():
    by, rest, th = calibrate_sim(RATES_144, amp=3.0, seed=7)
    d = ssvep.Decider(th, refractory_s=5.0)
    sig = SimSignal(fs=FS, ssvep_uv=3.0, seed=778)
    out = []
    for t, g, _, s, _ in scores_stream(sig, [(4.0, None, False), (8.0, 2, False)], RATES_144):
        c = d.update(t, s)
        if c is not None:
            out.append((t, c))
    assert out and out[0][1] == 2
    assert out[0][0] < 4.0 + 6.0          # chosen within six seconds of looking
    assert all(c == 2 for _, c in out)


def fake_scores(idx, n=5, hi=0.9, lo=0.1):
    s = np.full(n, lo)
    if idx is not None:
        s[idx] = hi
    return s


def test_refractory_time_and_no_double_order():
    th = ssvep.Thresholds(threshold=0.5, margin=0.2, hold=4)
    d = ssvep.Decider(th, refractory_s=5.0)
    fired = []
    t = 0.0
    for _ in range(80):                    # stare at tile 1 for 20 s
        c = d.update(t, fake_scores(1))
        if c is not None:
            fired.append((t, c))
        t += 0.25
    assert fired == [(0.75, 1)]            # once, after 4 windows, never again while staring
    for _ in range(4):                     # glance away for 1 s: not long enough to release
        d.update(t, fake_scores(None)); t += 0.25
    again = [d.update(t + 0.25 * k, fake_scores(1)) for k in range(8)]
    assert all(c is None for c in again)
    t += 2.0
    for _ in range(8):                     # look away for 2 s: released
        d.update(t, fake_scores(None)); t += 0.25
    later = [d.update(t + 0.25 * k, fake_scores(1)) for k in range(8)]
    assert 1 in later


def test_refractory_blocks_a_different_item_too():
    th = ssvep.Thresholds(threshold=0.5, margin=0.2, hold=4)
    d = ssvep.Decider(th, refractory_s=5.0)
    out = [d.update(k * 0.25, fake_scores(0)) for k in range(4)]
    assert out[-1] == 0
    soon = [d.update(1.0 + k * 0.25, fake_scores(3)) for k in range(12)]    # 1 to 4 s after
    assert all(c is None for c in soon)
    late = [d.update(5.75 + k * 0.25, fake_scores(3)) for k in range(4)]    # 5 s after the choice
    assert late[-1] == 3


def test_margin_is_required():
    th = ssvep.Thresholds(threshold=0.5, margin=0.2, hold=1)
    close = np.array([0.8, 0.7, 0.1, 0.1, 0.1])
    assert ssvep.classify(close, th)[0] is None
    assert ssvep.classify(np.array([0.8, 0.5, 0.1, 0.1, 0.1]), th)[0] == 0


# ---------------------------------------------------------------- the stop wins

def test_alpha_burst_fires_the_stop_and_suppresses_a_choice():
    """Looking at a tile, then closing the eyes before the choice completes: the stop fires
    and nothing is chosen, now or for the refractory time after the eyes open."""
    rates = RATES_144
    by, rest, th = calibrate_sim(rates, amp=3.0, seed=7)
    sig = SimSignal(fs=FS, ssvep_uv=3.0, seed=900)
    # notched alpha levels for the stop, measured on this simulator
    open_a, closed_a = [], []
    for t, g, closed, s, raw in scores_stream(sig, [(6.0, None, False), (6.0, None, True)], rates):
        a = ssvep.alpha_excluding(raw, FS, rates)
        if not closed and t > 3.0:
            open_a.append(a)
        elif closed and t > 9.0:      # eyes shut for 3 s, so the 2 s alpha window is all closed
            closed_a.append(a)
    stop_at = (np.mean(open_a) + np.mean(closed_a)) / 2
    gate = Gate(StopRule(stop_at, np.mean(open_a) + np.std(open_a)), ssvep.Decider(th, 5.0))

    # stray tile response kept on while the eyes are closed (the worst case: a choice that
    # looks valid to CCA at the same moment the stop wants to fire)
    events, choices = [], []
    sig2 = SimSignal(fs=FS, ssvep_uv=3.0, seed=901)
    segs = [(4.0, None, False), (1.0, 1, False), (6.0, 1, True), (3.0, 1, False)]
    for t, g, closed, s, raw in scores_stream(sig2, segs, rates):
        if closed:
            sig2.gaze_hz = rates[1]
            sig2._ssvep_level = sig2.ssvep_uv     # force the response to stay present
        a = ssvep.alpha_excluding(raw, FS, rates)
        ev, choice, _, _ = gate.step(t, a, True, s, True)
        if ev:
            events.append((t, ev))
        if choice is not None:
            choices.append((t, choice))
    assert any(ev == "stop" for _, ev in events)
    assert choices == []


def test_stop_is_deaf_to_a_flicker_inside_the_alpha_band():
    """A strong steady 10 Hz tile response must not look like closed eyes to the stop."""
    rates = [10.0, 15.0]
    sig = SimSignal(fs=FS, ssvep_uv=0.0, seed=5)
    x_open = sig.take(6.0)
    t = np.arange(len(x_open)) / FS
    flicker = 12.0 * np.sin(2 * np.pi * 10.0 * t)[:, None]
    plain = ssvep.alpha_excluding(x_open, FS, rates)
    with_flicker = ssvep.alpha_excluding(x_open + flicker, FS, rates)
    assert with_flicker < 1.5 * plain
    sig.eyes_closed = True
    sig.take(2.0)
    closed = ssvep.alpha_excluding(sig.take(6.0), FS, rates)
    assert closed > 5 * plain


def test_gate_blocks_choices_after_a_stop_clears():
    th = ssvep.Thresholds(threshold=0.5, margin=0.2, hold=2)
    gate = Gate(StopRule(stop_at=100.0, release_at=20.0, hold_s=1.5), ssvep.Decider(th, refractory_s=5.0))
    t = 0.0
    evs, picks = [], []
    plan = [(10.0, 8)] + [(500.0, 8)] + [(10.0, 40)]     # (alpha, ticks)
    for alpha, n in plan:
        for _ in range(n):
            ev, c, _, _ = gate.step(t, alpha, True, fake_scores(2), True)
            evs.append(ev); picks.append((t, c))
            t += 0.25
    chosen = [(tt, c) for tt, c in picks if c is not None]
    assert "stop" in evs and "clear" in evs
    assert chosen[0] == (0.25, 2)                      # before the stop
    clear_t = 0.25 * (8 + 8)                           # alpha drops at 4.0 s
    assert all(tt >= clear_t + 5.0 for tt, _ in chosen[1:])


# ---------------------------------------------------------------- rates

@pytest.mark.parametrize("refresh", [60.0, 59.94, 144.0, 120.0, 75.0])
def test_rates_divide_the_screen_and_do_not_collide(refresh):
    rates = ssvep.pick_rates(refresh, 5)
    assert len(rates) == 5
    for r in rates:
        assert r.hz == pytest.approx(refresh / r.frames)
        assert r.frames >= 2
    hz = sorted(r.hz for r in rates)
    for i, a in enumerate(hz):
        for b in hz[i + 1:]:
            assert b - a >= 0.5
            assert abs(2 * a - b) >= 0.4 and abs(3 * a - b) >= 0.4


def test_rates_at_60_hz():
    hz = [round(r.hz, 2) for r in ssvep.pick_rates(60.0, 5)]
    assert sorted(hz) == [4.62, 6.0, 8.57, 15.0, 20.0]
    assert sum(1 for f in hz if 7.5 <= f <= 13.5) <= 1     # at most one near alpha
    assert hz[0] == 15.0                                   # first item gets the strongest tile


def test_rates_at_144_hz_avoid_alpha():
    hz = [r.hz for r in ssvep.pick_rates(144.0, 5)]
    assert not any(7.5 <= f <= 13.5 for f in hz)
    assert sorted(round(f, 2) for f in hz) == [6.55, 14.4, 16.0, 18.0, 20.57]


def test_too_many_items_is_refused():
    with pytest.raises(ValueError):
        ssvep.pick_rates(60.0, 12)


# ---------------------------------------------------------------- config and files

def test_config_round_trip(tmp_path):
    cfg = ssvep.SsvepConfig(items=["cup", "pen"], sentence="please hand me the {item}",
                            channels={"O1": [1, 2], "Oz": [3, 4, 5], "O2": [6]})
    p = tmp_path / "cfg.json"
    ssvep.save_config(cfg, p)
    back = ssvep.load_config(p)
    assert back == cfg
    assert back.inputs() == [0, 1, 2, 3, 4, 5]
    assert back.text_for("pen") == "please hand me the pen"


def test_shipped_config_loads():
    cfg = ssvep.load_config()
    assert cfg.items == ["bottle", "red bull", "sharpie", "scissors", "glue stick"]
    assert cfg.text_for("bottle") == "bring me the bottle"
    assert cfg.inputs() == list(range(16))


@pytest.mark.parametrize("bad", [
    {"channels": {"Fp1": [1]}},
    {"channels": {"Oz": [17]}},
    {"channels": {"Oz": []}},
    {"sentence": "bring it"},
    {"items": []},
])
def test_config_rejects_mistakes(bad):
    with pytest.raises(ValueError):
        ssvep.SsvepConfig(**bad).validate()


def test_parse_channels():
    assert ssvep.parse_channels("Oz=1-16") == {"Oz": list(range(1, 17))}
    assert ssvep.parse_channels("O1=1-4,Oz=5-12,O2=13-16") == {
        "O1": [1, 2, 3, 4], "Oz": list(range(5, 13)), "O2": [13, 14, 15, 16]}
    assert ssvep.parse_channels("oz=1-8; poz=9,10") == {"Oz": list(range(1, 9)), "POz": [9, 10]}
    with pytest.raises(ValueError):
        ssvep.parse_channels("Cz=1")


def test_items_from_scene_payloads():
    assert ssvep.items_from_payload(["cup", "pen", "cup"]) == ["cup", "pen"]
    assert ssvep.items_from_payload({"objects": [{"name": "cup"}, {"label": "pen"}]}) == ["cup", "pen"]
    assert ssvep.items_from_payload({"nothing": 1}) == []


def test_calibration_round_trip(tmp_path):
    by, rest, th = calibrate_sim(RATES_144[:3], amp=3.0, seed=11, hold=5)
    from dataclasses import asdict
    cal = {"refresh": 144.0, "items": ["a", "b", "c"], "frames": [10, 9, 8],
           "thresholds": asdict(th), "stop_alpha": 500.0, "release_alpha": 60.0}
    p = tmp_path / "brain_ssvep.json"
    save_calibration(cal, p)
    back = load_ssvep_calibration(p)
    th2 = ssvep.Thresholds(**back["thresholds"])
    assert th2 == th
    s = by[1][-1]
    assert ssvep.classify(s, th2) == ssvep.classify(s, th)


def test_leave_one_out_reports_every_rate():
    by, rest, th = calibrate_sim(RATES_144, amp=4.0, seed=12)
    loo = ssvep.leave_one_out(by, rest, 6)
    assert set(loo["per_rate"]) == set(range(5))
    assert min(loo["per_rate"].values()) >= 0.7
    assert loo["rest_undecided"] >= 0.8


def test_missing_calibration_says_what_to_run(tmp_path):
    with pytest.raises(SystemExit, match="ssvep-calibrate"):
        load_ssvep_calibration(tmp_path / "none.json")


# ---------------------------------------------------------------- simulator and page state

def test_simulator_eyes_closed_raises_alpha_and_removes_the_response():
    sig = SimSignal(fs=FS, seed=2)
    sig.gaze_hz = 15.0
    sig.take(2.0)
    open_x = sig.take(4.0)
    sig.eyes_closed = True
    sig.take(1.0)
    closed_x = sig.take(4.0)
    a_open = ssvep.alpha_excluding(open_x, FS, [15.0])
    a_closed = ssvep.alpha_excluding(closed_x, FS, [15.0])
    assert a_closed > 5 * a_open
    s_open = ssvep.window_scores(open_x, FS, [15.0, 20.0])
    s_closed = ssvep.window_scores(closed_x, FS, [15.0, 20.0])
    assert s_open[0] > s_closed[0] + 0.1


def test_page_state_drives_the_simulated_gaze():
    class FakeSim:
        def __init__(self):
            self.calls = []

        def set(self, **kw):
            self.calls.append(kw)

    sim = FakeSim()
    st = ChooserState(["a", "b"], "run", sim)
    st.set(rates_hz=[14.4, 16.0])
    st.set_gaze(1, False)
    assert sim.calls[-1] == {"gaze_hz": 16.0, "eyes_closed": False}
    st.set_gaze(None, True)
    assert sim.calls[-1] == {"clear_gaze": True, "eyes_closed": True}
    st.set_gaze(7, None)          # out of range counts as looking away
    assert st.snapshot()["gaze"] is None


def test_sim_script_parsing():
    ev = parse_script("0:none,3:sharpie,12:closed,15:open,20:2", ["bottle", "red bull", "sharpie"])
    assert ev == [(0.0, "gaze", None), (3.0, "gaze", 2), (12.0, "eyes", True), (15.0, "eyes", False), (20.0, "gaze", 1)]


# ---------------------------------------------------------------- real sample rate and lost packets

def test_cca_works_at_the_rate_the_puck_really_streams():
    """The dashboard showed about 230 samples a second, not 250. References are built at the
    measured rate, so the right tile still wins."""
    fs = 230.0
    rates = RATES_144
    hits = 0
    for i, f in enumerate(rates):
        sig = SimSignal(fs=fs, ssvep_uv=3.0, seed=40 + i)
        sig.gaze_hz = f
        s = ssvep.window_scores(sig.take(6.0), fs, rates)
        hits += int(np.argmax(s) == i)
    assert hits == len(rates)


def test_counter_gap():
    from backend.brain.bridge import counter_gap
    assert counter_gap(None, 5) == 0
    assert counter_gap(5, 6) == 0
    assert counter_gap(5, 9) == 3
    assert counter_gap(254, 1) == 2          # wraps at 256
    assert counter_gap(100, 40) == 0         # a jump backwards is a restart, not a loss


def test_fill_draws_a_straight_line():
    from backend.brain.bridge import fill
    out = fill(np.array([0.0, 10.0]), np.array([4.0, 14.0]), 3)
    assert out.tolist() == [[1.0, 11.0], [2.0, 12.0], [3.0, 13.0]]
    assert fill(np.zeros(2), np.ones(2), 0).shape == (0, 2)


def test_filling_lost_packets_keeps_the_fast_rates_readable():
    """Drop 6 percent of samples at random, as a busy Bluetooth link does. Without filling,
    the 20 Hz response smears; with the bridge's straight-line fill it comes back."""
    from backend.brain.bridge import fill
    fs = 244.14
    f = 20.57
    sig = SimSignal(fs=fs, ssvep_uv=3.0, seed=21)
    sig.gaze_hz = f
    x = sig.take(8.0)
    rng = np.random.default_rng(0)
    keep = rng.random(len(x)) > 0.06
    keep[0] = True
    idx = np.flatnonzero(keep)
    dropped = x[idx]
    measured = len(idx) / (len(x) / fs)      # what the time stamps would say
    rows = [x[idx[0]][None, :]]
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a > 1:
            rows.append(fill(x[a], x[b], b - a - 1))
        rows.append(x[b][None, :])
    filled = np.vstack(rows)
    rates = [14.4, 16.0, 18.0, f]
    s_drop = ssvep.window_scores(dropped, measured, rates)
    s_fill = ssvep.window_scores(filled, fs, rates)
    assert s_fill[3] > s_drop[3] + 0.1
    assert int(np.argmax(s_fill)) == 3
