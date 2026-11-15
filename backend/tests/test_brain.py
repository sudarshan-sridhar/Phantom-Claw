"""Offline tests for the brain maths and rules. No LSL, no board, no server."""
import numpy as np
import pytest

from backend.brain import dsp
from backend.brain.calibrate import split_by_alpha
from backend.brain.rules import FocusMeter, StopRule


def sine(freq, amp, fs=250.0, seconds=4.0, channels=16, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * seconds)) / fs
    x = amp * np.sin(2 * np.pi * freq * t)[:, None] + rng.normal(0, noise, (len(t), channels))
    return x + 50.0  # a DC offset like a real electrode, which the filters must remove


@pytest.mark.parametrize("fs", [250.0, 244.140625])
def test_ten_hz_gives_high_alpha(fs):
    f = dsp.features(sine(10.0, 30.0, fs), fs)
    assert f["n_good"] == 16
    # a 30 uV sine carries 450 uV^2; most of it should land in alpha
    assert f["alpha"] > 300
    assert f["alpha"] > 20 * f["beta"]
    assert f["alpha"] > 20 * f["theta"]


def test_twenty_hz_goes_to_beta_not_alpha():
    f = dsp.features(sine(20.0, 30.0), 250.0)
    assert f["beta"] > 300
    assert f["beta"] > 20 * f["alpha"]


def test_sixty_hz_mains_is_notched_out():
    f = dsp.features(sine(60.0, 100.0, noise=0.5), 250.0)
    assert f["alpha"] < 1 and f["beta"] < 1
    assert f["rms"] < 5


def test_contact_rule():
    fs = 250
    x = np.zeros((fs, 4))
    rng = np.random.default_rng(1)
    x[:, 0] = rng.normal(0, 0.3, fs)                        # floating pin
    x[:, 1] = rng.normal(0, 20, fs)                         # good contact
    x[:, 2] = rng.normal(0, 400, fs)                        # hair or movement
    x[:, 3] = dsp.FULL_SCALE_UV - rng.uniform(0, 5, fs)     # stuck at full scale
    good, rms = dsp.contact(x)
    assert good.tolist() == [False, True, False, False]


def test_bad_channels_are_left_out_of_the_average():
    x = sine(10.0, 30.0, channels=4)
    x[:, 2:] = 0.0
    f = dsp.features(x, 250.0)
    assert f["n_good"] == 2
    assert f["alpha"] > 300


def test_measured_rate_uses_timestamps():
    ts = np.arange(0, 2441) / 244.140625
    assert dsp.measured_rate(ts, 250.0) == pytest.approx(244.14, abs=0.01)
    assert dsp.measured_rate(ts[:100], 250.0) == 250.0   # under 2 s: not enough to measure


def drive(rule, samples, dt=0.25, ok=True):
    """Feed (alpha) values at dt spacing; return list of (time, event) for events."""
    out = []
    for i, a in enumerate(samples):
        ev = rule.update(i * dt, a, ok)
        if ev:
            out.append((i * dt, ev))
    return out


def test_stop_fires_after_hold_and_not_before():
    rule = StopRule(stop_at=100, release_at=20, hold_s=1.5)
    events = drive(rule, [10] * 4 + [200] * 12)
    # alpha crosses at t=1.0 s, so the stop fires at t=2.5 s, exactly 1.5 s later
    assert events == [(2.5, "stop")]


def test_dip_resets_the_hold():
    rule = StopRule(stop_at=100, release_at=20, hold_s=1.5)
    events = drive(rule, [200] * 5 + [50] + [200] * 5)
    assert events == []   # 1.25 s, a dip, then 1.25 s: never 1.5 s straight


def test_release_needs_alpha_below_release_level():
    rule = StopRule(stop_at=100, release_at=20, hold_s=1.5)
    events = drive(rule, [200] * 7 + [60] * 4 + [15] * 2)
    # 60 is under the stop level but above release, so it stays stopped until 15
    assert events == [(1.5, "stop"), (2.75, "clear")]


def test_never_fires_on_bad_contact():
    rule = StopRule(stop_at=100, release_at=20, hold_s=1.5)
    assert drive(rule, [1000] * 40, ok=False) == []
    assert not rule.active


def test_focus_maps_and_freezes_on_clench():
    m = FocusMeter(rest_ratio=0.4, work_ratio=0.8, tau_s=2.0)
    assert m.update(0.0, beta=4, theta=6, rms=10, ok=True) == pytest.approx(0.0)   # ratio 0.4
    for i in range(1, 60):
        v = m.update(i * 0.25, beta=8, theta=2, rms=10, ok=True)                  # ratio 0.8
    assert v == pytest.approx(1.0, abs=0.01)
    frozen = m.update(15.0, beta=1, theta=9, rms=120, ok=True)                     # clench
    assert m.frozen and frozen == v


def test_focus_smoothing_takes_about_two_seconds():
    m = FocusMeter(rest_ratio=0.4, work_ratio=0.8, tau_s=2.0)
    m.update(0.0, beta=4, theta=6, rms=10, ok=True)
    for i in range(1, 9):
        v = m.update(i * 0.25, beta=8, theta=2, rms=10, ok=True)
    assert v == pytest.approx(1 - np.exp(-1), abs=0.02)   # one time constant in


def test_split_by_alpha_separates_bursts():
    rows = [{"alpha": a, "step": "open"} for a in [20, 21, 19, 22, 20, 18, 21, 20]]
    rows += [{"alpha": a, "step": "closed"} for a in [500, 520, 480, 510, 505, 495, 515, 490]]
    low, high = split_by_alpha(rows)
    assert all(r["alpha"] < 100 for r in low) and len(low) >= 6
    assert all(r["alpha"] > 100 for r in high) and len(high) >= 6
