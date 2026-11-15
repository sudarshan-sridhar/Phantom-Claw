"""Scanning chooser driven by the jaw clench: on the simulator and with a fake clock."""
import numpy as np
import pytest

from backend.brain import clench as C
from backend.brain.chooser import ChooserState
from backend.brain.scan import SCAN_TICK_S, Scanner
from backend.brain.ssvep_sim import SimSignal
from backend.tests.test_brain_clench import protocol_rows

FS = 237.0
DWELL = 1.5


@pytest.fixture(scope="module")
def cal():
    return C.fit_clench(protocol_rows(seed=21))


class ScanBench:
    """Simulator, detector and scanner at the scan tick, as scan-run wires them."""

    def __init__(self, cal, seed=31, n_items=5, **kw):
        self.sig = SimSignal(fs=FS, seed=seed, **kw)
        self.buf = self.sig.take(2.0)
        self.det = C.ClenchDetector(cal["base_hf"], cal["base_lf"], tick_s=SCAN_TICK_S, seed_ticks=100)
        self.thr, self.rel, self.tilt = cal["threshold"], cal["release"], cal["tilt_min"]
        self.sc = Scanner(n_items, dwell_s=DWELL, min_s=cal["hold_s"])
        self.t = 0.0
        self.events = []
        self.step(5.0)            # settle
        self.sc.start(self.t)
        self.t_start = self.t

    def step(self, seconds):
        for _ in range(int(round(seconds / SCAN_TICK_S))):
            self.buf = np.vstack([self.buf, self.sig.take(SCAN_TICK_S)])[-int(2 * FS):]
            self.t += SCAN_TICK_S
            m = self.det.measure(self.buf, FS)
            hot = m["score"] >= self.thr and m["tilt"] >= self.tilt
            for ev in self.sc.update(self.t, hot, m["score"] < self.rel, m["ok"]):
                self.events.append((round(self.t - self.t_start, 2), ev))

    def until(self, t_rel):
        self.step(max(0.0, t_rel - (self.t - self.t_start)))

    def kinds(self):
        return [e[0] for _, e in self.events if e[0] != "ignored"]

    def choices(self):
        return [e[1] for _, e in self.events if e[0] == "choose"]


@pytest.mark.parametrize("tile", [0, 2, 4])
def test_short_clench_chooses_the_lit_tile(cal, tile):
    b = ScanBench(cal)
    b.until(tile * DWELL + 0.5)
    b.sig.schedule("clench", True)
    b.sig.schedule("clench", False, delay_s=0.5)
    b.step(2.0)
    assert b.choices() == [tile]
    assert "stop" not in b.kinds()


def test_after_a_choice_scanning_pauses_then_starts_again_from_the_first_tile(cal):
    b = ScanBench(cal)
    b.until(1 * DWELL + 0.5)
    b.sig.schedule("clench", True)
    b.sig.schedule("clench", False, delay_s=0.5)
    b.step(1.5)
    t_choice = b.events[-1][0] + b.t_start
    assert b.sc.lit(b.t) is None                 # paused
    b.step(b.sc.pause_s)
    assert b.sc.lit(b.t) == 0
    assert b.sc.lit(t_choice + b.sc.pause_s + DWELL + 0.01) == 1


def test_long_clench_stops_and_chooses_nothing(cal):
    b = ScanBench(cal)
    b.until(2 * DWELL + 0.3)
    b.sig.schedule("clench", True)
    b.step(2.5)
    assert b.kinds() == ["stop"]
    b.sig.schedule("clench", False)
    b.step(3.0)
    assert b.kinds() == ["stop", "clear"]
    assert b.choices() == []


def test_a_clench_held_about_1_2_s_is_a_stop(cal):
    b = ScanBench(cal)
    b.until(0.3)
    b.sig.schedule("clench", True)
    b.sig.schedule("clench", False, delay_s=1.2)
    b.step(4.0)
    assert "stop" in b.kinds() and b.choices() == []


@pytest.mark.parametrize("seed", [31, 32, 33])
def test_blinks_choose_nothing(cal, seed):
    b = ScanBench(cal, seed=seed)
    for k in range(5):
        b.until(k * DWELL + 0.6)
        b.sig.schedule("blink")
    b.step(3.0)
    assert b.kinds() == []


def test_talking_chooses_nothing(cal):
    b = ScanBench(cal, seed=34)
    b.sig.schedule("talk", True)
    b.step(12.0)
    assert b.choices() == [] and "stop" not in b.kinds()


def test_late_clench_within_the_grace_time_counts_for_the_tile_before(cal):
    b = ScanBench(cal)
    b.until(3 * DWELL + 0.15)        # tile 3 has just lit; the person meant tile 2
    b.sig.schedule("clench", True)
    b.sig.schedule("clench", False, delay_s=0.5)
    b.step(2.0)
    assert b.choices() == [2]


def test_clench_after_the_grace_time_counts_for_the_lit_tile(cal):
    b = ScanBench(cal)
    b.until(3 * DWELL + 0.6)
    b.sig.schedule("clench", True)
    b.sig.schedule("clench", False, delay_s=0.5)
    b.step(2.0)
    assert b.choices() == [3]


# ---------------------------------------------------------------- the scanner alone

def feed(sc, pattern, t0=0.0, dt=0.1):
    """pattern: list of (seconds, hot) stretches. Returns the events with their times."""
    out = []
    t = t0
    for seconds, hot in pattern:
        for _ in range(int(round(seconds / dt))):
            t += dt
            for ev in sc.update(t, hot, not hot, True):
                out.append((round(t, 2), ev))
    return out


def test_scanner_tile_timing_and_grace():
    sc = Scanner(4, dwell_s=1.5, grace_s=0.3)
    sc.start(10.0)
    assert sc.lit(10.0) == 0 and sc.lit(11.49) == 0 and sc.lit(11.5) == 1 and sc.lit(16.1) == 0
    assert sc.tile_for(11.6) == 0       # 0.1 s into tile 1: meant tile 0
    assert sc.tile_for(11.9) == 1
    assert sc.tile_for(16.05) == 3      # wraps: 0.05 s into tile 0 of the next round
    assert sc.tile_for(10.1) == 0       # the very first tile has nothing before it


def test_scanner_long_clench_cancels_the_pending_choice():
    sc = Scanner(4)
    sc.start(0.0)
    ev = feed(sc, [(0.5, False), (1.0, True)])
    assert ev == []                      # a burst in progress, nothing yet
    ev = feed(sc, [(1.0, True), (2.0, False)], t0=1.5)
    kinds = [e[0] for _, e in ev]
    assert kinds == ["stop", "clear"]


def test_scanner_ignores_a_burst_shorter_than_the_hold():
    sc = Scanner(4, min_s=0.4)
    sc.start(0.0)
    ev = feed(sc, [(0.5, False), (0.2, True), (1.0, False)])
    assert [e[0] for _, e in ev] == ["ignored"]


def test_scanner_nothing_while_halted_and_stop_still_works():
    sc = Scanner(4)
    ev = feed(sc, [(0.5, False), (0.6, True), (1.0, False)])
    assert [e[0] for _, e in ev] == []    # not started: no tile to choose
    ev = feed(sc, [(2.0, True), (1.5, False)], t0=2.1)
    assert [e[0] for _, e in ev] == ["stop", "clear"]


def test_scanner_bad_contact_drops_a_burst():
    sc = Scanner(4)
    sc.start(0.0)
    feed(sc, [(0.5, False), (0.5, True)])
    assert sc.update(1.1, False, True, False) == []
    assert feed(sc, [(1.0, False)], t0=1.1) == []


def test_page_state_has_the_scan_fields():
    st = ChooserState(["cup", "pen"], "scan")
    st.set(lit=1, dwell_s=2.0)
    snap = st.snapshot()
    assert snap["mode"] == "scan" and snap["lit"] == 1 and snap["dwell_s"] == 2.0
