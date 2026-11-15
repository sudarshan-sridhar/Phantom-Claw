"""The two things run mode decides from the numbers: the eyes-closed STOP and the focus value.
Both take a time in seconds with every update so the tests can drive them with fake clocks."""
import math

from .dsp import focus_ratio

MIN_GOOD_CHANNELS = 3   # below this nothing fires, we just report no_signal
CLENCH_RMS_UV = 80.0    # jaw clench or chewing: broadband muscle noise, freeze focus


class StopRule:
    """Fires once alpha has stayed above stop_at for hold_s, releases when it drops below
    release_at. The gap between the two keeps it from flickering on the edge."""

    def __init__(self, stop_at: float, release_at: float, hold_s: float = 1.5):
        self.stop_at = stop_at
        self.release_at = min(release_at, stop_at)
        self.hold_s = hold_s
        self.active = False
        self.above_since = None

    def update(self, t: float, alpha: float, ok: bool) -> str | None:
        if not ok:
            # bad contact: forget any hold in progress and leave the latch as it is
            self.above_since = None
            return None
        if self.active:
            if alpha < self.release_at:
                self.active = False
                return "clear"
            return None
        if alpha > self.stop_at:
            if self.above_since is None:
                self.above_since = t
            if t - self.above_since >= self.hold_s:
                self.active = True
                self.above_since = None
                return "stop"
        else:
            self.above_since = None
        return None


class FocusMeter:
    """beta / (beta + theta), placed between the resting and the arithmetic means so 0 is
    resting and 1 is working hard, then smoothed with a time constant of tau_s."""

    def __init__(self, rest_ratio: float, work_ratio: float, spread: float = 0.0, tau_s: float = 2.0):
        lo, hi = rest_ratio, work_ratio
        if hi - lo < max(spread, 1e-3):
            # arithmetic did not move the ratio (or moved it the wrong way); fall back to
            # a range of two spreads above rest so the meter still means something
            hi = lo + max(2 * spread, 0.05)
        self.lo, self.hi = lo, hi
        self.tau_s = tau_s
        self.value = None
        self.last_t = None
        self.frozen = False

    def update(self, t: float, beta: float, theta: float, rms: float, ok: bool) -> float | None:
        self.frozen = (not ok) or rms > CLENCH_RMS_UV
        if self.frozen:
            self.last_t = t
            return self.value
        raw = (focus_ratio(beta, theta) - self.lo) / (self.hi - self.lo)
        raw = min(1.0, max(0.0, raw))
        if self.value is None:
            self.value = raw
        else:
            dt = t - self.last_t if self.last_t is not None else 0.25
            k = 1.0 - math.exp(-max(dt, 0.0) / self.tau_s)
            self.value += k * (raw - self.value)
        self.last_t = t
        return self.value
