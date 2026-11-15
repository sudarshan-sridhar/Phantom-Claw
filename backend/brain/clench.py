"""Jaw-clench stop: a second stop trigger that does not depend on alpha.

What it reads: clenching the jaw fires the chewing muscles (temporalis, masseter) and some
neck muscles. Their electrical activity is a broadband burst, strongest above about 40 Hz,
that the pins at the back of the head pick up together, tens of microvolts or more for as
long as the teeth stay clenched. This is a muscle signal picked up by the headset, not a
brain signal.

How it is told apart from other things:
  - blink and head movement put their power below about 20 Hz. The detector watches 45 to
    100 Hz, and the "tilt" guard asks that the high band rose clearly more than the 2 to
    20 Hz band did, so a slow swing or an electrode pop (whose spectrum falls fast with
    frequency) cannot fire it even when it is large.
  - talking is the same kind of muscle signal but weaker and broken into syllables. The
    calibration measures it and puts the threshold above it, and the burst has to stay
    above the threshold for about 0.4 s.
  - the flicker response sits at the tile rates and their second harmonic, all under 45 Hz,
    and is a few microvolts.
  - 55 to 65 Hz is left out so mains hum does not count.

Every number is a ratio to the person's own resting level, per pin, then the median over
the pins that are touching skin, so one noisy pin cannot fire it.
"""
from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import signal

from .dsp import CONTACT_MIN_UV, FULL_SCALE_UV
from .reader import ROOT
from .rules import MIN_GOOD_CHANNELS

CLENCH_CAL_PATH = ROOT / "setup" / "brain_clench.json"
SIM_CLENCH_CAL_PATH = ROOT / "backend" / "brain" / "test_out" / "brain_clench_sim.json"

HF_BAND = (45.0, 100.0)     # the muscle band we watch
LF_BAND = (2.0, 20.0)       # where blinks, head movement and most EEG live
MAINS_HZ = 60.0
MAINS_GAP_HZ = 5.0          # 55 to 65 Hz left out (wider for short windows, see hf_mask)
WINDOW_S = 0.5              # each measurement uses the last half second
TICK_S = 0.25
HOLD_S = 0.4                # above threshold this long fires the stop
RELEASE_S = 1.0             # below the release level this long clears it
TILT_DEFAULT = 2.0          # the high band must rise at least twice as much as the low band
MIN_THRESHOLD = 3.0         # never fire on less than three times the resting level
GOOD_MARGIN = 2.0           # weakest clench at least twice the strongest other thing
OK_MARGIN = 1.3


# ---------------------------------------------------------------- measuring

def hf_mask(freqs: np.ndarray, fs: float) -> np.ndarray:
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    gap = max(MAINS_GAP_HZ, 2.5 * df)   # a short window has wide bins; keep their mains leak out
    hi = min(HF_BAND[1], 0.45 * fs)
    return (freqs >= HF_BAND[0]) & (freqs <= hi) & (np.abs(freqs - MAINS_HZ) > gap)


def pin_powers(raw: np.ndarray, fs: float) -> dict:
    """Per pin: power (uV^2) in the high band, in the low band, at 60 Hz, the rms, and
    whether the pin can be used. raw is (samples, pins), unfiltered.

    A pin is usable if it is not floating (rms at least 2 uV) and not pinned at full scale.
    There is no upper rms limit here, unlike the alpha contact rule: a hard clench can push
    a well seated pin past 150 uV, and that is exactly when we need it."""
    x = np.asarray(raw, dtype=np.float64)
    centred = x - x.mean(axis=0, keepdims=True)
    rms = np.sqrt(np.mean(centred**2, axis=0))
    pinned = np.abs(x).max(axis=0) > 0.9 * FULL_SCALE_UV
    usable = (rms >= CONTACT_MIN_UV) & ~pinned
    freqs, psd = signal.periodogram(x, fs=fs, window="hann", detrend="linear", axis=0)
    df = freqs[1] - freqs[0]
    hf = psd[hf_mask(freqs, fs)].sum(axis=0) * df
    lf = psd[(freqs >= LF_BAND[0]) & (freqs <= LF_BAND[1])].sum(axis=0) * df
    mains = psd[np.abs(freqs - MAINS_HZ) <= max(2.0, df / 2)].sum(axis=0) * df
    return {"hf": hf, "lf": lf, "mains": mains, "rms": rms, "usable": usable}


def score_tilt(hf, lf, usable, base_hf, base_lf) -> tuple[float, float, int]:
    """(score, tilt, pins used). score is the median over usable pins of high band power
    over its resting level. tilt is score over the same ratio for the low band (floored at
    1, so a quieter low band never inflates it): a clench raises the high band far more than
    the low band; a blink or a head movement does the opposite."""
    use = np.asarray(usable, dtype=bool)
    n = int(use.sum())
    if n == 0:
        return 0.0, 0.0, 0
    r_hf = np.asarray(hf)[use] / np.maximum(np.asarray(base_hf)[use], 1e-6)
    r_lf = np.asarray(lf)[use] / np.maximum(np.asarray(base_lf)[use], 1e-6)
    score = float(np.median(r_hf))
    tilt = score / max(float(np.median(r_lf)), 1.0)
    return score, tilt, n


class ClenchDetector:
    """Turns the newest stretch of raw samples into (score, tilt) once a tick.

    The resting level per pin starts from calibration and then follows the stream: the 25th
    percentile (high band) and the median (low band) of the last history_s of ticks. A
    clench of a few seconds is far less than a quarter of that, so it does not lift the
    resting level, but a reseat or a change in contact is followed within half a minute."""

    def __init__(self, base_hf=None, base_lf=None, window_s: float = WINDOW_S, adapt: bool = True,
                 history_s: float = 30.0, tick_s: float = TICK_S, seed_ticks: int = 40):
        self.window_s = window_s
        self.adapt = adapt
        self.maxlen = max(8, int(round(history_s / tick_s)))
        self.seed_ticks = seed_ticks
        self._cal_hf = None if base_hf is None else np.asarray(base_hf, dtype=np.float64)
        self._cal_lf = None if base_lf is None else np.asarray(base_lf, dtype=np.float64)
        self.hist_hf: deque | None = None
        self.hist_lf: deque | None = None
        self.base_hf = None
        self.base_lf = None

    def _seed(self, n_ch: int, first: dict) -> None:
        def fit(cal, fallback):
            if cal is None:
                return fallback.copy()
            if len(cal) == n_ch:
                return cal.copy()
            return np.full(n_ch, float(np.median(cal)))   # another channel count: one level for all

        self.base_hf = fit(self._cal_hf, first["hf"])
        self.base_lf = fit(self._cal_lf, first["lf"])
        self.hist_hf = deque([self.base_hf] * self.seed_ticks, maxlen=self.maxlen)
        self.hist_lf = deque([self.base_lf] * self.seed_ticks, maxlen=self.maxlen)

    def measure(self, raw: np.ndarray, fs: float) -> dict:
        n = max(16, int(round(self.window_s * fs)))
        p = pin_powers(raw[-n:], fs)
        if self.base_hf is None or len(self.base_hf) != len(p["hf"]):
            self._seed(len(p["hf"]), p)
        score, tilt, n_used = score_tilt(p["hf"], p["lf"], p["usable"], self.base_hf, self.base_lf)
        if self.adapt:
            self.hist_hf.append(p["hf"])
            self.hist_lf.append(p["lf"])
            self.base_hf = np.percentile(np.array(self.hist_hf), 25, axis=0)
            self.base_lf = np.median(np.array(self.hist_lf), axis=0)
        use = p["usable"]
        med = (lambda v: float(np.median(v[use]))) if use.any() else (lambda v: 0.0)
        return {"ok": n_used >= MIN_GOOD_CHANNELS, "score": score, "tilt": tilt, "n_used": n_used,
                "hf": med(p["hf"]), "lf": med(p["lf"]), "mains": med(p["mains"])}


class ClenchRule:
    """Fires once the score has stayed at or above threshold (with the tilt guard passing)
    for hold_s; clears once it has stayed below release for release_s. Bad contact forgets
    a hold in progress and leaves the latch as it is, like the alpha stop."""

    def __init__(self, threshold: float, release: float, tilt_min: float = TILT_DEFAULT,
                 hold_s: float = HOLD_S, release_s: float = RELEASE_S):
        self.threshold = threshold
        self.release_at = min(release, threshold)
        self.tilt_min = tilt_min
        self.hold_s = hold_s
        self.release_s = release_s
        self.active = False
        self.above_since = None
        self.below_since = None

    @classmethod
    def from_calibration(cls, cal: dict) -> "ClenchRule":
        return cls(cal["threshold"], cal["release"], cal.get("tilt_min", TILT_DEFAULT),
                   cal.get("hold_s", HOLD_S), cal.get("release_s", RELEASE_S))

    def hot(self, score: float, tilt: float) -> bool:
        return score >= self.threshold and tilt >= self.tilt_min

    def near(self, m: dict | None) -> bool:
        """On, or above the threshold now and so maybe about to fire."""
        return self.active or (m is not None and m["ok"] and self.hot(m["score"], m["tilt"]))

    def update(self, t: float, score: float, tilt: float, ok: bool) -> str | None:
        if not ok:
            self.above_since = None
            self.below_since = None
            return None
        if self.active:
            if score < self.release_at:
                if self.below_since is None:
                    self.below_since = t
                if t - self.below_since >= self.release_s - 1e-9:
                    self.active = False
                    self.below_since = None
                    return "clear"
            else:
                self.below_since = None
            return None
        if self.hot(score, tilt):
            if self.above_since is None:
                self.above_since = t
            if t - self.above_since >= self.hold_s - 1e-9:
                self.active = True
                self.above_since = None
                return "stop"
        else:
            self.above_since = None
        return None


# ---------------------------------------------------------------- calibration maths

def held_level(times, values, hold_s: float = HOLD_S) -> float:
    """The highest level the values stayed at or above for hold_s without a break: the
    highest threshold that a hold rule would fire on. 0 when the stretch is too short."""
    t = np.asarray(times, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    best = 0.0
    k = 0
    for j in range(len(t)):
        k = max(k, j)
        while k < len(t) and t[k] - t[j] < hold_s - 1e-9:
            k += 1
        if k >= len(t):
            break
        best = max(best, float(v[j:k + 1].min()))
    return best


def fit_clench(rows: list[dict], hold_s: float = HOLD_S) -> dict:
    """rows: one per tick with t, label (rest, clench, gap, talk, blink), event (which
    clench, gap or blink), and per pin hf, lf and usable. Returns the fitted levels and a
    report. The threshold goes half way (on a log scale) between the weakest clench and the
    strongest anything else; if they overlap it goes just above the strongest other, so a
    false stop is never built in, and the verdict says so."""
    rest = [r for r in rows if r["label"] == "rest"]
    if len(rest) < 8:
        raise SystemExit(f"too few clean rest windows ({len(rest)}); check contact and run again")
    base_hf = np.percentile(np.array([r["hf"] for r in rest]), 25, axis=0)
    base_lf = np.median(np.array([r["lf"] for r in rest]), axis=0)
    for r in rows:
        r["score"], r["tilt"], r["n_used"] = score_tilt(r["hf"], r["lf"], r["usable"], base_hf, base_lf)

    def group(label):
        out: dict = {}
        for r in rows:
            if r["label"] == label:
                out.setdefault(r.get("event"), []).append(r)
        return [out[k] for k in sorted(out, key=lambda e: (e is None, e))]

    clenches = [g for g in group("clench") if g]
    if not clenches:
        raise SystemExit("no clean clench windows; check contact and run again")
    # tilt over each clench's strongest ticks, then a guard well under the weakest
    tilts = []
    for g in clenches:
        s = np.array([r["score"] for r in g])
        tilts.append(float(np.median([r["tilt"] for r in g if r["score"] >= 0.5 * s.max()])))
    tilt_min = float(np.clip(0.5 * min(tilts), 1.0, TILT_DEFAULT))

    def eff(g):
        return [r["score"] if r["tilt"] >= tilt_min else 0.0 for r in g]

    # a clench counts only if it passes the tilt guard too; everything else is judged on the
    # bare level, so the threshold clears it on its own and the guard is a second safety
    levels = [held_level([r["t"] for r in g], eff(g), hold_s) for g in clenches]
    others = {}
    for label in ("rest", "gap", "talk", "blink"):
        segs = group(label)
        others[label] = max([held_level([r["t"] for r in g], [r["score"] for r in g], hold_s) for g in segs],
                            default=0.0)
    weakest = min(levels)
    strongest = max(max(others.values()), 1.0)
    margin = weakest / strongest
    if margin > 1.0:
        threshold = float(np.sqrt(weakest * strongest))
        threshold = max(threshold, min(MIN_THRESHOLD, weakest / 1.2))
    else:
        threshold = strongest * 1.1
    release = float(max(np.sqrt(threshold), 1.5))
    caught = sum(1 for v in levels if v >= threshold)
    false = [k for k, v in others.items() if v >= threshold]
    if caught == len(levels) and not false and margin >= GOOD_MARGIN:
        verdict = "good"
    elif caught == len(levels) and not false and margin >= OK_MARGIN:
        verdict = "usable"
    else:
        verdict = "not good"
    return {
        "threshold": threshold, "release": release, "tilt_min": tilt_min, "hold_s": hold_s,
        "release_s": RELEASE_S, "window_s": WINDOW_S, "band_hz": list(HF_BAND),
        "mains_gap_hz": [MAINS_HZ - MAINS_GAP_HZ, MAINS_HZ + MAINS_GAP_HZ],
        "base_hf": base_hf.round(4).tolist(), "base_lf": base_lf.round(4).tolist(),
        "base_hf_median": float(np.median(base_hf)),
        "clench_levels": [round(v, 2) for v in levels], "clench_tilts": [round(v, 2) for v in tilts],
        "others": {k: round(v, 2) for k, v in others.items()},
        "margin": round(margin, 2), "caught": caught, "n_clenches": len(levels),
        "would_fire": false, "verdict": verdict,
    }


def save_clench(cal: dict, path: Path = CLENCH_CAL_PATH) -> None:
    cal = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), **cal}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cal, indent=2), encoding="utf-8")


def load_clench(path: Path = CLENCH_CAL_PATH, required: bool = True) -> dict | None:
    if not path.exists():
        if required:
            raise SystemExit(f"no clench calibration at {path}; run: python -m backend.brain clench-calibrate --board")
        return None
    cal = json.loads(path.read_text(encoding="utf-8"))
    for key in ("threshold", "release", "base_hf", "base_lf"):
        if key not in cal:
            raise SystemExit(f"{path} has no {key}; run clench-calibrate again")
    return cal


def detector_from(cal: dict, **kw) -> ClenchDetector:
    return ClenchDetector(cal.get("base_hf"), cal.get("base_lf"), **kw)


# ---------------------------------------------------------------- which stop

ALPHA_WARNING = "eyes closed did not clearly raise alpha"


def pick_stops(requested: str | None, have_alpha: bool, alpha_reliable: bool, have_clench: bool) -> tuple[set, list]:
    """Which stop triggers to run, and notes to print. requested is "alpha", "clench",
    "both" or None for the default: both when both calibrations exist, else whichever does.
    An alpha calibration that itself warned the stop would be unreliable is left out of the
    default when a clench calibration exists (it would fire on its own on that head); asking
    for it by name still turns it on."""
    notes = []
    if requested in ("alpha", "both") and not have_alpha:
        raise SystemExit("no eyes-closed calibration; run: python -m backend.brain calibrate --board")
    if requested in ("clench", "both") and not have_clench:
        raise SystemExit("no clench calibration; run: python -m backend.brain clench-calibrate --board")
    if requested == "alpha":
        stops = {"alpha"}
    elif requested == "clench":
        stops = {"clench"}
    elif requested == "both":
        stops = {"alpha", "clench"}
    elif requested is None:
        if have_alpha and have_clench:
            if alpha_reliable:
                stops = {"alpha", "clench"}
            else:
                stops = {"clench"}
                notes.append("the eyes-closed calibration says alpha barely changed, so only the clench "
                             "stop is on; add --stop both to use both anyway")
        elif have_clench:
            stops = {"clench"}
        elif have_alpha:
            stops = {"alpha"}
        else:
            raise SystemExit("no stop calibration; run: python -m backend.brain clench-calibrate --board "
                             "(or calibrate --board for the eyes-closed stop)")
    else:
        raise SystemExit(f"--stop must be alpha, clench or both, not {requested!r}")
    if "alpha" in stops and have_alpha and not alpha_reliable:
        notes.append("WARNING: the eyes-closed calibration warned that alpha barely changed; "
                     "that stop may fire on its own")
    return stops, notes
