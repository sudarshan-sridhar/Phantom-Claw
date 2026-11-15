"""Choosing an item by looking at it (SSVEP). Pure maths and config, no streams or sockets,
so the tests can feed it made-up signals.

Each tile on the chooser page flickers at its own rate. Looking at one tile makes the visual
cortex at the back of the head follow that rate. We compare a few seconds of the back-of-head
channels with sine and cosine references at each rate and its second harmonic (canonical
correlation, CCA) and pick the rate that matches best, but only when it clearly beats the
runner-up for several windows in a row.

Arrays are shaped (samples, channels) in microvolts.
"""
from __future__ import annotations

import itertools
import json
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import signal

from .reader import ROOT

CONFIG_PATH = Path(__file__).with_name("ssvep_config.json")
SSVEP_CAL_PATH = ROOT / "setup" / "brain_ssvep.json"

# Positions we accept in the channel map. These are the spots over the visual cortex and
# just above it where a flicker response shows up.
POSITIONS = ("O1", "Oz", "O2", "POz", "PO3", "PO4", "Pz")

# The eyes-closed stop watches 8 to 13 Hz. We keep flicker rates out of this range (plus a
# half hertz each side for spectral leakage) whenever the screen leaves enough other rates.
ALPHA_GUARD = (7.5, 13.5)

# CCA may be limited to the strongest few directions across the channels (None = all).
# On the simulator all 16 did best; try 4 on the real board if rest scores run high.
MAX_DIMS = None

# how much a rate inside the alpha guard costs when picking rates (a strong rate scores 1)
GUARD_PENALTY = 0.8


# ---------------------------------------------------------------- config

@dataclass
class SsvepConfig:
    items: list[str] = field(default_factory=lambda: ["bottle", "red bull", "sharpie", "scissors", "glue stick"])
    sentence: str = "bring me the {item}"
    # position name -> list of board inputs, numbered 1 to 16 as printed on the board
    channels: dict[str, list[int]] = field(default_factory=lambda: {"Oz": list(range(1, 17))})
    window_s: float = 3.0
    step_s: float = 0.25
    harmonics: int = 2
    # the 16 pins sit on one patch of scalp and see nearly the same signal, so CCA works on
    # the strongest few principal components of the good pins rather than 16 separate sites
    cca_dims: int | None = 3
    refractory_s: float = 5.0
    port: int = 8767
    server: str = "http://127.0.0.1:8765"
    items_url: str | None = None   # optional: fetch the item list from here (the arm scene, later)
    rates_hz: list[float] | None = None   # optional: force these rates instead of picking

    def inputs(self) -> list[int]:
        """Zero-based column numbers for every configured input, in a stable order."""
        cols = sorted({i - 1 for ins in self.channels.values() for i in ins})
        if not cols:
            raise ValueError("the channel map has no inputs; at least one is needed")
        return cols

    def text_for(self, item: str) -> str:
        return self.sentence.format(item=item)

    def validate(self) -> None:
        if not self.items:
            raise ValueError("no items to choose from")
        if len(set(self.items)) != len(self.items):
            raise ValueError("item names must be different from each other")
        for pos, ins in self.channels.items():
            if pos not in POSITIONS:
                raise ValueError(f"unknown position {pos!r}; use one of {', '.join(POSITIONS)}")
            for i in ins:
                if not 1 <= int(i) <= 16:
                    raise ValueError(f"input {i} at {pos} is not between 1 and 16")
        self.inputs()
        if self.cca_dims is not None and int(self.cca_dims) < 1:
            raise ValueError("cca_dims must be 1 or more, or null for all pins")
        if "{item}" not in self.sentence:
            raise ValueError("the sentence needs {item} in it, for example 'bring me the {item}'")


def load_config(path: Path = CONFIG_PATH) -> SsvepConfig:
    cfg = SsvepConfig()
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {k: v for k, v in data.items() if k in SsvepConfig.__dataclass_fields__}
        cfg = SsvepConfig(**{**asdict(cfg), **known})
    cfg.channels = {k: [int(i) for i in v] for k, v in cfg.channels.items()}
    cfg.validate()
    return cfg


def save_config(cfg: SsvepConfig, path: Path) -> None:
    cfg.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")


def parse_channels(text: str) -> dict[str, list[int]]:
    """'Oz=1-16' or 'O1=1-4,Oz=5-12,O2=13-16' or 'Oz=1-8;POz=9,10'."""
    out: dict[str, list[int]] = {}
    for part in re.split(r"\s*;\s*|,(?=\s*[A-Za-z])", text.strip()):
        part = part.strip()
        if not part:
            continue
        pos, _, spec = part.partition("=")
        pos = pos.strip()
        match = {p.lower(): p for p in POSITIONS}.get(pos.lower())
        if match is None:
            raise ValueError(f"unknown position {pos!r}; use one of {', '.join(POSITIONS)}")
        ins: list[int] = []
        for chunk in spec.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "-" in chunk:
                a, b = (int(v) for v in chunk.split("-"))
                ins += list(range(a, b + 1))
            else:
                ins.append(int(chunk))
        out[match] = ins
    return out


def items_from_payload(payload) -> list[str]:
    """Pull item names out of whatever the scene endpoint returns: a list of names, a list
    of objects with a name or label, or an object holding such a list under items/objects."""
    if isinstance(payload, dict):
        for key in ("items", "objects", "scene", "names"):
            if key in payload:
                return items_from_payload(payload[key])
        return []
    names = []
    for entry in payload or []:
        if isinstance(entry, str):
            names.append(entry)
        elif isinstance(entry, dict):
            name = entry.get("name") or entry.get("label") or entry.get("item")
            if name:
                names.append(str(name))
    seen = set()
    return [n for n in names if not (n in seen or seen.add(n))]


# ---------------------------------------------------------------- rate picking

@dataclass
class Rate:
    frames: int        # frames per flicker cycle on the screen
    hz: float          # refresh / frames

    def to_dict(self) -> dict:
        return {"frames": self.frames, "hz": round(self.hz, 4)}


def _strength(f: float) -> float:
    """How strong a flicker response we expect at f, roughly. Strongest from about 6 to
    16 Hz against the background, weaker above 18 Hz, and poor below 6 Hz where the
    background brain noise is loudest."""
    if f < 5.0:
        return 0.35
    if f < 8.0:
        return 0.55 + 0.45 * (f - 5.0) / 3.0
    if f <= 16.0:
        return 1.0
    return max(0.3, 1.0 - 0.06 * (f - 16.0))


def _in_guard(f: float) -> bool:
    return ALPHA_GUARD[0] <= f <= ALPHA_GUARD[1]


def pick_rates(refresh_hz: float, n: int) -> list[Rate]:
    """Pick n flicker rates that the screen can show exactly (refresh divided by a whole
    number of frames), so each tile is on for some frames and off for the rest of a fixed
    cycle.

    Rules, in order of weight: rates must be at least 0.5 Hz apart; no rate may sit on
    another's second or third harmonic (looking at 5 Hz also shows at 15 Hz, since a square
    flicker carries a strong third harmonic);
    stay out of the 7.5 to 13.5 Hz alpha guard where possible, and keep second harmonics out
    of it too; then prefer rates with strong responses and wide spacing."""
    if refresh_hz < 30:
        raise ValueError(f"refresh rate {refresh_hz} Hz is too low for flicker")
    cands = []
    for k in range(2, 64):
        f = refresh_hz / k
        if f < 4.5:
            break
        if f <= 24.0:
            cands.append(Rate(k, f))
    if len(cands) < n:
        raise ValueError(f"only {len(cands)} usable rates at {refresh_hz:.1f} Hz for {n} items")

    def ok(combo) -> bool:
        fs = [r.hz for r in combo]
        for a, b in itertools.combinations(fs, 2):
            if abs(a - b) < 0.5:
                return False
            lo, hi = min(a, b), max(a, b)
            if any(abs(m * lo - hi) < 0.4 for m in (2, 3)):
                return False
        return True

    def score(combo) -> float:
        fs = sorted(r.hz for r in combo)
        s = sum(_strength(f) for f in fs)
        s -= sum(GUARD_PENALTY for f in fs if _in_guard(f))
        s -= sum(0.1 for f in fs if _in_guard(2 * f))
        gaps = np.diff(fs) if len(fs) > 1 else np.array([5.0])
        s -= sum(0.8 * (1.0 - g) for g in gaps if g < 1.0)
        return float(s)

    pool = cands if len(cands) <= 22 else cands[:22]
    best, best_s = None, -1e9
    for combo in itertools.combinations(pool, n):
        if not ok(combo):
            continue
        s = score(combo)
        if s > best_s:
            best, best_s = combo, s
    if best is None:
        raise ValueError(f"no set of {n} clean rates at {refresh_hz:.1f} Hz; use fewer items")
    # strongest expected response first, so the first items in the list (the ones used most)
    # get the most reliable tiles; ties go to the higher rate
    return sorted(best, key=lambda r: (-_strength(r.hz), -r.hz))


def rates_from_frames(refresh_hz: float, frames: list[int]) -> list[Rate]:
    return [Rate(int(k), refresh_hz / int(k)) for k in frames]


# ---------------------------------------------------------------- filtering and CCA

@lru_cache(maxsize=16)
def _ssvep_filters(fs_rounded: float):
    hi = min(45.0, 0.45 * fs_rounded)
    sos = signal.butter(4, [3.0, hi], btype="bandpass", fs=fs_rounded, output="sos")
    b, a = signal.iirnotch(60.0, 30.0, fs=fs_rounded)
    return sos, b, a


def ssvep_clean(raw: np.ndarray, fs: float) -> np.ndarray:
    """Band-pass 3 to 45 Hz and notch 60 Hz, forwards only (see dsp.clean for why). The
    caller hands in about a second more than it needs and drops the start."""
    sos, b, a = _ssvep_filters(round(fs * 2) / 2)
    x = raw - raw[:1]
    zi = signal.sosfilt_zi(sos)[:, :, None] * x[0][None, None, :]
    x, _ = signal.sosfilt(sos, x, axis=0, zi=zi)
    zn = signal.lfilter_zi(b, a)[:, None] * x[0][None, :]
    x, _ = signal.lfilter(b, a, x, axis=0, zi=zn)
    return x


def references(freq: float, n: int, fs: float, harmonics: int = 2) -> np.ndarray:
    t = np.arange(n) / fs
    cols = []
    for h in range(1, harmonics + 1):
        cols.append(np.sin(2 * np.pi * h * freq * t))
        cols.append(np.cos(2 * np.pi * h * freq * t))
    return np.stack(cols, axis=1)


def _basis(x: np.ndarray, max_dims: int | None = None) -> np.ndarray:
    """Orthonormal basis for the columns of x, dropping directions with almost no variance
    and, with max_dims, keeping only the strongest few. The 16 pins of one puck are nearly
    copies of each other, so plain QR would be unstable, and letting CCA weigh all 16 freely
    mostly teaches it to fit noise."""
    x = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    if s.size == 0 or s[0] <= 0:
        return u[:, :0]
    keep = s > 1e-6 * s[0]
    u = u[:, keep]
    return u[:, :max_dims] if max_dims else u


def cca_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Largest canonical correlation between the columns of x and of y."""
    qx, qy = _basis(x), _basis(y)
    if qx.shape[1] == 0 or qy.shape[1] == 0:
        return 0.0
    s = np.linalg.svd(qx.T @ qy, compute_uv=False)
    return float(min(1.0, s[0])) if s.size else 0.0


def cca_scores(x: np.ndarray, fs: float, rates_hz: list[float], harmonics: int = 2,
               max_dims: int | None = MAX_DIMS) -> np.ndarray:
    """One canonical correlation per rate for a cleaned window x (samples, channels)."""
    qx = _basis(x, max_dims)
    out = np.zeros(len(rates_hz))
    if qx.shape[1] == 0:
        return out
    for i, f in enumerate(rates_hz):
        harm = harmonics if harmonics * f < 0.45 * fs else 1
        qy = _basis(references(f, x.shape[0], fs, harm))
        s = np.linalg.svd(qx.T @ qy, compute_uv=False)
        out[i] = min(1.0, s[0]) if s.size else 0.0
    return out


def window_scores(raw: np.ndarray, fs: float, rates_hz: list[float], window_s: float = 3.0,
                  harmonics: int = 2, settle_s: float = 1.0, max_dims: int | None = MAX_DIMS) -> np.ndarray | None:
    """CCA scores for the newest window_s of raw, or None if raw is too short. raw should
    already be just the configured, good channels."""
    n_win = int(round(window_s * fs))
    n_all = n_win + int(round(settle_s * fs))
    if raw.shape[0] < n_all or raw.shape[1] == 0:
        return None
    x = ssvep_clean(raw[-n_all:], fs)[-n_win:]
    return cca_scores(x, fs, rates_hz, harmonics, max_dims)


def alpha_excluding(raw: np.ndarray, fs: float, rates_hz: list[float], window_s: float = 2.0,
                    settle_s: float = 2.0, half_width: float = 0.6) -> float | None:
    """Alpha power (8 to 13 Hz, uV^2, mean over channels) with narrow bins around each
    flicker rate and its second harmonic left out and the rest scaled back up to the full
    band. A flicker response is a sharp line, eyes-closed alpha is a broad hump, so this
    keeps the stop deaf to the tiles while it still hears closed eyes."""
    from .dsp import clean
    n_win = int(round(window_s * fs))
    n_all = n_win + int(round(settle_s * fs))
    if raw.shape[0] < n_all or raw.shape[1] == 0:
        return None
    x = clean(raw[-n_all:], fs)[-n_win:]
    freqs, psd = signal.periodogram(x, fs=fs, window="hann", axis=0, detrend="constant")
    df = freqs[1] - freqs[0]
    band = (freqs >= 8.0) & (freqs < 13.0)
    keep = band.copy()
    for f in rates_hz:
        for h in (f, 2 * f):
            keep &= np.abs(freqs - h) > half_width
    if keep.sum() < 2:
        keep = band
    p = psd[keep].sum(axis=0) * df * band.sum() / keep.sum()
    return float(p.mean())


# ---------------------------------------------------------------- deciding

@dataclass
class Thresholds:
    threshold: float = 0.35     # best (baseline-corrected) score must reach this
    margin: float = 0.05        # and beat the runner-up by this much
    hold: int = 6               # for this many windows in a row (0.25 s apart)
    baseline: list[float] | None = None   # score per rate when not looking at it, subtracted first
    scale: list[float] | None = None      # its spread, divided out, so every rate counts alike

    def corrected(self, scores: np.ndarray) -> np.ndarray:
        c = np.asarray(scores, dtype=float)
        if self.baseline is not None and len(self.baseline) == c.shape[-1]:
            c = c - np.asarray(self.baseline)
        if self.scale is not None and len(self.scale) == c.shape[-1]:
            c = c / np.asarray(self.scale)
        return c


def classify(scores: np.ndarray, th: Thresholds) -> tuple[int | None, float, float]:
    """(index or None, best corrected score, margin over the runner-up) for one window."""
    c = th.corrected(scores)
    order = np.argsort(c)[::-1]
    best = float(c[order[0]])
    second = float(c[order[1]]) if len(c) > 1 else 0.0
    margin = best - second
    if best >= th.threshold and margin >= th.margin:
        return int(order[0]), best, margin
    return None, best, margin


class Decider:
    """Turns per-window guesses into a choice: the same guess for th.hold windows in a row,
    then a refractory time. After a choice the same item cannot fire again until the guess
    has stayed off it for release_windows in a row (default twice the hold), so staring at a
    tile does not order it twice. Times are passed in so tests can drive it with a fake clock."""

    def __init__(self, th: Thresholds, refractory_s: float = 5.0, release_windows: int | None = None):
        self.th = th
        self.refractory_s = refractory_s
        self.release_windows = release_windows or 2 * max(1, th.hold)
        self.current: int | None = None
        self.count = 0
        self.last_fire_t: float | None = None
        self.last_item: int | None = None
        self.released = True
        self.away = 0
        self.block_until = -1e9

    def reset(self) -> None:
        self.current, self.count = None, 0

    def suppress(self, t: float) -> None:
        """The stop is on or about to be, or the flicker is off: forget any choice in
        progress, and choose nothing until refractory_s after the last call. The item chosen
        last stays blocked until the gaze moves off it, so opening your eyes on the same
        tile after a stop does not order it again."""
        self.reset()
        self.block_until = t + self.refractory_s

    def progress(self) -> tuple[int | None, float]:
        if self.current is None:
            return None, 0.0
        return self.current, min(1.0, self.count / max(1, self.th.hold))

    def update(self, t: float, scores: np.ndarray) -> int | None:
        guess, _, _ = classify(scores, self.th)
        if t < self.block_until:
            # right after a stop or a pause: windows still hold closed eyes or no flicker,
            # so they say nothing about where the gaze went and do not count as looking away
            self.reset()
            return None
        if not self.released:
            self.away = self.away + 1 if guess != self.last_item else 0
            if self.away >= self.release_windows:
                self.released = True
        if self.last_fire_t is not None and t - self.last_fire_t < self.refractory_s:
            self.reset()
            return None
        if guess is None:
            self.reset()
            return None
        if guess == self.last_item and not self.released:
            self.reset()
            return None
        if guess == self.current:
            self.count += 1
        else:
            self.current, self.count = guess, 1
        if self.count >= self.th.hold:
            self.last_fire_t = t
            self.last_item = guess
            self.released = False
            self.away = 0
            self.reset()
            return guess
        return None


# ---------------------------------------------------------------- calibration maths

def fit_thresholds(target: dict[int, np.ndarray], rest: np.ndarray, hold: int = 4,
                   max_rest_false: float = 0.05) -> Thresholds:
    """Thresholds from calibration windows.

    target maps rate index -> array (windows, rates) of raw scores recorded while looking at
    that tile; rest is (windows, rates) recorded while looking at the centre.
    Each rate is first put on a common footing: its median and spread while nobody looks at
    it (the rest windows plus the windows of every other tile) are subtracted and divided out.
    Low rates sit on more background EEG and would otherwise win for nothing. Then a grid search over threshold and margin: among
    the pairs that let at most max_rest_false of the resting windows through, take the one
    that passes the most correct windows, with every wrong pick counted twice against it.
    The hold of several windows in a row cuts false choices down further at run time."""
    n_rates = rest.shape[1]
    baseline, scale = np.zeros(n_rates), np.ones(n_rates)
    for j in range(n_rates):
        # what rate j scores when nobody is looking at it: rest, plus every other tile
        away = [rest[:, j]] + [w[:, j] for i, w in target.items() if i != j and len(w)]
        v = np.concatenate(away)
        baseline[j] = np.median(v)
        scale[j] = max(1.4826 * np.median(np.abs(v - baseline[j])), 1e-3)
    norm = Thresholds(baseline=list(baseline), scale=list(scale))

    def best_and_margin(c):
        order = np.argsort(c, axis=1)
        top = np.take_along_axis(c, order[:, -1:], axis=1)[:, 0]
        second = np.take_along_axis(c, order[:, -2:-1], axis=1)[:, 0] if c.shape[1] > 1 else np.zeros(len(c))
        return order[:, -1], top, top - second

    _, rb, rm = best_and_margin(norm.corrected(rest))
    tc = norm.corrected(np.vstack([w for w in target.values()])) if target else np.zeros((0, n_rates))
    labels = np.concatenate([np.full(len(w), i) for i, w in target.items()]) if target else np.zeros(0)
    ti, tb, tm = best_and_margin(tc) if len(tc) else (np.zeros(0, int), np.zeros(0), np.zeros(0))
    correct = ti == labels

    qs = np.linspace(50, 100, 21)
    thr_grid = np.unique(np.concatenate([[0.0], np.percentile(rb, qs)]))
    mar_grid = np.unique(np.concatenate([[0.0], np.percentile(rm, qs)]))
    best_pair, best_val = (float(thr_grid[-1]), float(mar_grid[-1])), -np.inf
    for t in thr_grid:
        rest_pass_t = rb >= t
        pass_t = tb >= t
        for m in mar_grid:
            if np.mean(rest_pass_t & (rm >= m)) > max_rest_false:
                continue
            ok = pass_t & (tm >= m)
            val = np.mean(ok & correct) - 2.0 * np.mean(ok & ~correct) if len(ok) else 0.0
            # on a tie keep the stricter pair
            if val > best_val + 1e-9 or (abs(val - best_val) <= 1e-9 and t + m > sum(best_pair)):
                best_pair, best_val = (float(t), float(m)), val
    thr, mar = best_pair
    return Thresholds(threshold=max(thr, 0.0), margin=max(mar, 0.05), hold=hold,
                      baseline=[float(b) for b in baseline], scale=[float(v) for v in scale])


def leave_one_out(target: dict[int, np.ndarray], rest: np.ndarray, hold: int = 4) -> dict:
    """Leave-one-window-out check: each window is classified with thresholds fitted on all
    the other windows. Returns per-rate accuracy (correct / windows), how often rest stayed
    undecided, and the overall confusion counts.

    Windows overlap (3 s long, 0.25 s apart), so this is kinder than a fresh session would
    be. Treat it as a check that the setup works, not as a promise."""
    n_rates = rest.shape[1]
    per_rate = {}
    confusion = np.zeros((n_rates + 1, n_rates + 1), dtype=int)   # last row/col = none / rest

    def fit_without(key, j):
        t2 = {k: (np.delete(v, j, axis=0) if k == key else v) for k, v in target.items()}
        r2 = np.delete(rest, j, axis=0) if key == "rest" else rest
        return fit_thresholds(t2, r2, hold)

    for i, w in target.items():
        correct = 0
        for j in range(len(w)):
            th = fit_without(i, j)
            g, _, _ = classify(w[j], th)
            confusion[i, n_rates if g is None else g] += 1
            correct += int(g == i)
        per_rate[i] = correct / max(1, len(w))
    quiet = 0
    for j in range(len(rest)):
        th = fit_without("rest", j)
        g, _, _ = classify(rest[j], th)
        confusion[n_rates, n_rates if g is None else g] += 1
        quiet += int(g is None)
    return {"per_rate": per_rate, "rest_undecided": quiet / max(1, len(rest)), "confusion": confusion.tolist()}
