"""Signal maths for the brain reader. Pure functions on numpy arrays, no streams or sockets,
so the tests can feed them made-up signals.

Arrays are shaped (samples, channels) in microvolts.
"""
from functools import lru_cache

import numpy as np
from scipy import signal

BANDS = {"theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}

# PiEEG's contact rule for the dry pins: RMS between 2 and 150 uV is a pin touching skin.
# Below that the pin is floating (flat), above it hair or movement is in the way (railed).
CONTACT_MIN_UV = 2.0
CONTACT_MAX_UV = 150.0
# ADS131M08 at gain 4 tops out at 1.2 V / 4 = 300000 uV. A channel sitting near that is pinned.
FULL_SCALE_UV = 1.2 / 4.0 * 1e6


@lru_cache(maxsize=16)
def _filters(fs_rounded: float):
    """Band-pass 1 to 40 Hz plus a 60 Hz notch, designed for one sample rate.
    Cached by rate rounded to 0.5 Hz so a drifting estimate does not redesign every tick."""
    sos = signal.butter(4, [1.0, 40.0], btype="bandpass", fs=fs_rounded, output="sos")
    b, a = signal.iirnotch(60.0, 30.0, fs=fs_rounded)
    return sos, b, a


def clean(raw: np.ndarray, fs: float) -> np.ndarray:
    """Band-pass and notch along the time axis.

    This runs forwards only, over a few seconds of history each tick. A zero-phase filter
    (filtfilt) looks nicer offline but rings at the newest end of the buffer, and the newest
    second is exactly the part we measure. Starting the filter state from the first sample
    and giving it three seconds to settle leaves the last second clean."""
    sos, b, a = _filters(round(fs * 2) / 2)
    x = raw - raw[:1]
    zi = signal.sosfilt_zi(sos)[:, :, None] * x[0][None, None, :]
    x, _ = signal.sosfilt(sos, x, axis=0, zi=zi)
    zn = signal.lfilter_zi(b, a)[:, None] * x[0][None, :]
    x, _ = signal.lfilter(b, a, x, axis=0, zi=zn)
    return x


def contact(raw_last: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per channel (good flags, rms) from the last second of raw, unfiltered samples."""
    centred = raw_last - raw_last.mean(axis=0, keepdims=True)
    rms = np.sqrt(np.mean(centred**2, axis=0))
    pinned = np.abs(raw_last).max(axis=0) > 0.9 * FULL_SCALE_UV
    good = (rms >= CONTACT_MIN_UV) & (rms <= CONTACT_MAX_UV) & ~pinned
    return good, rms


def band_powers(x: np.ndarray, fs: float) -> dict:
    """Power in each band (uV^2) per channel, from a Hann-windowed periodogram of x."""
    n = x.shape[0]
    freqs, psd = signal.periodogram(x, fs=fs, window="hann", axis=0, detrend="constant")
    df = freqs[1] - freqs[0] if n > 1 else 1.0
    out = {}
    for name, (lo, hi) in BANDS.items():
        sel = (freqs >= lo) & (freqs < hi)
        out[name] = psd[sel].sum(axis=0) * df
    return out


def features(raw: np.ndarray, fs: float, window_s: float = 1.0) -> dict:
    """Everything the rules need from one tick.

    raw holds a few seconds of samples so the filters have room to settle; band power and
    contact use only the last window_s of it. Powers and rms are averaged over good channels.
    """
    n_win = max(8, int(round(window_s * fs)))
    last_raw = raw[-n_win:]
    good, rms_contact = contact(last_raw)
    out = {"n_good": int(good.sum()), "good": good.tolist(), "contact_rms": rms_contact.round(1).tolist()}
    if not good.any():
        out.update(alpha=0.0, beta=0.0, theta=0.0, rms=0.0)
        return out
    x = clean(raw[:, good], fs)[-n_win:]
    p = band_powers(x, fs)
    out.update({k: float(v.mean()) for k, v in p.items()})
    out["rms"] = float(np.sqrt(np.mean(x**2, axis=0)).mean())
    return out


def focus_ratio(beta: float, theta: float) -> float:
    return beta / (beta + theta) if beta + theta > 0 else 0.5


def measured_rate(timestamps: np.ndarray, nominal: float) -> float:
    """Sample rate from the stream's own time stamps, falling back to the nominal rate
    until there are a couple of seconds to measure over."""
    if len(timestamps) < 2:
        return nominal
    span = float(timestamps[-1] - timestamps[0])
    if span < 2.0:
        return nominal
    fs = (len(timestamps) - 1) / span
    # anything far outside the board's range means the stamps are junk, not the board
    return fs if 100.0 <= fs <= 1000.0 else nominal
