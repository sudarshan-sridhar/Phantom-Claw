"""Calibration: seat check, then eyes open, eyes closed and mental arithmetic, all led by
voice and beeps. Saves the means and spreads that run mode uses to setup/brain_calibration.json.

With auto_label (a test switch) the windows are labelled by their own measured alpha instead
of by which step they fell in. That lets the simulator's quiet stretches stand in for eyes
open and its bursts for eyes closed, so calibration can be run with nobody wearing anything.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import speech
from .dsp import focus_ratio
from .reader import ROOT, Reader
from .rules import MIN_GOOD_CHANNELS

CAL_PATH = ROOT / "setup" / "brain_calibration.json"
TICK_S = 0.25
SETTLE_S = 2.0   # ignore the start of each step while the person settles and the window fills


def _wait_tick(next_t: float) -> float:
    delay = next_t - time.monotonic()
    if delay > 0:
        time.sleep(delay)
    return next_t + TICK_S


def seat_check(reader: Reader, min_good: int, timeout_s: float) -> int:
    speech.say("Seat check. Press the puck straight down on the top back of your head and hold still.")
    start = time.monotonic()
    last_report = 0.0
    passes = 0
    n_good = 0
    nxt = time.monotonic()
    while True:
        nxt = _wait_tick(nxt)
        f = reader.tick()
        if f is None:
            continue
        n_good = 0 if f["stale"] else f["n_good"]
        now = time.monotonic()
        if now - last_report >= 4.0:
            last_report = now
            msg = f"{n_good} of {reader.n_ch} good"
            print(f"  seat check: {msg}  (contact rms {min(f['contact_rms']):.1f} to {max(f['contact_rms']):.1f} uV)")
            passes = passes + 1 if n_good >= min_good else 0
            if passes >= 2:
                speech.say(f"{msg}. Good. Keep it there.")
                return n_good
            speech.say(msg, block=False)
        if now - start > timeout_s:
            if n_good >= MIN_GOOD_CHANNELS:
                speech.say(f"Going on with {n_good} good.")
                return n_good
            speech.say("Not enough contact. Stopping calibration.")
            raise SystemExit(f"seat check failed: only {n_good} channels with good contact")


def collect(reader: Reader, label: str, prompt: str, length_s: float, end_beeps: int = 1) -> list[dict]:
    speech.say(prompt)
    speech.beep(880, 150)
    rows = []
    start = time.monotonic()
    nxt = start
    while time.monotonic() - start < length_s:
        nxt = _wait_tick(nxt)
        f = reader.tick()
        if f is None or f["stale"] or f["n_good"] < MIN_GOOD_CHANNELS:
            continue
        if time.monotonic() - start < SETTLE_S:
            continue
        f["step"] = label
        rows.append(f)
    speech.beep(660, 150, times=end_beeps)
    alphas = [r["alpha"] for r in rows]
    if alphas:
        print(f"  {label}: {len(rows)} windows, alpha median {np.median(alphas):.1f} uV^2")
    return rows


def _stats(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    if not rows:
        return out
    for key in ("alpha", "beta", "theta"):
        v = np.array([r[key] for r in rows])
        out[f"{key}_mean"], out[f"{key}_sd"] = float(v.mean()), float(v.std())
    ratio = np.array([focus_ratio(r["beta"], r["theta"]) for r in rows])
    out["ratio_mean"], out["ratio_sd"] = float(ratio.mean()), float(ratio.std())
    return out


def split_by_alpha(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Two-means on log alpha, then keep the middle 80% of each group so windows that
    straddle a change do not widen the spreads."""
    la = np.log(np.array([r["alpha"] for r in rows]) + 1e-9)
    lo_c, hi_c = np.percentile(la, 20), np.percentile(la, 80)
    for _ in range(20):
        high = np.abs(la - hi_c) < np.abs(la - lo_c)
        if high.all() or (~high).all():
            break
        lo_c, hi_c = la[~high].mean(), la[high].mean()

    def trim(mask):
        idx = np.flatnonzero(mask)
        if len(idx) < 5:
            return [rows[i] for i in idx]
        lo, hi = np.percentile(la[idx], [10, 90])
        return [rows[i] for i in idx if lo <= la[i] <= hi]

    return trim(~high), trim(high)


def calibrate(reader: Reader, quick: bool, auto_label: bool, min_good: int, out_path: Path = CAL_PATH) -> dict:
    length = 15.0 if quick else 45.0
    print(f"calibration: {'quick' if quick else 'full'}, {length:.0f} s per step"
          + (", windows labelled by measured alpha (test mode)" if auto_label else ""))
    n_good = seat_check(reader, min_good, timeout_s=40.0 if quick else 120.0)

    rows = []
    rows += collect(reader, "open", "Eyes open. Relax and look at one spot in front of you.", length)
    rows += collect(reader, "closed", "Close your eyes and relax. Keep them closed until you hear two beeps.", length, end_beeps=2)
    rows += collect(reader, "arith", "Open your eyes. In your head, count back from seven hundred in steps of seven.", length)
    speech.say("Calibration done. Thank you.")

    if auto_label:
        low, high = split_by_alpha(rows)
        groups = {"open": [r for r in low if r["step"] != "arith"],
                  "closed": high,
                  "arith": [r for r in low if r["step"] == "arith"]}
    else:
        groups = {k: [r for r in rows if r["step"] == k] for k in ("open", "closed", "arith")}
    stats = {k: _stats(v) for k, v in groups.items()}
    for k in groups:
        if stats[k]["n"] < 4:
            raise SystemExit(f"too few clean windows for '{k}' ({stats[k]['n']}); check contact and run again")

    o, c = stats["open"], stats["closed"]
    cal = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "quick": quick,
        "auto_label": auto_label,
        "sample_rate": rows[-1]["fs"],
        "good_channels_at_seat_check": n_good,
        "stop_alpha": (o["alpha_mean"] + c["alpha_mean"]) / 2,
        "release_alpha": o["alpha_mean"] + o["alpha_sd"],
        **stats,
    }
    if c["alpha_mean"] <= o["alpha_mean"] + 2 * o["alpha_sd"]:
        cal["warning"] = "eyes closed did not clearly raise alpha; the stop will be unreliable"
        speech.say("Warning. Closing your eyes did not change the signal much. Reseat the puck and try again.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    print(f"saved {out_path}")
    print(f"  eyes open alpha   {o['alpha_mean']:8.1f} +- {o['alpha_sd']:.1f}  (n={o['n']})")
    print(f"  eyes closed alpha {c['alpha_mean']:8.1f} +- {c['alpha_sd']:.1f}  (n={c['n']})")
    print(f"  arithmetic ratio  {stats['arith']['ratio_mean']:.3f} vs rest {o['ratio_mean']:.3f}")
    print(f"  stop above {cal['stop_alpha']:.1f}, release below {cal['release_alpha']:.1f}")
    if "warning" in cal:
        print("  WARNING:", cal["warning"])
    return cal
