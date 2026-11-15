"""clench-calibrate and signal: the spoken jaw-clench calibration, and a short read-out for
judging a seat without calibrating anything."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import speech
from .calibrate import seat_check
from .clench import CLENCH_CAL_PATH, fit_clench, pin_powers, save_clench
from .dsp import measured_rate
from .reader import STALE_S, Reader
from .clench import TICK_S, WINDOW_S

CLENCH_TAIL_S = 0.9   # after the let-go beep, still part of the clench (reaction time plus the window)


def _wait(nxt: float, step: float = TICK_S) -> float:
    delay = nxt - time.monotonic()
    if delay > 0:
        time.sleep(delay)
    return nxt + step


def _collect(reader: Reader, rows: list, label: str, seconds: float, event=None, settle_s: float = 0.0) -> int:
    """Record one stretch: per tick, per pin powers of the last half second."""
    start = time.monotonic()
    nxt = start
    n0 = len(rows)
    while time.monotonic() - start < seconds:
        nxt = _wait(nxt)
        reader.pull()
        if len(reader.ts) < 2 or reader.ts[-1] - reader.ts[0] < 1.0:
            continue
        if reader.last_arrival is None or time.monotonic() - reader.last_arrival > STALE_S:
            continue
        el = time.monotonic() - start
        if el < settle_s:
            continue
        fs = measured_rate(reader.ts, reader.nominal)
        p = pin_powers(reader.data[-int(round(WINDOW_S * fs)):], fs)
        rows.append({"t": time.monotonic(), "label": label, "event": event, "fs": fs,
                     "hf": p["hf"], "lf": p["lf"], "usable": p["usable"]})
    return len(rows) - n0


def clench_calibrate(reader: Reader, quick: bool = False, out_path: Path = CLENCH_CAL_PATH, sim=None,
                     min_good: int = 10, seat: bool = True) -> dict:
    rest_s = 6.0 if quick else 10.0
    talk_s = 6.0 if quick else 10.0
    n = 3 if quick else 5
    print(f"clench calibration: {'quick' if quick else 'full'}; rest {rest_s:.0f} s, {n} clenches, "
          f"talking {talk_s:.0f} s, {n} blinks")
    n_good = seat_check(reader, min_good, timeout_s=40.0 if quick else 90.0) if seat else None
    rows: list[dict] = []

    speech.say("Jaw clench calibration. First, rest. Sit still, eyes open, jaw loose.")
    speech.beep(880, 150)
    _collect(reader, rows, "rest", rest_s, settle_s=1.5)
    speech.beep(660, 150)

    speech.say(f"Now {n} jaw clenches. At the high beep, bite down hard and hold. At the low beep, let go.")
    for i in range(n):
        speech.beep(1000, 120)
        if sim is not None:
            sim.act("clench", True, delay_s=0.35)
        _collect(reader, rows, "clench", 2.0, event=i)
        speech.beep(500, 120)
        if sim is not None:
            sim.act("clench", False, delay_s=0.3)
        _collect(reader, rows, "clench", CLENCH_TAIL_S, event=i)
        _collect(reader, rows, "gap", 3.0 - CLENCH_TAIL_S, event=i)

    speech.say("Now count out loud, in a quiet normal voice, one, two, three, and keep going until the beep.")
    speech.beep(880, 150)
    if sim is not None:
        sim.act("talk", True, delay_s=0.4)
    _collect(reader, rows, "talk", talk_s, settle_s=0.5)
    if sim is not None:
        sim.act("talk", False)
    speech.beep(660, 150)

    speech.say(f"Last, {n} hard blinks, one at each beep.")
    for i in range(n):
        speech.beep(880, 100)
        if sim is not None:
            sim.act("blink", delay_s=0.3)
        _collect(reader, rows, "blink", 2.0, event=i)
    speech.say("Done.")

    fit = fit_clench(rows)
    fit.update({"quick": quick, "sim": sim is not None, "sample_rate": round(rows[-1]["fs"], 2),
                "good_channels_at_seat_check": n_good,
                "ticks": {k: sum(1 for r in rows if r["label"] == k) for k in ("rest", "clench", "gap", "talk", "blink")}})
    save_clench(fit, out_path)
    report(fit, out_path)
    say_verdict(fit)
    return fit


def report(fit: dict, path: Path | None = None) -> None:
    if path is not None:
        print(f"saved {path}")
    print(f"  resting 45 to 100 Hz power, median pin: {fit['base_hf_median']:.2f} uV^2")
    print("  each clench, highest level held 0.4 s (times resting): "
          + ", ".join(f"{v:.1f}" for v in fit["clench_levels"]))
    o = fit["others"]
    print(f"  everything else, highest level held 0.4 s: rest {o['rest']:.1f}, between clenches {o['gap']:.1f}, "
          f"talking {o['talk']:.1f}, blinks {o['blink']:.1f}")
    print(f"  stop at {fit['threshold']:.1f} x resting, clear below {fit['release']:.1f}; "
          f"high band must rise {fit['tilt_min']:.1f} times more than the low band")
    print(f"  margin {fit['margin']:.2f} (weakest clench over strongest other); "
          f"clenches caught {fit['caught']} of {fit['n_clenches']}"
          + (f"; would fire on: {', '.join(fit['would_fire'])}" if fit["would_fire"] else ""))
    print(f"  result: {fit['verdict'].upper()}")


def say_verdict(fit: dict) -> None:
    m = fit["margin"]
    if fit["verdict"] == "good":
        speech.say(f"Clench calibration is good. Your clench is {m:.0f} times stronger than anything else.")
    elif fit["verdict"] == "usable":
        speech.say("Clench calibration is usable, but close. Clench harder when you want to stop.")
    else:
        worst = max(fit["others"], key=fit["others"].get)
        what = {"talk": "talking", "blink": "blinking", "gap": "the rests", "rest": "rest"}[worst]
        speech.say(f"Clench calibration is not good. {what.capitalize()} came too close to a clench. "
                   "Clench harder, check the puck, and run it again.")


# ---------------------------------------------------------------- signal check

def signal_check(reader: Reader, seconds: float = 20.0) -> dict:
    """Once a second: alpha, 45 to 100 Hz power, 60 Hz power and good pins. Medians at the end."""
    print(f"signal check for {seconds:.0f} s: alpha (8 to 13 Hz), high (45 to 100 Hz, no 55 to 65), "
          "mains (60 Hz), all uV^2, median over pins")
    rows = {"alpha": [], "high": [], "mains": [], "good": []}
    t0 = time.monotonic()
    nxt = t0
    last_print = 0.0
    try:
        while time.monotonic() - t0 < seconds:
            nxt = _wait(nxt)
            f = reader.tick()
            if f is None:
                continue
            now = time.monotonic()
            if now - last_print < 1.0:
                continue
            last_print = now
            if f["stale"]:
                print(f"{now - t0:6.1f}s  stream stopped", flush=True)
                continue
            p = pin_powers(reader.data[-int(round(f["fs"])):], f["fs"])
            use = p["usable"]
            high = float(np.median(p["hf"][use])) if use.any() else 0.0
            mains = float(np.median(p["mains"][use])) if use.any() else 0.0
            for k, v in (("alpha", f["alpha"]), ("high", high), ("mains", mains), ("good", f["n_good"])):
                rows[k].append(v)
            print(f"{now - t0:6.1f}s  alpha {f['alpha']:7.1f}  high {high:7.2f}  mains {mains:7.2f}  "
                  f"good {f['n_good']:2d}/{reader.n_ch}  fs {f['fs']:.1f}", flush=True)
    except KeyboardInterrupt:
        print("stopped by keyboard")
    if not rows["good"]:
        speech.say("No signal.")
        return {}
    med = {k: float(np.median(v)) for k, v in rows.items()}
    print(f"medians: alpha {med['alpha']:.1f}, high {med['high']:.2f}, mains {med['mains']:.2f}, "
          f"good pins {med['good']:.0f} of {reader.n_ch}")
    speech.say(f"Alpha {med['alpha']:.0f}, high band {med['high']:.1f}, mains {med['mains']:.1f}, "
               f"{med['good']:.0f} of {reader.n_ch} pins good.")
    return med
