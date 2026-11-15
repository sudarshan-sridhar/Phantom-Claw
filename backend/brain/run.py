"""Run mode: watch alpha for the eyes-closed STOP and send a focus value once a second."""
import json
import time
from pathlib import Path

from . import speech
from .calibrate import CAL_PATH
from .link import ServerLink
from .reader import Reader
from .rules import MIN_GOOD_CHANNELS, FocusMeter, StopRule

TICK_S = 0.25
NO_SIGNAL_REPEAT_S = 2.0


def load_calibration(path: Path = CAL_PATH) -> dict:
    if not path.exists():
        raise SystemExit(f"no calibration at {path}; run: python -m backend.brain calibrate")
    return json.loads(path.read_text(encoding="utf-8"))


def run(reader: Reader, link: ServerLink, cal: dict, seconds: float | None = None, log_path: Path | None = None) -> dict:
    rule = StopRule(cal["stop_alpha"], cal["release_alpha"])
    o, a = cal["open"], cal["arith"]
    focus = FocusMeter(o["ratio_mean"], a["ratio_mean"], o["ratio_sd"])
    print(f"run: stop above alpha {rule.stop_at:.1f} held 1.5 s, release below {rule.release_at:.1f}")
    log = open(log_path, "w", encoding="utf-8") if log_path else None

    t0 = time.monotonic()
    nxt = t0
    last_focus_sent = last_print = last_no_signal = 0.0
    had_signal = None
    summary = {"stops": [], "clears": [], "no_signal_ticks": 0, "ticks": 0}
    try:
        while seconds is None or time.monotonic() - t0 < seconds:
            delay = nxt - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            nxt += TICK_S
            f = reader.tick()
            if f is None:
                continue
            now = time.monotonic()
            el = now - t0
            summary["ticks"] += 1
            ok = not f["stale"] and f["n_good"] >= MIN_GOOD_CHANNELS

            event = rule.update(f["t"], f["alpha"], ok)
            value = focus.update(f["t"], f["beta"], f["theta"], f["rms"], ok)

            if not ok:
                summary["no_signal_ticks"] += 1
                if had_signal is not False or now - last_no_signal >= NO_SIGNAL_REPEAT_S:
                    why = "stream stopped" if f["stale"] else f"{f['n_good']} good channels"
                    if had_signal is not False:
                        print(f"{el:7.2f}s  NO SIGNAL ({why})", flush=True)
                    link.event("no_signal", n_good=f["n_good"], reason=why)
                    last_no_signal = now
                had_signal = False
            else:
                if had_signal is False:
                    print(f"{el:7.2f}s  signal back, {f['n_good']} good channels", flush=True)
                had_signal = True

            if event == "stop":
                summary["stops"].append(round(el, 2))
                print(f"{el:7.2f}s  STOP  alpha {f['alpha']:.1f} > {rule.stop_at:.1f} for 1.5 s", flush=True)
                link.post("/api/stop", None, loud=True)
                link.event("stop", loud=True, alpha=round(f["alpha"], 2), threshold=round(rule.stop_at, 2), n_good=f["n_good"])
                speech.beep(220, 450, block=False)
            elif event == "clear":
                summary["clears"].append(round(el, 2))
                print(f"{el:7.2f}s  clear alpha {f['alpha']:.1f} < {rule.release_at:.1f}", flush=True)
                link.event("clear", loud=True, alpha=round(f["alpha"], 2))

            if ok and value is not None and now - last_focus_sent >= 1.0:
                last_focus_sent = now
                link.event("focus", value=round(value, 3), frozen=focus.frozen)

            if now - last_print >= 1.0:
                last_print = now
                fv = "--" if value is None else f"{value:.2f}"
                print(f"{el:7.2f}s  alpha {f['alpha']:7.1f}  beta {f['beta']:6.1f}  good {f['n_good']:2d}/{reader.n_ch}"
                      f"  fs {f['fs']:.1f}  focus {fv}{' (frozen)' if focus.frozen else ''}"
                      f"  {'STOPPED' if rule.active else ''}", flush=True)
            if log:
                log.write(json.dumps({"el": round(el, 3), "alpha": round(f["alpha"], 2), "n_good": f["n_good"],
                                      "ok": ok, "stopped": rule.active, "event": event,
                                      "focus": None if value is None else round(value, 3)}) + "\n")
    except KeyboardInterrupt:
        print("stopped by keyboard")
    finally:
        if log:
            log.close()
        link.flush()
    summary["server"] = link.results
    return summary
