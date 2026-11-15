"""Run mode: the STOP (eyes-closed alpha, jaw clench, or both) and a focus value once a second.

Either stop trigger sends the same thing: POST /api/stop and a brain "stop" event, with a
"trigger" field saying which one fired."""
import json
import time
from pathlib import Path

from . import speech
from .calibrate import CAL_PATH
from .clench import ALPHA_WARNING, ClenchRule, detector_from
from .link import ServerLink
from .reader import Reader
from .rules import MIN_GOOD_CHANNELS, FocusMeter, StopRule

TICK_S = 0.25
NO_SIGNAL_REPEAT_S = 2.0


def load_calibration(path: Path = CAL_PATH, required: bool = True) -> dict | None:
    if not path.exists():
        if required:
            raise SystemExit(f"no calibration at {path}; run: python -m backend.brain calibrate")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def alpha_reliable(cal: dict | None) -> bool:
    if cal is None:
        return False
    warnings = [cal.get("warning", "")] + list(cal.get("warnings", []))
    return not any(ALPHA_WARNING in w for w in warnings if w)


def run(reader: Reader, link: ServerLink, cal: dict | None, seconds: float | None = None,
        log_path: Path | None = None, clench_cal: dict | None = None, stops: set | None = None,
        sim=None, script: list | None = None) -> dict:
    stops = set(stops or {"alpha"})
    use_alpha, use_clench = "alpha" in stops, "clench" in stops
    rule = StopRule(cal["stop_alpha"], cal["release_alpha"]) if use_alpha else None
    focus = None
    if cal is not None and "open" in cal and "arith" in cal:
        o, a = cal["open"], cal["arith"]
        focus = FocusMeter(o["ratio_mean"], a["ratio_mean"], o["ratio_sd"])
    crule = ClenchRule.from_calibration(clench_cal) if use_clench else None
    det = detector_from(clench_cal) if use_clench else None
    if rule:
        print(f"run: eyes-closed stop above alpha {rule.stop_at:.1f} held 1.5 s, release below {rule.release_at:.1f}")
    if crule:
        print(f"run: clench stop at {crule.threshold:.1f} x resting held {crule.hold_s:.1f} s, "
              f"clear below {crule.release_at:.1f} for {crule.release_s:.1f} s")
    log = open(log_path, "w", encoding="utf-8") if log_path else None
    events = list(script or [])

    t0 = time.monotonic()
    nxt = t0
    last_focus_sent = last_print = last_no_signal = 0.0
    had_signal = None
    summary = {"stops": [], "clears": [], "no_signal_ticks": 0, "ticks": 0, "triggers": list(sorted(stops))}
    try:
        while seconds is None or time.monotonic() - t0 < seconds:
            delay = nxt - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            nxt += TICK_S
            while events and sim is not None and time.monotonic() - t0 >= events[0][0]:
                _, kind, val = events.pop(0)
                if kind != "gaze":
                    sim.act(kind, val)
            f = reader.tick()
            if f is None:
                continue
            now = time.monotonic()
            el = now - t0
            summary["ticks"] += 1
            ok = not f["stale"] and f["n_good"] >= MIN_GOOD_CHANNELS

            m = None
            fired = []
            if det is not None:
                m = det.measure(reader.data, f["fs"])
                m["ok"] = m["ok"] and not f["stale"]
                cev = crule.update(f["t"], m["score"], m["tilt"], m["ok"])
                if cev:
                    fired.append(("clench", cev))
            if rule is not None:
                aev = rule.update(f["t"], f["alpha"], ok)
                if aev:
                    fired.append(("alpha", aev))
            clenching = crule is not None and crule.near(m)
            value = None
            if focus is not None:
                value = focus.update(f["t"], f["beta"], f["theta"], f["rms"], ok and not clenching)

            # a hard clench can push pins past the alpha contact limit; that is not lost signal
            signal_ok = ok or (clenching and m is not None and m["ok"])
            if not signal_ok:
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

            for trigger, event in fired:
                if event == "stop":
                    summary["stops"].append({"t": round(el, 2), "trigger": trigger})
                    if trigger == "alpha":
                        print(f"{el:7.2f}s  STOP (eyes closed)  alpha {f['alpha']:.1f} > {rule.stop_at:.1f} for 1.5 s", flush=True)
                        detail = {"alpha": round(f["alpha"], 2), "threshold": round(rule.stop_at, 2)}
                    else:
                        print(f"{el:7.2f}s  STOP (jaw clench)  level {m['score']:.1f} x resting >= {crule.threshold:.1f} "
                              f"for {crule.hold_s:.1f} s", flush=True)
                        detail = {"level": round(m["score"], 2), "threshold": round(crule.threshold, 2)}
                    link.post("/api/stop", None, loud=True)
                    link.event("stop", loud=True, trigger=trigger, n_good=f["n_good"], **detail)
                    speech.beep(220, 450, block=False)
                else:
                    summary["clears"].append({"t": round(el, 2), "trigger": trigger})
                    if trigger == "alpha":
                        print(f"{el:7.2f}s  clear (eyes open)  alpha {f['alpha']:.1f} < {rule.release_at:.1f}", flush=True)
                        link.event("clear", loud=True, trigger=trigger, alpha=round(f["alpha"], 2))
                    else:
                        print(f"{el:7.2f}s  clear (jaw relaxed)  level {m['score']:.1f}", flush=True)
                        link.event("clear", loud=True, trigger=trigger, level=round(m["score"], 2))

            if ok and value is not None and now - last_focus_sent >= 1.0:
                last_focus_sent = now
                link.event("focus", value=round(value, 3), frozen=focus.frozen)

            stopped = (rule is not None and rule.active) or (crule is not None and crule.active)
            if now - last_print >= 1.0:
                last_print = now
                fv = "--" if value is None else f"{value:.2f}"
                ctxt = "" if m is None else f"  clench {m['score']:6.1f}"
                print(f"{el:7.2f}s  alpha {f['alpha']:7.1f}  beta {f['beta']:6.1f}  good {f['n_good']:2d}/{reader.n_ch}"
                      f"{ctxt}  fs {f['fs']:.1f}  focus {fv}{' (frozen)' if focus and focus.frozen else ''}"
                      f"  {'STOPPED' if stopped else ''}", flush=True)
            if log:
                log.write(json.dumps({"el": round(el, 3), "alpha": round(f["alpha"], 2), "n_good": f["n_good"],
                                      "clench": None if m is None else round(m["score"], 2),
                                      "ok": ok, "stopped": stopped, "events": fired,
                                      "focus": None if value is None else round(value, 3)}) + "\n")
    except KeyboardInterrupt:
        print("stopped by keyboard")
    finally:
        if log:
            log.close()
        link.flush()
    summary["server"] = link.results
    return summary
