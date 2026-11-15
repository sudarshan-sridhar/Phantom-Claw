"""ssvep-calibrate and ssvep-run: the chooser page plus the reader, with the stop (eyes
closed, jaw clench, or both) running the whole time and always taking priority over a choice."""
from __future__ import annotations

import json
import threading
import time
import webbrowser
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np

from . import speech
from .chooser import ChooserState, serve
from .clench import ClenchRule, detector_from
from .dsp import contact, measured_rate
from .link import ServerLink
from .reader import Reader
from .rules import MIN_GOOD_CHANNELS, StopRule
from .ssvep import (SSVEP_CAL_PATH, Decider, SsvepConfig, Thresholds, alpha_excluding, classify,
                    fit_thresholds, leave_one_out, pick_rates, rates_from_frames, window_scores)

SETTLE_S = 1.0       # after a prompt, time for the eyes to move before windows count
RECORD_S = 8.0       # recorded per calibration step
FILTER_SETTLE_S = 1.0
STALE_S = 1.0
WARMUP_S = 5.0     # reading before the first calibration step


class Front:
    """One tick of numbers from the configured channels: contact, CCA scores, notched alpha."""

    def __init__(self, reader: Reader, cfg: SsvepConfig):
        self.reader = reader
        self.cfg = cfg
        self.cols = [c for c in cfg.inputs() if c < reader.n_ch]
        if not self.cols:
            raise SystemExit(f"none of the configured inputs exist on this stream ({reader.n_ch} channels)")
        self.last_fs = reader.nominal
        # calibration scores both ways and keeps the better; run mode scores one way
        self.dims_options = [cfg.cca_dims]

    def tick(self, rates_hz: list[float]) -> dict | None:
        r = self.reader
        r.pull()
        need = self.cfg.window_s + FILTER_SETTLE_S + 0.25
        if len(r.ts) < 2 or r.ts[-1] - r.ts[0] < need:
            return None
        fs = measured_rate(r.ts, r.nominal)
        self.last_fs = fs
        raw = r.data[:, self.cols]
        good, rms = contact(raw[-int(fs):])
        stale = r.last_arrival is None or time.monotonic() - r.last_arrival > STALE_S
        use = raw[:, good]
        n_good = int(good.sum())
        by_dims = {}
        if n_good:
            for d in self.dims_options:
                by_dims[d] = window_scores(use, fs, rates_hz, self.cfg.window_s, self.cfg.harmonics,
                                           FILTER_SETTLE_S, d)
        scores = by_dims.get(self.cfg.cca_dims)
        alpha = alpha_excluding(use, fs, rates_hz) if n_good else None
        return {"fs": fs, "n_good": n_good, "stale": stale, "scores": scores, "by_dims": by_dims,
                "alpha": 0.0 if alpha is None else alpha, "t": time.monotonic()}

    def min_good_for_stop(self) -> int:
        return min(MIN_GOOD_CHANNELS, len(self.cols))


# ---------------------------------------------------------------- page and display

def start_page(state: ChooserState, cfg: SsvepConfig, open_browser: bool) -> str:
    serve(state, cfg.port)
    url = f"http://127.0.0.1:{cfg.port}/"
    print(f"chooser page: {url}")
    if open_browser:
        webbrowser.open(url)
    return url


def wait_for_display(state: ChooserState, auto: bool, timeout: float = 600.0) -> float:
    if auto:
        state.set(refresh=60.0)
        return 60.0
    print("waiting for the page to measure the screen...")
    if not state.display_event.wait(timeout):
        raise SystemExit("the chooser page never reported the screen; open it in a browser")
    return float(state.refresh)


def wait_for_flicker(state: ChooserState, auto: bool) -> None:
    if auto:
        state.set(flicker=True)
        return
    while not state.snapshot()["flicker"]:
        time.sleep(0.1)


def _wait_tick(nxt: float, step: float) -> float:
    delay = nxt - time.monotonic()
    if delay > 0:
        time.sleep(delay)
    return nxt + step


# ---------------------------------------------------------------- calibration

def _record_step(front: Front, state: ChooserState, rates: list[float], label, highlight, prompt: str,
                 sim_gaze, sim_closed: bool, auto: bool) -> tuple[list[np.ndarray], list[float]]:
    """One calibration step. Returns (score windows, alpha values). Redone from the start
    if the flicker is stopped part way through."""
    cfg = front.cfg
    while True:
        wait_for_flicker(state, auto)
        state.set(highlight=highlight, prompt=prompt, status=f"calibrating: {label}")
        if state.sim is not None:
            state.set_gaze(sim_gaze, sim_closed)
        speech.say(prompt)
        speech.beep(880, 150)
        windows, alphas = [], []
        start = time.monotonic()
        nxt = start
        broken = False
        while time.monotonic() - start < SETTLE_S + RECORD_S:
            nxt = _wait_tick(nxt, cfg.step_s)
            if not state.snapshot()["flicker"]:
                broken = True
                break
            f = front.tick(rates)
            if f is None or f["stale"] or f["n_good"] == 0:
                continue
            el = time.monotonic() - start
            if el >= SETTLE_S + cfg.window_s - 1e-6 and f["by_dims"]:
                windows.append(f["by_dims"])
            if el >= 2.5:
                alphas.append(f["alpha"])
        speech.beep(660, 150, times=2 if label == "eyes closed" else 1)
        if not broken:
            print(f"  {label}: {len(windows)} windows, notched alpha median "
                  f"{np.median(alphas) if alphas else float('nan'):.1f} uV^2", flush=True)
            return windows, alphas
        print(f"  {label}: flicker stopped, will redo this step when Start is pressed", flush=True)
        state.set(status="paused; press Start to redo this step")


def ssvep_calibrate(reader: Reader, cfg: SsvepConfig, out_path: Path = SSVEP_CAL_PATH, sim=None,
                    open_browser: bool = True, auto: bool = False, hold: int = 6) -> dict:
    state = ChooserState(cfg.items, "calibrate", sim)
    front = Front(reader, cfg)
    front.dims_options = list(dict.fromkeys([cfg.cca_dims, None]))
    start_page(state, cfg, open_browser)
    refresh = wait_for_display(state, auto)
    if cfg.rates_hz:
        frames = [max(2, round(refresh / f)) for f in cfg.rates_hz]
        rates = rates_from_frames(refresh, frames)
    else:
        rates = pick_rates(refresh, len(cfg.items))
    rates_hz = [r.hz for r in rates]
    state.set(frames=[r.frames for r in rates], rates_hz=rates_hz,
              status="Press Start. The steps are spoken; follow the highlighted tile.")
    print(f"screen {refresh:.2f} Hz; rates " + ", ".join(f"{it} {r.hz:.2f} Hz ({r.frames} frames)"
                                                       for it, r in zip(cfg.items, rates)))
    print(f"channels: {cfg.channels} -> {len(front.cols)} inputs")
    speech.say("Chooser calibration. Press Start on the screen when you are ready.", block=False)
    wait_for_flicker(state, auto)
    # let the stream and the filters settle before the first step counts
    warm = time.monotonic()
    while time.monotonic() - warm < WARMUP_S:
        time.sleep(cfg.step_s)
        front.tick(rates_hz)

    rest_w, open_alpha = _record_step(front, state, rates_hz, "rest", "centre",
                                      "Look at the cross in the middle. Do not look at any tile.", None, False, auto)
    target: dict[int, np.ndarray] = {}
    tile_alpha: dict[int, list[float]] = {}
    for i, item in enumerate(cfg.items):
        w, a = _record_step(front, state, rates_hz, item, i, f"Look at the {item} tile and keep looking.", i, False, auto)
        target[i] = w
        tile_alpha[i] = a
    _, closed_alpha = _record_step(front, state, rates_hz, "eyes closed", None,
                                   "Close your eyes and relax. Keep them closed until you hear two beeps.", None, True, auto)
    state.set(highlight=None, prompt="Calibration done. You can close this page.")
    if state.sim is not None:
        state.set_gaze(None, False)
    speech.say("Calibration done. Thank you.")

    if len(rest_w) < 8 or any(len(v) < 8 for v in target.values()) or len(closed_alpha) < 4:
        raise SystemExit("too few clean windows; check contact and run again")

    # score the windows both ways (a few principal components of the pins, and all pins) and
    # keep whichever reads this head better; a tie goes to the few components
    best = None
    for d in front.dims_options:
        rest_d = np.array([w[d] for w in rest_w])
        target_d = {i: np.array([w[d] for w in ws]) for i, ws in target.items()}
        th_d = fit_thresholds(target_d, rest_d, hold)
        loo_d = leave_one_out(target_d, rest_d, hold)
        mean_acc = float(np.mean(list(loo_d["per_rate"].values())))
        merit = mean_acc - max(0.0, 0.85 - loo_d["rest_undecided"])
        label = "all pins" if d is None else f"{d} components"
        print(f"  {label}: mean accuracy {mean_acc * 100:.1f}%, rest undecided {loo_d['rest_undecided'] * 100:.1f}%")
        if best is None or merit > best[0] + 0.02:
            best = (merit, d, th_d, loo_d)
    _, cfg.cca_dims, th, loo = best

    all_open = np.array(open_alpha + [a for v in tile_alpha.values() for a in v])
    closed = np.array(closed_alpha)
    stop_alpha = float((all_open.mean() + closed.mean()) / 2)
    release_alpha = float(all_open.mean() + all_open.std())
    warnings = []
    if closed.mean() <= all_open.mean() + 2 * all_open.std():
        warnings.append("eyes closed did not clearly raise alpha; the stop will be unreliable")
    for i, v in tile_alpha.items():
        if v and np.median(v) > 0.7 * stop_alpha:
            warnings.append(f"looking at {cfg.items[i]} raises alpha close to the stop level")
    weak = [cfg.items[i] for i, acc in loo["per_rate"].items() if acc < 0.6]
    if weak:
        warnings.append("weak responses for: " + ", ".join(weak))

    cal = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sim": sim is not None,
        "refresh": refresh,
        "items": cfg.items,
        "frames": [r.frames for r in rates],
        "rates_hz": [round(r.hz, 4) for r in rates],
        "channels": cfg.channels,
        "window_s": cfg.window_s,
        "step_s": cfg.step_s,
        "harmonics": cfg.harmonics,
        "cca_dims": cfg.cca_dims,
        "sample_rate": round(front.last_fs, 2),
        "thresholds": asdict(th),
        "accuracy": {cfg.items[i]: round(a, 3) for i, a in loo["per_rate"].items()},
        "rest_undecided": round(loo["rest_undecided"], 3),
        "confusion": loo["confusion"],
        "stop_alpha": stop_alpha,
        "release_alpha": release_alpha,
        "alpha_open_mean": float(all_open.mean()), "alpha_open_sd": float(all_open.std()),
        "alpha_closed_mean": float(closed.mean()), "alpha_closed_sd": float(closed.std()),
        "warnings": warnings,
    }
    save_calibration(cal, out_path)
    print(f"saved {out_path}")
    print(f"  threshold {th.threshold:.3f}, margin {th.margin:.3f}, hold {th.hold} windows")
    print("  leave-one-window-out accuracy (windows overlap, so this is on the kind side):")
    for item, acc in cal["accuracy"].items():
        print(f"    {item:12s} {acc * 100:5.1f}%")
    print(f"    rest stayed undecided {cal['rest_undecided'] * 100:.1f}% of windows")
    print(f"  stop above notched alpha {stop_alpha:.1f}, release below {release_alpha:.1f} "
          f"(open {all_open.mean():.1f}, closed {closed.mean():.1f})")
    for w in warnings:
        print("  WARNING:", w)
    return cal


def save_calibration(cal: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cal, indent=2), encoding="utf-8")


def load_ssvep_calibration(path: Path = SSVEP_CAL_PATH) -> dict:
    if not path.exists():
        raise SystemExit(f"no chooser calibration at {path}; run: python -m backend.brain ssvep-calibrate")
    cal = json.loads(path.read_text(encoding="utf-8"))
    for key in ("frames", "items", "thresholds", "stop_alpha", "release_alpha"):
        if key not in cal:
            raise SystemExit(f"{path} has no {key}; run ssvep-calibrate again")
    return cal


# ---------------------------------------------------------------- run

class Gate:
    """The order of things each tick, kept free of streams so the tests can drive it.

    The stop goes first and always wins: while it is on, or alpha is already above the stop
    level (so it may be about to fire), nothing is chosen, any choice in progress is dropped,
    and the decider stays blocked for its refractory time after the stop clears. The clench
    stop, when given, follows the same rules: on, or above its threshold now, blocks choosing.
    stop may be None when only the clench stop is in use."""

    def __init__(self, stop: StopRule | None, decider: Decider, clench: ClenchRule | None = None):
        self.stop = stop
        self.decider = decider
        self.clench = clench
        self.clench_event = None

    def step(self, t: float, alpha: float, ok_stop: bool, scores, choosing: bool, clench: dict | None = None):
        """Returns (eyes-closed stop event or None, chosen index or None, this window's guess,
        stop_near). The clench stop's event of this tick is left in self.clench_event.
        clench is the detector's measurement for this tick (score, tilt, ok)."""
        ev = self.stop.update(t, alpha, ok_stop) if self.stop is not None else None
        stop_near = self.stop is not None and (self.stop.active or (ok_stop and alpha > self.stop.stop_at))
        self.clench_event = None
        if self.clench is not None and clench is not None:
            self.clench_event = self.clench.update(t, clench["score"], clench["tilt"], clench["ok"])
        if self.clench is not None and self.clench.near(clench):
            stop_near = True
        if stop_near or not choosing or scores is None:
            self.decider.suppress(t)
            return ev, None, None, stop_near
        guess, _, _ = classify(scores, self.decider.th)
        return ev, self.decider.update(t, scores), guess, stop_near

class Requester:
    """Sends one 'bring me the X' at a time to the agent server, on its own thread, with a
    long timeout because the server runs the whole pipeline before it answers."""

    def __init__(self, base: str, dry: bool):
        self.base = base.rstrip("/")
        self.dry = dry
        self.busy = False
        self.sent: list[dict] = []

    def send(self, text: str) -> str:
        if self.dry:
            print(f"  [dry] would POST /api/request {json.dumps({'text': text})}", flush=True)
            self.sent.append({"text": text, "outcome": "dry"})
            return "dry"
        if self.busy:
            print("  [server] still busy with the last request; this one is not sent", flush=True)
            self.sent.append({"text": text, "outcome": "busy"})
            return "busy"
        self.busy = True
        threading.Thread(target=self._post, args=(text,), daemon=True).start()
        return "sent"

    def _post(self, text: str) -> None:
        try:
            r = httpx.post(self.base + "/api/request", json={"text": text}, timeout=httpx.Timeout(180.0, connect=1.0))
            outcome = str(r.status_code)
            extra = ""
            if r.status_code == 200:
                body = r.json()
                extra = f" {body.get('agent')}: {str(body.get('say'))[:120]}"
            print(f"  [server] POST /api/request -> {outcome}{extra}", flush=True)
        except httpx.HTTPError as e:
            outcome = f"not reachable ({type(e).__name__})"
            print(f"  [server] POST /api/request -> {outcome}", flush=True)
        self.sent.append({"text": text, "outcome": outcome})
        self.busy = False


def parse_script(text: str, items: list[str]) -> list[tuple[float, str, object]]:
    """'0:none,3:sharpie,12:closed,15:open' -> [(0, 'gaze', None), (3, 'gaze', 2), ...].
    Times are seconds after the flicker starts; a target is an item name, its number from 1,
    none, closed or open. Simulator muscle actions: clench and release, talk and quiet, blink
    and turn; they come out as (t, kind, value) with kind clench, talk, blink or turn."""
    out = []
    lower = [i.lower() for i in items]
    for part in text.split(","):
        if not part.strip():
            continue
        t, _, what = part.partition(":")
        what = what.strip().lower()
        if what == "none":
            out.append((float(t), "gaze", None))
        elif what in ("closed", "open"):
            out.append((float(t), "eyes", what == "closed"))
        elif what in ("clench", "release"):
            out.append((float(t), "clench", what == "clench"))
        elif what in ("talk", "quiet"):
            out.append((float(t), "talk", what == "talk"))
        elif what in ("blink", "turn"):
            out.append((float(t), what, None))
        elif what in lower:
            out.append((float(t), "gaze", lower.index(what)))
        elif what.isdigit() and 1 <= int(what) <= len(items):
            out.append((float(t), "gaze", int(what) - 1))
        else:
            raise SystemExit(f"sim script: do not know {what!r}")
    return sorted(out, key=lambda e: e[0])


def ssvep_run(reader: Reader, cfg: SsvepConfig, cal: dict, link: ServerLink, dry: bool = False, sim=None,
              seconds: float | None = None, open_browser: bool = True, auto: bool = False,
              script: str | None = None, log_path: Path | None = None, clench_cal: dict | None = None,
              stops: set | None = None) -> dict:
    stops = set(stops or {"alpha"})
    if len(cal["frames"]) != len(cfg.items):
        raise SystemExit(f"calibration has {len(cal['frames'])} tiles but there are {len(cfg.items)} items; "
                         "run ssvep-calibrate again with these items")
    if [i.lower() for i in cal.get("items", cfg.items)] != [i.lower() for i in cfg.items]:
        print("note: item names differ from calibration; tiles keep their calibrated rates in order")
    if cal.get("sim") and sim is None:
        print("WARNING: this calibration was made on the simulator; run ssvep-calibrate --board first")
    for key in ("window_s", "harmonics", "cca_dims"):
        if key in cal:   # score exactly the way calibration did, or the thresholds mean nothing
            setattr(cfg, key, cal[key])
    th = Thresholds(**cal["thresholds"])
    state = ChooserState(cfg.items, "run", sim)
    front = Front(reader, cfg)
    start_page(state, cfg, open_browser)
    refresh = wait_for_display(state, auto)
    frames = list(cal["frames"])
    warning = ""
    can_choose = True
    cal_refresh = float(cal.get("refresh", refresh))
    if abs(refresh - cal_refresh) > 1.0:
        # another screen: keep the calibrated rates if this screen can show them exactly
        cal_hz = [cal_refresh / k for k in frames]
        mapped = [max(2, round(refresh / f)) for f in cal_hz]
        if all(abs(refresh / k - f) < 0.1 for k, f in zip(mapped, cal_hz)):
            frames = mapped
            print(f"screen is {refresh:.1f} Hz, calibration was {cal_refresh:.1f} Hz; same rates fit, carrying on")
        else:
            can_choose = False
            warning = (f"This screen runs at {refresh:.0f} Hz but calibration was done at {cal_refresh:.0f} Hz. "
                       "Choosing is off; the stop still works. Calibrate again on this screen.")
            print("WARNING:", warning)
    rates = rates_from_frames(refresh, frames)
    rates_hz = [r.hz for r in rates]
    state.set(frames=frames, rates_hz=rates_hz, warning=warning, status="Press Start to begin choosing.")
    print(f"screen {refresh:.2f} Hz; rates " + ", ".join(f"{it} {r.hz:.2f} Hz" for it, r in zip(cfg.items, rates)))
    stop = StopRule(cal["stop_alpha"], cal["release_alpha"]) if "alpha" in stops else None
    crule = ClenchRule.from_calibration(clench_cal) if "clench" in stops else None
    det = detector_from(clench_cal) if crule else None
    decider = Decider(th, cfg.refractory_s)
    gate = Gate(stop, decider, crule)
    req = Requester(cfg.server, dry)
    print(f"run: choose when score >= {th.threshold:.3f} and {th.margin:.3f} ahead for {th.hold} windows; "
          + (f"eyes-closed stop above notched alpha {stop.stop_at:.1f}" if stop else "")
          + (f"{'; ' if stop else ''}clench stop at {crule.threshold:.1f} x resting" if crule else "")
          + ("  (dry: nothing is sent)" if dry else ""))

    events = parse_script(script, cfg.items) if script else []
    log = open(log_path, "w", encoding="utf-8") if log_path else None
    summary = {"choices": [], "stops": [], "clears": [], "ticks": 0, "suppressed_by_stop": 0}
    t0 = time.monotonic()
    flicker_t0 = None
    last_flicker = None
    nxt = t0
    last_print = 0.0
    try:
        if auto:
            wait_for_flicker(state, auto)
        while seconds is None or time.monotonic() - t0 < seconds:
            nxt = _wait_tick(nxt, cfg.step_s)
            snap = state.snapshot()
            if snap["flicker"] and flicker_t0 is None:
                flicker_t0 = time.monotonic()
            if events and flicker_t0 is not None and sim is not None:
                while events and time.monotonic() - flicker_t0 >= events[0][0]:
                    _, kind, val = events.pop(0)
                    if kind == "gaze":
                        state.set_gaze(val, None)
                    elif kind == "eyes":
                        state.set_gaze(state.snapshot()["gaze"], val)
                    else:
                        sim.act(kind, val)
            f = front.tick(rates_hz)
            if f is None:
                continue
            el = time.monotonic() - t0
            summary["ticks"] += 1
            ok_stop = not f["stale"] and f["n_good"] >= front.min_good_for_stop()
            choosing = can_choose and snap["flicker"] and not f["stale"]
            m = None
            if det is not None:
                m = det.measure(reader.data[:, front.cols], f["fs"])
                m["ok"] = m["ok"] and not f["stale"]
            ev, choice, guess, stop_near = gate.step(f["t"], f["alpha"], ok_stop, f["scores"], choosing, m)
            cev = gate.clench_event
            if snap["flicker"] != last_flicker:
                last_flicker = snap["flicker"]
                state.set(status="Choosing. Look at one tile for a few seconds." if last_flicker
                          else "Paused. Press Start to go on.")
            if ev == "stop":
                summary["stops"].append(round(el, 2))
                print(f"{el:7.2f}s  STOP  notched alpha {f['alpha']:.1f} > {stop.stop_at:.1f} for 1.5 s", flush=True)
                link.post("/api/stop", None, loud=True)
                link.event("stop", loud=True, trigger="alpha", alpha=round(f["alpha"], 2),
                           threshold=round(stop.stop_at, 2), n_good=f["n_good"])
                speech.beep(220, 450, block=False)
            elif ev == "clear":
                summary["clears"].append(round(el, 2))
                print(f"{el:7.2f}s  clear notched alpha {f['alpha']:.1f} < {stop.release_at:.1f}", flush=True)
                link.event("clear", loud=True, trigger="alpha", alpha=round(f["alpha"], 2))
            if cev == "stop":
                summary["stops"].append(round(el, 2))
                summary.setdefault("clench_stops", []).append(round(el, 2))
                print(f"{el:7.2f}s  STOP (jaw clench)  level {m['score']:.1f} x resting >= {crule.threshold:.1f}", flush=True)
                link.post("/api/stop", None, loud=True)
                link.event("stop", loud=True, trigger="clench", level=round(m["score"], 2),
                           threshold=round(crule.threshold, 2), n_good=f["n_good"])
                speech.beep(220, 450, block=False)
            elif cev == "clear":
                summary["clears"].append(round(el, 2))
                print(f"{el:7.2f}s  clear (jaw relaxed)  level {m['score']:.1f}", flush=True)
                link.event("clear", loud=True, trigger="clench", level=round(m["score"], 2))
            stopped = (stop is not None and stop.active) or (crule is not None and crule.active)

            if stop_near:
                summary["suppressed_by_stop"] += 1
            idx, prog = decider.progress()
            state.set(stopped=stopped, progress={"index": idx, "value": prog})

            if choice is not None:
                item = cfg.items[choice]
                text = cfg.text_for(item)
                print(f"{el:7.2f}s  CHOSEN {item}  ({rates_hz[choice]:.2f} Hz)  -> \"{text}\"", flush=True)
                outcome = req.send(text)
                speech.beep(990, 120, times=2, block=False)
                summary["choices"].append({"t": round(el, 2), "item": item, "outcome": outcome})
                state.set(chosen={"item": item, "at": time.time(), "sent": outcome == "sent"},
                          progress={"index": choice, "value": 1.0})

            if time.monotonic() - last_print >= 1.0:
                last_print = time.monotonic()
                sc = "--" if f["scores"] is None else " ".join(f"{v:.2f}" for v in th.corrected(f["scores"]))
                gtxt = "-" if guess is None else cfg.items[guess]
                ctxt = "" if m is None else f"  clench {m['score']:5.1f}"
                print(f"{el:7.2f}s  good {f['n_good']:2d}  alpha {f['alpha']:7.1f}{ctxt}  scores {sc}  guess {gtxt:10s}"
                      f"  {'STOPPED' if stopped else ''}{'' if snap['flicker'] else '  (no flicker)'}", flush=True)
            if log:
                log.write(json.dumps({"el": round(el, 3), "alpha": round(f["alpha"], 2), "n_good": f["n_good"],
                                      "scores": None if f["scores"] is None else [round(float(v), 4) for v in f["scores"]],
                                      "clench": None if m is None else round(m["score"], 2),
                                      "guess": guess, "stopped": stopped, "event": ev or cev,
                                      "choice": None if choice is None else cfg.items[choice]}) + "\n")
    except KeyboardInterrupt:
        print("stopped by keyboard")
    finally:
        if log:
            log.close()
        link.flush()
        end = time.time() + 3
        while req.busy and time.time() < end:
            time.sleep(0.05)
    summary["server"] = link.results
    summary["requests"] = req.sent
    return summary
