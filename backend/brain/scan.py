"""scan-run: choose an item with the jaw instead of the eyes.

The chooser page lights its tiles one at a time, in turn, with no flicker. A short clench
while the wanted tile is lit chooses it and sends the same request the flicker chooser sends.
A long clench is the STOP, always wins, and cancels a choice that was about to be made.

This reads a muscle signal (the jaw muscles, picked up by the headset at the back of the
head), not a brain signal. It needs only setup/brain_clench.json.

Timing, measured on the detector's own clock:
  - a burst starts at the first tick at or above the clench threshold (with the tilt guard)
    and ends once the level has been under the threshold for END_S;
  - a burst shorter than the calibration's hold time (0.4 s) is ignored: calibration showed
    nothing but a clench stays up that long;
  - a burst still going LONG_S after it started is the STOP;
  - otherwise, when it ends, it chooses the tile that was lit when it started. If it started
    within GRACE_S after the highlight moved on, it counts for the tile before, since the
    person was reacting to that one.
The half second measuring window (tapered at its ends) sees a burst start about 0.25 s
late and end about 0.3 s late, so the burst's length on the detector is close to the
clench's own length. ONSET_LAG_S takes the late start off before picking the tile. These
lags were measured on the simulator only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import speech
from .chooser import ChooserState
from .clench import ClenchDetector
from .dsp import measured_rate
from .link import ServerLink
from .reader import STALE_S, Reader
from .ssvep import SsvepConfig
from .ssvep_session import Requester, parse_script, start_page, wait_for_flicker

SCAN_TICK_S = 0.1
DWELL_S = 1.5        # each tile stays lit this long
GRACE_S = 0.3        # a clench this soon after the highlight moved still counts for the tile before
LONG_S = 1.1         # burst this long on the detector is a STOP (a clench held about 1.2 s)
END_S = 0.2          # under threshold this long ends a burst
ONSET_LAG_S = 0.25   # the detector sees a burst about this long after it starts
PAUSE_S = 4.0        # scanning rests this long after a choice or after a stop clears


class Scanner:
    """The scan and the clench rules, free of streams and pages so the tests can drive it
    with a fake clock. update() returns a list of events: ("choose", index), ("stop", None),
    ("clear", None), ("ignored", duration) for a burst too short to count."""

    def __init__(self, n_items: int, dwell_s: float = DWELL_S, grace_s: float = GRACE_S, long_s: float = LONG_S,
                 min_s: float = 0.4, end_s: float = END_S, pause_s: float = PAUSE_S, release_s: float = 1.0,
                 onset_lag_s: float = ONSET_LAG_S):
        self.n = n_items
        self.dwell_s, self.grace_s, self.long_s, self.min_s = dwell_s, grace_s, long_s, min_s
        self.end_s, self.pause_s, self.release_s, self.onset_lag_s = end_s, pause_s, release_s, onset_lag_s
        self.scan_t0: float | None = None      # when tile 0 was first lit in this round
        self.burst_t: float | None = None      # first hot tick of the burst in progress
        self.burst_tile: int | None = None
        self.last_hot: float | None = None
        self.stopped = False
        self.cool_since: float | None = None

    # ------------------------------------------------------------ the scan

    def start(self, t: float) -> None:
        self.scan_t0 = t

    def halt(self) -> None:
        """The page was paused (Esc): no scan, and no choice from a burst in progress."""
        self.scan_t0 = None
        self.burst_tile = None

    def running(self) -> bool:
        return self.scan_t0 is not None

    def lit(self, t: float) -> int | None:
        if self.scan_t0 is None or self.stopped or t < self.scan_t0:
            return None
        return int((t - self.scan_t0) // self.dwell_s) % self.n

    def tile_for(self, onset: float) -> int | None:
        """The tile a clench that started at onset was meant for."""
        if self.scan_t0 is None or onset < self.scan_t0 - self.grace_s:
            return None
        k = int((onset - self.scan_t0) // self.dwell_s)
        if k < 0:
            return None   # started in the pause, just before scanning resumed
        since = self.scan_t0 + k * self.dwell_s
        if onset - since < self.grace_s and k > 0:
            k -= 1
        return k % self.n

    # ------------------------------------------------------------ the clench

    def update(self, t: float, hot: bool, cool: bool, ok: bool) -> list:
        """hot: level at or above the clench threshold with the tilt guard passing.
        cool: level under the release level. ok: enough pins and a live stream."""
        out = []
        if not ok:
            self.burst_t = self.burst_tile = self.last_hot = None
            self.cool_since = None
            return out
        if self.stopped:
            if cool:
                if self.cool_since is None:
                    self.cool_since = t
                if t - self.cool_since >= self.release_s - 1e-9:
                    self.stopped = False
                    self.cool_since = None
                    if self.scan_t0 is not None:
                        self.scan_t0 = t + self.pause_s
                    out.append(("clear", None))
            else:
                self.cool_since = None
            return out
        if hot:
            if self.burst_t is None:
                self.burst_t = t
                self.burst_tile = self.tile_for(t - self.onset_lag_s)
            self.last_hot = t
            if t - self.burst_t >= self.long_s - 1e-9:
                # held: the STOP, and the choice it was heading for is dropped
                self.stopped = True
                self.burst_t = self.burst_tile = self.last_hot = None
                self.cool_since = None
                out.append(("stop", None))
            return out
        if self.burst_t is not None and t - self.last_hot >= self.end_s - 1e-9:
            dur = self.last_hot - self.burst_t + SCAN_TICK_S
            tile = self.burst_tile
            self.burst_t = self.burst_tile = self.last_hot = None
            if dur < self.min_s - 1e-9:
                out.append(("ignored", round(dur, 2)))
            elif tile is not None:
                out.append(("choose", tile))
                if self.scan_t0 is not None:
                    self.scan_t0 = t + self.pause_s   # rest, then start again from the first tile
        return out


def scan_run(reader: Reader, cfg: SsvepConfig, clench_cal: dict, link: ServerLink, dry: bool = False, sim=None,
             seconds: float | None = None, open_browser: bool = True, auto: bool = False,
             script: str | None = None, log_path: Path | None = None, dwell_s: float = DWELL_S,
             pause_s: float = PAUSE_S, long_s: float = LONG_S) -> dict:
    thr = float(clench_cal["threshold"])
    rel = min(float(clench_cal["release"]), thr)
    tilt_min = float(clench_cal.get("tilt_min", 2.0))
    det = ClenchDetector(clench_cal.get("base_hf"), clench_cal.get("base_lf"), tick_s=SCAN_TICK_S,
                         seed_ticks=100)
    sc = Scanner(len(cfg.items), dwell_s=dwell_s, pause_s=pause_s, long_s=long_s, min_s=float(clench_cal.get("hold_s", 0.4)),
                 release_s=float(clench_cal.get("release_s", 1.0)))
    cols = [c for c in cfg.inputs() if c < reader.n_ch] or list(range(reader.n_ch))
    if clench_cal.get("sim") and sim is None:
        print("WARNING: this clench calibration was made on the simulator; run clench-calibrate --board first")
    state = ChooserState(cfg.items, "scan", sim)
    state.set(dwell_s=dwell_s, status="Press Start. Tiles light up one at a time.")
    start_page(state, cfg, open_browser)
    req = Requester(cfg.server, dry)
    print(f"scan: {len(cfg.items)} tiles, {dwell_s:.1f} s each; a short clench chooses the lit tile, "
          f"held about {sc.long_s + 0.1:.1f} s it is the STOP; clench at {thr:.1f} x resting"
          + ("  (dry: nothing is sent)" if dry else ""))
    speech.say("Clench to choose. Press Start on the screen when you are ready.", block=False)

    events = parse_script(script, cfg.items) if script else []
    log = open(log_path, "w", encoding="utf-8") if log_path else None
    summary = {"choices": [], "stops": [], "clears": [], "ignored": [], "ticks": 0}
    t0 = time.monotonic()
    run_t0 = None
    was_running = False
    last_print = 0.0
    nxt = t0
    try:
        if auto:
            wait_for_flicker(state, auto)
        while seconds is None or time.monotonic() - t0 < seconds:
            delay = nxt - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            nxt += SCAN_TICK_S
            now = time.monotonic()
            snap = state.snapshot()
            running = bool(snap["flicker"])   # the page's Start/Stop; nothing flickers in this mode
            if running and not was_running:
                sc.start(now)
                run_t0 = run_t0 or now
                state.set(status="Scanning. Clench briefly when your item lights up. Hold the clench to stop.")
            elif was_running and not running:
                sc.halt()
                state.set(status="Paused. Press Start to go on.")
            was_running = running
            while events and run_t0 is not None and sim is not None and now - run_t0 >= events[0][0]:
                _, kind, val = events.pop(0)
                if kind not in ("gaze",):
                    sim.act(kind, val)
            reader.pull()
            if len(reader.ts) < 2 or reader.ts[-1] - reader.ts[0] < 1.0:
                continue
            fs = measured_rate(reader.ts, reader.nominal)
            stale = reader.last_arrival is None or now - reader.last_arrival > STALE_S
            m = det.measure(reader.data[:, cols], fs)
            ok = m["ok"] and not stale
            hot = m["score"] >= thr and m["tilt"] >= tilt_min
            cool = m["score"] < rel
            summary["ticks"] += 1
            el = now - t0
            for kind, val in sc.update(now, hot, cool, ok):
                if kind == "stop":
                    summary["stops"].append(round(el, 2))
                    print(f"{el:7.2f}s  STOP (long clench)  level {m['score']:.1f} x resting", flush=True)
                    link.post("/api/stop", None, loud=True)
                    link.event("stop", loud=True, trigger="clench", level=round(m["score"], 2),
                               threshold=round(thr, 2), n_good=m["n_used"])
                    speech.beep(220, 450, block=False)
                elif kind == "clear":
                    summary["clears"].append(round(el, 2))
                    print(f"{el:7.2f}s  clear (jaw relaxed); scanning again in {pause_s:.0f} s", flush=True)
                    link.event("clear", loud=True, trigger="clench", level=round(m["score"], 2))
                elif kind == "ignored":
                    summary["ignored"].append(round(el, 2))
                    print(f"{el:7.2f}s  short burst ({val:.2f} s), not a clench; ignored", flush=True)
                elif kind == "choose":
                    item = cfg.items[val]
                    text = cfg.text_for(item)
                    print(f"{el:7.2f}s  CHOSEN {item}  -> \"{text}\"", flush=True)
                    outcome = req.send(text)
                    speech.beep(990, 120, times=2, block=False)
                    summary["choices"].append({"t": round(el, 2), "item": item, "outcome": outcome})
                    state.set(chosen={"item": item, "at": time.time(), "sent": outcome == "sent"})
            lit = sc.lit(now)
            state.set(highlight=lit, lit=lit, stopped=sc.stopped,
                      progress={"index": None, "value": 0.0})
            if time.monotonic() - last_print >= 1.0:
                last_print = time.monotonic()
                ltxt = "-" if lit is None else cfg.items[lit]
                print(f"{el:7.2f}s  clench {m['score']:6.1f}  pins {m['n_used']:2d}  lit {ltxt:10s}"
                      f"{'  STOPPED' if sc.stopped else ''}{'' if running else '  (not started)'}", flush=True)
            if log:
                log.write(json.dumps({"el": round(el, 3), "clench": round(m["score"], 2), "tilt": round(m["tilt"], 2),
                                      "hot": hot, "lit": lit, "stopped": sc.stopped}) + "\n")
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
