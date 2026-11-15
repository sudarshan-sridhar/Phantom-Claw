"""Reads the Octopus 16 over LSL and turns it into one set of numbers every tick.

The board itself is read by PiEEG's own python_quickstart.py, which decodes the Bluetooth
packets and republishes them as an LSL stream called "Octopus16". Its simulator publishes the
same stream, so everything here works the same with or without the puck.
"""
import atexit
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pylsl

from .dsp import features, measured_rate

ROOT = Path(__file__).resolve().parents[2]
QUICKSTART = ROOT / "third_party" / "hello_world_octopus_16" / "python_quickstart.py"
SOURCE_LOG = ROOT / "backend" / "cache" / "brain_source.log"

KEEP_S = 10.0     # how much history we hold, mostly for measuring the sample rate
FILTER_S = 4.0    # how much of it goes through the filters each tick
STALE_S = 1.0     # no new samples for this long counts as no signal


def start_source(kind: str) -> subprocess.Popen:
    """Start PiEEG's quickstart as a child process: kind "sim" for the simulator, "board"
    for the real puck over Bluetooth. Its output goes to a log file, not our terminal."""
    if not QUICKSTART.exists():
        raise SystemExit(f"PiEEG starter not found at {QUICKSTART}. Clone it into third_party first.")
    args = [sys.executable, str(QUICKSTART), "--lsl"] + (["--sim"] if kind == "sim" else [])
    SOURCE_LOG.parent.mkdir(parents=True, exist_ok=True)
    log = open(SOURCE_LOG, "w", encoding="utf-8")
    proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, cwd=str(QUICKSTART.parent))

    def _stop():
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        log.close()

    atexit.register(_stop)
    proc.stop = _stop  # so callers can stop it early
    return proc


class Reader:
    def __init__(self, stream_name: str = "Octopus16", window_s: float = 1.0):
        self.stream_name = stream_name
        self.window_s = window_s
        self.inlet = None
        self.nominal = 250.0
        self.n_ch = 16
        self.data = np.zeros((0, 16))
        self.ts = np.zeros(0)
        self.last_arrival = None

    def open(self, timeout: float = 20.0, source: subprocess.Popen | None = None) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            found = pylsl.resolve_byprop("name", self.stream_name, minimum=1, timeout=1.0)
            if found:
                info = found[0]
                self.inlet = pylsl.StreamInlet(info, max_buflen=30)
                self.nominal = info.nominal_srate() or 250.0
                self.n_ch = info.channel_count()
                self.data = np.zeros((0, self.n_ch))
                return
            if source is not None and source.poll() is not None:
                raise SystemExit(f"the EEG source exited early; see {SOURCE_LOG}")
        raise SystemExit(f'no LSL stream named "{self.stream_name}" after {timeout:.0f} s')

    def pull(self) -> None:
        chunk, stamps = self.inlet.pull_chunk(timeout=0.0, max_samples=8192)
        if not stamps:
            return
        self.last_arrival = time.monotonic()
        self.data = np.vstack([self.data, np.asarray(chunk, dtype=np.float64)])
        self.ts = np.concatenate([self.ts, np.asarray(stamps, dtype=np.float64)])
        keep = self.ts >= self.ts[-1] - KEEP_S
        self.data, self.ts = self.data[keep], self.ts[keep]

    def tick(self) -> dict | None:
        """Pull whatever arrived and compute this tick's numbers, or None while the buffer
        is still filling. "stale" is set when samples have stopped coming."""
        self.pull()
        if len(self.ts) < 2 or self.ts[-1] - self.ts[0] < self.window_s + 0.5:
            return None
        fs = measured_rate(self.ts, self.nominal)
        recent = self.ts >= self.ts[-1] - FILTER_S
        f = features(self.data[recent], fs, self.window_s)
        f["t"] = float(self.ts[-1])
        f["fs"] = round(fs, 2)
        f["stale"] = self.last_arrival is None or time.monotonic() - self.last_arrival > STALE_S
        return f
