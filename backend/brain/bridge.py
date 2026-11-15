"""Bluetooth to LSL bridge for the chooser: PiEEG's own Bluetooth code, with lost packets
filled in.

Each Bluetooth packet from the puck carries one sample and a counter from 0 to 255. PiEEG's
quickstart passes samples on as they come, so a lost packet simply vanishes and the samples
after it slide early in time. For the eyes-closed stop that does not matter. For the chooser
it does: the flicker response is a steady sine, and a few percent of missing samples knock
its phase about enough to blur 15 to 20 Hz. This bridge watches the counter and, for each
gap of up to MAX_FILL packets, inserts straight-line samples between the last good sample
and the next, so the stream keeps the board's true spacing.

It reuses the packet decoder and the scan and connect code from
third_party/hello_world_octopus_16/python_quickstart.py and publishes the same LSL stream
name ("Octopus16"), so the reader cannot tell it from the quickstart.

    python -m backend.brain.bridge
"""
from __future__ import annotations

import asyncio
import atexit
import importlib.util
import subprocess
import sys
import time

import numpy as np

from .reader import QUICKSTART, ROOT

BRIDGE_LOG = ROOT / "backend" / "cache" / "brain_bridge.log"
MAX_FILL = 25     # longer gaps (0.1 s and up) are left as they are; the stream was really broken


def counter_gap(last: int | None, counter: int) -> int:
    """Packets lost between two counters (0 to 255, wrapping). A jump backwards (more than
    half the circle) is treated as a restart, not a loss."""
    if last is None:
        return 0
    gap = (counter - last - 1) % 256
    return gap if gap < 128 else 0


def fill(last: np.ndarray, current: np.ndarray, gap: int) -> np.ndarray:
    """gap samples on the straight line from last to current, shaped (gap, channels)."""
    if gap <= 0:
        return np.zeros((0, len(current)))
    w = (np.arange(1, gap + 1) / (gap + 1))[:, None]
    return last[None, :] * (1 - w) + current[None, :] * w


class GapFillMonitor:
    """Stands in for the quickstart's StreamMonitor: run_ble calls ingest() per sample."""

    def __init__(self, name: str = "Octopus16", nominal: float = 250.0, n_ch: int = 16):
        import pylsl
        info = pylsl.StreamInfo(name, "EEG", n_ch, nominal, "float32", "octopus16_ble_gapfill")
        self.outlet = pylsl.StreamOutlet(info)
        self.last_counter: int | None = None
        self.last: np.ndarray | None = None
        self.received = self.filled = self.lost_unfilled = 0
        self.t0 = self.t_print = time.perf_counter()

    def ingest(self, counter: int, channels: list[float]) -> None:
        cur = np.asarray(channels, dtype=np.float64)
        gap = counter_gap(self.last_counter, counter)
        rows = []
        if gap and self.last is not None:
            if gap <= MAX_FILL:
                rows.append(fill(self.last, cur, gap))
                self.filled += gap
            else:
                self.lost_unfilled += gap
        rows.append(cur[None, :])
        self.outlet.push_chunk(np.vstack(rows).astype(np.float32).tolist())
        self.last_counter, self.last = counter, cur
        self.received += 1
        now = time.perf_counter()
        if now - self.t_print >= 5.0:
            self.t_print = now
            el = now - self.t0
            total = self.received + self.filled
            print(f"[bridge] {self.received / el:.1f} packets/s received, {total / el:.1f} samples/s out, "
                  f"{self.filled} filled, {self.lost_unfilled} lost in long gaps", flush=True)

    def close(self) -> None:
        pass


def _load_quickstart():
    spec = importlib.util.spec_from_file_location("pieeg_quickstart", QUICKSTART)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    if not QUICKSTART.exists():
        raise SystemExit(f"PiEEG starter not found at {QUICKSTART}")
    qs = _load_quickstart()
    monitor = GapFillMonitor()
    print("[bridge] publishing LSL stream 'Octopus16' with lost packets filled in", flush=True)
    try:
        asyncio.run(qs.run_ble(monitor))
    except KeyboardInterrupt:
        pass


def start_bridge() -> subprocess.Popen:
    """Run the bridge as a child process, output to a log file, like reader.start_source."""
    BRIDGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    log = open(BRIDGE_LOG, "w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-u", "-m", "backend.brain.bridge"], stdout=log,
                            stderr=subprocess.STDOUT, cwd=str(ROOT))

    def _stop():
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        log.close()

    atexit.register(_stop)
    proc.stop = _stop
    return proc


if __name__ == "__main__":
    main()
