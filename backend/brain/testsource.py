"""A bad-contact LSL stream for testing that run mode refuses to fire on junk.

Every mode still carries a big 10 Hz burst every other 6 s on whatever channels have signal,
so a reader that ignored contact would happily fire STOP on it.
  flat    all 16 pins floating: well under 2 uV
  railed  pins 1 to 8 stuck at full scale, pins 9 to 16 swinging hundreds of uV
  few     two pins with good contact, the other fourteen flat
"""
import math
import time

import numpy as np
import pylsl

from .dsp import FULL_SCALE_UV

FS = 250.0


def publish(mode: str, name: str = "Octopus16Test", seconds: float | None = None) -> None:
    info = pylsl.StreamInfo(name, "EEG", 16, FS, "float32", f"phantom_test_{mode}")
    outlet = pylsl.StreamOutlet(info)
    rng = np.random.default_rng(1)
    print(f"publishing '{name}' in mode {mode}")
    n = 0
    start = time.perf_counter()
    while seconds is None or time.perf_counter() - start < seconds:
        due = int((time.perf_counter() - start) * FS)
        if due > n:
            t = (np.arange(n, due) / FS)[:, None]
            burst = np.where((t // 6) % 2 == 1, 32.0, 6.0)
            wave = burst * np.sin(2 * math.pi * 10.2 * t) + rng.uniform(-4, 4, (len(t), 16))
            if mode == "flat":
                x = rng.normal(0, 0.3, (len(t), 16)) + 0.02 * wave
            elif mode == "railed":
                x = np.empty((len(t), 16))
                x[:, :8] = FULL_SCALE_UV - rng.uniform(0, 5, (len(t), 8))
                x[:, 8:] = 20 * wave[:, 8:] + rng.normal(0, 200, (len(t), 8))
            elif mode == "few":
                x = rng.normal(0, 0.3, (len(t), 16))
                x[:, :2] = wave[:, :2]
            else:
                raise SystemExit(f"unknown mode {mode}")
            outlet.push_chunk(x.astype(np.float32).tolist())
            n = due
        time.sleep(0.02)
