"""A stand-in for the puck that can look at things.

It makes 16 channels the way a puck sitting over the visual cortex would see them: a shared
background of 1/f brain noise, some independent noise per pin, a little 60 Hz mains, a
resting alpha rhythm that grows about five times when the eyes close, and, while "looking"
at a tile, a small steady response at that tile's rate and its second harmonic. The response
is a couple of microvolts, well under the background, which is about what a real flicker
response looks like on dry pins.

SimSignal makes samples (used directly by the tests). SimSource publishes them as an LSL
stream so the reader, the chooser and the stop run exactly as they do with the board.
"""
from __future__ import annotations

import threading
import time

import numpy as np
from scipy import signal

# Paul Kellet's pink noise filter: white noise in, roughly 1/f power out.
_PINK_B = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
_PINK_A = np.array([1.0, -2.494956002, 2.017265875, -0.522189400])


class SimSignal:
    def __init__(self, fs: float = 250.0, n_ch: int = 16, ssvep_uv: float = 2.0, background_uv: float = 8.0,
                 pin_noise_uv: float = 3.0, alpha_open_uv: float = 4.0, alpha_closed_uv: float = 22.0,
                 mains_uv: float = 3.0, seed: int = 0):
        self.fs, self.n_ch = fs, n_ch
        self.ssvep_uv, self.background_uv, self.pin_noise_uv = ssvep_uv, background_uv, pin_noise_uv
        self.alpha_open_uv, self.alpha_closed_uv, self.mains_uv = alpha_open_uv, alpha_closed_uv, mains_uv
        self.rng = np.random.default_rng(seed)
        # pins in one 20 mm cluster: similar gains, a little spread
        self.gain = 1.0 + 0.08 * self.rng.standard_normal(n_ch)
        self.ssvep_gain = 1.0 + 0.15 * self.rng.standard_normal(n_ch)
        self.mix2 = 0.35 * self.rng.standard_normal(n_ch)   # a second, partly different source
        self.dc = self.rng.uniform(-200, 200, n_ch)
        self._pink_zi = [signal.lfilter_zi(_PINK_B, _PINK_A) * 0.0 for _ in range(2)]
        self._alpha_sos = signal.butter(2, [9.2, 11.2], btype="bandpass", fs=fs, output="sos")
        self._alpha_zi = signal.sosfilt_zi(self._alpha_sos) * 0.0
        self._pink_scale = None
        self.n = 0
        self.phase = self.rng.uniform(0, 2 * np.pi)
        # state that the page, a flag or calibration can change while it runs
        self.gaze_hz: float | None = None
        self.eyes_closed = False
        self._ssvep_level = 0.0
        self._alpha_level = alpha_open_uv
        self._calibrate_scales()

    def _calibrate_scales(self) -> None:
        # scale the filtered noises once so their RMS matches the requested microvolts
        w = self.rng.standard_normal(int(self.fs * 20))
        p = signal.lfilter(_PINK_B, _PINK_A, w)
        self._pink_scale = 1.0 / np.std(p[int(self.fs):])
        a = signal.sosfilt(self._alpha_sos, w)
        self._alpha_scale = 1.0 / np.std(a[int(self.fs):])

    def next(self, n: int) -> np.ndarray:
        """The next n samples, shaped (n, n_ch), in microvolts."""
        fs = self.fs
        t = (self.n + np.arange(n)) / fs
        self.n += n

        pinks = []
        for k in range(2):
            p, self._pink_zi[k] = signal.lfilter(_PINK_B, _PINK_A, self.rng.standard_normal(n), zi=self._pink_zi[k])
            pinks.append(p * self._pink_scale * self.background_uv)
        a, self._alpha_zi = signal.sosfilt(self._alpha_sos, self.rng.standard_normal(n), zi=self._alpha_zi)
        a = a * self._alpha_scale * np.sqrt(2)   # unit-amplitude narrowband alpha

        # smooth changes in alpha and in the flicker response (about 0.3 s)
        k = 1.0 - np.exp(-1.0 / (0.3 * fs))
        target_alpha = self.alpha_closed_uv if self.eyes_closed else self.alpha_open_uv
        target_ssvep = 0.0 if (self.eyes_closed or self.gaze_hz is None) else self.ssvep_uv
        alpha_lvl = np.empty(n)
        ssvep_lvl = np.empty(n)
        for i in range(n):
            self._alpha_level += k * (target_alpha - self._alpha_level)
            self._ssvep_level += k * (target_ssvep - self._ssvep_level)
            alpha_lvl[i], ssvep_lvl[i] = self._alpha_level, self._ssvep_level

        if self.gaze_hz is not None:
            # the response wanders a little in phase, as a real one does
            self.phase += 0.02 * self.rng.standard_normal()
            f = self.gaze_hz
            resp = (np.sin(2 * np.pi * f * t + self.phase)
                    + 0.5 * np.sin(2 * np.pi * 2 * f * t + 2 * self.phase + 0.7))
        else:
            resp = np.zeros(n)

        common = pinks[0] + alpha_lvl * a
        x = (common[:, None] * self.gain[None, :]
             + pinks[1][:, None] * self.mix2[None, :]
             + (ssvep_lvl * resp)[:, None] * self.ssvep_gain[None, :]
             + self.mains_uv * np.sin(2 * np.pi * 60.0 * t)[:, None]
             + self.pin_noise_uv * self.rng.standard_normal((n, self.n_ch))
             + self.dc[None, :])
        return x

    def take(self, seconds: float) -> np.ndarray:
        return self.next(int(round(seconds * self.fs)))


class SimSource:
    """Publishes SimSignal as an LSL stream from a background thread."""

    def __init__(self, name: str = "Octopus16Sim", fs: float = 250.0, **kw):
        import pylsl
        self.name, self.fs = name, fs
        self.sig = SimSignal(fs=fs, **kw)
        info = pylsl.StreamInfo(name, "EEG", self.sig.n_ch, fs, "float32", f"phantom_ssvep_sim_{int(time.time())}")
        self.outlet = pylsl.StreamOutlet(info)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._work, daemon=True)
        self.pid = "thread"

    def start(self) -> "SimSource":
        self._thread.start()
        return self

    def set(self, gaze_hz: float | None = None, eyes_closed: bool | None = None, clear_gaze: bool = False) -> None:
        if clear_gaze:
            self.sig.gaze_hz = None
        elif gaze_hz is not None:
            self.sig.gaze_hz = float(gaze_hz)
        if eyes_closed is not None:
            self.sig.eyes_closed = bool(eyes_closed)

    def poll(self):
        return None if self._thread.is_alive() or not self._stop.is_set() else 0

    def stop(self) -> None:
        self._stop.set()

    def _work(self) -> None:
        n = 0
        start = time.perf_counter()
        while not self._stop.is_set():
            due = int((time.perf_counter() - start) * self.fs)
            if due > n:
                x = self.sig.next(due - n)
                self.outlet.push_chunk(x.astype(np.float32).tolist())
                n = due
            time.sleep(0.02)
