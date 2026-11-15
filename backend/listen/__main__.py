"""Run the listener.

    python -m backend.listen                      microphone, starts asleep
    python -m backend.listen --awake              microphone, starts awake
    python -m backend.listen --file a.wav --file b.wav
                                                  play files through the same path as the mic
    python -m backend.listen --mic-test 3         open the mic for 3 s and report what came in

A file item written as reply:path waits until a character is speaking before it plays,
which is how the "does not hear itself" check is done without a person in the room.
"""
from __future__ import annotations

import argparse
import queue
import threading
import time
import wave

import numpy as np

from . import config
from .listener import Listener, Server, log, watch_server
from .stt import Transcriber
from .vad import Segmenter, make_vad

FRAME = config.RATE * config.FRAME_MS // 1000  # 480 samples


def resample(x: np.ndarray, rate: int) -> np.ndarray:
    if rate == config.RATE or len(x) == 0:
        return x
    n = int(round(len(x) * config.RATE / rate))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x.astype(np.float32)).astype(np.int16)


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        ch, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise SystemExit(f"{path}: only 16 bit PCM wav files are supported")
    x = np.frombuffer(raw, dtype=np.int16)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1).astype(np.int16)
    return resample(x, rate)


def file_frames(listener: Listener, items: list[str], gap_s: float, fast: bool):
    """Frames from a list of files, paced like a live mic, with quiet in between."""
    t = time.monotonic()
    silence = np.zeros(int(gap_s * config.RATE), dtype=np.int16)
    for item in items:
        during_reply = item.startswith("reply:")
        path = item[len("reply:"):] if during_reply else item
        if during_reply:
            # Wait for a character to start talking, then speak over them.
            log(f"waiting for a reply to start before playing {path}")
            deadline = time.monotonic() + 30
            while not listener.gate.muted() and time.monotonic() < deadline:
                time.sleep(0.02)
            time.sleep(0.3)
        else:
            # Wait for the room to be quiet again, like a person would.
            deadline = time.monotonic() + 60
            while not listener.gate.idle() and time.monotonic() < deadline:
                time.sleep(0.05)
        log(f"playing {path}")
        audio = np.concatenate([read_wav(path), silence])
        t = max(t, time.monotonic())
        for i in range(0, len(audio) - FRAME + 1, FRAME):
            t += config.FRAME_MS / 1000
            if not fast:
                wait = t - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
            yield audio[i : i + FRAME].tobytes(), t
        # Let the utterance get transcribed and the request go out before the next file.
        listener.utts.join()
        time.sleep(0.2)


def pick_mic(choice=None):
    """The mic to use: --mic or LISTEN_MIC if given, else the laptop's built-in array, else the default."""
    import os

    import sounddevice as sd

    choice = choice if choice is not None else (os.environ.get("LISTEN_MIC") or None)
    if choice is not None:
        return int(choice) if str(choice).isdigit() else choice
    try:
        mme = [h["name"] for h in sd.query_hostapis()].index("MME")
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0 and d["hostapi"] == mme and "microphone array" in d["name"].lower():
                return i
    except Exception:  # noqa: BLE001
        pass
    return None


def mic_frames(stop: threading.Event, device=None):
    """Frames from the microphone at 16 kHz mono. Resamples if the device will not do 16 kHz."""
    import sounddevice as sd

    q: queue.Queue = queue.Queue(maxsize=400)

    def cb(indata, _frames, _time, status):
        try:
            q.put_nowait((bytes(indata), time.monotonic()))
        except queue.Full:
            pass

    rate = config.RATE
    try:
        stream = sd.RawInputStream(samplerate=rate, channels=1, dtype="int16", blocksize=FRAME, device=device, callback=cb)
    except Exception:  # noqa: BLE001
        rate = int(sd.query_devices(device, "input")["default_samplerate"])
        stream = sd.RawInputStream(samplerate=rate, channels=1, dtype="int16", blocksize=int(rate * config.FRAME_MS / 1000), device=device, callback=cb)
    info = sd.query_devices(stream.device, "input")
    log(f"microphone: {info['name']} at {rate} Hz{' (resampled to 16 kHz)' if rate != config.RATE else ''}")
    buf = np.zeros(0, dtype=np.int16)
    with stream:
        while not stop.is_set():
            try:
                data, t = q.get(timeout=0.2)
            except queue.Empty:
                continue
            x = np.frombuffer(data, dtype=np.int16)
            buf = np.concatenate([buf, resample(x, rate)])
            while len(buf) >= FRAME:
                yield buf[:FRAME].tobytes(), t
                buf = buf[FRAME:]


def mic_test(seconds: float, device=None) -> None:
    stop = threading.Event()
    n, peak, rms = 0, 0, []
    t_end = time.monotonic() + seconds
    is_speech = make_vad()
    voiced = 0
    for frame, _t in mic_frames(stop, device):
        x = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        n += 1
        peak = max(peak, int(np.abs(x).max()))
        rms.append(float(np.sqrt(np.mean(x * x))))
        voiced += is_speech(frame)
        if time.monotonic() >= t_end:
            stop.set()
            break
    log(f"mic test: {n} frames ({n * config.FRAME_MS / 1000:.1f} s of audio in {seconds:.1f} s), peak {peak}, "
        f"median level {np.median(rms) if rms else 0:.0f}, loudest frame {max(rms) if rms else 0:.0f}, voiced frames {voiced}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m backend.listen")
    ap.add_argument("--file", action="append", help="wav file to play through the listener (repeatable; prefix reply: to play it over a reply)")
    ap.add_argument("--gap", type=float, default=1.0, help="seconds of quiet after each file")
    ap.add_argument("--fast", action="store_true", help="do not pace files in real time")
    ap.add_argument("--awake", action="store_true", help="start awake")
    ap.add_argument("--require-name", dest="require_name", action="store_true", default=None)
    ap.add_argument("--no-require-name", dest="require_name", action="store_false")
    ap.add_argument("--stt", choices=["auto", "cuda", "cpu", "server"], default=config.DEVICE)
    ap.add_argument("--model", default=config.MODEL, help="faster-whisper model name, for example base.en or small.en")
    ap.add_argument("--mic", default=None, help="microphone device index or name (default: the built-in mic array, else the system default)")
    ap.add_argument("--mic-test", type=float, default=0, help="open the mic for this many seconds, report, exit")
    ap.add_argument("--no-hotkey", action="store_true")
    args = ap.parse_args()

    mic = pick_mic(args.mic)
    if args.mic_test:
        mic_test(args.mic_test, mic)
        return

    import os

    for item in args.file or []:
        path = item[len("reply:"):] if item.startswith("reply:") else item
        if not os.path.isfile(path):
            raise SystemExit(f"no such file: {path}")

    require_name = config.REQUIRE_NAME if args.require_name is None else args.require_name
    stt = Transcriber(device=args.stt, model=args.model)
    log("loading speech to text...")
    log(f"speech path: {stt.load()}")
    server = Server()
    listener = Listener(stt, Segmenter(make_vad()), server, require_name=require_name)
    log(f"require a name: {'yes' if require_name else 'no, unnamed requests go to Merlin'}")

    threading.Thread(target=watch_server, args=(listener,), daemon=True, name="ws").start()
    threading.Thread(target=listener.stt_worker, daemon=True, name="stt").start()

    current = server.get_listen()
    if current is None:
        log(f"server not reachable at {config.SERVER}; start it with python -m backend.server")
    if args.awake:
        listener.set_awake(True, "start flag")
    elif current:
        listener.set_awake(True, "server", tell_server=False)
    else:
        log("ASLEEP, waiting for Phantom Claw start")

    if not args.file and not args.no_hotkey:
        from .hotkey import start_hotkey

        ok, msg = start_hotkey(lambda: listener.toggle("shortcut"))
        log(("shortcut: " if ok else "shortcut not available: ") + msg)

    try:
        if args.file:
            listener.feed_frames(file_frames(listener, args.file, args.gap, args.fast))
            listener.utts.join()
            deadline = time.monotonic() + 90
            while not listener.gate.idle() and time.monotonic() < deadline:
                time.sleep(0.1)
            listener.print_summary()
        else:
            log("listening on the microphone, Ctrl+C to quit")
            listener.feed_frames(mic_frames(listener._stop, mic))
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        log("stopped")


if __name__ == "__main__":
    main()
