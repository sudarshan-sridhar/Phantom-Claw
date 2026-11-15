"""Offline check of the listener on synthetic speech.

    python -m backend.listen.evalcheck            make the clips (cached) and score every pipeline
    python -m backend.listen.evalcheck --make     only make the clips
    python -m backend.listen.evalcheck --old DIR  also score the old listener, a copy of the package in DIR/listen_old

The sentences are spoken by three of the cast voices through the project's own ElevenLabs text
to speech (backend.agents.voice.tts and its cache), then turned down to the level the laptop mic
gives (peak about 400 of 32767), with mic hiss and room noise (other voices plus a motor hum)
at two levels. This is synthetic speech, not the owner's voice: it shows whether the pipeline
holds up at a low level and with noise, not how it does with his accent.

The clips go to backend/listen/test_out/. Nothing is sent to the Phantom Claw server.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import re
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

RATE = 16000
OUT = Path(__file__).resolve().parent / "test_out"
VOICES = ["merlin", "king", "escanor"]  # Monika Sogam (Indian English), Kryuu, Arthur

# (text, agent, item, verb); agent "wake" means the wake phrase must be found.
SENTENCES = [
    ("Phantom Claw, start.", "wake", None, None),
    ("Hey Phantom Claw, start.", "wake", None, None),
    ("Phantom Claw start.", "wake", None, None),
    ("Escanor, give me the bottle.", "escanor", "bottle", "give me"),
    ("Elizabeth, bring me the Sharpie.", "elizabeth", "sharpie", "bring me"),
    ("Ban, hand me the scissors.", "ban", "scissors", "hand me"),
    ("King, point at the glue stick.", "king", "glue stick", "point at"),
    ("Gowther, pick up the earbuds.", "gowther", "earbuds", "pick up"),
    ("Escanor, put the bottle on the left.", "escanor", "bottle", "put"),
    ("Elizabeth, give me the scissors please.", "elizabeth", "scissors", "give me"),
    ("Ban, bring me the glue stick.", "ban", "glue stick", "bring me"),
    ("King, hand me the Sharpie.", "king", "sharpie", "hand me"),
    ("Gowther, point at the bottle.", "gowther", "bottle", "point at"),
    ("Escanor, pick up the earbuds.", "escanor", "earbuds", "pick up"),
    ("Elizabeth, put the glue stick down.", "elizabeth", "glue stick", "put"),
    ("Ban, point at the Sharpie.", "ban", "sharpie", "point at"),
    ("King, give me the earbuds.", "king", "earbuds", "give me"),
    ("Gowther, hand me the bottle.", "gowther", "bottle", "hand me"),
    ("Merlin, open Notepad.", "merlin", None, "open"),
    ("Merlin, play some music.", "merlin", None, "play"),
    ("King, search for glue sticks online.", "king", "glue stick", "search"),
    ("Escanor, stop.", "escanor", None, "stop"),
    ("Ban, bring me the scissors.", "ban", "scissors", "bring me"),
    ("Elizabeth, pick up the bottle.", "elizabeth", "bottle", "pick up"),
    ("Gowther, give me the Sharpie.", "gowther", "sharpie", "give me"),
]

# (name, speech peak, noise SNR in dB or None for hiss only)
CONDITIONS = [("quiet", 400, None), ("quiet_noise15", 400, 15), ("quiet_noise5", 400, 5)]


# ---------------------------------------------------------------- making the clips

def decode(mp3: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp3), "-ac", "1", "-ar", str(RATE), "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32)


def write_wav(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(np.clip(x, -32767, 32767).astype(np.int16).tobytes())


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).copy()


def speech_clips() -> list[np.ndarray]:
    from backend.agents import voice
    from backend.agents.config import TTS_CACHE_DIR

    async def go():
        out = []
        for i, (text, *_rest) in enumerate(SENTENCES):
            r = await voice.tts(VOICES[i % len(VOICES)], text)
            out.append(TTS_CACHE_DIR / r["file"])
        return out

    return [decode(p) for p in asyncio.run(go())]


def room_noise(n: int, clips: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Other people talking (reversed speech from three of the clips) plus a servo like hum."""
    babble = np.zeros(n, dtype=np.float32)
    for _ in range(3):
        c = clips[rng.integers(len(clips))][::-1]
        reps = np.tile(c, n // len(c) + 2)
        off = rng.integers(len(c))
        babble += reps[off: off + n] / (np.sqrt(np.mean(c * c)) + 1e-6)
    t = np.arange(n) / RATE
    hum = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) / k for k, f in enumerate((120, 240, 360, 1800), 1))
    hum *= 0.6 + 0.4 * np.sin(2 * np.pi * 0.7 * t)  # the motor speeds up and slows down
    whine = np.convolve(rng.standard_normal(n), np.ones(4) / 4, mode="same")
    noise = babble / 3 + 0.5 * hum + 0.3 * whine
    return noise / np.sqrt(np.mean(noise * noise))


def make_clips() -> None:
    clips = speech_clips()
    rng = np.random.default_rng(7)
    for name, peak, snr in CONDITIONS:
        for i, c in enumerate(clips):
            s = c * (peak / max(1.0, np.abs(c).max()))
            body = np.concatenate([np.zeros(int(1.0 * RATE)), s, np.zeros(int(1.5 * RATE))])
            hiss = rng.standard_normal(len(body)) * 4.0  # the mic's own noise
            x = body + hiss
            if snr is not None:
                srms = np.sqrt(np.mean(s * s))
                x = x + room_noise(len(body), clips, rng) * srms / (10 ** (snr / 20))
            write_wav(OUT / name / f"{i:02d}.wav", x)
    (OUT / "sentences.json").write_text(json.dumps(SENTENCES, indent=1), encoding="utf-8")
    print(f"clips in {OUT}", flush=True)


# ---------------------------------------------------------------- scoring

def words(t: str) -> list[str]:
    return re.findall(r"[a-z]+", (t or "").lower().replace("'s", ""))


def wer(ref: str, hyp: str) -> tuple[int, int]:
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)], len(r)


def understood(text: str, agent: str, item: str | None, verb: str | None, names_mod) -> bool:
    if agent == "wake":
        return names_mod.find_wake(text)[0] == "start"
    if names_mod.find_agent(text)[0] != agent:
        return False
    plain = " ".join(words(text))
    if item and not re.search(r"\b" + item + r"s?\b", plain):
        return False
    if verb and not re.search(r"\b" + verb + r"\b", plain):
        return False
    return True


def run(name: str, vad_mod, seg_kwargs: dict, agc_vad: bool | None, transcribe, names_mod, fix=None, cond=None) -> dict:
    """Play every clip of one condition back to back through one VAD and segmenter, like a live
    mic in a room that never goes quiet (the room noise runs between the sentences too), and score
    it. An utterance belongs to the clip it started in, so two sentences run together count as a
    miss for the second one."""
    make_vad = vad_mod.make_vad
    is_speech = make_vad(agc=agc_vad) if agc_vad is not None else make_vad()
    seg = vad_mod.Segmenter(is_speech, **seg_kwargs)
    ok = 0
    errs = n_ref = 0
    ms = []
    rows = []
    t = 0.0
    starts, utts = [], []
    for i in range(len(SENTENCES)):
        audio = read_wav(OUT / cond / f"{i:02d}.wav")
        starts.append(t)
        for k in range(0, len(audio) - 480 + 1, 480):
            t += 0.03
            u = seg.feed(audio[k: k + 480].tobytes(), t)
            if u is not None:
                utts.append(u)
    u = seg.flush(t)
    if u is not None:
        utts.append(u)
    per = [[] for _ in SENTENCES]
    for u in utts:
        at = u.t_start + 0.5
        i = max(j for j, s in enumerate(starts) if s <= at) if at >= 0 else 0
        t0 = time.monotonic()
        h = transcribe(u.audio)
        ms.append((time.monotonic() - t0) * 1000)
        per[i].append(h)
    lens = [round(len(u.audio) / RATE, 1) for u in utts]
    cut = sum(1 for p in per if p)
    for i, (text, agent, item, verb) in enumerate(SENTENCES):
        hyp = per[i]
        raw = " ".join(x for x in hyp if x)
        out = fix(raw)[0] if (fix and raw) else raw
        e, nr = wer(text, out)
        errs += e
        n_ref += nr
        good = understood(out, agent, item, verb, names_mod)
        ok += good
        rows.append({"ref": text, "raw": raw, "out": out, "ok": good})
    return {"run": name, "cond": cond, "word_acc": max(0.0, 1 - errs / n_ref), "cmd_acc": ok / len(SENTENCES), "utts": len(utts), "max_len": max(lens) if lens else 0,
            "cut": cut, "ms_mean": float(np.mean(ms)) if ms else 0.0, "ms_p90": float(np.percentile(ms, 90)) if ms else 0.0, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--old", default=None, help="folder holding listen_old (a copy of the package before the changes)")
    ap.add_argument("--runs", default="", help="comma list of runs to do (default all)")
    ap.add_argument("--vad-agc", type=int, default=None, help="1 or 0: gain before the VAD in the after runs (default: LISTEN_AGC_VAD)")
    args = ap.parse_args()
    if not (OUT / CONDITIONS[-1][0] / f"{len(SENTENCES) - 1:02d}.wav").exists() or args.make:
        make_clips()
    if args.make:
        return

    from backend.listen import names as new_names
    from backend.listen import stt as new_stt
    from backend.listen import vad as new_vad
    from backend.listen.config import MIN_SPEECH_MS, PREROLL_MS, SILENCE_MS
    from backend.listen.fix import repair

    new_seg = {"silence_ms": SILENCE_MS, "preroll_ms": PREROLL_MS, "min_speech_ms": MIN_SPEECH_MS}
    old_seg = {"silence_ms": 450, "preroll_ms": 300, "min_speech_ms": 300}

    plans = []
    old_names = old_vad = None
    if args.old:
        sys.path.insert(0, args.old)
        old_vad = importlib.import_module("listen_old.vad")
        old_names = importlib.import_module("listen_old.names")
        old_stt = importlib.import_module("listen_old.stt")
        old_local = old_stt.Transcriber(device="cuda", model="")
        print("before, local:", old_local.load(), flush=True)
        plains = new_stt.ElevenSTT()
        plains.extras = False  # model_id only, like the server's /api/stt

        plans.append(("before local small.en", old_vad, old_seg, None, old_local.transcribe, old_names, None))
        plans.append(("before eleven", old_vad, old_seg, None, lambda a: plains(a), old_names, None))

    loc = new_stt.Transcriber(engine="local", device="cuda")
    print("after, local:", loc.load(), flush=True)
    small = new_stt.Transcriber(engine="local", device="cuda", model="small.en")
    small.load()
    elev = new_stt.Transcriber(engine="eleven", device="cuda")
    elev.model, elev.kind, elev.path, elev.model_name = loc.model, loc.kind, loc.path, loc.model_name  # share the GPU model
    elev.engine = "eleven"
    elev.eleven = new_stt.ElevenSTT()
    fallbacks = []

    def elev_tr(a):
        r = elev.transcribe(a)
        if elev.last.get("fallback"):
            fallbacks.append(elev.last["fallback"])
        return r

    from backend.listen.config import AGC_VAD
    va = bool(AGC_VAD if args.vad_agc is None else args.vad_agc)
    plans.append((f"after local {loc.model_name}", new_vad, new_seg, va, loc.transcribe, new_names, repair))
    plans.append(("after local small.en", new_vad, new_seg, va, small.transcribe, new_names, repair))
    plans.append(("after eleven", new_vad, new_seg, va, elev_tr, new_names, repair))

    want = [w.strip() for w in args.runs.split(",") if w.strip()]
    results = []
    for p in plans:
        if want and not any(w in p[0] for w in want):
            continue
        for cond, _peak, _snr in CONDITIONS:
            r = run(p[0], *p[1:6], fix=p[6], cond=cond)
            results.append(r)
            print(f"{r['run']:<28} {cond:<14} words {r['word_acc']*100:5.1f}%  commands {r['cmd_acc']*100:5.1f}%  "
                  f"utterances {r['utts']} (longest {r['max_len']} s)  stt {r['ms_mean']:.0f} ms (p90 {r['ms_p90']:.0f})", flush=True)
    if fallbacks:
        print(f"ElevenLabs fell back to local {len(fallbacks)} times: {sorted(set(fallbacks))}")
    (OUT / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"details in {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
