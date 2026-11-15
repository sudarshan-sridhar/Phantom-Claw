"""The listener: two states (asleep, awake), the mute gate, and the link to the server."""
from __future__ import annotations

import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Callable, Iterable

import httpx

from . import config
from .names import called_text, find_agent, find_wake, name_alone
from .vad import Segmenter, Utterance


def log(msg: str) -> None:
    print(f"[listen {datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {msg}", flush=True)


def ms(a: float | None, b: float | None) -> str:
    return "-" if a is None or b is None else f"{(b - a) * 1000:.0f} ms"


class Gate:
    """Keeps the mic shut while a character is talking, and tracks requests in flight."""

    def __init__(self, tail_s: float = config.MUTE_TAIL_S, clock: Callable[[], float] = time.monotonic):
        self.tail = tail_s
        self.clock = clock
        self.until = 0.0
        self.inflight: dict[str, int] = {}
        self.lock = threading.Lock()

    def speak(self, seconds: float) -> float:
        """A line of this length is about to play. The overlay plays lines one after another,
        so a new line starts after any line already queued. Returns how long the mic stays shut."""
        with self.lock:
            now = self.clock()
            self.until = max(self.until, now) + max(0.0, float(seconds)) + self.tail
            # A pile of queued lines should not leave the mic deaf for ever.
            self.until = min(self.until, now + 25.0)
            return self.until - now

    def muted(self, now: float | None = None) -> bool:
        return (self.clock() if now is None else now) < self.until

    def begin(self, key: str) -> None:
        with self.lock:
            self.inflight[key] = self.inflight.get(key, 0) + 1

    def end(self, key: str) -> None:
        with self.lock:
            n = self.inflight.get(key, 0) - 1
            if n > 0:
                self.inflight[key] = n
            else:
                self.inflight.pop(key, None)

    def busy(self, key: str) -> bool:
        with self.lock:
            return self.inflight.get(key, 0) > 0

    def idle(self) -> bool:
        with self.lock:
            return not self.inflight and self.clock() >= self.until


class Server:
    """Plain HTTP calls to the Phantom Claw server."""

    def __init__(self, base: str = config.SERVER):
        self.base = base
        self.http = httpx.Client(base_url=base, timeout=httpx.Timeout(5.0, read=180.0))

    def event(self, ev: dict) -> None:
        self.http.post("/api/event", json=ev, timeout=3)

    def set_listen(self, awake: bool, source: str, say: bool = True) -> dict:
        r = self.http.post("/api/listen", json={"awake": awake, "source": source, "say": say}, timeout=15)
        r.raise_for_status()
        return r.json()

    def get_listen(self) -> bool | None:
        try:
            r = self.http.get("/api/listen", timeout=3)
            r.raise_for_status()
            return bool(r.json().get("awake"))
        except Exception:  # noqa: BLE001
            return None

    def request(self, text: str, agent: str | None) -> dict:
        r = self.http.post("/api/request", json={"text": text, "agent": agent})
        r.raise_for_status()
        return r.json()


class Listener:
    def __init__(
        self,
        transcriber,
        segmenter: Segmenter,
        server: Server | None = None,
        require_name: bool = config.REQUIRE_NAME,
        gate: Gate | None = None,
    ):
        self.stt = transcriber
        self.seg = segmenter
        self.server = server
        self.require_name = require_name
        self.gate = gate or Gate()
        self.awake = False
        self.lock = threading.Lock()
        self.utts: queue.Queue = queue.Queue()
        self.pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="request")
        self.records: list[dict] = []
        self.waiting_reply: dict[str, dict] = {}  # who we are waiting to hear from -> record
        self.muted_voice_frames = 0
        self.muted_utterance_cut = 0
        self._was_muted = False
        self._stop = threading.Event()
        self._n = 0

    # ------------------------------------------------------------ state

    def set_awake(self, awake: bool, source: str, tell_server: bool = True) -> None:
        with self.lock:
            if awake == self.awake:
                return
            self.awake = awake
        log(f"{'AWAKE, listening for names' if awake else 'ASLEEP, waiting for Phantom Claw start'} ({source})")
        if tell_server and self.server is not None:
            def tell():
                # Counts as busy until the server has queued Merlin's line, so a file test
                # waits for it like a person would.
                self.gate.begin("_listen")
                try:
                    self.server.set_listen(awake, source)
                except Exception as e:  # noqa: BLE001
                    log(f"could not tell the server about the change: {type(e).__name__}")
                finally:
                    self.gate.end("_listen")
            self.pool.submit(tell)

    def on_connect(self) -> None:
        """The server (re)appeared. Bring the two states together: awake on either side wins."""
        if self.server is None:
            return
        cur = self.server.get_listen()
        if cur is None:
            return
        if self.awake and not cur:
            try:
                self.server.set_listen(True, "listener", say=False)
            except Exception:  # noqa: BLE001
                pass
        elif cur and not self.awake:
            self.set_awake(True, "server", tell_server=False)

    def toggle(self, source: str) -> None:
        self.set_awake(not self.awake, source)

    # ------------------------------------------------------------ server events

    def on_server_event(self, ev: dict) -> None:
        t = time.monotonic()
        typ = ev.get("type")
        if typ == "say":
            secs = float(ev.get("duration_s") or 0)
            left = self.gate.speak(secs)
            self._note_reply(ev.get("agent"), t)
            log(f"mic muted for {left:.1f} s, {ev.get('agent')} is speaking")
        elif typ == "merlin":
            secs = sum(float(a.get("duration_s") or 0) for a in ev.get("argued") or []) + float(ev.get("duration_s") or 0)
            left = self.gate.speak(secs)
            self._note_reply("merlin", t)
            log(f"mic muted for {left:.1f} s, Merlin is speaking")
        elif typ == "listen" and "awake" in ev:
            self.set_awake(bool(ev["awake"]), "server", tell_server=False)

    def _note_reply(self, agent: str | None, t: float) -> None:
        with self.lock:
            keys = [agent] if agent in self.waiting_reply else []
            if agent == "merlin" or (agent and not keys):
                # A request with no named agent is answered first by Merlin, or by whoever he picks.
                keys += [k for k in ("any",) if k in self.waiting_reply]
            for k in keys:
                rec = self.waiting_reply.pop(k)
                rec["t_reply"] = t
                rec["reply_from"] = agent
                log(f"#{rec['n']} first reply from {agent}, {ms(rec['t_sent'], t)} after the request went out")

    # ------------------------------------------------------------ audio

    def feed_frames(self, frames: Iterable[tuple[bytes, float]]) -> None:
        """Run frames through the mute gate and the segmenter. Same path for the mic and for files."""
        for frame, t in frames:
            if self._stop.is_set():
                break
            if self.gate.muted(t):
                if not self._was_muted:
                    self._was_muted = True
                if self.seg.active:
                    self.muted_utterance_cut += 1
                    self.seg.reset()
                if self.seg.is_speech(frame):
                    self.muted_voice_frames += 1
                continue
            if self._was_muted:
                self._was_muted = False
                if self.muted_voice_frames:
                    log(f"mic open again; ignored {self.muted_voice_frames * config.FRAME_MS / 1000:.1f} s of voice while muted")
                self.muted_voice_frames = 0
            utt = self.seg.feed(frame, t)
            if utt is not None:
                self.utts.put(utt)

    def stt_worker(self) -> None:
        while not self._stop.is_set():
            try:
                utt = self.utts.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.process(utt)
            except Exception as e:  # noqa: BLE001
                log(f"could not handle an utterance: {type(e).__name__}: {e}")
            finally:
                self.utts.task_done()

    # ------------------------------------------------------------ one utterance

    def process(self, utt: Utterance) -> dict:
        self._n += 1
        rec = {"n": self._n, "seconds": round(utt.seconds, 2), "t_end": utt.t_end, "t_cut": utt.t_cut}
        t0 = time.monotonic()
        text = self.stt.transcribe(utt.audio)
        rec["t_text"] = time.monotonic()
        rec["stt_ms"] = round((rec["t_text"] - t0) * 1000)
        rec["text"] = text
        self.records.append(rec)
        if not text:
            rec["action"] = "nothing heard"
            log(f"#{rec['n']} {utt.seconds:.1f} s clip, no words")
            return rec
        log(f'#{rec["n"]} heard "{text}" ({utt.seconds:.1f} s clip, speech end to text {ms(utt.t_end, rec["t_text"])}, model {rec["stt_ms"]} ms)')
        self.decide(text, rec)
        return rec

    def decide(self, text: str, rec: dict) -> None:
        cmd, _a, _b = find_wake(text)
        if not self.awake:
            if cmd == "start":
                rec["action"] = "wake"
                self.set_awake(True, "voice")
            else:
                rec["action"] = "ignored, asleep"
                log(f"#{rec['n']} asleep, ignored")
            return
        if cmd == "stop":
            rec["action"] = "sleep"
            self.set_awake(False, "voice")
            return
        if cmd == "start":
            rec["action"] = "ignored, already awake"
            log(f"#{rec['n']} already awake")
            return

        agent, _i = find_agent(text)
        if agent is None and self.require_name:
            rec["action"] = "ignored, no name"
            log(f"#{rec['n']} no name in it, ignored")
            return
        key = agent or "any"
        if self.gate.busy(key):
            rec["action"] = f"ignored, {key} is still busy"
            log(f"#{rec['n']} {key} is still working on the last one, ignored")
            return
        alone = agent is not None and name_alone(text, agent)
        send_text = called_text(agent) if alone else text
        # Merlin is not one of the five agents; naming him means he picks, same as no name.
        req_agent = agent if agent not in (None, "merlin") else None
        rec.update({"action": "request", "agent": agent, "sent_text": send_text, "name_alone": alone})
        self.gate.begin(key)
        self.pool.submit(self._send, rec, key, agent or "merlin", text, send_text, req_agent)

    def _send(self, rec: dict, key: str, hear_agent: str, heard: str, send_text: str, req_agent: str | None) -> None:
        try:
            if self.server is None:
                return
            try:
                # Tell the avatar first so it reacts before the LLM even starts.
                self.server.event({"type": "hear", "agent": hear_agent, "text": heard})
            except Exception as e:  # noqa: BLE001
                log(f"hear event failed: {type(e).__name__}")
            with self.lock:
                rec["t_sent"] = time.monotonic()
                self.waiting_reply[key] = rec
            log(f"#{rec['n']} request to {req_agent or 'Merlin to pick'}: \"{send_text}\" (text to request {ms(rec['t_text'], rec['t_sent'])})")
            try:
                res = self.server.request(send_text, req_agent)
                rec["replied_agent"] = res.get("agent")
                rec["reply"] = res.get("say")
            except Exception as e:  # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {str(e)[:120]}"
                log(f"#{rec['n']} request failed: {rec['error']}")
            rec["t_done"] = time.monotonic()
            with self.lock:
                self.waiting_reply.pop(key, None)
            log(
                f"#{rec['n']} TIMING speech end to text {ms(rec['t_end'], rec['t_text'])} "
                f"(silence wait {ms(rec['t_end'], rec['t_cut'])}, model {rec['stt_ms']} ms), "
                f"text to request sent {ms(rec['t_text'], rec['t_sent'])}, "
                f"request to first reply {ms(rec['t_sent'], rec.get('t_reply'))}, "
                f"speech end to first reply {ms(rec['t_end'], rec.get('t_reply'))}, "
                f"whole request {ms(rec['t_sent'], rec['t_done'])}; {rec.get('replied_agent')} said \"{rec.get('reply')}\""
            )
        finally:
            self.gate.end(key)

    # ------------------------------------------------------------ wrap up

    def stop(self) -> None:
        self._stop.set()
        self.pool.shutdown(wait=False, cancel_futures=True)

    def summary(self) -> list[dict]:
        out = []
        for r in self.records:
            d = {k: r.get(k) for k in ("n", "text", "action", "agent", "sent_text", "replied_agent", "reply", "error")}
            d["end_to_text_ms"] = _d(r.get("t_end"), r.get("t_text"))
            d["stt_ms"] = r.get("stt_ms")
            d["text_to_sent_ms"] = _d(r.get("t_text"), r.get("t_sent"))
            d["sent_to_reply_ms"] = _d(r.get("t_sent"), r.get("t_reply"))
            d["end_to_reply_ms"] = _d(r.get("t_end"), r.get("t_reply"))
            out.append({k: v for k, v in d.items() if v is not None})
        return out

    def print_summary(self) -> None:
        log("summary " + json.dumps({"awake": self.awake, "utterances": self.summary(), "muted_cuts": self.muted_utterance_cut}))


def _d(a, b):
    return None if a is None or b is None else round((b - a) * 1000)


def watch_server(listener: Listener, base: str = config.SERVER, stop: threading.Event | None = None) -> None:
    """Follow the server's WebSocket: say and merlin events mute the mic, listen events sync the state."""
    from websockets.sync.client import connect

    url = base.replace("http", "ws", 1) + "/ws"
    stop = stop or listener._stop
    warned = False
    while not stop.is_set():
        try:
            with connect(url, open_timeout=3, close_timeout=1) as ws:
                log(f"following {url}")
                warned = False
                listener.on_connect()
                while not stop.is_set():
                    try:
                        msg = ws.recv(timeout=0.5)
                    except TimeoutError:
                        continue
                    try:
                        ev = json.loads(msg)
                    except ValueError:
                        continue
                    if isinstance(ev, dict):
                        listener.on_server_event(ev)
        except Exception as e:  # noqa: BLE001
            if not warned:
                log(f"server WebSocket not reachable ({type(e).__name__}), retrying")
                warned = True
            stop.wait(1.5)
