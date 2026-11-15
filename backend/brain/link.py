"""Sends brain events to the Phantom Claw server without ever holding up the reader.

Requests go into a queue and a background thread posts them with short timeouts. If the
server is down, or does not have /api/event yet (404), we say so once and carry on.
"""
import os
import queue
import threading
import time

import httpx

DEFAULT_SERVER = os.environ.get("PHANTOM_SERVER", "http://127.0.0.1:8765")


class ServerLink:
    def __init__(self, base: str = DEFAULT_SERVER, verbose: bool = True):
        self.base = base.rstrip("/")
        self.verbose = verbose
        self.q: queue.Queue = queue.Queue(maxsize=200)
        self.results: dict[str, dict[str, int]] = {}
        self._told: set = set()
        self._thread = threading.Thread(target=self._work, daemon=True)
        self._thread.start()

    def post(self, path: str, body: dict | None = None, loud: bool = False) -> None:
        try:
            self.q.put_nowait((path, body, loud, time.time()))
        except queue.Full:
            pass  # the server has been gone a while; dropping old focus values is fine

    def event(self, state: str, loud: bool = False, **fields) -> None:
        self.post("/api/event", {"type": "brain", "state": state, "time": time.time(), **fields}, loud)

    def _note(self, path: str, outcome: str, loud: bool) -> None:
        counts = self.results.setdefault(path, {})
        counts[outcome] = counts.get(outcome, 0) + 1
        first = (path, outcome) not in self._told
        self._told.add((path, outcome))
        if self.verbose and (loud or first):
            extra = "" if loud else " (shown once)"
            print(f"  [server] POST {path} -> {outcome}{extra}", flush=True)

    def _work(self) -> None:
        with httpx.Client(timeout=httpx.Timeout(1.0, connect=0.5)) as client:
            while True:
                path, body, loud, _ = self.q.get()
                try:
                    r = client.post(self.base + path, json=body)
                    outcome = str(r.status_code)
                except httpx.HTTPError as e:
                    outcome = f"not reachable ({type(e).__name__})"
                self._note(path, outcome, loud)
                self.q.task_done()

    def flush(self, timeout: float = 3.0) -> None:
        end = time.time() + timeout
        while self.q.unfinished_tasks and time.time() < end:
            time.sleep(0.05)
