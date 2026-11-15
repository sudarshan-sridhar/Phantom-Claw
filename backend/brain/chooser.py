"""The chooser page and the small local server behind it (default port 8767).

The page measures the screen's refresh rate, flickers each tile at a whole number of frames
per cycle, and polls /api/state a few times a second to show the progress bar, what was
chosen and whether the stop is on. Nothing flickers until Start is pressed; Esc stops it.

Endpoints (all JSON, local only):
  GET  /                 the page
  GET  /api/state        everything the page shows
  POST /api/display      {"refresh": 59.94}   the page reports the measured refresh rate
  POST /api/flicker      {"on": true|false}   Start pressed, or Esc / tab hidden
  POST /api/sim          {"gaze": index|null, "eyes_closed": bool}   simulator only
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = Path(__file__).with_name("chooser.html")


class ChooserState:
    def __init__(self, items: list[str], mode: str, sim=None):
        self.lock = threading.Lock()
        self.items = list(items)
        self.mode = mode                 # "calibrate" or "run"
        self.sim = sim                   # SimSource or None
        self.refresh: float | None = None
        self.frames: list[int] | None = None   # frames per cycle for each tile
        self.rates_hz: list[float] = []
        self.flicker = False
        self.highlight = None            # tile index, "centre", or None
        self.prompt = ""
        self.progress = {"index": None, "value": 0.0}
        self.chosen = None               # {"item", "at", "sent"}
        self.stopped = False
        self.status = "Press Start when you are ready."
        self.warning = ""
        self.gaze = None                 # simulator: which tile is being looked at
        self.eyes_closed = False         # simulator
        self.display_event = threading.Event()

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "mode": self.mode, "items": self.items, "frames": self.frames,
                "rates_hz": [round(r, 3) for r in self.rates_hz], "refresh": self.refresh,
                "flicker": self.flicker, "highlight": self.highlight, "prompt": self.prompt,
                "progress": self.progress, "chosen": self.chosen, "stopped": self.stopped,
                "status": self.status, "warning": self.warning, "sim": self.sim is not None,
                "gaze": self.gaze, "eyes_closed": self.eyes_closed, "now": time.time(),
            }

    def set(self, **kw) -> None:
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def set_gaze(self, gaze, eyes_closed=None) -> None:
        """Simulator only: point the simulated eyes at a tile (or nowhere), open or close them."""
        with self.lock:
            if gaze is not None and not (isinstance(gaze, int) and 0 <= gaze < len(self.items)):
                gaze = None
            self.gaze = gaze
            if eyes_closed is not None:
                self.eyes_closed = bool(eyes_closed)
            rates = list(self.rates_hz)
            sim, closed = self.sim, self.eyes_closed
        if sim is not None:
            if gaze is None or gaze >= len(rates):
                sim.set(clear_gaze=True, eyes_closed=closed)
            else:
                sim.set(gaze_hz=rates[gaze], eyes_closed=closed)


def _handler(state: ChooserState):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # keep the terminal for the reader's own lines
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode("utf-8"))

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/state":
                self._json(state.snapshot())
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._json({"error": "bad json"}, 400)
            path = self.path.split("?", 1)[0]
            if path == "/api/display":
                try:
                    hz = float(body.get("refresh"))
                except (TypeError, ValueError):
                    return self._json({"error": "refresh missing"}, 400)
                if not 30.0 <= hz <= 400.0:
                    return self._json({"error": "refresh out of range"}, 400)
                state.set(refresh=hz)
                state.display_event.set()
                return self._json(state.snapshot())
            if path == "/api/flicker":
                state.set(flicker=bool(body.get("on")))
                return self._json({"ok": True})
            if path == "/api/sim":
                if state.sim is None:
                    return self._json({"error": "not in simulator mode"}, 400)
                g = body.get("gaze")
                state.set_gaze(int(g) if isinstance(g, (int, float)) else None, body.get("eyes_closed"))
                return self._json({"ok": True})
            self._json({"error": "not found"}, 404)

    return H


def serve(state: ChooserState, port: int) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _handler(state))
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd
