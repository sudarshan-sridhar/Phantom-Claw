"""Listens to the server and turns what it hears into rows.

Every WebSocket event goes into `events` as it is. On top of that a request row is built
from the request, merlin, say and arm events of one job, brain events also go into `brain`,
and GET /api/status is polled once a second for the joint readings, for which LLM
provider and model answered each character and how long it took (a `replies` row per answer),
and for which character holds the arm.
"""
import asyncio
import json
import logging
import time

import httpx
import websockets

from .store import Store

log = logging.getLogger("record")

NOT_A_JOB = {"gesture"}   # body language while talking; every other skill is an arm job
REQUEST_TIMEOUT_S = 90.0


def outcome(skill: str, phases: list[str]) -> str:
    """How an arm job ended, in the words the charts use."""
    if "missed" in phases:
        return "missed"
    if any("hand" in p for p in phases):
        return "handed over"
    if skill in ("point", "half_reach"):
        return "refused"       # pointing or stopping halfway instead of doing the job
    if "error" in phases:
        return "error"
    if "put_back" in phases:
        return "put back"
    if "done" in phases:
        return "done"
    return "stopped"


class RequestTracker:
    """Follows one job at a time from its request event to the end of its arm run."""

    def __init__(self, store: Store):
        self.store = store
        self.cur = None

    def _start(self, ev: dict, t: float) -> None:
        self.cur = {"time": t, "text": ev.get("text"), "agent": ev.get("named_agent"), "provider": None,
                    "reply_ms": None, "arm_skill": None, "arm_t0": None, "arm_ms": None, "phases": [],
                    "said": False, "arm_done": False}

    def finish(self) -> None:
        c, self.cur = self.cur, None
        if c is None:
            return
        result = None
        if c["arm_skill"]:
            result = outcome(c["arm_skill"], c["phases"])
            if c["arm_ms"] is None and c["arm_t0"] is not None:
                c["arm_ms"] = int((time.time() - c["arm_t0"]) * 1000)
        self.store.add("requests", {"time": c["time"], "agent": c["agent"], "text": c["text"],
                                    "provider": c["provider"], "reply_ms": c["reply_ms"],
                                    "arm_skill": c["arm_skill"], "arm_result": result, "arm_ms": c["arm_ms"]})

    def on_event(self, ev: dict, t: float) -> None:
        kind = ev.get("type")
        if kind == "request":
            self.finish()
            self._start(ev, t)
            return
        c = self.cur
        if c is None:
            return
        if kind == "merlin" and ev.get("chosen"):
            c["agent"] = ev["chosen"]
        elif kind == "say" and not c["said"]:
            c["said"] = True
            c["agent"] = c["agent"] or ev.get("agent")
        elif kind == "arm" and ev.get("skill") and ev["skill"] not in NOT_A_JOB and ev.get("agent") == c["agent"]:
            phase = ev.get("phase") or ""
            if c["arm_skill"] is None:
                c["arm_skill"], c["arm_t0"] = ev["skill"], t
            c["phases"].append(phase)
            if phase in ("done", "error") and c["arm_ms"] is None:
                c["arm_ms"] = int((t - c["arm_t0"]) * 1000)
                c["arm_done"] = True
        elif kind == "agent_state" and ev.get("state") == "idle" and ev.get("agent") == c["agent"] and c["said"]:
            if c["arm_skill"] is None or c["arm_done"]:
                self.finish()

    def on_providers(self, providers: dict) -> None:
        """Take the first answer from the chosen agent's provider that came after the request."""
        c = self.cur
        if c is None or c["provider"] or not c["agent"]:
            return
        info = (providers.get("by_agent") or {}).get(c["agent"]) or {}
        if info.get("provider") and (info.get("at") or 0) >= c["time"] - 0.5:
            c["provider"], c["reply_ms"] = info["provider"], info.get("ms")

    def check_timeout(self) -> None:
        if self.cur and time.time() - self.cur["time"] > REQUEST_TIMEOUT_S:
            self.finish()


class Collector:
    def __init__(self, store: Store, server: str = "http://127.0.0.1:8765"):
        self.store = store
        self.server = server.rstrip("/")
        self.ws_url = self.server.replace("http", "ws", 1) + "/ws"
        self.tracker = RequestTracker(store)
        self.connected = False
        self.events_seen = 0
        self.last_status_ok = None
        self.arm_holder = None    # the character whose turn it is with the arm, if any
        self.reply_at = {}        # agent -> "at" of the last answer already written to replies

    def on_replies(self, by_agent: dict) -> None:
        """Write one replies row for every answer not seen before. The server keeps only the
        latest answer per character, so a new "at" means a new answer."""
        for agent, info in (by_agent or {}).items():
            if not isinstance(info, dict) or not info.get("provider"):
                continue
            at = info.get("at")
            if not isinstance(at, (int, float)) or self.reply_at.get(agent) == at:
                continue
            if agent not in self.reply_at and self._already_recorded(agent, at):
                # first look this run at an answer an earlier run of the record already wrote
                self.reply_at[agent] = at
                continue
            self.reply_at[agent] = at
            ms = info.get("ms")
            self.store.add("replies", {"time": float(at), "agent": agent, "provider": info["provider"],
                                       "model": info.get("model"), "fallback": bool(info.get("fallback")),
                                       "reply_ms": int(ms) if isinstance(ms, (int, float)) else None})

    def _already_recorded(self, agent: str, at: float) -> bool:
        rows = self.store.select("replies", at - 0.01)
        return any(r.get("agent") == agent and abs(r["time"] - at) < 0.01 for r in rows)

    def on_status(self, st: dict) -> None:
        """Everything taken from one GET /api/status answer."""
        t = time.time()
        joints = ((st.get("arm") or {}).get("joints")) or {}
        for name, pos in joints.items():
            if isinstance(pos, (int, float)):
                self.store.add("joints", {"time": t, "joint": name, "position": float(pos)})
        states = st.get("states")
        if isinstance(states, dict):
            self.arm_holder = next((a for a, s in states.items() if s == "has_arm"), None)
        providers = st.get("providers") or {}
        self.on_replies(providers.get("by_agent") or {})
        self.tracker.on_providers(providers)

    def handle(self, ev: dict, t: float | None = None) -> None:
        t = time.time() if t is None else t
        self.events_seen += 1
        kind = ev.get("type")
        text = ev.get("text") or ev.get("say") or ev.get("message") or ev.get("label")
        agent = ev.get("agent") or ev.get("chosen") or ev.get("named_agent")
        self.store.add("events", {"time": t, "type": kind, "agent": agent, "skill": ev.get("skill"),
                                  "phase": ev.get("phase") or ev.get("state"), "text": text, "data": ev})
        if kind == "brain":
            v = ev.get("value")
            self.store.add("brain", {"time": t, "state": ev.get("state"),
                                     "value": float(v) if isinstance(v, (int, float)) else None})
        elif kind == "agent_state" and ev.get("agent"):
            # the status poll says the same once a second; the event says it sooner
            if ev.get("state") == "has_arm":
                self.arm_holder = ev["agent"]
            elif self.arm_holder == ev["agent"]:
                self.arm_holder = None
        self.tracker.on_event(ev, t)

    async def ws_loop(self) -> None:
        delay = 1.0
        while True:
            try:
                async with websockets.connect(self.ws_url, open_timeout=3, ping_interval=20) as ws:
                    self.connected, delay = True, 1.0
                    log.info("record: listening on %s", self.ws_url)
                    async for msg in ws:
                        try:
                            ev = json.loads(msg)
                        except ValueError:
                            continue
                        if isinstance(ev, dict):
                            self.handle(ev)
                            if ev.get("type") == "say":
                                # the server notes which provider answered just before it sends
                                # say; read it now, before a quick job can end and close the row
                                await self.poll_once()
            except (OSError, websockets.WebSocketException, asyncio.TimeoutError) as e:
                if self.connected:
                    log.warning("record: lost the server WebSocket (%s), reconnecting", type(e).__name__)
                self.connected = False
            await asyncio.sleep(delay)
            delay = min(delay * 2, 5.0)

    async def poll_once(self, client: httpx.AsyncClient | None = None) -> None:
        own = client is None
        client = client or httpx.AsyncClient(timeout=1.0)
        try:
            r = await client.get(self.server + "/api/status")
            r.raise_for_status()
            st = r.json()
        except (httpx.HTTPError, ValueError):
            self.last_status_ok = False
            self.arm_holder = None   # with the server gone nobody holds the arm
            return
        finally:
            if own:
                await client.aclose()
        self.last_status_ok = True
        if isinstance(st, dict):
            self.on_status(st)

    async def status_loop(self) -> None:
        async with httpx.AsyncClient(timeout=1.0) as client:
            while True:
                start = time.monotonic()
                await self.poll_once(client)
                self.tracker.check_timeout()
                await asyncio.sleep(max(0.05, 1.0 - (time.monotonic() - start)))
