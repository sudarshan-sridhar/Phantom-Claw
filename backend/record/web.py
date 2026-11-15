"""The charts page on 127.0.0.1:8766 and the JSON behind it."""
import asyncio
import json
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .store import ROOT, Store

PAGE = Path(__file__).with_name("page.html")
FONTS = ROOT / "overlay" / "assets" / "hd" / "fonts"
AGENTS = ["escanor", "elizabeth", "ban", "king", "gowther"]
CHARACTERS = AGENTS + ["merlin"]
OUTCOMES = ["handed over", "put back", "missed", "refused", "done", "stopped", "error"]
PANEL_HOURS = 24.0


def panel_data(store: Store, now: float | None = None, hours: float = PANEL_HOURS,
               arm_holder: str | None = None, server_up: bool | None = None) -> dict:
    """Who is running what: per character, the provider and model of its last recorded answer,
    how fast it was, and how many answers came from the other provider instead."""
    now = time.time() if now is None else now
    rows = store.select("replies", now - hours * 3600)
    out = []
    for a in CHARACTERS:
        mine = sorted((r for r in rows if r.get("agent") == a), key=lambda r: r["time"])
        last = mine[-1] if mine else None
        out.append({
            "agent": a,
            "replies": len(mine),
            "fallbacks": sum(1 for r in mine if r.get("fallback")),
            "last": None if last is None else {
                "provider": last.get("provider"), "model": last.get("model"),
                "fallback": bool(last.get("fallback")), "ms": last.get("reply_ms"),
                "at": last["time"], "ago_s": round(max(0.0, now - last["time"]), 1)},
            "holds_arm": a == arm_holder,
        })
    return {"now": now, "hours": hours, "server_up": server_up, "arm_holder": arm_holder, "characters": out}


def charts_data(store: Store, minutes: float = 60.0, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    reqs = store.select("requests", now - minutes * 60)
    joints = store.select("joints", now - 60)
    brain = store.select("brain", now - 120)

    jobs = {a: 0 for a in AGENTS}
    arm = []
    replies = {}
    for r in reqs:
        a = r.get("agent")
        if a in jobs:
            jobs[a] += 1
        if r.get("arm_skill") and r.get("arm_ms") is not None:
            arm.append({"t": r["time"], "agent": a, "skill": r["arm_skill"], "ms": r["arm_ms"], "result": r["arm_result"]})
        if r.get("provider") and r.get("reply_ms") is not None:
            replies.setdefault(r["provider"], []).append({"t": r["time"], "agent": a, "ms": r["reply_ms"]})

    arm_summary = {}
    for a in AGENTS:
        mine = [j for j in arm if j["agent"] == a]
        arm_summary[a] = {"n": len(mine),
                          "median_ms": statistics.median([j["ms"] for j in mine]) if mine else None,
                          "results": {o: sum(1 for j in mine if j["result"] == o) for o in OUTCOMES if any(j["result"] == o for j in mine)}}

    traces = {}
    for row in joints:
        traces.setdefault(row["joint"], []).append([round(row["time"] - now, 2), row["position"]])

    focus = [[round(b["time"] - now, 2), b["value"]] for b in brain if b["state"] == "focus" and b["value"] is not None]
    marks = [{"t": round(b["time"] - now, 2), "state": b["state"]} for b in brain if b["state"] in ("stop", "clear", "no_signal")]

    return {
        "now": now,
        "minutes": minutes,
        "store": store.status(),
        "jobs": [{"agent": a, "count": jobs[a]} for a in AGENTS],
        "arm_jobs": arm,
        "arm_summary": arm_summary,
        "replies": {p: {"median_ms": statistics.median([x["ms"] for x in v]), "items": v} for p, v in sorted(replies.items())},
        "joints": traces,
        "brain": {"focus": focus, "marks": marks},
        "requests_seen": len(reqs),
    }


LOGS = ROOT / "backend" / "cache" / "logs"
BRAIN_CMD = ["-u", "-m", "backend.brain", "gtec-run", "--subject", "1", "--live", "--targets", "sharpie"]
BRAIN_PICTURES = ["last_wrist_aim_1_read.jpg", "last_handover_view.jpg"]


def _launch_brain(log_path: Path):
    """Start the brain demo as a hidden child of the record process. Its output goes to the log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    out = open(log_path, "w", encoding="utf-8", errors="replace")
    flags = 0x08000000 if sys.platform == "win32" else 0        # no console window
    return subprocess.Popen([sys.executable] + BRAIN_CMD, cwd=str(ROOT), stdout=out, stderr=subprocess.STDOUT,
                            creationflags=flags)


def _fetch_json(url: str, timeout: float = 1.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:  # noqa: BLE001 - the page it reads is only up while a run is going
        return None


def _post(url: str, timeout: float = 3.0) -> bool:
    try:
        urllib.request.urlopen(urllib.request.Request(url, data=b"", method="POST"), timeout=timeout).read()
        return True
    except Exception:  # noqa: BLE001
        return False


class BrainDemo:
    """The button on the page: runs the recorded g.tec brain signals through the decoder and on
    to the agents and the real arm, one run at a time, and reports what the run is doing."""

    def __init__(self, launcher=_launch_brain, log_path: Path | None = None, fetch=_fetch_json, post=_post,
                 state_url: str = "http://127.0.0.1:8767/state", stop_url: str = "http://127.0.0.1:8765/api/stop",
                 setup_dir: Path | None = None):
        self.launcher, self.fetch, self.post = launcher, fetch, post
        self.log_path = log_path or (LOGS / "gtec.log")
        self.state_url, self.stop_url = state_url, stop_url
        self.setup_dir = setup_dir or (ROOT / "setup")
        self.proc = None
        self.started = None

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def run(self) -> dict:
        if self.running():
            return {"ok": False, "message": "A brain demo is already running. Wait for it to finish, or press Stop."}
        self.proc = self.launcher(self.log_path)
        self.started = time.time()
        return {"ok": True, "message": "Started. Nothing is sent for the first five seconds."}

    def stop(self) -> dict:
        was = self.running()
        if was:
            try:
                self.proc.kill()
            except Exception:  # noqa: BLE001
                pass
        self.proc = None
        arm_stopped = self.post(self.stop_url)
        return {"ok": True, "was_running": was, "arm_stop_sent": arm_stopped}

    def log_lines(self, n: int = 12) -> list[str]:
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        keep = [ln.rstrip() for ln in text.splitlines() if ln.strip() and "data_receiver.cpp" not in ln
                and "netinterfaces.cpp" not in ln and "api_config.cpp" not in ln and "common.cpp" not in ln]
        return [ln[:220] for ln in keep[-n:]]

    def picture(self) -> Path | None:
        found = [self.setup_dir / n for n in BRAIN_PICTURES if (self.setup_dir / n).is_file()]
        return max(found, key=lambda p: p.stat().st_mtime) if found else None

    def state(self) -> dict:
        running = self.running()
        pic = self.picture()
        return {"running": running, "started": self.started, "log": self.log_lines(),
                "page": self.fetch(self.state_url) if running else None,
                "picture": None if pic is None else {"name": pic.name, "at": pic.stat().st_mtime}}


def make_app(store: Store, collector=None, brain: BrainDemo | None = None) -> FastAPI:
    app = FastAPI(title="Phantom Claw record")
    if FONTS.exists():
        # mounted at the same relative place the page's CSS points to, so the page also
        # finds the fonts when opened straight from disk
        app.mount("/overlay/assets/hd/fonts", StaticFiles(directory=str(FONTS)), name="fonts")

    @app.get("/")
    async def page():
        return FileResponse(PAGE, media_type="text/html")

    def panel():
        holder = getattr(collector, "arm_holder", None)
        up = getattr(collector, "last_status_ok", None)
        return panel_data(store, arm_holder=holder, server_up=up)

    @app.get("/api/charts")
    async def charts(minutes: float = 60.0):
        data = await asyncio.to_thread(charts_data, store, minutes)
        data["panel"] = await asyncio.to_thread(panel)
        if collector is not None:
            data["server"] = {"websocket": collector.connected, "status_poll": collector.last_status_ok,
                              "events_seen": collector.events_seen}
        return JSONResponse(data)

    @app.get("/api/panel")
    async def panel_only():
        return JSONResponse(await asyncio.to_thread(panel))

    brain = brain or BrainDemo()

    @app.post("/api/brain/run")
    async def brain_run():
        return JSONResponse(await asyncio.to_thread(brain.run))

    @app.post("/api/brain/stop")
    async def brain_stop():
        return JSONResponse(await asyncio.to_thread(brain.stop))

    @app.get("/api/brain/state")
    async def brain_state():
        return JSONResponse(await asyncio.to_thread(brain.state))

    @app.get("/api/brain/picture")
    async def brain_picture():
        pic = brain.picture()
        if pic is None:
            return JSONResponse({"picture": None}, status_code=404)
        return FileResponse(pic, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    return app