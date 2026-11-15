"""The charts page on 127.0.0.1:8766 and the JSON behind it."""
import asyncio
import statistics
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .store import ROOT, Store

PAGE = Path(__file__).with_name("page.html")
FONTS = ROOT / "overlay" / "assets" / "hd" / "fonts"
AGENTS = ["escanor", "elizabeth", "ban", "king", "gowther"]
OUTCOMES = ["handed over", "put back", "missed", "refused", "done", "stopped", "error"]


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


def make_app(store: Store, collector=None) -> FastAPI:
    app = FastAPI(title="Phantom Claw record")
    if FONTS.exists():
        # mounted at the same relative place the page's CSS points to, so the page also
        # finds the fonts when opened straight from disk
        app.mount("/overlay/assets/hd/fonts", StaticFiles(directory=str(FONTS)), name="fonts")

    @app.get("/")
    async def page():
        return FileResponse(PAGE, media_type="text/html")

    @app.get("/api/charts")
    async def charts(minutes: float = 60.0):
        data = await asyncio.to_thread(charts_data, store, minutes)
        if collector is not None:
            data["server"] = {"websocket": collector.connected, "status_poll": collector.last_status_ok,
                              "events_seen": collector.events_seen}
        return JSONResponse(data)

    return app
