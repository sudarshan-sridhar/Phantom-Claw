"""Phantom Claw server: FastAPI on 127.0.0.1:8765.

Run from the project root:
    C:\\Users\\sudar\\.conda\\envs\\lerobot\\python.exe -m backend.server
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.agents import providers, voice
from backend.agents.cast import escanor_mood, public_cast
from backend.agents.config import HOST, PORT
from backend.agents.pipeline import Pipeline, clock_now

log = logging.getLogger("phantom")


def load_arm() -> tuple[Any, str]:
    """The arm module from the other builder, or the stand-in if it is not ready."""
    try:
        from backend.arm import make_arm  # FakeArm unless PHANTOM_ARM=real

        return make_arm(), "backend.arm"
    except Exception as e:  # noqa: BLE001
        log.warning("backend.arm not usable (%s); using the stand-in arm", type(e).__name__)
        from backend.agents._standin_arm import make_arm as standin

        return standin(), "standin"


class Hub:
    """Sends events to every WebSocket client, in order, through one queue."""

    def __init__(self):
        self.clients: set[WebSocket] = set()
        self.queue: asyncio.Queue = asyncio.Queue()
        self.task: asyncio.Task | None = None

    async def emit(self, event: dict) -> None:
        self.queue.put_nowait(event)

    async def pump(self) -> None:
        while True:
            ev = await self.queue.get()
            dead = []
            for ws in list(self.clients):
                try:
                    await ws.send_json(ev)
                except Exception:  # noqa: BLE001
                    dead.append(ws)
            for ws in dead:
                self.clients.discard(ws)


hub = Hub()
state: dict[str, Any] = {"arm": None, "arm_source": None, "pipeline": None}


def get_arm():
    if state["arm"] is None:
        state["arm"], state["arm_source"] = load_arm()
    return state["arm"]


def get_pipeline() -> Pipeline:
    if state["pipeline"] is None:
        state["pipeline"] = Pipeline(get_arm(), hub.emit)
    return state["pipeline"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    hub.task = asyncio.create_task(hub.pump())
    get_pipeline()
    yield
    hub.task.cancel()


app = FastAPI(title="Phantom Claw", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(https?://(localhost|127\.0\.0\.1)(:\d+)?|file://.*|null|app://.*)$",
    allow_methods=["*"],
    allow_headers=["*"],
)


class RequestBody(BaseModel):
    text: str
    agent: str | None = None


@app.post("/api/request")
async def api_request(body: RequestBody):
    if not body.text.strip():
        raise HTTPException(400, "text is empty")
    return await get_pipeline().handle(body.text, body.agent)


@app.post("/api/stt")
async def api_stt(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty audio")
    try:
        text = await voice.stt(data, file.filename or "audio.webm", file.content_type or "application/octet-stream")
    except voice.VoiceError as e:
        await hub.emit({"type": "error", "where": "stt", "message": str(e)[:300]})
        raise HTTPException(502, "speech to text failed")
    return {"text": text}


@app.post("/api/stop")
async def api_stop():
    arm = get_arm()
    await asyncio.to_thread(arm.stop)
    return {"ok": True}


@app.get("/api/cast")
async def api_cast():
    return {"cast": public_cast(get_pipeline().states), "escanor_mood": escanor_mood(clock_now().hour + clock_now().minute / 60)}


@app.get("/api/status")
async def api_status():
    arm = get_arm()
    try:
        arm_status = await asyncio.to_thread(arm.status)
    except Exception as e:  # noqa: BLE001
        arm_status = {"connected": False, "error": str(e)[:200]}
    return {
        "arm": arm_status,
        "arm_source": state["arm_source"],
        "providers": {"last": providers.LAST["last"], "by_agent": providers.LAST["by_agent"]},
        "states": get_pipeline().states,
    }


@app.get("/audio/{file}")
async def audio(file: str):
    p = voice.safe_cache_path(file)
    if p is None:
        raise HTTPException(404, "no such clip")
    return FileResponse(p, media_type="audio/mpeg")


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    hub.clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()  # clients do not need to send anything
    except WebSocketDisconnect:
        pass
    finally:
        hub.clients.discard(websocket)


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
