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
_in_flight = False      # a request is being handled right now
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
    # One request at a time. Anything said while a job is running is dropped, so a misheard
    # word cannot start a second character talking or moving over the first.
    global _in_flight
    if _in_flight:
        await hub.emit({"type": "ignored", "text": body.text[:200], "why": "busy"})
        return {"ignored": "busy"}
    _in_flight = True
    try:
        return await get_pipeline().handle(body.text, body.agent)
    finally:
        _in_flight = False


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
    get_pipeline().request_stop()     # a desk plan runs no further steps
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


# ---------------------------------------------------------------- voice control (backend/listen)
# Additions only: an event relay for the listener and the brain process, and the shared
# listening state so the listener and the overlay agree on asleep or awake.

EVENT_TYPES = {"hear", "listen", "brain"}
listen_state: dict[str, Any] = {"awake": False, "source": None, "changed_at": None}

_WAKE_LINES = ["We're listening.", "All ears. Go ahead.", "Awake. Who do you need?", "We're here. Go on."]
_SLEEP_LINES = ["Resting now.", "Going quiet.", "Say the word when you need us."]


@app.post("/api/event")
async def api_event(body: dict[str, Any]):
    if body.get("type") not in EVENT_TYPES:
        raise HTTPException(400, "type must be one of: " + ", ".join(sorted(EVENT_TYPES)))
    await hub.emit(dict(body))
    return {"ok": True}


class ListenBody(BaseModel):
    awake: bool
    say: bool = True
    source: str | None = None


@app.get("/api/listen")
async def api_listen_get():
    return {"awake": listen_state["awake"], "source": listen_state["source"], "changed_at": listen_state["changed_at"]}


async def _listen_line(awake: bool) -> None:
    """Merlin says one short line so the owner hears the change worked."""
    import random

    pipeline = get_pipeline()
    line = random.choice(_WAKE_LINES if awake else _SLEEP_LINES)
    clip = await pipeline.speak("merlin", line)
    await hub.emit(
        {"type": "merlin", "chosen": None, "say": line, "audio_url": clip["audio_url"], "duration_s": clip["duration_s"], "argued": []}
    )
    # The arm is the body of whoever speaks, so Merlin gets her gesture too.
    asyncio.ensure_future(pipeline._gesture("merlin", clip["duration_s"], set_states=True))


@app.post("/api/listen")
async def api_listen_set(body: ListenBody):
    import time

    changed = body.awake != listen_state["awake"]
    if changed:
        listen_state.update({"awake": body.awake, "source": body.source, "changed_at": time.time()})
        await hub.emit({"type": "listen", "awake": body.awake, "source": body.source})
        if body.say:
            try:
                await _listen_line(body.awake)
            except Exception as e:  # noqa: BLE001
                await hub.emit({"type": "error", "where": "listen", "message": f"{type(e).__name__}: {e}"[:300]})
    return {"awake": listen_state["awake"], "changed": changed}


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
