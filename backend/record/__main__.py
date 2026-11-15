"""python -m backend.record

Records every server event and the arm's joints, and serves the charts on 127.0.0.1:8766.
Writes to Tiger Data when TIGER_DATABASE_URL is set in .env, otherwise to
backend/cache/record.sqlite.
"""
import argparse
import asyncio
import logging
import os

import uvicorn
from dotenv import load_dotenv

from .collector import Collector
from .store import ROOT, Store
from .web import make_app

log = logging.getLogger("record")


async def main_async(server: str, port: int) -> None:
    load_dotenv(ROOT / ".env")
    url = (os.environ.get("TIGER_DATABASE_URL") or "").strip() or None
    store = Store(url).start()
    log.info("record: database is %s", "Tiger Data (TIGER_DATABASE_URL)" if url else "local SQLite backend/cache/record.sqlite")
    collector = Collector(store, server)
    app = make_app(store, collector)
    web = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    log.info("record: charts on http://127.0.0.1:%d/", port)
    tasks = [asyncio.create_task(collector.ws_loop()), asyncio.create_task(collector.status_loop())]
    try:
        await web.serve()
    finally:
        for t in tasks:
            t.cancel()
        collector.tracker.finish()   # keep a job that was still open
        store.stop()
        log.info("record: stopped, %s", store.status()["written"])


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m backend.record")
    p.add_argument("--server", default=os.environ.get("PHANTOM_SERVER", "http://127.0.0.1:8765"))
    p.add_argument("--port", type=int, default=8766)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # it logs every status poll otherwise
    try:
        asyncio.run(main_async(args.server, args.port))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
