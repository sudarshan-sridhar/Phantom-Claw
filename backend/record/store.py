"""Where the record goes: Tiger Data (PostgreSQL with TimescaleDB) when TIGER_DATABASE_URL is
set, a local SQLite file when it is not. Same five tables either way.

Rows are buffered in memory and written by a background thread, so a slow or missing
database never holds up the WebSocket reader. If a write fails the rows stay in the buffer
and we try again a little later.
"""
import json
import logging
import sqlite3
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("record")

ROOT = Path(__file__).resolve().parents[2]
SQLITE_PATH = ROOT / "backend" / "cache" / "record.sqlite"

# column name -> (sqlite type, postgres type). time is always first.
TABLES = {
    "events": {"time": ("REAL", "TIMESTAMPTZ NOT NULL"), "type": ("TEXT", "TEXT"), "agent": ("TEXT", "TEXT"),
               "skill": ("TEXT", "TEXT"), "phase": ("TEXT", "TEXT"), "text": ("TEXT", "TEXT"),
               "data": ("TEXT", "JSONB")},
    "joints": {"time": ("REAL", "TIMESTAMPTZ NOT NULL"), "joint": ("TEXT", "TEXT"),
               "position": ("REAL", "DOUBLE PRECISION")},
    "requests": {"time": ("REAL", "TIMESTAMPTZ NOT NULL"), "agent": ("TEXT", "TEXT"), "text": ("TEXT", "TEXT"),
                 "provider": ("TEXT", "TEXT"), "reply_ms": ("INTEGER", "INTEGER"),
                 "arm_skill": ("TEXT", "TEXT"), "arm_result": ("TEXT", "TEXT"), "arm_ms": ("INTEGER", "INTEGER")},
    "brain": {"time": ("REAL", "TIMESTAMPTZ NOT NULL"), "state": ("TEXT", "TEXT"), "value": ("REAL", "DOUBLE PRECISION")},
    # one row per language model answer, as the server reports it in /api/status providers.by_agent.
    # time is when the answer came back. fallback is true when the character's own provider failed
    # and the other one answered.
    "replies": {"time": ("REAL", "TIMESTAMPTZ NOT NULL"), "agent": ("TEXT", "TEXT"), "provider": ("TEXT", "TEXT"),
                "model": ("TEXT", "TEXT"), "fallback": ("INTEGER", "BOOLEAN"), "reply_ms": ("INTEGER", "INTEGER")},
}
MAX_BUFFERED = 200_000   # about an hour of joints at worst; past this the oldest go


class SQLiteDB:
    kind = "sqlite"

    def __init__(self, path: Path = SQLITE_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(str(path), check_same_thread=False, timeout=5)
        self.conn.execute("PRAGMA journal_mode=WAL")
        for name, cols in TABLES.items():
            spec = ", ".join(f"{c} {t[0]}" for c, t in cols.items())
            self.conn.execute(f"CREATE TABLE IF NOT EXISTS {name} ({spec})")
            self.conn.execute(f"CREATE INDEX IF NOT EXISTS {name}_time ON {name} (time)")
            # a file made by an older version may lack columns added since; add them empty
            have = {r[1] for r in self.conn.execute(f"PRAGMA table_info({name})")}
            for c, t in cols.items():
                if c not in have:
                    self.conn.execute(f"ALTER TABLE {name} ADD COLUMN {c} {t[0]}")
        self.conn.commit()

    def write(self, table: str, rows: list[dict]) -> None:
        cols = list(TABLES[table])
        vals = [[json.dumps(r.get(c)) if c == "data" and r.get(c) is not None else r.get(c) for c in cols] for r in rows]
        self.conn.executemany(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", vals)
        self.conn.commit()

    def select(self, table: str, since: float) -> list[dict]:
        cur = self.conn.execute(f"SELECT * FROM {table} WHERE time >= ? ORDER BY time", (since,))
        names = [d[0] for d in cur.description]
        rows = [dict(zip(names, r)) for r in cur.fetchall()]
        for r in rows:
            if r.get("data"):
                r["data"] = json.loads(r["data"])
        return rows

    def counts(self) -> dict:
        return {t: self.conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in TABLES}


class PostgresDB:
    kind = "postgres"

    def __init__(self, url: str):
        import psycopg  # only needed when Tiger Data is switched on
        self.psycopg = psycopg
        self.url = url
        self.conn = None
        self.hypertables = False
        self._connect()

    def _connect(self):
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:  # noqa: BLE001
                pass
        self.conn = self.psycopg.connect(self.url, autocommit=True, connect_timeout=5)
        self.hypertables = self._make_tables()

    def _exec(self, sql: str, args=None) -> bool:
        try:
            with self.conn.cursor() as cur:
                cur.execute(sql, args)
            return True
        except self.psycopg.Error as e:
            log.debug("record: %s failed: %s", sql.split("(")[0], str(e).splitlines()[0][:120])
            return False

    def _make_tables(self) -> bool:
        """Create the tables as hypertables. Tiger Data's own syntax first, then the
        older create_hypertable call, and on a plain PostgreSQL they stay ordinary tables."""
        self._exec("CREATE EXTENSION IF NOT EXISTS timescaledb")
        for name, cols in TABLES.items():
            spec = ", ".join(f"{c} {t[1]}" for c, t in cols.items())
            with self.conn.cursor() as cur:
                cur.execute("SELECT to_regclass(%s)", (name,))
                exists = cur.fetchone()[0] is not None
            if not exists:
                if not self._exec(f"CREATE TABLE {name} ({spec}) WITH (tsdb.hypertable, tsdb.orderby = 'time DESC')"):
                    self._exec(f"CREATE TABLE IF NOT EXISTS {name} ({spec})")
            else:
                # a table made by an older version may lack columns added since
                for c, t in cols.items():
                    if c != "time":
                        self._exec(f"ALTER TABLE {name} ADD COLUMN IF NOT EXISTS {c} {t[1]}")
            self._exec("SELECT create_hypertable(%s, 'time', if_not_exists => TRUE, migrate_data => TRUE)", (name,))
            self._exec(f"CREATE INDEX IF NOT EXISTS {name}_time ON {name} (time DESC)")
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT hypertable_name FROM timescaledb_information.hypertables")
                found = {r[0] for r in cur.fetchall()}
        except self.psycopg.Error:
            found = set()
        ok = set(TABLES) <= found
        log.info("record: Tiger Data tables ready, %s", "all hypertables" if ok else f"hypertables: {sorted(found & set(TABLES)) or 'none'}")
        return ok

    def _retry(self, fn):
        try:
            return fn()
        except self.psycopg.OperationalError:
            self._connect()  # one fresh try after a dropped connection
            return fn()

    def write(self, table: str, rows: list[dict]) -> None:
        from psycopg.types.json import Jsonb
        cols = list(TABLES[table])

        def conv(r, c):
            v = r.get(c)
            if c == "time":
                return datetime.fromtimestamp(v, timezone.utc)
            if c == "data" and v is not None:
                return Jsonb(v)
            return v

        vals = [[conv(r, c) for c in cols] for r in rows]

        def go():
            with self.conn.cursor() as cur:
                cur.executemany(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))})", vals)
        self._retry(go)

    def select(self, table: str, since: float) -> list[dict]:
        cols = [c for c in TABLES[table] if c != "time"]

        def go():
            with self.conn.cursor() as cur:
                cur.execute(f"SELECT extract(epoch from time)::float8, {', '.join(cols)} FROM {table} "
                            f"WHERE time >= to_timestamp(%s) ORDER BY time", (since,))
                return [dict(zip(["time"] + cols, r)) for r in cur.fetchall()]
        return self._retry(go)

    def counts(self) -> dict:
        def go():
            with self.conn.cursor() as cur:
                out = {}
                for t in TABLES:
                    cur.execute(f"SELECT count(*) FROM {t}")
                    out[t] = cur.fetchone()[0]
                return out
        return self._retry(go)


class Store:
    """Buffer in front of the database. add() never blocks and never raises.

    With a Tiger Data URL the rows go there. If Tiger Data cannot be reached they go to the
    local SQLite file instead, so nothing is lost, and Tiger Data is tried again every
    RETRY_S. When it comes back, the rows written locally during the outage are copied up.
    """

    RETRY_S = 15.0

    def __init__(self, url: str | None = None, sqlite_path: Path = SQLITE_PATH, flush_s: float | None = None):
        self.url = url
        self.sqlite_path = sqlite_path
        self.pg = None
        self.lite = None
        self.active = None              # "tiger" or "sqlite": where rows are going right now
        self.outage_since = None        # first row time that went to SQLite instead of Tiger
        self.next_pg_try = 0.0
        self.ok = False
        self.last_error = None
        # Tiger is a shared free tier, so write in batches every two seconds there
        self.flush_s = flush_s if flush_s is not None else (2.0 if url else 1.0)
        self.buf: dict[str, deque] = {t: deque() for t in TABLES}
        self.written = {"tiger": {t: 0 for t in TABLES}, "sqlite": {t: 0 for t in TABLES}}
        self.lock = threading.Lock()      # guards the buffers
        self.db_lock = threading.Lock()   # one database call at a time
        self._stop = threading.Event()
        self._thread = None

    @property
    def kind(self) -> str:
        return "postgres" if self.url else "sqlite"

    def start(self) -> "Store":
        self._thread = threading.Thread(target=self._loop, daemon=True, name="record-writer")
        self._thread.start()
        return self

    def add(self, table: str, row: dict) -> None:
        with self.lock:
            q = self.buf[table]
            q.append(row)
            if len(q) > MAX_BUFFERED:
                q.popleft()

    def _sqlite(self) -> SQLiteDB:
        if self.lite is None:
            self.lite = SQLiteDB(self.sqlite_path)
        return self.lite

    def _tiger(self):
        """The Tiger connection, or None while it is down and not yet due for a retry."""
        if not self.url:
            return None
        if self.pg is not None:
            return self.pg
        if time.time() < self.next_pg_try:
            return None
        try:
            self.pg = PostgresDB(self.url)
            return self.pg
        except Exception as e:  # noqa: BLE001
            self._tiger_down(e)
            return None

    def _tiger_down(self, e: Exception) -> None:
        msg = f"{type(e).__name__}: {str(e).splitlines()[0][:160] if str(e) else ''}"
        if self.active != "sqlite" or self.last_error is None:
            log.warning("record: Tiger Data not reachable, writing to local SQLite instead (%s)", msg)
        self.last_error = msg
        self.pg = None
        self.next_pg_try = time.time() + self.RETRY_S

    def _take(self, table: str) -> list[dict]:
        with self.lock:
            return list(self.buf[table])

    def _drop(self, table: str, n: int) -> None:
        with self.lock:
            for _ in range(n):
                self.buf[table].popleft()

    def _backfill(self, pg) -> None:
        """Copy what went to SQLite during an outage up to Tiger Data."""
        since = self.outage_since
        copied = {}
        for table in TABLES:
            rows = self._sqlite().select(table, since)
            for i in range(0, len(rows), 5000):
                pg.write(table, rows[i:i + 5000])
            copied[table] = len(rows)
        self.outage_since = None
        log.info("record: Tiger Data is back, copied up the rows from the outage %s", copied)

    def flush(self) -> None:
        with self.db_lock:
            pg = self._tiger()
            if pg is not None:
                try:
                    if self.outage_since is not None:
                        self._backfill(pg)
                    for table in TABLES:
                        rows = self._take(table)
                        if rows:
                            pg.write(table, rows)
                            self._drop(table, len(rows))
                            self.written["tiger"][table] += len(rows)
                    if self.active != "tiger":
                        log.info("record: writing to Tiger Data")
                    self.active, self.ok, self.last_error = "tiger", True, None
                    return
                except Exception as e:  # noqa: BLE001
                    self._tiger_down(e)
            # no URL, or Tiger is down: the local file
            lite = self._sqlite()
            for table in TABLES:
                rows = self._take(table)
                if rows:
                    if self.url and self.outage_since is None:
                        self.outage_since = min(r["time"] for r in rows)
                    lite.write(table, rows)
                    self._drop(table, len(rows))
                    self.written["sqlite"][table] += len(rows)
            if self.active != "sqlite" and not self.url:
                log.info("record: writing to local SQLite %s", self.sqlite_path)
            self.active = "sqlite"
            self.ok = not self.url   # with a URL set, falling back counts as not ok

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.flush()
            except Exception as e:  # noqa: BLE001
                # even the local file failed (disk full, locked): rows wait in memory
                if self.last_error is None:
                    log.warning("record: could not write anywhere, holding rows in memory (%s)", type(e).__name__)
                self.ok, self.last_error = False, f"{type(e).__name__}"
            self._stop.wait(self.flush_s)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        try:
            self.flush()
        except Exception:  # noqa: BLE001
            pass

    def select(self, table: str, since: float) -> list[dict]:
        """Rows since a unix time from wherever rows are going now, plus anything still
        buffered. During an outage that is the local file, which has the recent rows."""
        rows = []
        with self.db_lock:  # held across both reads so a flush cannot slip in between
            try:
                db = self.pg if (self.active == "tiger" and self.pg is not None) else self._sqlite()
                rows = db.select(table, since)
            except Exception as e:  # noqa: BLE001
                if self.active == "tiger":
                    self._tiger_down(e)
            with self.lock:
                rows += [dict(r) for r in self.buf[table] if r["time"] >= since]
        return rows

    def status(self) -> dict:
        with self.lock:
            buffered = {t: len(q) for t, q in self.buf.items()}
        return {"configured": "tiger" if self.url else "sqlite", "backend": self.active, "ok": self.ok,
                "error": self.last_error, "buffered": buffered, "written": self.written,
                "hypertables": getattr(self.pg, "hypertables", None)}
