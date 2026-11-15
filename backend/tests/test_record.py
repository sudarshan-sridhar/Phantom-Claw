"""Offline tests for the record: table writes, request rows built from events, chart data."""
import sqlite3
import time

from backend.record.collector import Collector, outcome
from backend.record.store import Store
from backend.record.web import charts_data, panel_data


def make_store(tmp_path):
    return Store(None, sqlite_path=tmp_path / "rec.sqlite")


def one_job(col, t0, agent="ban", skill="grip_bottle", phases=("start", "at_target", "gripped", "lifted", "put_back", "returning", "done")):
    col.handle({"type": "request", "text": "grab the bottle", "named_agent": None}, t0)
    col.handle({"type": "merlin", "chosen": agent, "say": "Yours.", "audio_url": "x", "argued": []}, t0 + 0.5)
    col.handle({"type": "agent_state", "agent": agent, "state": "thinking"}, t0 + 0.6)
    col.tracker.on_providers({"by_agent": {agent: {"provider": "gemini", "ms": 812, "at": t0 + 1.4}}})
    col.handle({"type": "say", "agent": agent, "text": "Mine.", "audio_url": "y"}, t0 + 1.5)
    col.handle({"type": "agent_state", "agent": agent, "state": "has_arm"}, t0 + 2.0)
    for i, p in enumerate(phases):
        col.handle({"type": "arm", "agent": agent, "skill": skill, "phase": p}, t0 + 2.0 + i)
    col.handle({"type": "agent_state", "agent": agent, "state": "idle"}, t0 + 2.0 + len(phases))


def test_tables_are_written_to_sqlite(tmp_path):
    store = make_store(tmp_path)
    col = Collector(store)
    t0 = time.time() - 30
    one_job(col, t0)
    col.handle({"type": "brain", "state": "focus", "value": 0.62}, t0 + 12)
    store.add("joints", {"time": t0 + 13, "joint": "gripper", "position": 1547.0})
    store.flush()

    db = sqlite3.connect(tmp_path / "rec.sqlite")
    assert db.execute("SELECT count(*) FROM events").fetchone()[0] == 14   # 13 job events and one brain event
    req = db.execute("SELECT agent, text, provider, reply_ms, arm_skill, arm_result, arm_ms FROM requests").fetchall()
    assert req == [("ban", "grab the bottle", "gemini", 812, "grip_bottle", "put back", 6000)]
    assert db.execute("SELECT state, value FROM brain").fetchall() == [("focus", 0.62)]
    assert db.execute("SELECT joint, position FROM joints").fetchall() == [("gripper", 1547.0)]
    # the whole event is kept as JSON too
    data = db.execute("SELECT data FROM events WHERE type='say'").fetchone()[0]
    assert '"Mine."' in data


def test_gestures_are_not_counted_as_arm_jobs(tmp_path):
    store = make_store(tmp_path)
    col = Collector(store)
    t0 = time.time() - 20
    col.handle({"type": "request", "text": "hello", "named_agent": "king"}, t0)
    col.handle({"type": "say", "agent": "king", "text": "Hm.", "audio_url": "z"}, t0 + 1)
    for p in ("start", "returning", "done"):
        col.handle({"type": "arm", "agent": "king", "skill": "gesture", "phase": p}, t0 + 2)
    col.handle({"type": "agent_state", "agent": "king", "state": "idle"}, t0 + 3)
    store.flush()
    rows = store.select("requests", 0)
    assert len(rows) == 1 and rows[0]["agent"] == "king" and rows[0]["arm_skill"] is None


def test_a_desk_plan_is_one_arm_job(tmp_path):
    store = make_store(tmp_path)
    col = Collector(store)
    t0 = time.time() - 60
    col.handle({"type": "request", "text": "hand me the sharpie", "named_agent": "ban"}, t0)
    col.handle({"type": "say", "agent": "ban", "text": "Sure.", "audio_url": "y"}, t0 + 1)
    col.handle({"type": "agent_state", "agent": "ban", "state": "has_arm"}, t0 + 1.5)
    col.handle({"type": "arm_plan", "agent": "ban", "phase": "start", "steps": []}, t0 + 2)
    steps = []
    at = t0 + 2

    def step(i, skill, result, ok, **flags):
        nonlocal at
        for p in ("start", "done"):
            at += 1
            col.handle({"type": "arm", "agent": "ban", "skill": skill, "phase": p, "step": i, "steps": 3, **flags}, at)
        rec = {"step": i, "skill": skill, "params": {}, "result": result, "ok": ok, "detail": "", **flags}
        steps.append(rec)
        col.handle({"type": "arm_step", "agent": "ban", **rec}, at)
    step(0, "look", "done", True)
    step(1, "pick", "holding", True)
    step(2, "hand_over", "not_taken", False)
    step(2, "place", "placed", True, recovery=True)
    col.handle({"type": "arm_plan", "agent": "ban", "phase": "done", "result": "not_taken", "ok": False,
                "steps": steps}, at + 1)
    col.handle({"type": "agent_state", "agent": "ban", "state": "idle"}, at + 2)
    store.flush()
    rows = store.select("requests", 0)
    assert len(rows) == 1
    assert (rows[0]["arm_skill"], rows[0]["arm_result"], rows[0]["arm_ms"]) == ("plan", "put back", 9000)
    # stopped wins; a single skill job is recorded as before
    from backend.record.collector import plan_outcome
    assert plan_outcome(steps[:2] + [{"skill": "hand_over", "result": "stopped", "ok": False}], "stopped") == "stopped"
    assert plan_outcome([{"skill": "pick", "result": "missed", "ok": False}] * 2, "missed") == "missed"
    assert plan_outcome(steps[:2] + [{"skill": "hand_over", "result": "handed_over", "ok": True}]) == "handed over"


def test_outcomes():
    assert outcome("grip_bottle", ["start", "at_target", "missed", "returning", "done"]) == "missed"
    assert outcome("grip_bottle", ["start", "gripped", "lifted", "put_back", "done"]) == "put back"
    assert outcome("grip_bottle", ["start", "gripped", "handed_over", "done"]) == "handed over"
    assert outcome("point", ["start", "at_target", "returning", "done"]) == "refused"
    assert outcome("grip_bottle", ["start", "at_target"]) == "stopped"


def test_charts_data_summarises(tmp_path):
    store = make_store(tmp_path)
    col = Collector(store)
    now = time.time()
    one_job(col, now - 300, "ban")
    one_job(col, now - 200, "elizabeth", phases=("start", "at_target", "missed", "returning", "done"))
    one_job(col, now - 100, "escanor", skill="point", phases=("start", "at_target", "returning", "done"))
    for i in range(10):
        store.add("joints", {"time": now - 10 + i, "joint": "elbow_flex", "position": 2400.0 + i})
    store.add("brain", {"time": now - 50, "state": "stop", "value": None})
    store.add("brain", {"time": now - 45, "state": "clear", "value": None})
    store.flush()

    d = charts_data(store, minutes=60, now=now)
    jobs = {j["agent"]: j["count"] for j in d["jobs"]}
    assert jobs == {"escanor": 1, "elizabeth": 1, "ban": 1, "king": 0, "gowther": 0}
    assert {j["agent"]: j["result"] for j in d["arm_jobs"]} == {"ban": "put back", "elizabeth": "missed", "escanor": "refused"}
    assert d["replies"]["gemini"]["median_ms"] == 812
    assert len(d["joints"]["elbow_flex"]) == 10
    assert [m["state"] for m in d["brain"]["marks"]] == ["stop", "clear"]


def test_unreachable_tiger_falls_back_to_sqlite(tmp_path):
    # nothing listens on port 1, so the connection is refused
    store = Store("postgresql://nobody:nothing@127.0.0.1:1/none", sqlite_path=tmp_path / "fb.sqlite")
    store.add("brain", {"time": time.time(), "state": "focus", "value": 0.5})
    store.flush()   # must not raise
    st = store.status()
    assert st["configured"] == "tiger" and st["backend"] == "sqlite" and not st["ok"] and st["error"]
    assert st["buffered"]["brain"] == 0 and st["written"]["sqlite"]["brain"] == 1
    assert len(store.select("brain", 0)) == 1   # the charts read the local copy


def status(by_agent=None, states=None):
    """A fake GET /api/status answer."""
    return {"arm": {"joints": {}}, "providers": {"last": None, "by_agent": by_agent or {}},
            "states": states or {a: "idle" for a in ("escanor", "elizabeth", "ban", "king", "gowther", "merlin")}}


def test_replies_table_added_to_an_old_database(tmp_path):
    path = tmp_path / "old.sqlite"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE requests (time REAL, agent TEXT, text TEXT, provider TEXT, reply_ms INTEGER)")
    db.execute("CREATE TABLE replies (time REAL, agent TEXT, provider TEXT)")   # an early half-made one
    db.execute("INSERT INTO requests VALUES (1.0, 'ban', 'hi', 'gemini', 500)")
    db.commit()
    db.close()
    store = Store(None, sqlite_path=path)
    store.add("replies", {"time": 2.0, "agent": "ban", "provider": "openai", "model": "gpt-4o-mini",
                          "fallback": True, "reply_ms": 640})
    store.flush()
    db = sqlite3.connect(path)
    cols = {r[1] for r in db.execute("PRAGMA table_info(requests)")}
    assert {"arm_skill", "arm_result", "arm_ms"} <= cols
    assert {r[1] for r in db.execute("PRAGMA table_info(replies)")} == {"time", "agent", "provider", "model", "fallback", "reply_ms"}
    assert db.execute("SELECT agent, model, fallback, reply_ms FROM replies").fetchall() == [("ban", "gpt-4o-mini", 1, 640)]
    assert db.execute("SELECT text, provider FROM requests").fetchall() == [("hi", "gemini")]   # old rows kept


def test_each_reply_recorded_once(tmp_path):
    store = make_store(tmp_path)
    col = Collector(store)
    t = time.time() - 10
    a = {"provider": "anthropic", "model": "claude-haiku-5-5", "fallback": False, "ms": 900, "errors": [], "at": t}
    m = {"provider": "gemini", "model": "gemini-3.5-flash-lite", "fallback": False, "ms": 300, "errors": [], "at": t - 1}
    for _ in range(3):   # the same answers seen on three polls
        col.on_status(status({"escanor": a, "merlin": m}))
    col.on_status(status({"escanor": {**a, "ms": 700, "at": t + 5}, "merlin": m}))
    store.flush()
    rows = store.select("replies", 0)
    assert sorted((r["agent"], r["reply_ms"]) for r in rows) == [("escanor", 700), ("escanor", 900), ("merlin", 300)]
    # a fresh record process sees the same latest answers again and does not write them twice
    col2 = Collector(store)
    col2.on_status(status({"escanor": {**a, "ms": 700, "at": t + 5}, "merlin": m}))
    store.flush()
    assert len(store.select("replies", 0)) == 3


def test_panel_counts_fallbacks_and_shows_arm_holder(tmp_path):
    store = make_store(tmp_path)
    col = Collector(store)
    now = time.time()
    base = {"errors": [], "ms": 500}
    col.on_status(status({"ban": {**base, "provider": "gemini", "model": "gemini-3.5-flash-lite", "fallback": False, "at": now - 60}}))
    col.on_status(status({"ban": {**base, "provider": "openai", "model": "gpt-4o-mini", "fallback": True, "ms": 1200, "at": now - 30}},
                         states={"ban": "has_arm", "king": "idle"}))
    store.flush()
    assert col.arm_holder == "ban"

    d = panel_data(store, now=now, arm_holder=col.arm_holder, server_up=True)
    assert [c["agent"] for c in d["characters"]] == ["escanor", "elizabeth", "ban", "king", "gowther", "merlin"]
    ban = next(c for c in d["characters"] if c["agent"] == "ban")
    assert ban["replies"] == 2 and ban["fallbacks"] == 1 and ban["holds_arm"]
    assert ban["last"]["provider"] == "openai" and ban["last"]["model"] == "gpt-4o-mini"
    assert ban["last"]["fallback"] is True and ban["last"]["ms"] == 1200 and 29 <= ban["last"]["ago_s"] <= 31
    king = next(c for c in d["characters"] if c["agent"] == "king")
    assert king["last"] is None and king["replies"] == 0 and not king["holds_arm"]
    assert d["arm_holder"] == "ban" and d["server_up"] is True
    # charts endpoint data and the panel share the store without getting in each other's way
    assert "replies" in charts_data(store, now=now)


def test_arm_holder_follows_events_and_clears_when_server_goes(tmp_path):
    import asyncio
    col = Collector(make_store(tmp_path), server="http://127.0.0.1:1")   # nothing listens on port 1
    col.handle({"type": "agent_state", "agent": "king", "state": "has_arm"})
    assert col.arm_holder == "king"
    col.handle({"type": "agent_state", "agent": "ban", "state": "idle"})
    assert col.arm_holder == "king"
    col.handle({"type": "agent_state", "agent": "king", "state": "idle"})
    assert col.arm_holder is None
    col.arm_holder = "king"
    asyncio.run(col.poll_once())
    assert col.last_status_ok is False and col.arm_holder is None
    d = panel_data(col.store, arm_holder=col.arm_holder, server_up=col.last_status_ok)
    assert d["server_up"] is False and all(c["last"] is None for c in d["characters"])


class FlakyTiger:
    """Stands in for PostgresDB: refuses to connect until `up` is set, then keeps rows."""
    up = False
    rows: dict = {}

    def __init__(self, url):
        if not FlakyTiger.up:
            raise OSError("connection refused")
        self.hypertables = True

    def write(self, table, rows):
        FlakyTiger.rows.setdefault(table, []).extend(rows)

    def select(self, table, since):
        return [r for r in FlakyTiger.rows.get(table, []) if r["time"] >= since]


def test_outage_rows_are_copied_up_when_tiger_returns(tmp_path, monkeypatch):
    from backend.record import store as store_mod
    monkeypatch.setattr(store_mod, "PostgresDB", FlakyTiger)
    FlakyTiger.up, FlakyTiger.rows = False, {}
    store = Store("postgresql://example", sqlite_path=tmp_path / "fb.sqlite")
    t = time.time()
    store.add("joints", {"time": t, "joint": "gripper", "position": 1310.0})
    store.flush()
    assert store.active == "sqlite"
    FlakyTiger.up = True
    store.next_pg_try = 0   # do not wait out the retry delay
    store.add("joints", {"time": t + 1, "joint": "gripper", "position": 1320.0})
    store.flush()
    assert store.active == "tiger" and store.ok
    assert [r["position"] for r in FlakyTiger.rows["joints"]] == [1310.0, 1320.0]
