# Record and charts

Keeps a timestamped record of everything Phantom Claw does and draws it live at http://127.0.0.1:8766/.

## Run it

From the project root, with the server already running (or started later, it reconnects by itself):

    C:\Users\sudar\.conda\envs\lerobot\python.exe -m backend.record

It listens to ws://127.0.0.1:8765/ws and stores every event. It also reads GET /api/status once a second for the six joint readings and for which language model answered and how fast. Open http://127.0.0.1:8766/ for the charts, which refresh every 1.5 s. The JSON behind them is at /api/charts (add `?minutes=N` to change the job window, 60 by default).

Options: `--server http://host:port` and `--port N`.

## Tables

| table | columns |
|---|---|
| events | time, type, agent, skill, phase, text, data (the whole event as JSON) |
| joints | time, joint, position |
| requests | time, agent, text, provider, reply_ms, arm_skill, arm_result, arm_ms |
| brain | time, state, value |

One row in `requests` is one job. It is built from that job's request, merlin, say and arm events. `arm_result` is one of handed over, put back, missed, refused (the arm pointed or stopped halfway instead), done, stopped or error. Gestures while talking are not counted as arm jobs.

## Tiger Data

Set `TIGER_DATABASE_URL` in the project's .env to the service's connection string (postgres://...). That is the only switch. With it set, the next start:

1. connects with psycopg
2. creates the four tables as hypertables with `CREATE TABLE ... WITH (tsdb.hypertable, tsdb.orderby = 'time DESC')`, falling back to `create_hypertable` on an older TimescaleDB, and to ordinary tables on plain PostgreSQL
3. writes in batches every 2 s (joints arrive once a second per joint)

The page footer says where rows are going. With the variable empty, rows go to backend\cache\record.sqlite with the same tables.

If Tiger Data cannot be reached, rows go to the SQLite file instead, so nothing is lost and nothing else in the system waits. Tiger Data is tried again every 15 s. When it answers, the rows saved locally during the outage are copied up.

## Queries to show

Average arm job time per character, in 5 minute buckets over the last hour:

```sql
SELECT time_bucket('5 minutes', time) AS bucket,
       agent,
       count(*)                          AS jobs,
       round(avg(arm_ms) / 1000.0, 1)    AS avg_seconds
FROM requests
WHERE arm_skill IS NOT NULL
  AND time > now() - interval '1 hour'
GROUP BY bucket, agent
ORDER BY bucket, agent;
```

How each character's jobs end, today:

```sql
SELECT agent, arm_result, count(*) AS jobs
FROM requests
WHERE arm_skill IS NOT NULL
  AND time > date_trunc('day', now())
GROUP BY agent, arm_result
ORDER BY agent, jobs DESC;
```

Reply time per language model, median and slowest, by minute:

```sql
SELECT time_bucket('1 minute', time) AS minute,
       provider,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY reply_ms) AS median_ms,
       max(reply_ms) AS slowest_ms
FROM requests
WHERE provider IS NOT NULL
GROUP BY minute, provider
ORDER BY minute DESC;
```

Gripper position averaged per 5 seconds over the last 10 minutes, with gaps filled:

```sql
SELECT time_bucket_gapfill('5 seconds', time) AS t,
       locf(avg(position)) AS gripper
FROM joints
WHERE joint = 'gripper'
  AND time > now() - interval '10 minutes' AND time <= now()
GROUP BY t
ORDER BY t;
```
