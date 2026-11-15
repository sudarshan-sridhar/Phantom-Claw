"""Smoothness numbers from a motion trace, so jitter shows up as numbers.

Record one on the real arm, from the project root:
    C:\\Users\\sudar\\.conda\\envs\\lerobot\\python.exe -m backend.arm.try_real pick place --target bottle --trace run.json
Then:
    C:\\Users\\sudar\\.conda\\envs\\lerobot\\python.exe -m backend.arm.trace_report run.json

For each skill, phase and joint it prints:
- stops: the joint was moving, stood still for at least STOP_S, then moved again with no new
  goal (a stop in the middle of a leg: a stall, a sag push, a hiccup)
- pauses: the same, but it was given a new goal in between (one leg ended dead, the next began)
- reversals: it changed direction with less than REVERSAL_GAP_S of stillness between
- peak speed in ticks per second, and the longest stop or pause
- goals written, and the most goals one leg wrote to that joint (a stepping loop shows up here)
Then every time the whole arm (joints 1 to 5) stood still between two motions, with the reason
when the skill marked it as on purpose (picture, grip, release, offer, point, hold, look,
look_first). An UNPLANNED one is a stop-and-go to look at.

Speeds come from positions about 0.13 s apart (two samples each side at 30 Hz), so a tick of
read noise is not taken for motion.
"""
import argparse
import json
import sys

MOVE_V, STILL_V = 20.0, 8.0       # ticks/s: above MOVE_V it moves, under STILL_V it is still
STOP_S = 0.10                     # a joint still this long between two motions is a stop
ARM_STOP_S = 0.12                 # the whole arm still this long between two motions
REVERSAL_GAP_S = 0.3
HOLD_BEFORE_S = 0.5               # an on-purpose mark explains a stop that started up to this much before it
ARM = (1, 2, 3, 4, 5)


def _speeds(rows, j, span=2):
    n = len(rows)
    out = []
    for k in range(n):
        a, b = max(0, k - span), min(n - 1, k + span)
        dt = rows[b][0] - rows[a][0]
        out.append((rows[b][j] - rows[a][j]) / dt if dt > 1e-6 else 0.0)
    return out


def _moving_flags(v):
    """Hysteresis: moving from MOVE_V up, still again under STILL_V."""
    flags, moving = [], False
    for x in v:
        if moving and abs(x) < STILL_V:
            moving = False
        elif not moving and abs(x) >= MOVE_V:
            moving = True
        flags.append(moving)
    return flags


def _runs(times, flags, v=None):
    """[(moving, t_start, t_end, sign)] from per-sample flags."""
    runs = []
    for k, (t, f) in enumerate(zip(times, flags)):
        sign = (1 if v[k] > 0 else -1) if (v is not None and f) else 0
        if runs and (runs[-1][0], runs[-1][3]) == (f, sign):
            runs[-1][2] = t
        else:
            if runs:
                runs[-1][2] = t
            runs.append([f, t, t, sign])
    return runs


def _stops(runs, min_s):
    """Still runs of at least min_s with motion on both sides: (t_start, t_end)."""
    out = []
    for k in range(1, len(runs) - 1):
        m, t0, t1, _ = runs[k]
        if not m and runs[k - 1][0] and runs[k + 1][0] and t1 - t0 >= min_s:
            out.append((t0, t1))
    return out


def _reversals(runs):
    out, last_sign, last_end = [], 0, None
    for m, t0, t1, s in runs:
        if not m:
            continue
        if last_sign and s and s != last_sign and t0 - last_end < REVERSAL_GAP_S:
            out.append(t0)
        last_sign, last_end = s or last_sign, t1
    return out


def _at(marks, t):
    """The last (t, value) mark at or before t."""
    val = None
    for tm, x in marks:
        if tm <= t + 1e-9:
            val = x
        else:
            break
    return val


def _why(holds, t0, t1):
    """The reason for a stop from t0 to t1: a mark during it (or just before), or a hold that began
    earlier and is still on (no arm leg started since). holds: [(t, why, t_next_arm_leg)]."""
    hit = [w for t, w, _ in holds if t0 - HOLD_BEFORE_S <= t <= t1 + 0.05]
    if hit:
        return hit[0]
    on = [w for t, w, until in holds if t <= t0 and (until is None or until >= t0 - 0.05)]
    return on[-1] if on else None


def analyse(trace):
    """{"skills": [{"skill", "t0", "t1", "rows", "phases": {phase: {joint: stats}}, "arm_stops": [...],
    "joint_stops": [...], "max_goals_per_leg": {joint: n}}]}"""
    rows = sorted(trace["rows"], key=lambda r: r[0])
    ev = sorted(trace["events"], key=lambda e: e[1])
    names = {int(k): v for k, v in trace.get("joints", {}).items()}
    spans, cur = [], None
    for e in ev:
        if e[0] == "skill":
            cur = [e[2], e[1], None]
            spans.append(cur)
        elif e[0] == "end" and cur is not None:
            cur[2] = e[1]
            cur = None
    if not spans and rows:
        spans = [["run", rows[0][0], rows[-1][0]]]
    out = []
    for name, s0, s1 in spans:
        s1 = s1 if s1 is not None else (rows[-1][0] if rows else s0)
        sr = [r for r in rows if s0 - 1e-9 <= r[0] <= s1 + 1e-9]
        se = [e for e in ev if s0 - 1e-9 <= e[1] <= s1 + 1e-9]
        phases = [(e[1], e[2]) for e in se if e[0] == "phase"]
        legs = [e[1] for e in se if e[0] == "leg"]
        arm_legs = [e[1] for e in se if e[0] == "leg" and any(g[0] in ARM for g in e[2])]
        holds = [(e[1], e[2], next((t for t in arm_legs if t > e[1]), None)) for e in se if e[0] == "hold"]
        goals = [(e[1], e[2]) for e in se if e[0] == "goal"]
        rep = {"skill": name, "t0": s0, "t1": s1, "rows": len(sr), "phases": {}, "arm_stops": [],
               "joint_stops": [], "max_goals_per_leg": {}}

        def phase_of(t):
            return _at(phases, t) or "start"

        # goals: per phase and joint, and the most in one leg
        per_leg = {}
        for t, j in goals:
            leg = sum(1 for lt in legs if lt <= t + 1e-9)
            per_leg[(leg, j)] = per_leg.get((leg, j), 0) + 1
        for (leg, j), n in per_leg.items():
            rep["max_goals_per_leg"][j] = max(rep["max_goals_per_leg"].get(j, 0), n)

        def stats(ph, j):
            d = rep["phases"].setdefault(ph, {})
            return d.setdefault(j, {"name": names.get(j, str(j)), "stops": 0, "pauses": 0, "reversals": 0,
                                    "peak": 0.0, "longest_pause_s": 0.0, "goals": 0})
        for t, j in goals:
            stats(phase_of(t), j)["goals"] += 1
        if len(sr) < 3:
            out.append(rep)
            continue
        times = [r[0] for r in sr]
        arm_flags = [False] * len(sr)
        for j in range(1, 7):
            v = _speeds(sr, j)
            flags = _moving_flags(v)
            if j in ARM:
                arm_flags = [a or f for a, f in zip(arm_flags, flags)]
            for t, x in zip(times, v):
                if abs(x) >= MOVE_V:
                    st = stats(phase_of(t), j)
                    st["peak"] = max(st["peak"], round(abs(x), 1))
            runs = _runs(times, flags, v)
            mine = [t for t, jj in goals if jj == j]
            for t0, t1 in _stops(runs, STOP_S):
                st = stats(phase_of(t0), j)
                st["longest_pause_s"] = max(st["longest_pause_s"], round(t1 - t0, 3))
                if any(t0 - HOLD_BEFORE_S <= t <= t1 + 0.2 for t in mine):
                    st["pauses"] += 1         # a new goal just before or during it: one leg ending, the next starting
                    continue
                st["stops"] += 1              # stopped and went on with no new goal: mid-leg
                if j in ARM:
                    rep["joint_stops"].append({"joint": j, "t": round(t0, 3), "s": round(t1 - t0, 3),
                                               "phase": phase_of(t0), "why": _why(holds, t0, t1)})
            for t in _reversals(runs):
                stats(phase_of(t), j)["reversals"] += 1
        for t0, t1 in _stops(_runs(times, arm_flags), ARM_STOP_S):
            rep["arm_stops"].append({"t": round(t0, 3), "s": round(t1 - t0, 3), "phase": phase_of(t0),
                                     "why": _why(holds, t0, t1)})
        out.append(rep)
    return {"skills": out}


def unplanned(report, joints=True):
    """Every stop no skill marked as on purpose."""
    bad = []
    for sk in report["skills"]:
        for s in sk["arm_stops"] + (sk["joint_stops"] if joints else []):
            if s["why"] is None:
                bad.append({"skill": sk["skill"], **s})
    return bad


def show(report, out=None):
    out = out or sys.stdout
    for sk in report["skills"]:
        print(f"skill {sk['skill']}  {sk['t1'] - sk['t0']:.1f} s, {sk['rows']} samples", file=out)
        print(f"  {'phase':<14}{'joint':<15}{'stops':>6}{'pauses':>7}{'revs':>6}{'peak t/s':>10}{'pause s':>9}"
              f"{'goals':>7}", file=out)
        for ph, joints in sk["phases"].items():
            for j in sorted(joints):
                st = joints[j]
                if st["peak"] < MOVE_V and not st["goals"] and not st["stops"]:
                    continue
                print(f"  {ph:<14}{st['name']:<15}{st['stops']:>6}{st['pauses']:>7}{st['reversals']:>6}"
                      f"{st['peak']:>10.0f}{st['longest_pause_s']:>9.2f}{st['goals']:>7}", file=out)
        mg = sk["max_goals_per_leg"]
        if mg:
            print("  most goals one leg wrote: " + ", ".join(f"{j}:{n}" for j, n in sorted(mg.items())), file=out)
        stops = sk["arm_stops"]
        if stops:
            print("  whole arm still between motions:", file=out)
            for s in stops:
                print(f"    t={s['t']:7.2f}  {s['s']:5.2f} s  in {s['phase']:<12} "
                      + (f"on purpose ({s['why']})" if s["why"] else "UNPLANNED"), file=out)
        js = [s for s in sk["joint_stops"] if s["why"] is None]
        for s in js:
            print(f"    joint {s['joint']} stopped {s['s']:.2f} s at t={s['t']:.2f} in {s['phase']}: UNPLANNED", file=out)
        print(file=out)
    bad = unplanned(report)
    print(f"{len(bad)} unplanned stops" + (" (look at those first)" if bad else ""), file=out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stops, reversals and peak speeds from a try_real --trace file.")
    ap.add_argument("file")
    ap.add_argument("--json", action="store_true", help="print the numbers as JSON")
    a = ap.parse_args(argv)
    with open(a.file) as f:
        trace = json.load(f)
    rep = analyse(trace)
    if a.json:
        print(json.dumps(rep, indent=1))
    else:
        show(rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
