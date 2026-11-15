"""Brain recordings made with g.tec amplifiers, decoded by our own code, choosing what the arm does.

The data is the SSVEP set g.tec hands out for its BR41N.IO hackathons: two people, two sessions
each, eight electrodes over the back of the head (PO7, PO3, POz, PO4, PO8, O1, Oz, O2), 256
samples a second. In each trial the person looks at one of four lights flickering at 9, 10, 12 or
15 times a second. The file also carries which light it was, so every answer can be checked.

    python -m backend.brain gtec-check                    accuracy on every trial of every file
    python -m backend.brain gtec-run [--subject 1]        replay one session as a live stream and decode it
    python -m backend.brain gtec-run --live               the same, and really send the choices to the arm

gtec-run publishes the recording as a live stream and reads it back the way a headset would be
read. A g.tec Unicorn headset streaming through the Unicorn LSL app can take the file's place:
--stream NAME --channels 8 --no-replay.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from .ssvep import cca_scores, ssvep_clean

DATA_DIR = Path(__file__).with_name("data") / "gtec_ssvep"
RATES = [9.0, 10.0, 12.0, 15.0]
# what each light means to the arm
ACTIONS = {9: ("give", "bottle"), 10: ("give", "sharpie"), 12: ("give", "scissors"), 15: ("stop", None)}
ONSET_S = 0.5           # the response takes about this long to build after a light comes on
SETTLE_S = 1.0          # extra signal in front of a window so the filter has settled
STEP_S = 0.25
DEFAULT_SERVER = "http://127.0.0.1:8765"
PAGE_PORT = 8767


# ---------------------------------------------------------------- the files

@dataclass
class Recording:
    name: str
    subject: int
    session: int
    fs: float
    eeg: np.ndarray         # (samples, 8), microvolts
    trigger: np.ndarray     # (samples,), 0 when no light is on, else 9, 10, 12 or 15


def load_recording(path: Path) -> Recording:
    """Rows of y: 0 time, 1 to 8 EEG, 9 which light is on, 10 the amplifier's own guess."""
    from scipy.io import loadmat

    m = loadmat(str(path))
    y = np.asarray(m["y"], dtype=float)
    if y.shape[0] > y.shape[1]:
        y = y.T
    if y.shape[0] < 10:
        raise ValueError(f"{path.name}: expected at least 10 rows, found {y.shape[0]}")
    fs = float(np.asarray(m.get("fs", 256)).ravel()[0])
    parts = path.stem.split("_")
    subject = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    session = int(parts[-1]) if parts[-1].isdigit() else 0
    return Recording(path.name, subject, session, fs, y[1:9].T.copy(), np.round(y[9]).astype(int))


NAMES = {item or "stop": rate for rate, (_kind, item) in ACTIONS.items()}      # "bottle" -> 9, "stop" -> 15


def splice(rec: Recording, names: list[str], gap_s: float = 12.0, lead_s: float = 5.0) -> Recording:
    """A short recording made of chosen trials from a real one, in a chosen order: for each name
    ("sharpie", "stop", ...) one real trial of that light, with real no-light stretches before
    and between them. Nothing is made up: every sample comes from the file; only the order and
    the length of the pauses are ours. Used to show one choice at a time on the arm."""
    found = trials(rec.trigger)
    rests = [(b, a2) for (_, b, _), (a2, _, _) in zip(found[:-1], found[1:]) if a2 - b > int(1.0 * rec.fs)]
    if not rests:
        raise ValueError("the recording has no pauses to fill the gaps with")
    used: dict[int, int] = {}
    eeg, trig = [], []
    r_i = 0

    def rest(seconds: float) -> None:
        nonlocal r_i
        need = int(round(seconds * rec.fs))
        while need > 0:
            a, b = rests[r_i % len(rests)]
            r_i += 1
            take = min(need, b - a)
            eeg.append(rec.eeg[a:a + take])
            trig.append(np.zeros(take, dtype=int))
            need -= take

    for k, name in enumerate(names):
        rate = NAMES.get(str(name).strip().lower())
        if rate is None:
            raise ValueError(f'unknown target "{name}": use one of {", ".join(NAMES)}')
        mine = [tr for tr in found if tr[2] == rate]
        if not mine:
            raise ValueError(f"the recording has no {rate} Hz trial")
        a, b, _ = mine[used.get(rate, 0) % len(mine)]
        used[rate] = used.get(rate, 0) + 1
        rest(lead_s if k == 0 else gap_s)
        eeg.append(rec.eeg[a:b])
        trig.append(np.full(b - a, rate, dtype=int))
    rest(4.0)
    return Recording(rec.name + " (trials: " + ", ".join(names) + ")", rec.subject, rec.session, rec.fs,
                     np.vstack(eeg), np.concatenate(trig))

def find_recordings(data_dir: Path = DATA_DIR) -> list[Path]:
    return sorted(data_dir.glob("subject_*_fvep_led_training_*.mat"))


def trials(trigger: np.ndarray) -> list[tuple[int, int, int]]:
    """(first sample, one past the last sample, rate) for each stretch with a light on."""
    out, start = [], None
    for i in range(len(trigger)):
        v = int(trigger[i])
        if start is None and v != 0:
            start = i
        elif start is not None and v != int(trigger[start]):
            out.append((start, i, int(trigger[start])))
            start = i if v != 0 else None
    if start is not None:
        out.append((start, len(trigger), int(trigger[start])))
    return out


def scores_at(eeg: np.ndarray, fs: float, end: int, window_s: float) -> np.ndarray | None:
    """CCA score per rate for the window that ends at sample `end`."""
    n_win = int(round(window_s * fs))
    n_all = n_win + int(round(SETTLE_S * fs))
    if end < n_all:
        return None
    x = ssvep_clean(eeg[end - n_all:end], fs)[-n_win:]
    return cca_scores(x, fs, RATES, harmonics=2, max_dims=None)


# ---------------------------------------------------------------- gtec-check

def check(data_dir: Path = DATA_DIR, windows=(2.0, 3.0, 4.0), out=print) -> dict:
    """Every trial of every file: take one window starting half a second after the light comes
    on, and pick the rate with the highest score. Nothing is fitted to these trials."""
    paths = find_recordings(data_dir)
    if not paths:
        out(f"no recordings in {data_dir}")
        return {}
    recs = [load_recording(p) for p in paths]
    result: dict = {"files": [r.name for r in recs], "windows": {}}
    for w in windows:
        conf = np.zeros((4, 4), dtype=int)
        per_file = {}
        for r in recs:
            good = total = 0
            for a, b, rate in trials(r.trigger):
                end = a + int(round((ONSET_S + w) * r.fs))
                if rate not in ACTIONS or end > b:
                    continue
                s = scores_at(r.eeg, r.fs, end, w)
                if s is None:
                    continue
                guess = int(np.argmax(s))
                conf[RATES.index(float(rate)), guess] += 1
                total += 1
                good += RATES[guess] == rate
            per_file[r.name] = (good, total)
        result["windows"][w] = {"per_file": per_file, "confusion": conf.tolist(),
                                "correct": int(np.trace(conf)), "total": int(conf.sum())}
    out("g.tec SSVEP recordings, our CCA decoder, one window per trial, highest score wins")
    out("guessing would be right 25% of the time\n")
    out(f"{'file':38}" + "".join(f"{w:>4.0f} s window  " for w in windows))
    for r in recs:
        row = f"{r.name:38}"
        for w in windows:
            g, t = result["windows"][w]["per_file"][r.name]
            row += f"{g:>3}/{t:<3} {100 * g / max(t, 1):>4.0f}%   "
        out(row)
    for subj in sorted({r.subject for r in recs}):
        row = f"{'subject ' + str(subj):38}"
        for w in windows:
            g = sum(v[0] for k, v in result["windows"][w]["per_file"].items() if f"subject_{subj}_" in k)
            t = sum(v[1] for k, v in result["windows"][w]["per_file"].items() if f"subject_{subj}_" in k)
            row += f"{g:>3}/{t:<3} {100 * g / max(t, 1):>4.0f}%   "
        out(row)
    row = f"{'all':38}"
    for w in windows:
        d = result["windows"][w]
        row += f"{d['correct']:>3}/{d['total']:<3} {100 * d['correct'] / max(d['total'], 1):>4.0f}%   "
    out(row)
    w = windows[len(windows) // 2]
    out(f"\nwhat we said (columns) against what the person looked at (rows), {w:.0f} s window")
    out("          " + "".join(f"{int(f):>6} Hz" for f in RATES))
    for i, f in enumerate(RATES):
        out(f"{int(f):>6} Hz " + "".join(f"{v:>9}" for v in result["windows"][w]["confusion"][i]))
    return result


# ---------------------------------------------------------------- thresholds from another session

@dataclass
class Limits:
    threshold: float        # best score must reach this
    margin: float           # and lead the runner-up by this
    hold: int = 4           # for this many windows in a row


def limits_from(rec: Recording, window_s: float, hold: int = 4) -> Limits:
    """Set the two limits from a session that is NOT the one being replayed: high enough that
    95 in 100 windows with no light on stay silent."""
    best, lead = [], []
    step = int(round(STEP_S * rec.fs))
    n_win = int(round(window_s * rec.fs))
    edges = [(a, b) for a, b, _ in trials(rec.trigger)]
    for (_, b), (a2, _) in zip(edges[:-1], edges[1:]):
        # windows that lie wholly inside the pause between two lights
        for end in range(b + n_win, a2, step):
            s = scores_at(rec.eeg, rec.fs, end, window_s)
            if s is not None:
                o = np.sort(s)[::-1]
                best.append(o[0])
                lead.append(o[0] - o[1])
    if not best:       # pauses shorter than the window: fall back to the stretch before the first light
        return Limits(0.35, 0.05, hold)
    return Limits(float(np.percentile(best, 95)), float(np.percentile(lead, 95)), hold)


# ---------------------------------------------------------------- live decoding

class LiveDecoder:
    """Feed it samples as they arrive; it answers with a rate when one has been held."""

    def __init__(self, fs: float, limits: Limits, window_s: float = 3.0, refractory_s: float = 8.0,
                 gate: float = 1.0):
        # gate scales both limits: 1 is the calibrated gate, 0 is no gate at all (the best guess
        # always counts), above 1 is stricter
        self.fs, self.limits, self.window_s, self.refractory_s, self.gate = fs, limits, window_s, refractory_s, gate
        self.keep = int(round((window_s + SETTLE_S) * fs)) + 8
        self.buf = np.zeros((0, 8))
        self.n = 0
        self.next_at = int(round((window_s + SETTLE_S) * fs))
        self.current, self.count = None, 0
        self.unsure, self.unsure_count = None, 0      # a best guess that keeps coming up but is not sure
        self.block_until = -1.0
        self.refused: list[tuple[float, int, float, float]] = []   # (time, rate, score, lead) the gate held back
        self.last: dict = {"scores": [0.0] * 4, "guess": None, "progress": 0.0, "best": None, "score": 0.0, "lead": 0.0}

    def feed(self, chunk: np.ndarray) -> list[tuple[float, int]]:
        """Returns (time in seconds, rate) for each choice made while taking in this chunk.
        Moments the gate held back go to self.refused."""
        fired = []
        x = np.atleast_2d(chunk)[:, :8]
        i = 0
        while i < len(x):
            take = min(len(x) - i, max(1, self.next_at - self.n))
            self.buf = np.vstack([self.buf, x[i:i + take]])[-self.keep:]
            self.n += take
            i += take
            if self.n < self.next_at:
                continue
            self.next_at = self.n + int(round(STEP_S * self.fs))
            t = self.n / self.fs
            s = scores_at(self.buf, self.fs, len(self.buf), self.window_s)
            if s is None:
                continue
            o = np.argsort(s)[::-1]
            score, lead = float(s[o[0]]), float(s[o[0]] - s[o[1]])
            sure = score >= self.limits.threshold * self.gate and lead >= self.limits.margin * self.gate
            best = int(o[0])
            guess = best if sure else None
            blocked = t < self.block_until
            if blocked or guess is None:
                self.current, self.count = None, 0
            elif guess == self.current:
                self.count += 1
            else:
                self.current, self.count = guess, 1
            # the same best guess, window after window, but never sure: the gate is holding it back
            if blocked or sure:
                self.unsure, self.unsure_count = None, 0
            elif best == self.unsure:
                self.unsure_count += 1
            else:
                self.unsure, self.unsure_count = best, 1
            if self.unsure is not None and self.unsure_count == self.limits.hold:
                self.refused.append((t, int(RATES[self.unsure]), score, lead))
            self.last = {"scores": [round(float(v), 3) for v in s],
                         "guess": None if guess is None else int(RATES[guess]),
                         "best": int(RATES[best]), "score": round(score, 3), "lead": round(lead, 3),
                         "progress": min(1.0, self.count / max(1, self.limits.hold))}
            if self.current is not None and self.count >= self.limits.hold:
                fired.append((t, int(RATES[self.current])))
                self.current, self.count = None, 0
                self.block_until = t + self.refractory_s
        return fired


# ---------------------------------------------------------------- talking to the agent server

def request_for(rate: int) -> tuple[str, str | None]:
    kind, item = ACTIONS[rate]
    return ("stop", None) if kind == "stop" else ("request", f"Gowther, give me the {item}")   # one named character, so nobody adds extras


def answer_parts(ans: dict) -> dict:
    """What came back from the agents for one request: who took it, the line, the arm steps and
    how it ended. The server sends either a list of steps or one skill."""
    ans = ans if isinstance(ans, dict) else {}
    arm = (ans.get("plan") or {}).get("arm")
    if isinstance(arm, list):
        steps = [str(s.get("skill")) + (" " + str((s.get("params") or {}).get("target") or "")).rstrip() for s in arm]
    elif isinstance(arm, dict):
        steps = [str(arm.get("skill"))]
    else:
        steps = []
    res = ans.get("arm_result") or {}
    return {"agent": ans.get("agent"), "say": str(ans.get("say") or ""), "steps": steps,
            "result": res.get("result"), "ok": res.get("ok"),
            "routed_by": (ans.get("routed") or {}).get("by")}


class Sender:
    """Sends a choice to the server without holding up the decoding. A request stays open until
    the arm has finished, so while one is out, new choices wait."""

    def __init__(self, server: str = DEFAULT_SERVER, live: bool = False, post=None, quiet_s: float = 5.0):
        self.server, self.live = server.rstrip("/"), live
        self.post = post or self._post
        self.busy = False
        self.quiet_s = quiet_s      # nothing goes out in the first seconds, while the decoder fills up
        self.log: list[dict] = []

    def _post(self, path: str, body: dict | None, timeout: float) -> dict:
        import httpx
        r = httpx.post(self.server + path, json=body, timeout=timeout) if body is not None else httpx.post(
            self.server + path, timeout=timeout)
        return r.json() if r.headers.get("content-type", "").startswith("application/json") else {}

    def send(self, rate: int, t: float) -> dict:
        kind, text = request_for(rate)
        rec = {"t": round(t, 2), "rate": rate, "kind": kind, "text": text or "stop", "sent": False, "reply": None,
               "agent": None, "say": "", "steps": [], "result": None, "ok": None}
        self.log.append(rec)
        if not self.live:
            rec["reply"] = "dry run, nothing sent"
            return rec
        if t < self.quiet_s:
            rec["reply"] = f"held: nothing is sent in the first {self.quiet_s:.0f} seconds"
            return rec
        if kind == "stop":
            # a stop always goes out at once, even while a job is running
            threading.Thread(target=self._go, args=(rec, "/api/stop", None, 5.0), daemon=True).start()
            return rec
        if self.busy:
            rec["reply"] = "waiting: the arm is still on the last job"
            return rec
        self.busy = True
        threading.Thread(target=self._go, args=(rec, "/api/request", {"text": text}, 180.0), daemon=True).start()
        return rec

    def _go(self, rec: dict, path: str, body, timeout: float) -> None:
        try:
            ans = self.post(path, body, timeout)
            rec["sent"] = True
            if isinstance(ans, dict) and ans.get("ignored") == "busy":
                rec["reply"] = "server busy with another job, not retried"
            elif path == "/api/stop":
                rec["reply"] = "stop sent"
            else:
                rec.update(answer_parts(ans))
                rec["reply"] = f"{rec['agent'] or '?'}: {rec['say'][:80]}"
        except Exception as e:  # noqa: BLE001 - the page shows it, decoding goes on
            rec["reply"] = f"not reachable ({type(e).__name__})"
        finally:
            if path != "/api/stop":
                self.busy = False


# ---------------------------------------------------------------- the status page

FONTS = Path(__file__).resolve().parents[2] / "overlay" / "assets" / "hd" / "fonts"

PAGE = """<!doctype html><meta charset="utf-8"><title>Brain to arm</title>
<style>
@font-face{font-family:"Aseina";src:url("/fonts/Aseina-Typeface.woff2") format("woff2")}
@font-face{font-family:"KC Obra";src:url("/fonts/KCObraLetraRegular.woff2") format("woff2")}
@font-face{font-family:"IM Fell SC";src:url("/fonts/IMFellEnglishSC-Regular.woff2") format("woff2")}
*{box-sizing:border-box}
body{margin:0;background:#4b6359;color:#fff;font:20px/1.25 "KC Obra",Georgia,serif}
header{display:flex;justify-content:space-between;align-items:flex-end;padding:20px 28px 14px}
h1{margin:0;font:normal 64px/0.9 "Aseina",Georgia,serif;color:#d9c5a3}
header span{font:17px "IM Fell SC",Georgia,serif;letter-spacing:.04em;text-align:right}
.g{display:grid;border-top:3px solid #000}
.g>div{padding:20px 24px;border-right:3px solid #000;border-bottom:3px solid #000;color:#000;min-width:0}
.g>div:last-child{border-right:0}
.g b{display:block;font:normal 54px/1 "Aseina",Georgia,serif;overflow-wrap:anywhere}
.g span{display:block;margin-top:8px;font:15px "IM Fell SC",Georgia,serif;letter-spacing:.06em}
#tiles{grid-template-columns:repeat(4,1fr)}
#tiles>div{background:#d9c5a3}
#tiles>div.look{background:#b0afc8}
#tiles>div.said{outline:7px solid #000;outline-offset:-7px}
#tiles i{display:block;height:10px;margin-top:14px;background:#000;width:0}
#tiles em{display:block;margin-top:6px;font:15px "IM Fell SC",Georgia,serif;letter-spacing:.06em;font-style:normal}
#nums{grid-template-columns:repeat(5,1fr)}
#nums>div{background:#fff}
#arm{grid-template-columns:1.1fr 1fr 2fr 1fr}
#arm>div{background:#b0afc8}
#arm b{font-size:40px}
#arm .sm b{font:26px/1.15 "KC Obra",Georgia,serif}
#log{padding:16px 28px 28px;font:16px/1.55 Consolas,monospace;white-space:pre-wrap;color:#fff}
.warn{background:#000!important;color:#fff!important}
</style>
<header><h1>Brain to arm</h1><span id="file"></span></header>
<div class="g" id="tiles"></div>
<div class="g" id="nums">
 <div><b id="looking">-</b><span>looking at</span></div>
 <div><b id="said">-</b><span>decoded</span></div>
 <div><b id="conf">-</b><span>confidence, gate at 100</span></div>
 <div><b id="acc">-</b><span>right so far</span></div>
 <div id="refbox"><b id="ref">0</b><span>held back by the gate</span></div>
</div>
<div class="g" id="arm">
 <div><b id="req">-</b><span>sent to the agents</span></div>
 <div><b id="who">-</b><span>taken by</span></div>
 <div class="sm"><b id="steps">-</b><span>arm steps</span></div>
 <div><b id="res">-</b><span>result</span></div>
</div>
<div id="log"></div>
<script>
const names={9:"bottle",10:"sharpie",12:"scissors",15:"stop"};
const tiles=document.getElementById("tiles");
for(const r of [9,10,12,15]){const d=document.createElement("div");d.id="t"+r;
 d.innerHTML="<b>"+names[r]+"</b><span>"+r+" Hz</span><i></i><em></em>";tiles.appendChild(d);}
const put=(id,v)=>{document.getElementById(id).textContent=v;};
async function tick(){try{const s=await (await fetch("/state")).json();
 [9,10,12,15].forEach((r,k)=>{const d=document.getElementById("t"+r);
  d.classList.toggle("look",s.truth===r);d.classList.toggle("said",s.last_choice===r);
  d.querySelector("i").style.width=(s.guess===r?Math.round(100*s.progress):0)+"%";
  d.querySelector("em").textContent="score "+(s.scores[k]||0).toFixed(2);});
 put("file",s.file+(s.live?" . live":" . dry run"));
 put("looking",s.truth?names[s.truth]:"nothing");
 put("said",s.last_choice?names[s.last_choice]:"-");
 put("conf",s.confidence==null?"-":s.confidence);
 put("acc",s.trials?s.correct+" of "+s.trials:"-");
 put("ref",s.refused);
 const a=s.arm||{};
 put("req",a.text||"-");put("who",a.agent||(a.text?"...":"-"));
 put("steps",((a.steps&&a.steps.length)?a.steps.join(", then "):(a.note||"-")).replace(/_/g," "));
 put("res",(a.result||(a.text?"working":"-")).replace(/_/g," "));
 document.getElementById("arm").classList.toggle("warn",false);
 document.getElementById("log").textContent=s.log.join("\\n");}catch(e){}}
setInterval(tick,300);tick();
</script>"""


def serve_page(state: dict, port: int = PAGE_PORT) -> ThreadingHTTPServer | None:
    class H(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path.startswith("/state"):
                body, kind = json.dumps(state).encode(), "application/json"
            elif self.path.startswith("/fonts/"):
                p = FONTS / Path(self.path).name
                if not p.is_file():
                    self.send_response(404)
                    self.end_headers()
                    return
                body, kind = p.read_bytes(), "font/woff2"
            else:
                body, kind = PAGE.encode(), "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):  # quiet
            pass

    try:
        srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    except OSError:
        return None
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


# ---------------------------------------------------------------- scoring a run against the file

@dataclass
class Score:
    trials: int = 0
    correct: int = 0
    wrong: int = 0
    missed: int = 0
    false_in_pause: int = 0
    refused: int = 0
    times: list = field(default_factory=list)


class Judge:
    """Marks each choice against what the person in the recording was really looking at."""

    def __init__(self, trial_list, fs: float):
        self.trials, self.fs = trial_list, fs
        self.answered: set[int] = set()
        self.score = Score()

    def mark(self, t: float, rate: int) -> tuple[str, int | None]:
        n = int(round(t * self.fs))
        # which light was on when the choice was made (a choice may land just after it went off)
        k = next((i for i, (a, b, _) in enumerate(self.trials) if a <= n < b + int(1.0 * self.fs)), None)
        if k is None:
            self.score.false_in_pause += 1
            return "WRONG: no light was on", None
        truth = self.trials[k][2]
        if k in self.answered:
            return "again in the same trial, not counted", truth
        self.answered.add(k)
        ok = truth == rate
        self.score.correct += ok
        self.score.wrong += not ok
        self.score.times.append(round((n - self.trials[k][0]) / self.fs, 2))
        return ("right" if ok else f"WRONG, was {truth} Hz"), truth

    def finish(self, n_seen: int, refused: int = 0) -> Score:
        shown = sum(1 for a, b, _ in self.trials if b <= n_seen)
        self.score.trials = shown
        self.score.missed = max(0, shown - len([k for k in self.answered if self.trials[k][1] <= n_seen + self.fs]))
        self.score.refused = refused
        return self.score


def offline(rec: Recording, limits: Limits, window_s: float = 3.0, gate: float = 1.0) -> Score:
    """The live decoder run over a whole recording at once, with no stream and nothing sent.
    The same code path as gtec-run, so the numbers are the live numbers."""
    dec = LiveDecoder(rec.fs, limits, window_s, gate=gate)
    judge = Judge(trials(rec.trigger), rec.fs)
    step = int(round(STEP_S * rec.fs))
    for i in range(0, len(rec.eeg), step):
        for t, rate in dec.feed(rec.eeg[i:i + step]):
            judge.mark(t, rate)
    return judge.finish(dec.n, len(dec.refused))


def gate_table(data_dir: Path = DATA_DIR, window_s: float = 3.0, gates=(0.0, 1.0, 1.5), out=print) -> dict:
    """What the confidence gate does, for every recording: no gate (the best guess always moves
    the arm), the calibrated gate, and a stricter one. Limits always come from the same person's
    other session."""
    recs = {(r.subject, r.session): r for r in map(load_recording, find_recordings(data_dir))}
    rows = {}
    out(f"live decoder over each whole recording, {window_s:.0f} s window, held 4 steps, 8 s between choices")
    out(f"{'recording':<22}{'gate':>6}{'right':>7}{'wrong':>7}{'none':>6}{'in pause':>10}{'held back':>11}")
    for (sub, ses), rec in sorted(recs.items()):
        other = recs.get((sub, 1 if ses == 2 else 2))
        limits = limits_from(other, window_s) if other is not None else Limits(0.35, 0.05)
        for g in gates:
            s = offline(rec, limits, window_s, g)
            rows[(sub, ses, g)] = s
            label = "off" if g == 0 else f"x{g:g}"
            out(f"subject {sub} session {ses:<5}{label:>6}{s.correct:>7}{s.wrong:>7}{s.missed:>6}{s.false_in_pause:>10}{s.refused:>11}")
    for sub in sorted({k[0] for k in rows}):
        for g in gates:
            ss = [v for k, v in rows.items() if k[0] == sub and k[2] == g]
            n = sum(s.trials for s in ss)
            bad = sum(s.wrong + s.false_in_pause for s in ss)
            out(f"subject {sub}, gate {'off' if g == 0 else 'x' + format(g, 'g')}: {sum(s.correct for s in ss)} of {n} right, "
                f"{bad} wrong moves, {sum(s.missed for s in ss)} lights with no move")
    return rows


# ---------------------------------------------------------------- gtec-run

def replay_stream(rec: Recording, name: str, speed: float, stop: threading.Event, sent: list) -> None:
    """Publish the recording as a live stream, a few samples at a time, at its own rate."""
    from pylsl import StreamInfo, StreamOutlet

    info = StreamInfo(name, "EEG", 8, rec.fs, "float32", "gtec-replay-" + rec.name)
    outlet = StreamOutlet(info, chunk_size=8)
    time.sleep(1.0)     # let the reader find it
    chunk, t0, i = 8, time.perf_counter(), 0
    while i < len(rec.eeg) and not stop.is_set():
        due = int((time.perf_counter() - t0) * rec.fs * speed)
        while i < min(due, len(rec.eeg)):
            j = min(i + chunk, len(rec.eeg))
            outlet.push_chunk(rec.eeg[i:j].astype(np.float32).tolist())
            i = j
            sent[0] = i
        time.sleep(0.004)
    sent[1] = True


def confidence(last: dict, limits: Limits, gate: float) -> int | None:
    """How far the decoder is past its gate, as one number: 100 is exactly at the gate."""
    if not last.get("best"):
        return None
    need_s, need_m = max(1e-6, limits.threshold * max(gate, 1e-6)), max(1e-6, limits.margin * max(gate, 1e-6))
    return int(round(100 * min(last["score"] / need_s, last["lead"] / need_m)))


def run(subject: int = 1, session: int = 2, live: bool = False, speed: float = 1.0, window_s: float = 3.0,
        stream: str = "GtecSSVEP", channels: int = 8, replay: bool = True, server: str = DEFAULT_SERVER,
        port: int = PAGE_PORT, seconds: float | None = None, data_dir: Path = DATA_DIR, out=print,
        gate: float = 1.0, quiet_s: float = 5.0, evidence: Path | None = None,
        targets: list[str] | None = None, gap_s: float = 12.0) -> Score:
    from pylsl import StreamInlet, resolve_byprop

    paths = {(r.subject, r.session): r for r in map(load_recording, find_recordings(data_dir))}
    rec = paths.get((subject, session))
    other = paths.get((subject, 1 if session == 2 else 2))
    if replay and rec is None:
        raise SystemExit(f"no recording for subject {subject}, session {session} in {data_dir}")
    if replay and targets:
        # show chosen trials only, in this order (real trials from the file, see splice)
        rec = splice(rec, targets, gap_s, lead_s=max(6.0, quiet_s + 1.0))
    # the limits come from the same person's other session, never from the one being played
    limits = limits_from(other, window_s) if other is not None else Limits(0.35, 0.05)
    fs = rec.fs if rec is not None else 250.0
    out(f"limits from {other.name if other else 'defaults'}: score at least {limits.threshold:.2f}, "
        f"lead at least {limits.margin:.2f}, held {limits.hold} windows of {window_s:.0f} s, gate x{gate:g}")
    out(f"LIVE: choices go to {server}" if live else "dry run: choices are shown, nothing is sent (add --live to send)")

    state = {"truth": None, "guess": None, "progress": 0.0, "last_choice": None, "trials": 0, "correct": 0,
             "live": live, "log": [], "file": rec.name if rec else stream, "scores": [0.0] * 4,
             "confidence": None, "refused": 0, "arm": {}}
    if serve_page(state, port):
        out(f"status page: http://127.0.0.1:{port}/")
    stop, sent = threading.Event(), [0, False]
    if replay:
        threading.Thread(target=replay_stream, args=(rec, stream, speed, stop, sent), daemon=True).start()
    found = resolve_byprop("name", stream, timeout=15)
    if not found:
        raise SystemExit(f'no live stream called "{stream}"')
    inlet = StreamInlet(found[0], max_buflen=30)
    if not replay:
        fs = float(found[0].nominal_srate()) or fs
    dec = LiveDecoder(fs, limits, window_s, gate=gate)
    sender = Sender(server, live, quiet_s=quiet_s)
    trial_list = trials(rec.trigger) if rec is not None else []
    judge = Judge(trial_list, fs)
    rows: list[dict] = []           # one per choice, for the evidence file
    shown_refused = 0
    t_start = time.time()
    try:
        while True:
            chunk, _ = inlet.pull_chunk(timeout=0.2)
            if chunk:
                x = np.asarray(chunk, dtype=float)[:, :channels]
                if x.shape[1] < 8:
                    x = np.pad(x, ((0, 0), (0, 8 - x.shape[1])))
                for t, rate in dec.feed(x):
                    verdict, truth = judge.mark(t, rate) if rec is not None else ("no file to check against", None)
                    recd = sender.send(rate, t)
                    recd["verdict"], recd["truth"] = verdict, truth
                    rows.append(recd)
                    kind, text = request_for(rate)
                    line = f"{t:6.1f} s  {rate:>2} Hz  {text or 'STOP':22} {verdict}"
                    out(line)
                    state["last_choice"] = rate
                    state["log"] = (state["log"] + [line])[-10:]
                for t, rate, score, lead in dec.refused[shown_refused:]:
                    line = f"{t:6.1f} s  {rate:>2} Hz  held back by the gate (score {score:.2f}, lead {lead:.2f})"
                    out(line)
                    state["log"] = (state["log"] + [line])[-10:]
                shown_refused = len(dec.refused)
                n_now = dec.n
                cur = next((r for a, b, r in trial_list if a <= n_now < b), None)
                done = sum(1 for a, b, _ in trial_list if b + int(1.0 * fs) <= n_now)
                # the last thing that really went to the agents, with what came back
                went = next((r for r in reversed(sender.log) if r["kind"] == "request" and
                             (r["sent"] or r["reply"] is None)), None)
                arm = {} if went is None else {"text": went["text"], "agent": went["agent"], "steps": went["steps"],
                                               "result": went["result"],
                                               "note": went["reply"] if went["sent"] and not went["steps"] else None}
                state.update(truth=cur, guess=dec.last["guess"], progress=dec.last["progress"],
                             trials=max(done, len(judge.answered)), correct=judge.score.correct,
                             scores=dec.last["scores"], confidence=confidence(dec.last, limits, gate),
                             refused=len(dec.refused), arm=arm)
            if replay and sent[1] and not chunk:
                break
            if seconds is not None and time.time() - t_start > seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
    # a request that is still out gets a moment to come back, so the record is complete
    t_wait = time.time()
    while sender.busy and time.time() - t_wait < 120:
        time.sleep(0.5)
    score = judge.finish(dec.n, len(dec.refused))
    if rec is not None:
        mean_t = float(np.mean(score.times)) if score.times else 0.0
        out(f"\n{rec.name}: {score.trials} lights shown; {score.correct} right, {score.wrong} wrong, {score.missed} "
            f"with no choice, {score.false_in_pause} choices while no light was on, {score.refused} moments held back "
            f"by the gate; a choice came {mean_t:.1f} s after the light on average")
    if live:
        out("\nwhat went to the agents and what came back:")
        for r in rows:
            if r["kind"] == "stop":
                out(f"  {r['t']:6.1f} s  STOP ({r['verdict']}) -> {r['reply']}")
            elif r["sent"]:
                out(f"  {r['t']:6.1f} s  \"{r['text']}\" ({r['verdict']}) -> {r['agent']}: {', then '.join(r['steps']) or 'no arm steps'}"
                    f" -> {r['result']}")
            else:
                out(f"  {r['t']:6.1f} s  \"{r['text']}\" ({r['verdict']}) -> not sent: {r['reply']}")
    if evidence is not None:
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_text(json.dumps({"file": rec.name if rec else stream, "server": server, "live": live,
                                        "gate": gate, "score": score.__dict__, "choices": rows}, indent=1, default=str))
        out(f"saved {evidence}")
    return score
