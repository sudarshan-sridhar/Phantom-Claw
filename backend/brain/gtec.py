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

    def __init__(self, fs: float, limits: Limits, window_s: float = 3.0, refractory_s: float = 8.0):
        self.fs, self.limits, self.window_s, self.refractory_s = fs, limits, window_s, refractory_s
        self.keep = int(round((window_s + SETTLE_S) * fs)) + 8
        self.buf = np.zeros((0, 8))
        self.n = 0
        self.next_at = int(round((window_s + SETTLE_S) * fs))
        self.current, self.count = None, 0
        self.block_until = -1.0
        self.last: dict = {"scores": [0.0] * 4, "guess": None, "progress": 0.0}

    def feed(self, chunk: np.ndarray) -> list[tuple[float, int]]:
        """Returns (time in seconds, rate) for each choice made while taking in this chunk."""
        fired = []
        for row in np.atleast_2d(chunk):
            self.buf = np.vstack([self.buf, row[None, :8]])[-self.keep:]
            self.n += 1
            if self.n < self.next_at:
                continue
            self.next_at = self.n + int(round(STEP_S * self.fs))
            t = self.n / self.fs
            s = scores_at(self.buf, self.fs, len(self.buf), self.window_s)
            if s is None:
                continue
            o = np.argsort(s)[::-1]
            sure = s[o[0]] >= self.limits.threshold and s[o[0]] - s[o[1]] >= self.limits.margin
            guess = int(o[0]) if sure else None
            if t < self.block_until or guess is None:
                self.current, self.count = None, 0
            elif guess == self.current:
                self.count += 1
            else:
                self.current, self.count = guess, 1
            self.last = {"scores": [round(float(v), 3) for v in s],
                         "guess": None if guess is None else int(RATES[guess]),
                         "progress": min(1.0, self.count / max(1, self.limits.hold))}
            if self.current is not None and self.count >= self.limits.hold:
                fired.append((t, int(RATES[self.current])))
                self.current, self.count = None, 0
                self.block_until = t + self.refractory_s
        return fired


# ---------------------------------------------------------------- talking to the agent server

def request_for(rate: int) -> tuple[str, str | None]:
    kind, item = ACTIONS[rate]
    return ("stop", None) if kind == "stop" else ("request", f"give me the {item}")


class Sender:
    """Sends a choice to the server without holding up the decoding. A request stays open until
    the arm has finished, so while one is out, new choices wait."""

    def __init__(self, server: str = DEFAULT_SERVER, live: bool = False, post=None):
        self.server, self.live = server.rstrip("/"), live
        self.post = post or self._post
        self.busy = False
        self.log: list[dict] = []

    def _post(self, path: str, body: dict | None, timeout: float) -> dict:
        import httpx
        r = httpx.post(self.server + path, json=body, timeout=timeout) if body is not None else httpx.post(
            self.server + path, timeout=timeout)
        return r.json() if r.headers.get("content-type", "").startswith("application/json") else {}

    def send(self, rate: int, t: float) -> dict:
        kind, text = request_for(rate)
        rec = {"t": round(t, 2), "rate": rate, "kind": kind, "text": text or "stop", "sent": False, "reply": None}
        self.log.append(rec)
        if not self.live:
            rec["reply"] = "dry run, nothing sent"
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
                rec["reply"] = f"{ans.get('agent', '?')}: {str(ans.get('say', ''))[:80]}"
        except Exception as e:  # noqa: BLE001 - the page shows it, decoding goes on
            rec["reply"] = f"not reachable ({type(e).__name__})"
        finally:
            if path != "/api/stop":
                self.busy = False


# ---------------------------------------------------------------- the status page

PAGE = """<!doctype html><meta charset="utf-8"><title>g.tec recording to arm</title>
<style>
body{margin:0;background:#4b6359;color:#fff;font:18px/1.3 Georgia,serif}
header{padding:18px 28px;font-size:22px;letter-spacing:.04em}
main{display:grid;grid-template-columns:repeat(4,1fr);gap:0;border-top:3px solid #000}
.t{padding:26px 22px;background:#d9c5a3;color:#000;border-right:3px solid #000;border-bottom:3px solid #000}
.t:last-child{border-right:0}
.t b{display:block;font-size:64px;line-height:1}
.t span{display:block;margin-top:6px;font-size:20px;text-transform:uppercase;letter-spacing:.06em}
.t.look{background:#b0afc8}
.t.said{outline:6px solid #000;outline-offset:-6px}
.t i{display:block;height:10px;margin-top:14px;background:#000;width:0}
.row{display:grid;grid-template-columns:repeat(4,1fr);border-bottom:3px solid #000;background:#fff;color:#000}
.row div{padding:20px 22px;border-right:3px solid #000}
.row div:last-child{border-right:0}
.row b{display:block;font-size:44px;line-height:1.05}
.row span{font-size:14px;text-transform:uppercase;letter-spacing:.08em}
#log{padding:18px 28px;font:16px/1.5 Consolas,monospace;white-space:pre-wrap}
</style>
<header>g.tec brain recording, decoded live, choosing for the arm</header>
<main id="tiles"></main>
<div class="row"><div><b id="looking">-</b><span>looking at</span></div><div><b id="said">-</b><span>decoded</span></div>
<div><b id="acc">-</b><span>right so far</span></div><div><b id="mode">-</b><span>arm</span></div></div>
<div id="log"></div>
<script>
const names={9:"bottle",10:"sharpie",12:"scissors",15:"stop"};
const tiles=document.getElementById("tiles");
for(const r of [9,10,12,15]){const d=document.createElement("div");d.className="t";d.id="t"+r;
 d.innerHTML="<b>"+r+" Hz</b><span>"+names[r]+"</span><i></i>";tiles.appendChild(d);}
async function tick(){try{const s=await (await fetch("/state")).json();
 for(const r of [9,10,12,15]){const d=document.getElementById("t"+r);
  d.classList.toggle("look",s.truth===r);d.classList.toggle("said",s.last_choice===r);
  d.querySelector("i").style.width=(s.guess===r?Math.round(100*s.progress):0)+"%";}
 document.getElementById("looking").textContent=s.truth?names[s.truth]:"nothing";
 document.getElementById("said").textContent=s.last_choice?names[s.last_choice]:"-";
 document.getElementById("acc").textContent=s.trials?s.correct+" of "+s.trials:"-";
 document.getElementById("mode").textContent=s.live?"live":"dry run";
 document.getElementById("log").textContent=s.log.join("\\n");}catch(e){}}
setInterval(tick,300);tick();
</script>"""


def serve_page(state: dict, port: int = PAGE_PORT) -> ThreadingHTTPServer | None:
    class H(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path.startswith("/state"):
                body, kind = json.dumps(state).encode(), "application/json"
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


# ---------------------------------------------------------------- gtec-run

@dataclass
class Score:
    trials: int = 0
    correct: int = 0
    wrong: int = 0
    missed: int = 0
    false_in_pause: int = 0
    times: list = field(default_factory=list)


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


def run(subject: int = 1, session: int = 2, live: bool = False, speed: float = 1.0, window_s: float = 3.0,
        stream: str = "GtecSSVEP", channels: int = 8, replay: bool = True, server: str = DEFAULT_SERVER,
        port: int = PAGE_PORT, seconds: float | None = None, data_dir: Path = DATA_DIR, out=print) -> Score:
    from pylsl import StreamInlet, resolve_byprop

    paths = {(r.subject, r.session): r for r in map(load_recording, find_recordings(data_dir))}
    rec = paths.get((subject, session))
    other = paths.get((subject, 1 if session == 2 else 2))
    if replay and rec is None:
        raise SystemExit(f"no recording for subject {subject}, session {session} in {data_dir}")
    # the limits come from the same person's other session, never from the one being played
    limits = limits_from(other, window_s) if other is not None else Limits(0.35, 0.05)
    fs = rec.fs if rec is not None else 250.0
    out(f"limits from {other.name if other else 'defaults'}: score at least {limits.threshold:.2f}, "
        f"lead at least {limits.margin:.2f}, held {limits.hold} windows of {window_s:.0f} s")
    out("LIVE: choices go to the arm" if live else "dry run: choices are shown, nothing is sent (add --live to send)")

    state = {"truth": None, "guess": None, "progress": 0.0, "last_choice": None, "trials": 0, "correct": 0,
             "live": live, "log": [], "file": rec.name if rec else stream}
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
    dec = LiveDecoder(fs, limits, window_s)
    sender = Sender(server, live)
    score = Score()
    trial_list = trials(rec.trigger) if rec is not None else []
    answered: set[int] = set()
    t_start = time.time()
    try:
        while True:
            chunk, _ = inlet.pull_chunk(timeout=0.2)
            if chunk:
                x = np.asarray(chunk, dtype=float)[:, :channels]
                if x.shape[1] < 8:
                    x = np.pad(x, ((0, 0), (0, 8 - x.shape[1])))
                for t, rate in dec.feed(x):
                    n = int(round(t * fs))
                    # which light was on when the choice was made (a choice may land just after it went off)
                    k = next((i for i, (a, b, _) in enumerate(trial_list) if a <= n < b + int(1.0 * fs)), None)
                    truth = trial_list[k][2] if k is not None else None
                    recd = sender.send(rate, t)
                    verdict = "no file to check against"
                    if rec is not None:
                        if k is None:
                            score.false_in_pause += 1
                            verdict = "WRONG: no light was on"
                        elif k in answered:
                            verdict = "again in the same trial, not counted"
                        else:
                            answered.add(k)
                            ok = truth == rate
                            score.correct += ok
                            score.wrong += not ok
                            score.times.append(round((n - trial_list[k][0]) / fs, 2))
                            verdict = "right" if ok else f"WRONG, was {truth} Hz"
                    kind, text = request_for(rate)
                    line = f"{t:6.1f} s  {rate:>2} Hz  {text or 'STOP':22} {verdict}"
                    out(line)
                    state["last_choice"] = rate
                    state["log"] = (state["log"] + [line + ("  |  " + str(recd["reply"]) if recd["reply"] else "")])[-12:]
                n_now = dec.n
                cur = next((r for a, b, r in trial_list if a <= n_now < b), None)
                done = sum(1 for a, b, _ in trial_list if b + int(1.0 * fs) <= n_now)
                state.update(truth=cur, guess=dec.last["guess"], progress=dec.last["progress"],
                             trials=max(done, len(answered)), correct=score.correct)
            if replay and sent[1] and not chunk:
                break
            if seconds is not None and time.time() - t_start > seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
    if rec is not None:
        shown = sum(1 for a, b, _ in trial_list if b <= dec.n)
        score.trials = shown
        score.missed = shown - len([k for k in answered if trial_list[k][1] <= dec.n + fs])
        mean_t = float(np.mean(score.times)) if score.times else 0.0
        out(f"\n{rec.name}: {shown} lights shown; {score.correct} right, {score.wrong} wrong, {max(score.missed, 0)} "
            f"with no choice, {score.false_in_pause} choices while no light was on; "
            f"a choice came {mean_t:.1f} s after the light on average")
    return score
