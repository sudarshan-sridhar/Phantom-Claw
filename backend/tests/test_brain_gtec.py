"""The g.tec recording route: reading a file, the mapping to arm jobs, and the busy handling.
No network and no real recordings: the .mat file here is made up in the test."""
import time

import numpy as np
from scipy.io import savemat

from backend.brain import gtec


def fake_file(tmp_path, fs=256, noise=4.0, seed=0):
    """Four lights, 5 s on and 3 s off, with a small sine at the light's rate on all 8 channels."""
    rng = np.random.default_rng(seed)
    order = [15, 12, 10, 9]
    n_on, n_off = 5 * fs, 3 * fs
    n = n_off + len(order) * (n_on + n_off)
    trig = np.zeros(n)
    eeg = rng.normal(0, noise, (8, n))
    t = np.arange(n) / fs
    for i, rate in enumerate(order):
        a = n_off + i * (n_on + n_off)
        trig[a:a + n_on] = rate
        eeg[:, a:a + n_on] += 3.0 * np.sin(2 * np.pi * rate * t[a:a + n_on])
    y = np.vstack([t, eeg, trig, np.zeros(n)])
    p = tmp_path / "subject_1_fvep_led_training_1.mat"
    savemat(str(p), {"y": y, "fs": np.array([[fs]], dtype=np.uint16)})
    return p, order


def test_loads_a_recording_and_finds_its_trials(tmp_path):
    p, order = fake_file(tmp_path)
    rec = gtec.load_recording(p)
    assert (rec.subject, rec.session, rec.fs) == (1, 1, 256.0)
    assert rec.eeg.shape[1] == 8 and rec.eeg.shape[0] == len(rec.trigger)
    found = gtec.trials(rec.trigger)
    assert [r for _, _, r in found] == order
    assert all(b - a == 5 * 256 for a, b, _ in found)


def test_check_gets_the_made_up_file_right(tmp_path):
    fake_file(tmp_path)
    lines = []
    res = gtec.check(tmp_path, windows=(2.0, 3.0), out=lines.append)
    assert res["windows"][3.0]["correct"] == res["windows"][3.0]["total"] == 4
    assert any("guessing" in s for s in lines)


def test_live_decoder_chooses_each_light_once(tmp_path):
    p, order = fake_file(tmp_path)
    rec = gtec.load_recording(p)
    dec = gtec.LiveDecoder(rec.fs, gtec.Limits(0.3, 0.05, hold=3), window_s=2.0)
    fired = []
    for i in range(0, len(rec.eeg), 32):
        fired += dec.feed(rec.eeg[i:i + 32])
    assert [r for _, r in fired] == order


def test_mapping_of_lights_to_jobs():
    assert gtec.request_for(9) == ("request", "give me the bottle")
    assert gtec.request_for(10) == ("request", "give me the sharpie")
    assert gtec.request_for(12) == ("request", "give me the scissors")
    assert gtec.request_for(15) == ("stop", None)


def test_dry_run_sends_nothing():
    calls = []
    s = gtec.Sender(live=False, post=lambda *a: calls.append(a) or {})
    rec = s.send(9, 1.0)
    assert calls == [] and rec["sent"] is False and "dry run" in rec["reply"]


def _wait(cond, secs=2.0):
    end = time.time() + secs
    while time.time() < end and not cond():
        time.sleep(0.01)
    return cond()


def test_busy_reply_is_normal_and_not_retried():
    calls = []

    def post(path, body, timeout):
        calls.append((path, body))
        return {"ignored": "busy"}

    s = gtec.Sender(live=True, post=post)
    rec = s.send(10, 2.0)
    assert _wait(lambda: rec["reply"] is not None)
    assert calls == [("/api/request", {"text": "give me the sharpie"})]
    assert "busy" in rec["reply"] and s.busy is False


def test_a_choice_waits_while_a_job_is_out_but_stop_goes_through():
    calls, release = [], []

    def post(path, body, timeout):
        calls.append(path)
        if path == "/api/request":
            while not release:
                time.sleep(0.01)
            return {"agent": "ban", "say": "Yours."}
        return {}

    s = gtec.Sender(live=True, post=post)
    first = s.send(9, 1.0)
    assert _wait(lambda: "/api/request" in calls)
    second = s.send(12, 6.0)
    assert "waiting" in second["reply"] and calls.count("/api/request") == 1
    stop = s.send(15, 7.0)
    assert _wait(lambda: "/api/stop" in calls) and _wait(lambda: stop["reply"] == "stop sent")
    release.append(1)
    assert _wait(lambda: first["reply"] is not None and s.busy is False)
    assert first["reply"].startswith("ban")
