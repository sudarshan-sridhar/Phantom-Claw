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

    s = gtec.Sender(live=True, post=post, quiet_s=0.0)
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

    s = gtec.Sender(live=True, post=post, quiet_s=0.0)
    first = s.send(9, 1.0)
    assert _wait(lambda: "/api/request" in calls)
    second = s.send(12, 6.0)
    assert "waiting" in second["reply"] and calls.count("/api/request") == 1
    stop = s.send(15, 7.0)
    assert _wait(lambda: "/api/stop" in calls) and _wait(lambda: stop["reply"] == "stop sent")
    release.append(1)
    assert _wait(lambda: first["reply"] is not None and s.busy is False)
    assert first["reply"].startswith("ban")


# ---- the gate, the quiet start, and what comes back from the agents

def test_gate_holds_back_what_it_is_not_sure_of(tmp_path):
    p, order = fake_file(tmp_path, noise=4.0)
    rec = gtec.load_recording(p)
    # limits set far too high: the lights are still the best guess, but never sure
    strict = gtec.Limits(0.95, 0.9, hold=3)
    dec = gtec.LiveDecoder(rec.fs, strict, window_s=2.0)
    fired = []
    for i in range(0, len(rec.eeg), 32):
        fired += dec.feed(rec.eeg[i:i + 32])
    assert fired == []
    assert {r for _, r, _, _ in dec.refused} >= set(order)      # every light was held back, none moved the arm
    # with the gate off the same decoder moves on its best guess
    loose = gtec.LiveDecoder(rec.fs, strict, window_s=2.0, gate=0.0)
    got = []
    for i in range(0, len(rec.eeg), 32):
        got += loose.feed(rec.eeg[i:i + 32])
    assert set(order) <= {r for _, r in got}


def test_offline_run_matches_the_file(tmp_path):
    p, order = fake_file(tmp_path)
    rec = gtec.load_recording(p)
    s = gtec.offline(rec, gtec.Limits(0.3, 0.05, hold=3), window_s=2.0)
    assert (s.trials, s.correct, s.wrong) == (4, 4, 0)


def test_feeding_in_big_or_small_pieces_gives_the_same_choices(tmp_path):
    p, _ = fake_file(tmp_path)
    rec = gtec.load_recording(p)

    def choices(step):
        dec = gtec.LiveDecoder(rec.fs, gtec.Limits(0.3, 0.05, hold=3), window_s=2.0)
        out = []
        for i in range(0, len(rec.eeg), step):
            out += dec.feed(rec.eeg[i:i + step])
        return out

    assert choices(1) == choices(8) == choices(500)


def test_nothing_is_sent_in_the_first_seconds():
    calls = []
    s = gtec.Sender("http://x", live=True, post=lambda path, body, timeout: calls.append(path) or {}, quiet_s=5.0)
    rec = s.send(9, 2.0)
    assert not rec["sent"] and "first 5 seconds" in rec["reply"]
    s.send(15, 3.0)                      # even a stop waits out the quiet start
    time.sleep(0.1)
    assert calls == []


def test_only_one_request_is_ever_out_and_stop_still_goes(tmp_path):
    import threading
    gate, calls = threading.Event(), []

    def post(path, body, timeout):
        calls.append(path)
        if path == "/api/request":
            gate.wait(2)
        return {"agent": "ban", "say": "Mine.", "plan": {"arm": [{"skill": "pick", "params": {"target": "bottle"}},
                                                               {"skill": "hand_over", "params": {}}]},
                "arm_result": {"ok": True, "result": "handed_over"}}

    s = gtec.Sender("http://x", live=True, post=post, quiet_s=0.0)
    first = s.send(9, 10.0)
    time.sleep(0.05)
    second = s.send(10, 11.0)            # arrives while the first is still out: dropped, not queued
    s.send(15, 12.0)                     # a stop goes out anyway
    time.sleep(0.1)
    assert calls.count("/api/request") == 1 and "/api/stop" in calls
    assert not second["sent"] and "waiting" in second["reply"]
    gate.set()
    time.sleep(0.2)
    assert first["agent"] == "ban" and first["steps"] == ["pick bottle", "hand_over"] and first["result"] == "handed_over"
    assert not s.busy


def test_answer_parts_reads_both_plan_shapes():
    one = gtec.answer_parts({"agent": "ban", "say": "On it.", "plan": {"arm": {"skill": "fetch_bottle"}},
                             "arm_result": {"result": "handed_over", "ok": True}})
    assert one["steps"] == ["fetch_bottle"] and one["result"] == "handed_over"
    none = gtec.answer_parts({"agent": "gowther", "say": "No.", "plan": {"arm": None}})
    assert none["steps"] == [] and none["result"] is None
    assert gtec.answer_parts({"ignored": "busy"})["agent"] is None


def test_confidence_is_100_at_the_gate():
    lim = gtec.Limits(0.4, 0.1)
    assert gtec.confidence({"best": 9, "score": 0.4, "lead": 0.1}, lim, 1.0) == 100
    assert gtec.confidence({"best": 9, "score": 0.8, "lead": 0.05}, lim, 1.0) == 50
    assert gtec.confidence({"best": None}, lim, 1.0) is None


def test_splice_plays_chosen_real_trials_in_order(tmp_path):
    p, _ = fake_file(tmp_path)
    rec = gtec.load_recording(p)
    sp = gtec.splice(rec, ["sharpie", "stop"], gap_s=10.0, lead_s=4.0)
    found = gtec.trials(sp.trigger)
    assert [r for _, _, r in found] == [10, 15]
    assert found[0][0] == int(4.0 * rec.fs)                       # the lead-in is no-light data
    assert found[1][0] - found[0][1] == int(10.0 * rec.fs)        # and so is the gap
    # every light sample is copied from the file, not made
    a, b, _ = found[0]
    src = next(t for t in gtec.trials(rec.trigger) if t[2] == 10)
    assert np.array_equal(sp.eeg[a:b], rec.eeg[src[0]:src[1]])
    # limits high enough that plain noise in the gap never counts
    s = gtec.offline(sp, gtec.Limits(0.5, 0.1, hold=3), window_s=2.0)
    assert (s.correct, s.wrong, s.false_in_pause) == (2, 0, 0)


def test_splice_refuses_a_name_it_does_not_know(tmp_path):
    import pytest
    p, _ = fake_file(tmp_path)
    with pytest.raises(ValueError):
        gtec.splice(gtec.load_recording(p), ["mug"])
