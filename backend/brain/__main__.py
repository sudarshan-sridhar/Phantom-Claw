"""python -m backend.brain calibrate | clench-calibrate | run | signal | ssvep-* | scan-run | testsource

  calibrate [--quick]        spoken eyes-closed calibration, saves setup/brain_calibration.json
  clench-calibrate [--quick] spoken jaw-clench calibration, saves setup/brain_clench.json
  run [--stop ...]           the STOP (alpha, clench or both) and focus, sent to the server
  signal [--seconds 20]      once a second: alpha, 45 to 100 Hz, 60 Hz and good pins
  ssvep-calibrate            calibrate choosing an item by looking at its flickering tile
  ssvep-run [--dry]          choose by looking, with the STOP on at the same time
  scan-run [--dry]           tiles lit in turn, a short clench chooses, a long clench stops
  --sim                      a simulator instead of the puck (no board needed)
  --board                    start PiEEG's Bluetooth reader ourselves (real puck)
  neither                    use an Octopus16 LSL stream that is already running
"""
import argparse
import json
from pathlib import Path

from . import speech
from .calibrate import CAL_PATH, calibrate
from .clench import CLENCH_CAL_PATH, SIM_CLENCH_CAL_PATH, load_clench, pick_stops
from .link import DEFAULT_SERVER, ServerLink
from .reader import Reader, start_source
from .run import alpha_reliable, load_calibration, run


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m backend.brain")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        src = sp.add_mutually_exclusive_group()
        src.add_argument("--sim", action="store_true", help="start the PiEEG simulator as a child process")
        src.add_argument("--board", action="store_true", help="start the PiEEG Bluetooth reader as a child process")
        sp.add_argument("--stream", default="Octopus16", help="LSL stream name to read")
        sp.add_argument("--mute", action="store_true", help="no speech or beeps (printing only)")
        sp.add_argument("--calibration", type=Path, default=CAL_PATH)

    c = sub.add_parser("calibrate")
    common(c)
    c.add_argument("--quick", action="store_true", help="about 15 s per step instead of 45")
    c.add_argument("--min-good", type=int, default=10, help="good channels wanted at the seat check")
    c.add_argument("--auto-label", action="store_true",
                   help="test only: label windows by measured alpha instead of by step")

    r = sub.add_parser("run")
    common(r)
    r.add_argument("--server", default=DEFAULT_SERVER)
    r.add_argument("--seconds", type=float, default=None, help="stop after this long (for tests)")
    r.add_argument("--log", type=Path, default=None, help="write every tick to this JSON lines file")
    r.add_argument("--stop", choices=["alpha", "clench", "both"], default=None,
                   help="which stop trigger (default: both when both calibrations exist, else whichever does)")
    r.add_argument("--clench-calibration", type=Path, default=None, help="default setup/brain_clench.json")
    r.add_argument("--sim-script", default=None,
                   help='with --sim, use our simulator and script it: "5:clench,7:release,12:blink,15:closed,18:open"')
    r.add_argument("--sim-rate", type=float, default=237.0, help="our simulator: samples per second")

    cc = sub.add_parser("clench-calibrate", help="spoken jaw-clench calibration")
    common(cc)
    cc.add_argument("--quick", action="store_true", help="shorter: 6 s rest, 3 clenches, 6 s talking, 3 blinks")
    cc.add_argument("--min-good", type=int, default=10, help="good channels wanted at the seat check")
    cc.add_argument("--no-seat-check", action="store_true", help="skip the seat check")
    cc.add_argument("--clench-calibration", type=Path, default=None, help="where to save (default setup/brain_clench.json)")
    cc.add_argument("--sim-rate", type=float, default=237.0)
    cc.add_argument("--sim-clench-uv", type=float, default=30.0, help="simulator: clench size in microvolts")
    cc.add_argument("--sim-talk-uv", type=float, default=8.0, help="simulator: talking size in microvolts")

    sg = sub.add_parser("signal", help="judge a seat: alpha, 45 to 100 Hz, 60 Hz and good pins, once a second")
    common(sg)
    sg.add_argument("--seconds", type=float, default=20.0)
    sg.add_argument("--sim-rate", type=float, default=237.0)
    sg.add_argument("--sim-script", default=None)

    t = sub.add_parser("testsource", help="publish a bad-contact stream for tests")
    t.add_argument("--mode", choices=["flat", "railed", "few"], required=True)
    t.add_argument("--stream", default="Octopus16Test")
    t.add_argument("--seconds", type=float, default=None)

    def ssvep_common(sp):
        common(sp)
        sp.add_argument("--config", type=Path, default=None, help="chooser config JSON (default backend/brain/ssvep_config.json)")
        sp.add_argument("--items", default=None, help='comma separated, for example "bottle,red bull,sharpie"')
        sp.add_argument("--items-url", default=None, help="fetch the item list from this URL instead")
        sp.add_argument("--channels", default=None, help='which inputs sit where, for example "Oz=1-16" or "O1=1-5,Oz=6-11,O2=12-16"')
        sp.add_argument("--port", type=int, default=None, help="chooser page port (default 8767)")
        sp.add_argument("--ssvep-calibration", type=Path, default=None, help="default setup/brain_ssvep.json")
        sp.add_argument("--no-browser", action="store_true", help="do not open the page by itself")
        sp.add_argument("--auto", action="store_true", help="simulator only: no page needed, assume 60 Hz and start at once")
        sp.add_argument("--sim-ssvep-uv", type=float, default=2.0, help="simulator: flicker response size in microvolts")
        sp.add_argument("--sim-rate", type=float, default=250.0, help="simulator: samples per second (the puck has shown about 230)")
        sp.add_argument("--plain-reader", action="store_true",
                        help="board: use PiEEG's quickstart as is, without filling lost packets")

    sc = sub.add_parser("ssvep-calibrate", help="calibrate choosing by looking at a flickering tile")
    ssvep_common(sc)
    sc.add_argument("--hold", type=int, default=6, help="windows in a row needed for a choice (0.25 s each)")

    sr = sub.add_parser("ssvep-run", help="choose by looking, with the eyes-closed stop on at the same time")
    ssvep_common(sr)
    sr.add_argument("--dry", action="store_true", help="print the choice and send nothing to the agent server")
    sr.add_argument("--server", default=None)
    sr.add_argument("--sentence", default=None, help='what a choice says to the server, default "bring me the {item}"')
    sr.add_argument("--seconds", type=float, default=None)
    sr.add_argument("--log", type=Path, default=None)
    sr.add_argument("--sim-script", default=None,
                    help='simulator: "0:none,3:sharpie,12:closed,16:open,20:clench,22:release" (seconds after flicker starts)')
    sr.add_argument("--stop", choices=["alpha", "clench", "both"], default=None,
                    help="which stop trigger (default: both when both calibrations exist, else whichever does)")
    sr.add_argument("--clench-calibration", type=Path, default=None, help="default setup/brain_clench.json")

    sn = sub.add_parser("scan-run", help="tiles lit in turn; a short jaw clench chooses, a long one stops")
    ssvep_common(sn)
    sn.add_argument("--dry", action="store_true", help="print the choice and send nothing to the agent server")
    sn.add_argument("--server", default=None)
    sn.add_argument("--sentence", default=None, help='what a choice says to the server, default "bring me the {item}"')
    sn.add_argument("--seconds", type=float, default=None)
    sn.add_argument("--log", type=Path, default=None)
    sn.add_argument("--dwell", type=float, default=1.5, help="seconds each tile stays lit")
    sn.add_argument("--pause", type=float, default=4.0, help="seconds scanning rests after a choice or a stop")
    sn.add_argument("--long", type=float, default=1.1,
                    help="seconds a clench must read on the detector to be the STOP (about a 1.2 s clench)")
    sn.add_argument("--clench-calibration", type=Path, default=None, help="default setup/brain_clench.json")
    sn.add_argument("--sim-script", default=None,
                    help='simulator: "3:clench,3.5:release,9:blink,14:clench,16:release" (seconds after Start)')

    args = p.parse_args()
    if args.cmd == "testsource":
        from .testsource import publish
        publish(args.mode, args.stream, args.seconds)
        return
    if args.cmd.startswith("ssvep") or args.cmd == "scan-run":
        ssvep_main(args)
        return

    speech.MUTED = args.mute
    source = sim = None
    stream = args.stream
    clench_path = clench_cal_path(args)
    # our own simulator has clenches, talking, blinks and closed eyes on demand; PiEEG's
    # simulator (plain run --sim and calibrate --sim) only alternates alpha bursts
    if args.sim and (args.cmd in ("clench-calibrate", "signal") or getattr(args, "sim_script", None)):
        from .ssvep_sim import SimSource
        sim = SimSource(name="Octopus16Sim", fs=args.sim_rate,
                        clench_uv=getattr(args, "sim_clench_uv", 30.0), talk_uv=getattr(args, "sim_talk_uv", 8.0)).start()
        stream = sim.name
        print(f"started our simulator at {args.sim_rate:.0f} samples a second")
    elif args.sim or args.board:
        source = start_source("sim" if args.sim else "board")
        print(f"started PiEEG {'simulator' if args.sim else 'Bluetooth reader'} (pid {source.pid})")
    reader = Reader(stream)
    try:
        reader.open(timeout=40.0 if args.board else 20.0, source=source)
        print(f'reading LSL stream "{stream}": {reader.n_ch} channels, nominal {reader.nominal:.0f} Hz')
        if args.cmd == "calibrate":
            calibrate(reader, args.quick, args.auto_label, args.min_good, args.calibration)
        elif args.cmd == "clench-calibrate":
            from .clench_session import clench_calibrate
            clench_calibrate(reader, args.quick, clench_path, sim, args.min_good, not args.no_seat_check)
        elif args.cmd == "signal":
            from .clench_session import signal_check
            if sim is not None and args.sim_script:
                from .ssvep_session import parse_script
                for t, kind, val in parse_script(args.sim_script, []):
                    if kind != "gaze":
                        sim.act(kind, val, delay_s=t)
            signal_check(reader, args.seconds)
        else:
            cal = load_calibration(args.calibration, required=False)
            clench_cal = load_clench(clench_path, required=False)
            stops, notes = pick_stops(args.stop, cal is not None, alpha_reliable(cal), clench_cal is not None)
            for n in notes:
                print(n)
            print(f"stop triggers: {', '.join(sorted(stops))}")
            script = None
            if sim is not None and args.sim_script:
                from .ssvep_session import parse_script
                script = parse_script(args.sim_script, [])
            summary = run(reader, ServerLink(args.server), cal, args.seconds, args.log,
                          clench_cal if "clench" in stops else None, stops, sim, script)
            print("summary:", json.dumps(summary))
    finally:
        if source is not None:
            source.stop()
        if sim is not None:
            sim.stop()


def clench_cal_path(args) -> Path:
    """The clench calibration file: as given, else the simulator's own file under --sim (so a
    simulator run never overwrites the real one), else setup/brain_clench.json."""
    given = getattr(args, "clench_calibration", None)
    if given:
        return given
    return SIM_CLENCH_CAL_PATH if getattr(args, "sim", False) else CLENCH_CAL_PATH


def ssvep_main(args) -> None:
    from . import ssvep
    from .ssvep_session import load_ssvep_calibration, ssvep_calibrate, ssvep_run

    speech.MUTED = args.mute
    cfg = ssvep.load_config(args.config or ssvep.CONFIG_PATH)
    if args.items:
        cfg.items = [s.strip() for s in args.items.split(",") if s.strip()]
    if args.items_url or cfg.items_url:
        url = args.items_url or cfg.items_url
        try:
            import httpx
            names = ssvep.items_from_payload(httpx.get(url, timeout=3.0).json())
            if names:
                cfg.items = names
                print(f"items from {url}: {names}")
            else:
                print(f"no item names at {url}; keeping {cfg.items}")
        except Exception as e:  # any failure here just means we keep the configured items
            print(f"could not fetch items from {url} ({type(e).__name__}); keeping {cfg.items}")
    if args.channels:
        cfg.channels = ssvep.parse_channels(args.channels)
    if args.port:
        cfg.port = args.port
    if getattr(args, "server", None):
        cfg.server = args.server
    if getattr(args, "sentence", None):
        cfg.sentence = args.sentence
    cfg.validate()
    if args.auto and not args.sim:
        raise SystemExit("--auto is for the simulator only; with the board, use the page and press Start")
    cal_path = args.ssvep_calibration or ssvep.SSVEP_CAL_PATH

    source = sim = None
    stream = args.stream
    if args.sim:
        from .ssvep_sim import SimSource
        sim = SimSource(name="Octopus16Sim", fs=args.sim_rate, ssvep_uv=args.sim_ssvep_uv).start()
        stream = sim.name
        print(f"started the chooser simulator (flicker response {args.sim_ssvep_uv} uV)")
    elif args.board and args.plain_reader:
        source = start_source("board")
        print(f"started PiEEG Bluetooth reader (pid {source.pid})")
    elif args.board:
        from .bridge import BRIDGE_LOG, start_bridge
        source = start_bridge()
        print(f"started the Bluetooth bridge that fills lost packets (pid {source.pid}, log {BRIDGE_LOG})")
    reader = Reader(stream)
    try:
        reader.open(timeout=40.0 if args.board else 20.0, source=source)
        print(f'reading LSL stream "{stream}": {reader.n_ch} channels, nominal {reader.nominal:.0f} Hz')
        if args.cmd == "ssvep-calibrate":
            ssvep_calibrate(reader, cfg, cal_path, sim, not args.no_browser, args.auto, args.hold)
        elif args.cmd == "scan-run":
            from .scan import scan_run
            clench_cal = load_clench(clench_cal_path(args))
            summary = scan_run(reader, cfg, clench_cal, ServerLink(cfg.server), args.dry, sim, args.seconds,
                               not args.no_browser, args.auto, args.sim_script, args.log, args.dwell, args.pause,
                               args.long)
            print("summary:", json.dumps(summary))
        else:
            from .run import alpha_reliable
            cal = load_ssvep_calibration(cal_path)
            clench_cal = load_clench(clench_cal_path(args), required=False)
            stops, notes = pick_stops(args.stop, True, alpha_reliable(cal), clench_cal is not None)
            for n in notes:
                print(n)
            print(f"stop triggers: {', '.join(sorted(stops))}")
            summary = ssvep_run(reader, cfg, cal, ServerLink(cfg.server), args.dry, sim, args.seconds,
                                not args.no_browser, args.auto, args.sim_script, args.log,
                                clench_cal if "clench" in stops else None, stops)
            print("summary:", json.dumps(summary))
    finally:
        if source is not None:
            source.stop()
        if sim is not None:
            sim.stop()


if __name__ == "__main__":
    main()
