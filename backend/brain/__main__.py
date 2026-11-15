"""python -m backend.brain calibrate | run | testsource

  calibrate [--quick]       spoken calibration, saves setup/brain_calibration.json
  run                       eyes-closed STOP and focus, sent to the server
  ssvep-calibrate           calibrate choosing an item by looking at its flickering tile
  ssvep-run [--dry]         choose by looking, with the eyes-closed STOP on at the same time
  --sim                     start PiEEG's simulator ourselves (no board needed)
  --board                   start PiEEG's Bluetooth reader ourselves (real puck)
  neither                   use an Octopus16 LSL stream that is already running
"""
import argparse
import json
from pathlib import Path

from . import speech
from .calibrate import CAL_PATH, calibrate
from .link import DEFAULT_SERVER, ServerLink
from .reader import Reader, start_source
from .run import load_calibration, run


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
                    help='simulator: "0:none,3:sharpie,12:closed,16:open" (seconds after flicker starts)')

    args = p.parse_args()
    if args.cmd == "testsource":
        from .testsource import publish
        publish(args.mode, args.stream, args.seconds)
        return
    if args.cmd.startswith("ssvep"):
        ssvep_main(args)
        return

    speech.MUTED = args.mute
    source = None
    if args.sim or args.board:
        source = start_source("sim" if args.sim else "board")
        print(f"started PiEEG {'simulator' if args.sim else 'Bluetooth reader'} (pid {source.pid})")
    reader = Reader(args.stream)
    try:
        reader.open(timeout=40.0 if args.board else 20.0, source=source)
        print(f'reading LSL stream "{args.stream}": {reader.n_ch} channels, nominal {reader.nominal:.0f} Hz')
        if args.cmd == "calibrate":
            calibrate(reader, args.quick, args.auto_label, args.min_good, args.calibration)
        else:
            cal = load_calibration(args.calibration)
            summary = run(reader, ServerLink(args.server), cal, args.seconds, args.log)
            print("summary:", json.dumps(summary))
    finally:
        if source is not None:
            source.stop()


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
        else:
            cal = load_ssvep_calibration(cal_path)
            summary = ssvep_run(reader, cfg, cal, ServerLink(cfg.server), args.dry, sim, args.seconds,
                                not args.no_browser, args.auto, args.sim_script, args.log)
            print("summary:", json.dumps(summary))
    finally:
        if source is not None:
            source.stop()
        if sim is not None:
            sim.stop()


if __name__ == "__main__":
    main()
