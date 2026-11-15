"""python -m backend.brain calibrate | run | testsource

  calibrate [--quick]       spoken calibration, saves setup/brain_calibration.json
  run                       eyes-closed STOP and focus, sent to the server
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

    args = p.parse_args()
    if args.cmd == "testsource":
        from .testsource import publish
        publish(args.mode, args.stream, args.seconds)
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


if __name__ == "__main__":
    main()
