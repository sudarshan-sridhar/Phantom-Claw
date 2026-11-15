# Brain stop (PiEEG Octopus 16)

The puck sits on Pz and reads alpha. Closing your eyes for about a second and a half stops the arm. Opening them clears the stop. A focus value from 0 to 1 also goes to the server once a second.

## Before you start

1. Flash the board if it has not been flashed yet. Plug it into the laptop by USB, open https://octobridge.pieeg.com/flash in Chrome or Edge, click Flash and pick the serial port. Unplug it when it is done.
2. Do not pair the puck in Windows Bluetooth settings. Our reader finds it by itself, and a Windows pairing can get in the way.
3. Close any PiEEG browser tab (cloud.pieeg.com, OctoBridge). The puck takes one Bluetooth connection at a time, and a tab will hold it.
4. Power the puck from a 5 V power bank only, never a wall charger or a laptop that is plugged in. Keep it away from chargers and power bricks, which add mains noise. A slow 1 Hz LED blink means it is advertising and ready.

## Seating the puck

Pz is on the midline of the head, a little behind the crown. Measure from the bridge of the nose over the top of the head to the bump at the back of the skull (the inion). Pz is about 70% of the way back along that line.

Part the hair there with your fingers. Press the puck straight down so all the pins go through to the scalp, then hold it in place with a headband or a beanie. Sit still and keep your jaw relaxed. Chewing and clenching show up as noise.

## Commands, in order

Run these from the project root (C:\Users\sudar\projects\third-hand) with C:\Users\sudar\.conda\envs\lerobot\python.exe. The server should already be running for step 3.

1. Calibrate (about 3 minutes, all spoken, eyes on nothing):

       python -m backend.brain calibrate --board

   It starts PiEEG's Bluetooth reader itself. First it does a seat check and says how many of the 16 pins have good contact, every few seconds, until at least 10 do. Then eyes open, eyes closed, and counting back from 700 in sevens, about 45 s each. A beep starts each step, and two beeps tell you to open your eyes. The results go to setup\brain_calibration.json. For a short version, add `--quick` (about 15 s per step).

2. If it warns that closing your eyes did not change the signal much, reseat the puck and run step 1 again.

3. Run:

       python -m backend.brain run --board

   One line a second shows alpha, good pins and focus. When the stop fires you hear a low beep, the arm stops (POST /api/stop) and a `brain` event goes out. Ctrl+C ends it and closes the Bluetooth link.

If fewer than 3 pins have good contact, or the stream stops, it sends `no_signal` and never fires a stop.

## Without the board

Add `--sim` in place of `--board` and PiEEG's simulator stands in for the puck. It switches between a quiet stretch and a strong 10 Hz burst every 6 s. Because nobody's eyes are involved, calibrate it with the test switch that sorts windows by their measured alpha:

    python -m backend.brain calibrate --sim --quick --auto-label
    python -m backend.brain run --sim

Other switches: `--mute` (no speech or beeps), `--seconds N` (stop after N seconds), `--log file.jsonl` (every tick to a file). If the reader is killed hard rather than with Ctrl+C, the simulator or Bluetooth reader it started can be left running. End it in Task Manager (a python.exe running python_quickstart.py).

## What it sends

All to the server on 127.0.0.1:8765:

- `POST /api/stop` when the stop fires
- `POST /api/event` with `{"type": "brain", "state": "stop" | "clear" | "focus" | "no_signal", ...}`. Focus has `value` (0 to 1) and `frozen` (true while a jaw clench is holding it still).

If the server is down, or has no /api/event, the reader says so once and keeps going.
