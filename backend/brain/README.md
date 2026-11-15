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

- `POST /api/stop` when the stop fires (either trigger)
- `POST /api/event` with `{"type": "brain", "state": "stop" | "clear" | "focus" | "no_signal", ...}`. Stop and clear carry `trigger`: `"alpha"` (eyes closed) or `"clench"`. Focus has `value` (0 to 1) and `frozen` (true while a jaw clench is holding it still).

If the server is down, or has no /api/event, the reader says so once and keeps going.

# Jaw-clench stop

Clench your teeth hard for about half a second and the arm stops. Relax your jaw and the stop clears about a second later. It sends the same stop as closing your eyes.

This is a muscle signal, not a brain signal. Clenching fires the jaw and neck muscles, and the puck picks up their electrical activity through the scalp: a burst of fast, broadband noise on every pin at once. The reader watches 45 to 100 Hz (leaving out 55 to 65 Hz for mains hum) and compares it with your own resting level. It is there because on some heads, the owner's included, closing the eyes barely changes alpha, so the eyes-closed stop cannot be trusted.

What it ignores, and why:

- Blinks and head movement put most of their power below 20 Hz. The stop also needs the fast band to rise at least twice as much as the slow band, so even a large slow swing or a pin popping cannot fire it.
- Talking is the same kind of muscle noise, but weaker and broken up by syllables. Calibration measures your talking and puts the threshold above it, and the clench has to stay above the threshold for 0.4 s.
- The chooser's flicker response sits under 45 Hz and is a few microvolts.

The likeliest false stop is a quick, hard head turn, because the neck muscles under the puck tense up. That has only been tried on the simulator.

Steps (puck seated, power bank, no PiEEG tab open):

1. Judge the seat, 20 seconds, nothing to do but sit still:

       python -m backend.brain signal --board

   One line a second: alpha, the 45 to 100 Hz level, the 60 Hz mains level and good pins, and the medians spoken at the end. You want 10 or more good pins, and mains well under the 45 to 100 Hz level. A high mains level means a charger or power brick is too close.

2. Calibrate, about a minute and a half, all spoken:

       python -m backend.brain clench-calibrate --board

   Rest for 10 s. Then five clenches: bite down hard at each high beep and let go at the low beep (2 s each, 3 s apart). Then count out loud in a quiet, normal voice for 10 s. Then five hard blinks, one at each beep. It prints how strong each clench was against your resting level, the strongest thing that was not a clench, and the margin between them, saves `setup\brain_clench.json`, and says whether the result is good. Good means every clench cleared the threshold, nothing else did, and the weakest clench was at least twice the strongest other thing. If it says not good, clench harder, press the puck down and run it again. `--quick` does three of each.

3. Run with the clench stop:

       python -m backend.brain run --board --stop clench

   `--stop` takes `alpha`, `clench` or `both`. Without it you get both when both calibrations exist and clench alone when only that one does. If the eyes-closed calibration itself warned that alpha barely changed, that stop is left out of the default (ask for it with `--stop both`). `ssvep-run` takes the same `--stop` switch, and a clench there blocks choosing under the same rules as closing the eyes.

On the simulator (no board): `--sim` with `clench-calibrate` and `signal` uses our own simulator and saves to `backend\brain\test_out\brain_clench_sim.json`, so it never overwrites your real calibration. `run --sim --sim-script "6:clench,8:release,12:blink,14:talk,19:quiet,22:turn"` scripts clenches, talking, blinks, head turns and `closed`/`open` eyes, in seconds from the start.

# Choosing an item by looking at it

A page shows one big black and white tile per item (bottle, red bull, sharpie, scissors, glue stick by default). Each tile flickers at its own rate. Look at one tile for a few seconds and the reader sends `POST /api/request {"text": "bring me the sharpie"}` to the agent server, the same request as saying it out loud. Closing your eyes still stops the arm, at the same time, and the stop always wins.

What this is: when you look at something flickering, the visual cortex at the back of your head picks up the same rate, a few microvolts strong. The reader finds which rate is there. It knows where you are looking. It does not read thoughts or intentions, and looking at a tile by accident for a few seconds will choose it.

Warning: the page flickers. Flicker can set off seizures in people with photosensitive epilepsy. Do not run it if the wearer, or anyone who can see the screen, has ever had a seizure. Nothing flickers until Start is pressed, and Esc stops it.

## Where the puck goes

The Octopus 16 is one puck with all 16 pins on one small patch of scalp, so there is a single seat for the whole head.

Seat it at Oz: on the midline at the back of the head, about 3 to 4 cm above the bump at the base of the skull (the inion). Run a finger up the middle of the back of the head from the bump; Oz is where the skull starts to flatten out. The visual cortex sits right under it, and eyes-closed alpha is strong there too, so this one seat serves both the chooser and the stop.

Part the hair, press the puck straight down so the pins reach skin, and hold it there with a snug headband or a beanie pulled over it. A headband across Oz works best if it sits just above the bump; check it does not tilt the puck. If contact at Oz stays poor (thick hair, the band keeps slipping), move to Pz, the seat in the section above. The stop works well there; the chooser will be weaker, since Pz is further from the visual cortex.

What to tell us: only where the puck sits (Oz or Pz) and whether some pins never get contact. The config file `backend/brain/ssvep_config.json` maps board inputs 1 to 16 to a seat name, all 16 at Oz by default. If pins are dead, drop them there or pass `--channels "Oz=1-6,9-16"`. The reader also leaves out any pin that fails the contact check each moment, so a dead pin mostly costs nothing.

## Session with the headset on

Run everything from the project root with `C:\Users\sudar\.conda\envs\lerobot\python.exe`, the agent server already running.

1. Close cloud.pieeg.com and OctoBridge tabs. The puck takes one Bluetooth connection at a time.
2. Seat the puck at Oz as above. Power it from the power bank.
3. Signal check: `python -m backend.brain signal --board` (20 s). Aim for 10 or more good pins. Then calibrate the stop: `python -m backend.brain clench-calibrate --board` (see "Jaw-clench stop" above). The eyes-closed stop is optional; if you use it, run its calibration with `--quick` and reseat if closing your eyes does not raise alpha.
4. Calibrate the chooser (about 90 seconds):

       python -m backend.brain ssvep-calibrate --board

   The page opens in the browser. Put it full screen on the screen you will use for the demo, sit at your normal distance, and press Start. A voice leads you: look at the cross in the middle, then at each highlighted tile in turn (8 seconds each), then close your eyes until two beeps. Keep your jaw loose and blink normally. It prints how well each tile was read and saves `setup/brain_ssvep.json`. Calibration is tied to that screen's refresh rate; calibrate again if you change screens.
5. Run:

       python -m backend.brain ssvep-run --board

   Press Start, look at a tile and keep looking. The bar under it fills; at full it says "chosen: sharpie" and sends the request. Look away for a couple of seconds before choosing the same item again. Clench your jaw (or close your eyes, if that stop is on) any time to stop the arm. Add `--dry` to print the choice and send nothing. Ctrl+C ends it.

If calibration shows weak responses for most tiles, use "Choosing by clenching" below instead.

When it picks wrong or never picks:

- One tile keeps getting chosen when you look elsewhere: that tile read weakly at calibration (see the printed accuracy). Calibrate again, or move the item you need most to the first slot with `--items`; the first items get the most reliable rates.
- Nothing gets chosen: check the terminal line for good pins. Fewer than a few good pins means reseat. Sit closer to the screen, turn the room lights down, and look at the middle of the tile, not its name.
- It picks while you are only glancing: calibrate with a longer hold, `--hold 8` (2 seconds of steady looking).
- Accuracy below about 60% for most tiles: the puck is probably not over Oz. Reseat lower on the head and calibrate again.

## How the chooser decides

- Rates: the page measures the screen's refresh rate, and each tile is white for half of a whole number of frames and black for the rest, so its rate is exact on that screen. Rates stay apart by at least 0.5 Hz, no rate sits on another's second or third harmonic, and rates between 7.5 and 13.5 Hz (the alpha band the stop watches) are avoided where the screen allows. At 144 Hz the five rates are 14.4, 16, 18, 20.57 and 6.55 Hz, none in the alpha band. At 60 Hz they are 15, 8.57, 20, 6 and 4.62 Hz; only 8.57 falls inside, and the lowest two read less reliably, so a 120 or 144 Hz screen is better.
- Detector: every 0.25 s, the last 3 s of the good pins are filtered (3 to 45 Hz, 60 Hz notch), reduced to their strongest few principal components (or all pins, whichever calibration found better), and compared by canonical correlation with sines and cosines at each rate and twice the rate. The reference waves use the sample rate measured from the stream's time stamps (the puck has shown about 230 a second, not 250). Each rate's score is set against how it scores when nobody looks at it. A choice needs the best score above a threshold, ahead of the runner-up by a margin, for 6 windows in a row (1.5 s), then nothing for 5 s.
- Keeping the stop and the chooser apart: the stop's alpha leaves out narrow bins around every flicker rate and its harmonic, so a tile response cannot look like closed eyes. The stop is checked first each tick; while it is on, or alpha is already above the stop level, nothing is chosen and choosing stays blocked for 5 s after it clears. Closing your eyes also removes the flicker response itself. Calibration records the stop level again at this seat with the tiles flickering.
- Lost Bluetooth packets: `--board` here starts `backend/brain/bridge.py`, which uses PiEEG's own Bluetooth code but fills each short gap in the packet counter with a straight line, so the samples keep their true spacing. Lost packets otherwise blur the faster rates. `--plain-reader` uses PiEEG's quickstart unchanged.

## Without the board

    python -m backend.brain ssvep-calibrate --sim
    python -m backend.brain ssvep-run --sim --dry

The simulator makes 16 pins of brain-like noise with a resting alpha, a five times bigger alpha with the eyes closed, mains hum, and a 2 microvolt response at whichever tile it is "looking" at. On the page in simulator mode, click a tile to look at it, click empty space to look away, and press E to close or open the eyes. Calibration moves the simulated gaze by itself. With no browser at all, add `--auto --no-browser` (assumes a 60 Hz screen) and drive the gaze with a script:

    python -m backend.brain ssvep-run --sim --dry --auto --no-browser --sim-script "2:bottle,10:none,12:sharpie,19:closed,24:open"

Other switches: `--sim-ssvep-uv 1.5` (weaker response), `--sim-rate 230`, `--items "cup,pen,tape"`, `--sentence "please pass the {item}"`, `--port 8767`, `--log file.jsonl`, `--ssvep-calibration path`.

## Choosing by clenching

Use this when the flicker calibration shows weak responses (on the owner, every tile read under 20%). Nothing flickers. The page lights its tiles one at a time, 1.5 s each. When the one you want is lit, clench your jaw briefly, about half a second. The page shows "chosen: sharpie" and the request goes to the agent server exactly as the flicker chooser sends it. To stop the arm, clench and hold for about 1.2 s; that stop always wins and also cancels a choice it interrupts. After a choice, and after a stop clears, scanning rests for 4 s and starts again from the first tile.

Like the clench stop, this reads a muscle signal picked up by the headset, not a brain signal.

1. Seat the puck and calibrate the clench as in "Jaw-clench stop" above. Nothing else is needed; it uses `setup\brain_clench.json`.
2. Run:

       python -m backend.brain scan-run --board

   Press Start on the page. Esc pauses. Add `--dry` to print the choice and send nothing.

Timing: a clench that starts within 0.3 s after the highlight moved on still counts for the tile that was lit before, since you were reacting to that one. A burst shorter than 0.4 s is ignored. Switches: `--dwell 2` (seconds per tile; give yourself more time if you miss tiles), `--pause 4`, `--long 1.1` (how long a burst must read on the detector to be the stop), `--items`, `--sentence`, `--port`, `--log`.

On the simulator: `python -m backend.brain scan-run --sim --dry`, then on the page press S for a short clench, L for a long one and B for a blink. Without a browser: `scan-run --sim --dry --auto --no-browser --sim-script "3.5:clench,4.0:release,10:blink,17:clench,19.5:release"` (seconds after Start). That one needs a simulator clench calibration first: `clench-calibrate --sim --quick`.

# If our Bluetooth reader cannot connect

The PiEEG dashboard (cloud.pieeg.com) has a Local Bridge desktop app that relays the stream over UDP, and an OSC stream out. We do not use them. They are the fallback if the direct reader cannot hold the puck; feeding either into an LSL stream named Octopus16 would let everything here run unchanged.

## g.tec recordings to the arm (no headset needed)

This is the route to show for the Mind and Machine track while no g.tec headset is in the room.

What it is: real brain recordings made with g.tec amplifiers, from the SSVEP set g.tec hands out for its BR41N.IO hackathons. Two people, two sessions each, eight electrodes over the back of the head, 256 samples a second. In each trial the person looks at one of four lights flickering at 9, 10, 12 or 15 times a second. We replay a session as a live stream, our own decoder (canonical correlation analysis) reads which light the person is looking at, and that choice goes to the characters as a request for the arm.

What it is not: the owner's own brain. The file also says which light was on, so every decoded choice is checked against it on the page.

| Light | Job |
|---|---|
| 9 Hz | give me the bottle |
| 10 Hz | give me the sharpie |
| 12 Hz | give me the scissors |
| 15 Hz | stop the arm |

The recordings go in `backend\brain\data\gtec_ssvep\` (four files named `subject_N_fvep_led_training_M.mat`). They came from https://github.com/bkowshik/br41n-ssvep, folder `data`.

Commands, from the project root:

1. Accuracy on every trial of every file:

       python -m backend.brain gtec-check

2. Replay a session live and decode it. Nothing is sent to the arm:

       python -m backend.brain gtec-run --subject 1

   The page at http://127.0.0.1:8767/ shows the four lights, the one the person is looking at, what we decoded, and how many were right.

3. The same, with the choices really sent to the running app (the arm moves):

       python -m backend.brain gtec-run --subject 1 --live

The limits that decide when a choice counts come from the same person's other session, never from the session being replayed. While the arm is on a job the next choice waits. A stop always goes out at once.

Measured with our decoder (guessing would give 25%):

| | One 3 s window per trial | Live replay |
|---|---|---|
| Subject 1 | 39 of 40 | 40 of 40 lights chosen right, one extra choice in a pause |
| Subject 2 | 26 of 40 | weak: most lights get no choice |

Use subject 1 for a demo. Subject 2's recordings have noisy channels and our decoder does not cope with them.

With a real g.tec Unicorn headset: start the Unicorn LSL app from the Unicorn Suite so the headset is on the network as a live stream, then

       python -m backend.brain gtec-run --no-replay --stream <the stream's name> --channels 8

This path has not been run, because no headset was available. The flicker lights would also be needed in front of the wearer at 9, 10, 12 and 15 Hz.
