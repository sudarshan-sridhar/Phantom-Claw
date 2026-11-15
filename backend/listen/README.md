# Voice control (backend/listen)

A separate process that owns the microphone. It sits asleep until it hears "Phantom Claw, start" (or you tap Ctrl+Shift), then sends what you say to the server. You don't have to use a name: an unnamed request goes to Merlin, who picks who takes it by the kind of job. "Phantom Claw, stop" (or "sleep", or Ctrl+Shift again) puts it back to sleep.

## Run it

From the project root, with the server already running or about to start (order does not matter, it waits and retries):

```
C:\Users\sudar\.conda\envs\lerobot\python.exe -m backend.listen
```

Useful flags:

```
--awake                 start awake
--require-name          only send utterances that name somebody (for a noisy room)
--no-require-name       the default: unnamed requests go to the server with agent null, so Merlin picks
--engine eleven|local|auto  speech to text (default auto: the local model when large-v3-turbo or another big model runs on the GPU, else ElevenLabs scribe if ELEVENLABS_API_KEY is set)
--stt cuda|cpu|server   where the local model runs (default: try GPU, then CPU, then the server)
--model small.en        a different faster-whisper model (default: best one already downloaded, see below)
--mic 3 | --mic headset a microphone by index or part of its name (default: the built-in Microphone Array)
--list-mics             list input devices and exit
--mic-test [5]          talk for 5 s; prints peak, RMS, noise floor, speech level and a verdict (TOO QUIET, OK, TOO LOUD)
--no-fix                do not repair near misses of the names and items
--no-hotkey             do not hook the keyboard
--file a.wav            play a 16 bit wav through exactly the same path as the mic (repeatable)
--file reply:b.wav      wait until a character is talking, then play b.wav over them (the self-hearing check)
```

Settings can also come from the environment: `LISTEN_ENGINE` (auto, eleven, local), `LISTEN_ELEVEN_TIMEOUT` (2.5 s, then the local answer is used), `LISTEN_ELEVEN_MODEL` (scribe_v1), `LISTEN_REQUIRE_NAME` (default 0; set 1 for a noisy room), `LISTEN_DEVICE`, `LISTEN_MODEL`, `LISTEN_BEAM` (5), `LISTEN_MIC` (index or part of the name), `LISTEN_VAD` (0 to 3, default 2), `LISTEN_ENERGY_MIN` (an extra loudness floor for a noisy hall, int16 RMS after the gain, default off), `LISTEN_SILENCE_MS` (700), `LISTEN_PREROLL_MS` (500), `LISTEN_MIN_MS` (300), `LISTEN_END_DB` (18; 0 turns it off), `LISTEN_MAX_S` (9), `LISTEN_AGC` (1; 0 stops normalising clips), `LISTEN_AGC_VAD` (0; 1 boosts frames before the VAD), `LISTEN_AGC_MAX_GAIN` (16), `LISTEN_FIX` (1), `LISTEN_MUTE_TAIL` (0.4), `PHANTOM_SERVER` (http://127.0.0.1:8765).

The microphone level: software cannot change the Windows input volume. If `--mic-test` says TOO QUIET, open Settings, System, Sound, Input, pick the microphone and set Input volume to about 90 (and Microphone Boost +10 or +20 dB if the Realtek panel offers it). The listener normalises each clip on its own, so it still works, but the VAD and the speech model do better with a real signal.

Stop it with Ctrl+C, or just end the process. It holds no files and no state of its own, so killing it is safe.

## How it works

1. 16 kHz mono audio from sounddevice in 30 ms frames. webrtcvad decides which frames are voice. With `LISTEN_AGC_VAD=1` each frame is first boosted by an automatic gain that follows the room's noise floor (`gain.LevelTracker`, at most 16 times); that is off by default because in the offline check it made far away voices count as speech, and quiet speech at the measured mic level was found without it. `--mic-test` prints voiced frames with and without it, so you can tell if your mic needs it. Three voiced frames out of five start an utterance (500 ms from before that is kept so the first syllable is not lost), 700 ms of quiet ends it (once it is going, anything 18 dB below its loudest part counts as quiet, so it still ends while other people talk further away), anything with less than 300 ms of voice is dropped, and nothing runs longer than 9 s.
2. The utterance is normalised to a peak of about half full scale (a clip with nothing above its own background is left alone), then transcribed. By default that is the local model when a big one is on the GPU (it beat ElevenLabs in the offline check, and is faster). With ElevenLabs (`LISTEN_ENGINE=eleven`, or the default when no big local model is there) the clip goes as a WAV straight to scribe_v1 with English and the names and items as keyterms; the local model works on the same clip at the same time and its answer is used if ElevenLabs fails or takes more than 2.5 s. The local model is faster-whisper on the GPU (float16, beam 5, temperature 0, English, the names, items and verbs in the prompt), using the best model already downloaded: large-v3-turbo, distil-large-v3, medium.en, else small.en. Nothing is downloaded at start. The CUDA DLLs come from the `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` pip packages. If those are missing it uses base.en on the CPU (int8), and if faster-whisper will not load at all it posts the clip to the server's `/api/stt`. The path in use is printed at start.
3. A small fixer (`fix.py`) repairs near misses of the six names (in the first one or two words, after "Phantom Claw start", or after a final comma), "Phantom Claw" anywhere, and the item names from a fixed list ("s kanor", "go there,", "marlin", "sharp e", "phantom clock", "fan tom claw"). It never rewrites anything else. The log shows the raw and the fixed text.
4. The wake and stop phrases: "Phantom Claw" anywhere in the utterance, matched by sound ("fantom", "fan tom", "phantum", "panthom", "phanton"; "clock", "claws", "clo", "cloth", "call", "clah", "close", "law", "glow"), followed within two words by a start word (start, starts, started, stat, star, go, begin, wake up, on) or a stop word (stop, stopped, stops, stap, top, halt, sleep, off, end, quit, "go to sleep"). "Phantom Claw" alone is a start. Asleep, only the wake phrase counts.
   Stop: on a stop phrase the listener calls `POST /api/stop` at once (arm and desk plan stop), before anything else, then goes to sleep. That works asleep too. While awake a bare "stop", "stop it", "stop now", "arm stop", "freeze", "halt", "please stop" or "Escanor, stop" does the same. Short clips are transcribed before long ones, so a stop is not stuck in the queue. The phrase lists the tests use are in `backend/tests/listen_phrases.py`.
5. Awake, a name is found with Merlin's matcher (`backend.agents.merlin.find_named_agent`) plus common speech-recognition misses ("band", "van", "ben", "gother", "gauther", "scanner", "merlyn", "elisabeth"). A named utterance first sends a `hear` event so the avatar perks up at once, then `POST /api/request` with that agent. A name said alone becomes "Ban, are you there?". Naming Merlin sends agent null so she picks (or answers herself). With no name the utterance still goes to the server with agent null, unless it is obviously not a request: fewer than three words (two are fine when they start like an order, "open notepad", "play music") or nothing verb-like in it (`backend.agents.merlin.looks_like_request`). Those print "ignored, not a request". With `--require-name` (or `LISTEN_REQUIRE_NAME=1`) every unnamed utterance is ignored, as before.
6. The listener follows the WebSocket. Every `say` and `merlin` event shuts the mic for the length of the clip plus 0.4 s (lines queue one after another, like the overlay plays them). Speech heard while muted is still cut and transcribed, but only a stop phrase in it is acted on; everything else is dropped as before. A second request to an agent that is still working on the last one is ignored, and when the server answers a request with `{"ignored": "busy"}` the listener logs it and does not retry.
7. Ctrl+Shift: the `keyboard` package hooks the keyboard (no admin needed). Press Ctrl and Shift together with no other key and let go within 600 ms to toggle. Ctrl+Shift+Space and other shortcuts are not affected. The overlay also registers Ctrl+Shift+L as a backup, which toggles through the server.

Each utterance prints its timing: speech end to text (this includes the 700 ms of silence needed to know you stopped), text to request sent, request to first reply (the first `say` or `merlin` event), and the whole request.

## Tests

```
C:\Users\sudar\.conda\envs\lerobot\python.exe -m pytest -q backend/tests/test_listen.py backend/tests/test_listen_audio.py
```

Accuracy check on synthetic speech (three cast voices through the project's ElevenLabs TTS, turned down to the laptop mic's level, with room noise at two levels; clips in `backend/listen/test_out/`, nothing goes to the server): run `python -m backend.listen.evalcheck` from the project root. Add `--old DIR` to score a copy of an older listener package kept in `DIR/listen_old`.

## Server additions (for docs/contract.md)

All in `backend/server.py`, nothing existing changed.

- `POST /api/event` body `{"type": "hear" | "listen" | "brain", ...any other fields}`. Broadcasts the object unchanged on `/ws`. Any other type is refused with 400. Used by the listener, and open to the brain process.
- `GET /api/listen` returns `{"awake": bool, "source": str | null, "changed_at": float | null}`.
- `POST /api/listen` body `{"awake": bool, "say": bool = true, "source": str | null}`. Sets the shared listening state. When it changes the server broadcasts `{"type": "listen", "awake": bool, "source"}` and, if `say` is true, Merlin says one short line through the normal speech path (a `merlin` event with `chosen: null`, `argued: []`, plus her gesture). Returns `{"awake", "changed"}`. The server starts asleep.

New WebSocket events:

| type | Fields |
|---|---|
| `hear` | `agent` (an agent id, or `merlin` when she will pick), `text` (what the listener heard). Sent before the request, so the avatar can react before the LLM runs. |
| `listen` | `awake`, `source` (`voice`, `shortcut`, `overlay`, `listener`, ...) |
| `brain` | whatever the brain process sends |

Overlay behaviour that goes with it: while asleep the avatars are off screen and only the tray icon and two small marks (linked, asleep) show. On `listen` awake they walk or drift in from the edges to their homes in about a second, and leave the same way on asleep (after Merlin's line). On `hear` the named avatar switches to its listening look and shows the heard words in its bubble until its reply. `npm start -- --show` (and demo mode) keep the avatars on screen all the time.

## For a launcher

- Working directory: the project root (`third-hand`) for the listener, `third-hand/overlay` for `npm start`.
- Any start order works. The listener retries the server WebSocket every 1.5 s and syncs the listening state when it connects. The overlay retries its WebSocket and stays hidden until the server says awake.
- No environment variables are needed. Do not set `PHANTOM_ARM` unless the real arm is meant to be used.
- Stopping: the listener stops on Ctrl+C (SIGINT) or a plain kill. The overlay quits from its tray menu, or by killing the electron process tree.
- The local model is the best one already in the Hugging Face cache (large-v3-turbo, 1.6 GB, is there now). Nothing is downloaded at start unless LISTEN_MODEL names a model that is not cached.

## Sharing the mic (Wispr Flow and other apps)

The listener opens the mic in shared mode only (MME by default, or DirectSound or WASAPI shared). It never asks for exclusive access, and it will not pick a WDM-KS (kernel streaming) device by name, because those can lock the mic away from other apps; a WDM-KS index given with `--mic` is swapped for the same mic under MME when there is one. So Wispr Flow (or Zoom, or the browser) can use the same mic at the same time. Both hear everything: what you dictate into Wispr Flow is also heard by the listener while it is awake, so say "Phantom Claw, stop" (or Ctrl+Shift) before long dictation, or run with `LISTEN_REQUIRE_NAME=1`. Wispr Flow's own hold-to-talk keys include the Windows key, which does not trip the Ctrl+Shift toggle.
