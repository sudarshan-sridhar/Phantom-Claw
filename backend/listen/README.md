# Voice control (backend/listen)

A separate process that owns the microphone. It sits asleep until it hears "Phantom Claw, start" (or you tap Ctrl+Shift), then sends anything that names a character to the server. "Phantom Claw, stop" (or "sleep", or Ctrl+Shift again) puts it back to sleep.

## Run it

From the project root, with the server already running or about to start (order does not matter, it waits and retries):

```
C:\Users\sudar\.conda\envs\lerobot\python.exe -m backend.listen
```

Useful flags:

```
--awake                 start awake
--no-require-name       unnamed requests go to the server with agent null, so Merlin picks
--stt cuda|cpu|server   force a speech path (default: try GPU, then CPU, then the server)
--model base.en         a different faster-whisper model
--mic 3                 a microphone by index or name (default: the built-in Microphone Array, else the system default)
--mic-test 3            open the mic for 3 s, print frame count and levels, quit
--no-hotkey             do not hook the keyboard
--file a.wav            play a 16 bit wav through exactly the same path as the mic (repeatable)
--file reply:b.wav      wait until a character is talking, then play b.wav over them (the self-hearing check)
```

Settings can also come from the environment: `LISTEN_REQUIRE_NAME` (default 1), `LISTEN_DEVICE`, `LISTEN_MODEL`, `LISTEN_MIC`, `LISTEN_VAD` (0 to 3, default 2), `LISTEN_ENERGY_MIN` (an extra loudness floor for a noisy hall, int16 RMS, default off), `LISTEN_SILENCE_MS` (450), `LISTEN_MIN_MS` (300), `LISTEN_MUTE_TAIL` (0.4), `PHANTOM_SERVER` (http://127.0.0.1:8765).

Stop it with Ctrl+C, or just end the process. It holds no files and no state of its own, so killing it is safe.

## How it works

1. 16 kHz mono audio from sounddevice in 30 ms frames. webrtcvad decides which frames are voice. Three voiced frames out of five start an utterance, 450 ms of quiet ends it, anything with less than 300 ms of voice is dropped, and nothing runs longer than 12 s.
2. Each utterance goes to faster-whisper small.en on the GPU (float16). The CUDA DLLs come from the `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` pip packages. If those are missing it uses base.en on the CPU (int8), and if faster-whisper will not load at all it posts the clip to the server's `/api/stt`. The path in use is printed at start. The model is loaded once and warmed with a dummy run.
3. Asleep, only the wake phrase counts. Near misses are accepted ("fantom claw", "phantom clock", "phantom cloud", "phantomclaw").
4. Awake, a name is found with Merlin's matcher (`backend.agents.merlin.find_named_agent`) plus common speech-recognition misses ("band", "van", "ben", "gother", "gauther", "scanner", "merlyn", "elisabeth"). A named utterance first sends a `hear` event so the avatar perks up at once, then `POST /api/request` with that agent. A name said alone becomes "Ban, are you there?". Naming Merlin sends agent null so she picks. With no name the utterance is ignored unless `--no-require-name`.
5. The listener follows the WebSocket. Every `say` and `merlin` event shuts the mic for the length of the clip plus 0.4 s (lines queue one after another, like the overlay plays them). Frames that arrive while muted never reach the speech model. A second request to an agent that is still working on the last one is ignored.
6. Ctrl+Shift: the `keyboard` package hooks the keyboard (no admin needed). Press Ctrl and Shift together with no other key and let go within 600 ms to toggle. Ctrl+Shift+Space and other shortcuts are not affected. The overlay also registers Ctrl+Shift+L as a backup, which toggles through the server.

Each utterance prints its timing: speech end to text (this includes the 450 ms of silence needed to know you stopped), text to request sent, request to first reply (the first `say` or `merlin` event), and the whole request.

## Tests

```
C:\Users\sudar\.conda\envs\lerobot\python.exe -m pytest -q backend/tests/test_listen.py
```

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
- First start downloads the speech model (about 480 MB for small.en) into the Hugging Face cache. After that it loads in about 2 s.
