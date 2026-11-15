# Phantom Claw build contract

Every part builds against this file. If something here is wrong or missing, say so in your report. Do not silently change it.

## Layout

```
third-hand/
  .env                  keys (never print, never commit)
  docs/contract.md      this file
  backend/              Python 3.12
    server.py           FastAPI app, port 8765
    arm/                skill server (real arm and pretend arm)
    agents/             cast, Merlin, LLM providers, voice
    tools/              laptop tools
    cache/tts/          saved speech (ignored by Git)
  overlay/              Electron app: avatars on the desktop + control panel
  setup/                calibration, taught poses, hand-written arm tools (reference)
```

Python to use for everything: `C:\Users\sudar\.conda\envs\lerobot\python.exe`. Install packages into that environment with its own pip. Do not create another environment.

Keys are read from `third-hand/.env`: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ELEVENLABS_API_KEY`. All three work. `ANTHROPIC_API_KEY` is empty.

Models known to work: OpenAI `gpt-4o-mini`. Gemini `gemini-3.5-flash-lite` (the larger Gemini flash models were overloaded). Model names live in one config dict so they can be changed in one place.

## The cast

| id | Name | Ego | LLM | Voice file key |
|---|---|---|---|---|
| escanor | Escanor | proud | openai | escanor |
| elizabeth | Elizabeth | caring | gemini | elizabeth |
| ban | Ban | greedy and eager | gemini | ban |
| king | King | lazy | openai | king |
| gowther | Gowther | neutral | openai (stands in for Claude) | gowther |
| merlin | Merlin | the Conductor | openai (stands in for Claude) | merlin |

Voice ids are in `setup/cast_voices.json` as `{id: {"voice": name, "voice_id": id}}`.

Character notes (lines are short, never longer than the action they go with):
- Escanor: thinks most jobs are beneath him. Refuses an easy arm job the first time and points instead. Does it properly when asked politely ("please") or asked a second time. His pride follows the laptop clock: full from 9:00 to 17:00, and from 21:00 to 6:00 he is timid, apologetic and does whatever he is asked.
- Elizabeth: gentle, checks before acting, asks if you need anything else.
- Ban: wants every job, quick, cocky, does a little more than asked.
- King: does the least that counts. Short answers. Sighs.
- Gowther: flat, literal, no feelings. Reports the result and nothing else.
- Merlin: calm, a little amused. Speaks only to hand a job over or to settle an argument.

## The plan (what an agent returns)

One JSON object. Free text never drives the arm.

```json
{
  "agent": "ban",
  "say": "Mine. Got it. What else you got?",
  "arm": {"skill": "grip_bottle", "pace": 0.8, "hold_s": 3, "flourish": true},
  "laptop": [{"tool": "note", "text": "buy solder"}]
}
```

- `agent`: one of the five agent ids.
- `say`: the spoken line, at most 140 characters.
- `arm`: `null`, or an object with `skill` from the list below. Anything else is rejected by Merlin's check.
- `laptop`: a list, possibly empty, of tool calls from the list below.

### Arm skills (the only ones that exist)

| skill | What the arm does | Parameters |
|---|---|---|
| `grip_bottle` | From rest: go to the bottle, grip it by the cap, lift, hold in the air, put it back, return to rest | `pace` 0.0 to 1.0 (0 slowest, 1 fastest), `hold_s` 1 to 8, `flourish` bool (small wrist roll while holding), `look_first` bool (pause above the bottle before lowering) |
| `point` | Raise the arm toward the bottle, hold 2 s, return to rest. No grip. | `pace` |
| `half_reach` | Go part of the way toward the bottle, stop, return to rest. No grip. | `pace` |
| `wave` | Raise to the photo pose, play the 5 s lively move, return to rest | none |
| `rest` | Return to rest and hold | none |
| `gesture` | Body language while a character talks. Rises from rest to a talking pose, moves in that character's manner for the length of the line, returns to rest. Never grips, never goes near the bottle. | `style`: an agent id including `merlin`; `seconds` 1.5 to 10 |

### The arm is the body of whoever is speaking

The arm moves whenever a character talks, to the user or to another character, even when there is no arm job. The server does this itself. It is not something the LLM chooses.

- A line with no arm skill: the server runs `gesture` with `style` set to the speaker and `seconds` set to the length of the audio clip, starting at the same moment the audio starts.
- A line that comes with an arm skill: the skill runs as before. Its ego settings already carry the manner.
- Merlin's lines and the argument lines each get their own `gesture` in that speaker's style.
- If the arm is busy with a skill, the gesture is skipped and the line is only spoken.

Gesture styles, all small, all above the desk, all inside the joint limits:
- escanor: rises tall and holds his head high, slow grand sweeps left and right.
- elizabeth: stays low and soft, small slow nods, a gentle sway.
- ban: quick darts left and right, wrist flicks, jaws snapping.
- king: barely lifts, sags, one slow shrug, flops back early.
- gowther: rises to a neutral pose, perfectly still, one precise head tilt.
- merlin: slow floating circles, jaws a little open.

Ego to arm mapping, applied by the agent prompt and clamped by the check:
- Gowther: `grip_bottle`, pace 0.5, hold 5, no extras.
- Escanor: `point` when refusing; `grip_bottle` pace 0.8, hold 5 when he agrees.
- Elizabeth: `grip_bottle`, pace 0.2, hold 4, `look_first` true.
- Ban: `grip_bottle`, pace 1.0, hold 3, `flourish` true.
- King: `half_reach` the first time; `grip_bottle` pace 0.4, hold 1 when asked again.

### Laptop tools

| tool | Fields | Effect |
|---|---|---|
| `note` | `text` | Append a line to `backend/notes.md` and send a `note` event |
| `todo` | `text` | Same, as a to-do item |
| `timer` | `seconds`, `label` | Start a timer; when it ends send a `timer_done` event |
| `open_url` | `url` (http or https only) | Open in the default browser |
| `lookup` | `query` | Ask the agent's own LLM for a two sentence answer; no web access yet |

No tool sends, deletes or pays for anything.

## Skill server interface (backend/arm)

```python
class Arm(Protocol):
    def run(self, skill: str, params: dict, on_phase: Callable[[str], None]) -> dict: ...
    def stop(self) -> None: ...          # hold where it is, drop the queue
    def status(self) -> dict: ...        # {"connected": bool, "busy": bool, "torque": bool, "joints": {...}}
```

- `run` blocks until the skill ends and returns `{"ok": bool, "result": "done" | "missed" | "stopped" | "error", "detail": str}`.
- `on_phase` is called with: `start`, `at_target`, `gripped`, `lifted`, `put_back`, `missed`, `returning`, `done`, `error`.
- Two classes: `RealArm` (serial port COM3) and `FakeArm` (sleeps for realistic durations, no hardware). The server uses `FakeArm` unless the environment variable `PHANTOM_ARM=real` is set.
- Only one skill runs at a time. A second request while busy is refused with result `busy`.

## HTTP and WebSocket API (backend/server.py, 127.0.0.1:8765)

- `POST /api/request` body `{"text": str, "agent": str | null}`. Runs the whole pipeline. Returns `{"agent", "say", "plan", "audio_url", "arm_result"}`.
- `POST /api/stt` multipart audio file. Returns `{"text"}` (ElevenLabs speech to text).
- `POST /api/stop`. Stops the arm.
- `GET /api/cast`. Returns the cast with names, egos, colours and current state.
- `GET /api/status`. Returns arm status and which providers answered last.
- `GET /audio/{file}`. Serves saved speech.
- `GET /ws`. WebSocket, server to client events, one JSON object per message:

| type | Fields |
|---|---|
| `request` | `text`, `named_agent` |
| `merlin` | `chosen`, `say`, `audio_url`, `argued` (list of `{agent, say, audio_url}`, may be empty) |
| `agent_state` | `agent`, `state`: `idle`, `listening`, `thinking`, `speaking`, `has_arm` |
| `say` | `agent`, `text`, `audio_url` |
| `arm` | `agent`, `skill`, `phase` |
| `note` | `agent`, `text`, `kind`: `note` or `todo` |
| `timer_done` | `agent`, `label` |
| `error` | `where`, `message` |

Pipeline for one request:
1. Send `request`.
2. If the text names an agent (tolerate near misses such as "Gauther", "Escanoor"), that agent takes it and Merlin says nothing. Otherwise Merlin picks by role and says one short line. About one request in four, when no agent is named, two agents each say one line wanting or dodging the job before Merlin decides.
3. The chosen agent's LLM returns the plan. If its provider fails or takes more than 8 s, fall through: gemini to openai, openai to gemini.
4. Merlin's check validates the plan against this file and clamps the numbers. A bad plan is replaced by a safe one: the agent says it cannot do that, no arm skill.
5. Speech is generated (cached on disk by voice id and text), `say` is sent, laptop tools run, then the arm skill runs with `arm` events.
6. Agent states are sent throughout. Everyone returns to `idle` at the end.

## Avatar colours

| id | Animal | Main colours |
|---|---|---|
| escanor | lion cub | #F5A623 gold, #C1440E deep orange |
| elizabeth | winged sprite | #E8F1F8 silver white, #8ECAE6 soft blue, #F4A6C0 pink |
| ban | fox | #C1121F crimson, #DCE3EA pale silver |
| king | bear cub | #2D6A4F green, #F4A261 orange |
| gowther | goat kid | #D63384 magenta pink |
| merlin | boar | #3C096C violet, #F5C400 gold eyes |

How the avatars are made:
- The base sprite for each character is generated once with the OpenAI image API as original pixel art: a cute chibi animal, front or three-quarter view, chunky readable pixels, limited palette in the colours above, transparent background. Model `gpt-image-1-mini` first; `gpt-image-2` only if the result is not usable. Hard cap: 12 image calls in total, because the account has about 5 dollars. Save the files in `overlay/assets/sprites/` and keep the prompts in `overlay/assets/sprites/prompts.json`.
- After generation each sprite is snapped to a true pixel grid in code (downscale to about 48 to 64 px with nearest colour, then draw scaled up with image smoothing off), so it is real pixel art on screen and not a blurry picture.
- Movement is done in code, like a desktop pet: each animal wanders along the bottom of the screen on its own, hops when it walks, flips to face the way it is going, breathes while idle, perks up when listening, bounces while speaking, and walks to the centre and grows a little when it has the arm. King mostly sleeps. Ban darts. Gowther stands still. Merlin floats above the others.
- A fallback sprite drawn in code must exist for every character, so the overlay still works if image generation fails.
- Nothing is traced or copied from the anime. No anime logos, no screenshots, no character art from the series. Only the names and the colour mood are shared.

## Rules for everyone building

- Never print, log or commit a key. Never put a key in a URL.
- Do not connect to the arm or send it commands. Build and test against `FakeArm` only. The real arm is tested by a person standing next to it.
- Stay inside your own folder. Do not edit another part's files or this contract.
- Plain code, plain comments, no em dashes. No Claude or AI attribution anywhere.
- Do not run `git commit` or `git push`.
- Finish with a short report: what you built, how to run it, what you tested and the result, and anything that does not work yet.
