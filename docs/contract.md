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
    vision/             overhead and wrist cameras, Gemini boxes, desk mapping, taught grips
    agents/             cast, Merlin, LLM providers, voice
    tools/              laptop tools
    cache/tts/          saved speech (ignored by Git)
  overlay/              Electron app: avatars on the desktop + control panel
  setup/                calibration, taught poses, hand-written arm tools (reference)
```

Python to use for everything: `C:\Users\sudar\.conda\envs\lerobot\python.exe`. Install packages into that environment with its own pip. Do not create another environment.

Keys are read from `third-hand/.env`: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ELEVENLABS_API_KEY`, `ANTHROPIC_API_KEY`. All four work.

Models known to work: OpenAI `gpt-4o-mini`. Gemini `gemini-3.5-flash-lite` (the larger Gemini flash models were overloaded). Anthropic `claude-haiku-5-5`. Model names live in one config dict so they can be changed in one place.

## The cast

| id | Name | Ego | LLM | Voice file key |
|---|---|---|---|---|
| escanor | Escanor | proud | openai | escanor |
| elizabeth | Elizabeth | caring | gemini | elizabeth |
| ban | Ban | greedy and eager | gemini | ban |
| king | King | lazy | openai | king |
| gowther | Gowther | neutral | anthropic (falls through to openai) | gowther |
| merlin | Merlin | the Conductor | anthropic (falls through to openai) | merlin |

Voice ids are in `setup/cast_voices.json` as `{id: {"voice": name, "voice_id": id}}`.

Character notes (lines are short, never longer than the action they go with):
- Escanor: thinks most jobs are beneath him. Refuses an easy arm job the first time and points instead. Does it properly when asked politely ("please") or asked a second time. His pride follows the laptop clock: full from 9:00 to 17:00, and from 21:00 to 6:00 he is timid, apologetic and does whatever he is asked.
- Elizabeth: gentle, checks before acting, now and then (at most one line in three) asks if you need anything else.
- Ban: wants every job, quick, cocky, does a little more than asked.
- King: does the least that counts. Short answers. Sighs.
- Gowther: flat, literal, no feelings. Reports the result and nothing else.
- Merlin: calm, a little amused. Speaks to hand a job over, to settle an argument, or when spoken to directly.

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
- `arm`: `null`, or an object with `skill` from the list below (the single-skill form), or a list of up to 6 steps (the desk plan form, see "Desk plans"). Anything else is rejected by Merlin's check.
- `laptop`: a list, possibly empty, of tool calls from the list below.

### Desk plans (the general skills)

For anything on the desk other than the plain bottle jobs, `arm` is a list of at most 6 steps, run in order:

```json
"arm": [{"skill": "pick", "params": {"target": "red bull can"}}, {"skill": "hand_over", "params": {}}]
```

A step may also be written flat (`{"skill": "pick", "target": "cup"}`); the check turns it into `{skill, params}`. A single general skill written as an object is read as a one-step list, and a one-step list holding an old skill is turned back into the single-skill form. Allowed in a step: every old skill above except `gesture`, plus `look`, `point_at`, `pick`, `place`, `hand_over` (parameters, results and phases in `backend/arm/README.md`).

Merlin's check, for every step list:
- An unknown or forbidden skill anywhere, or a step that isn't an object, replaces the whole plan with the safe one.
- Unknown parameters are dropped. `target` and `where` are cleaned by the arm's own `validate`; an item name is changed to the matching name in the scene list. `pick` and `point_at` without a target are dropped.
- The manner knobs are set from the character, whatever the model wrote (below).
- Order: `place` and `hand_over` only after a `pick` in the same plan or when `status()["held"]` says something is held; no `pick` while holding; while holding only `look`, `place`, `hand_over` and `rest`; a `place` written right after a `hand_over` (for "nobody took it") is dropped because the runner does that. A plan never ends holding: unless the last step is `hand_over`, a `place` with `where: "back"` is added.
- Extras (a `pick` or `point_at` of an item the owner didn't name): allowed only when the item is graspable and 15 to 45 cm from the base, within the character's cap (Ban 1, Elizabeth 1, the others 0), and never when the owner says "only" or "just". For a request that names no item ("something to drink") the first item is the job, not an extra; "everything", "all", "clear" and "tidy" make every item part of the job.
- A desk question, or no picture from the overhead camera: no motion at all.
- In a desk job with the bottle in the scene, `fetch_bottle` becomes `pick` the bottle plus `hand_over`, and `grip_bottle` becomes `pick` plus `place` `"back"`, so the plan works from the scene like any other item.
- Anything the line promises must be in the steps (the prompt says so; the check can't read the line).
- King, the first time he's asked to fetch something from the desk: the plan becomes one `point_at` at the item (he complains); asked again ("I said...", or the same request again) he picks it and hands it over. He never adds a `look`.

Manner knobs per character (the fetch_bottle ego numbers, pushed apart so they show across a table):

| id | pace | approach | lift_cm_style | wait_s |
|---|---|---|---|---|
| escanor | 0.8 | direct | high | 2.5 |
| elizabeth | 0.25 | look_first | low | 6 |
| ban | 1.0 | direct | normal | 2 |
| king | 0.4 | direct | normal | 2 |
| gowther | 0.6 | direct | normal | 3 |

Planning habits in the prompts: Escanor does every desk job grandly and boasts, straight route, no extras. Elizabeth puts things down neatly next to something and may add one kind extra (the scissors away from the edge). Ban grabs and goes and may take one more thing for himself or bring a second item. King points when pointing answers it. Gowther does exactly what was asked and says the distance in cm. Merlin never carries; she hands desk jobs on and settles arguments.

What counts as a desk job (`cast.desk_kind`): a moving or pointing verb with a desk item, "something to drink", "everything" and similar, or "just/only the <item>". A desk question: "what's on my desk", "what do you see", "where is the <item>", "how far is the <item>". A request about timers, notes, searches, music, the screen or files is never a desk job. A request that names another desk item ("pick up the cup") is not a bottle job.

### Arm skills (the only ones that exist)

| skill | What the arm does | Parameters |
|---|---|---|
| `fetch_bottle` | Find the bottle and the person with the overhead camera and Gemini, turn to the bottle, grip the cap with the taught grips blended by distance, lift and tuck it in, turn to the person, hold it out and beep, let go when the person tugs it or the wrist camera sees a hand; if nobody takes it, carry it back to its spot; return to rest. Ported from `setup/tools/fetch_bottle.py`. | `pace` 0.0 to 1.0 (1.0 is the script's timings; lower stretches every leg, 1.8 times longer at 0), `wait_s` 2 to 6 (how long it holds it out), `look_first` bool (pause 1 s above the cap before the final approach) |
| `grip_bottle` | From rest: go to the bottle, grip it by the cap, lift, hold in the air, put it back, return to rest | `pace` 0.0 to 1.0 (0 slowest, 1 fastest), `hold_s` 1 to 8, `flourish` bool (small wrist roll while holding), `look_first` bool (pause above the bottle before lowering) |
| `point` | Raise the arm toward the bottle, hold 2 s, return to rest. No grip. | `pace` |
| `half_reach` | Go part of the way toward the bottle, stop, return to rest. No grip. | `pace` |
| `wave` | Raise to the photo pose, play the 5 s lively move, return to rest | none |
| `rest` | Return to rest and hold, from any pose a skill can leave the arm in. Holding the bottle out of rest: stand it on the table where the arm is, let go, lift clear, then fold back. Jaws open with the arm reaching out: lift straight up first. | none |
| `gesture` | Body language while a character talks. Rises from rest to a talking pose, moves in that character's manner for the length of the line, returns to rest. The whole gesture, rise and return included, lasts about the line's length (at most about 0.5 s longer); a short line rises less. Never grips, never goes near the bottle. | `style`: an agent id including `merlin`; `seconds` 1.5 to 10 |

Request kinds: being given, handed, passed or brought the bottle ("get me the bottle", "hand me the bottle", "pass it here") is `fetch_bottle`. Picking it up, lifting it or holding it up ("pick it up", "lift the bottle", "hold it up", "grab it") is `grip_bottle`. A request that only mentions the bottle ("what's in the bottle?") is not an arm job. Arm skills are only for physical requests.

fetch_bottle details:
- Where things are: the overhead camera (index 3) and Gemini give boxes for the bottle and the person, mapped onto the desk with `setup/table_map.json`. The bottle must be 19 to 41 cm from the base (`out_of_reach` otherwise). Person not seen: it holds the bottle out to the right (1.2 rad).
- Wrist camera only: when the overhead camera gives no picture (phone off: black frames or the iVCam "waiting" card), the arm tilts the wrist up from rest (wrist flex 2150), steps the base through five angles (-40 to +40 degrees, small slow steps), asks Gemini at each for the bottle and the person in the upright wrist picture, then turns until the bottle is in the middle of the picture. Distance cannot be measured with one camera, so it uses the taught middle grip (the result detail says so). Person not found: 45 degrees to the right. Switch it off with `PHANTOM_WRIST_FALLBACK=0`.
- The base runs at full torque (1000) during the fetch and goes back to 500 at rest. The bottle is tucked in before the base turns. While carrying, a joint 110 ticks short and still counts as arrived.
- A turn that cannot finish (stall or too much current): it stands the bottle on the table where it stopped, lets go, lifts clear and goes home (`turn_failed`).

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

Ego to arm mapping, applied by the agent prompt and clamped by the check (hand it over / pick it up):
- Gowther: `fetch_bottle` pace 0.6, wait 3 / `grip_bottle` pace 0.5, hold 5, no extras.
- Escanor: `point` when refusing (first time, during the day, not polite); when he agrees (asked politely, asked again, or at night) `fetch_bottle` pace 0.8, wait 3 / `grip_bottle` pace 0.8, hold 5.
- Elizabeth: `fetch_bottle` pace 0.3, wait 5, `look_first` true / `grip_bottle` pace 0.2, hold 4, `look_first` true.
- Ban: `fetch_bottle` pace 1.0, wait 2 / `grip_bottle` pace 1.0, hold 3, `flourish` true.
- King: `half_reach` the first time; asked again `fetch_bottle` pace 0.4, wait 2 / `grip_bottle` pace 0.4, hold 1.

Spoken lines: always contractions ("I'm", "it's", "don't"); the check applies common contractions to any line that misses them, and keeps Elizabeth's "anything else?" to at most one line in three.

### Laptop tools

All tools live in one registry, `backend/tools` (`TOOLS`). Merlin's check knows only those names and cleans every field.

| tool | Fields | Effect |
|---|---|---|
| `note` | `text` | Append a line to `backend/notes.md` and send a `note` event |
| `todo` | `text` | Same, as a to-do item |
| `timer` | `seconds` (1 to 3600), `label` | Start a timer; when it ends send a `timer_done` event |
| `open_url` | `url` (http or https only) | Open in the default browser |
| `lookup` | `query` | Ask the agent's own LLM for a short answer in that character's voice; no web access |
| `open_app` | `name` | Open an installed program by spoken name: Start menu shortcuts (user and all-users, searched recursively, fuzzy match, uninstallers and setup entries skipped) plus a table of common names (chrome, edge, notepad, calculator, file explorer, vs code, spotify, discord, terminal, settings, word, excel, powerpoint, paint, camera, snipping tool, and a few sites). Launched with `os.startfile`. No match: says so, never guesses a path |
| `open_folder` | `name` | Downloads, Documents, Desktop, Pictures, Videos, Music, Screenshots, or the project folder |
| `open_file` | `query` | The newest file in Downloads, Documents or Desktop (3 levels deep) whose name has every word; opened with its usual program. Programs, scripts, installers and shortcuts are never opened |
| `web_search` | `query` | Open a web search in the default browser |
| `youtube` | `query` | Open a YouTube search |
| `play_music` | `query` | Open Spotify on a search (`spotify:search:`), or the web player if the app isn't installed |
| `media` | `action`: `play_pause`, `next`, `previous`, `volume_up`, `volume_down`, `mute` | Press the media key (volume moves about 10 percent) |
| `type_text` | `text` (at most 600 characters) | Type into the window in front, like dictation. Text only: line breaks become spaces, no key combinations, nothing that mentions a password |
| `read_screen` | `question` | One screenshot, shrunk in memory, sent to Gemini (`gemini-3.5-flash-lite`, key in the `x-goog-api-key` header) with the question; nothing is saved |
| `clipboard` | `question` | Read the clipboard text and answer about it or sum it up (same Gemini path, text only) |
| `tell` | `what`: `time`, `date`, `battery`, `window` | Read the clock, the battery, or the title of the window in front |
| `show_desktop` | none | Win+D |
| `screenshot` | none | Save a new PNG in `Pictures\Screenshots` (never overwrites) and say where |

Gated tools: `read_screen`, `clipboard` and `type_text` are dropped by the check unless the request is plainly about the screen, what was copied, or typing.

Dry run: `tools.set_dry_run(True)` or `PHANTOM_DRY_TOOLS=1`. Each tool works out what it would do (which shortcut, folder, file or address) and reports it as `{"ok": true, "dry": true, "would": "..."}` without opening, typing, pressing or capturing anything. `python -m backend.agents.demo "open notepad" --dry-tools` uses it.

Results that need saying: `lookup`, `read_screen`, `clipboard`, `tell`, a program or file that isn't there, a screenshot's place, a refusal. After the first line (which only says "hang on" for answer tools), the character speaks one more short line with the result as a second `say` event. Answers from `lookup`, `read_screen` and `clipboard` come back already in the character's voice (the tool's own LLM call is told who is talking); simple facts use a fixed line per character, no extra LLM call. Answer tools start while the first line is being made and played. A second line follows the same rules as the first, may run to 220 characters, and at most two follow per request. A missing extra (Ban's bonus app) is not announced when the main thing opened.

How each character uses the laptop (prompts, with the caps enforced by the check):
- Escanor: does every laptop job, even at night; a trivial one with a remark that it was beneath him. Research always gets `lookup` plus one page (the check adds whichever he forgot). At most 2 things opened, 1 extra.
- Elizabeth: says back what she understood in a few words. At most one extra, and only a gentle one (`timer`, `note`, `todo`). At most 2 opened.
- Ban: the thing plus one related thing. His extra must be something opened. At most 3 opened.
- King: exactly one tool call, the shortest route.
- Gowther: exactly what was asked; extras are dropped.
- Everyone: at most 3 or 4 calls in total, duplicates dropped.

Never, whoever asks: delete, move, rename or overwrite files; send messages or email; pay or buy; type or read passwords; shut down, restart, sign out or lock the machine; change system or security settings; install anything. `backend/tools/safety.py` spots these in the request (a note or reminder about them, "remind me to pay rent", and a how-to question, "how do I delete a branch?", are fine; passwords are refused in every form). The check then drops every tool and the arm, and if the line doesn't already say no it is replaced by that character's refusal line. Only `notes.md` is ever written to (appended), plus new screenshot files.

## Skill server interface (backend/arm)

```python
class Arm(Protocol):
    def run(self, skill: str, params: dict, on_phase: Callable[[str], None]) -> dict: ...
    def stop(self) -> None: ...          # hold where it is, drop the queue
    def status(self) -> dict: ...        # {"connected": bool, "busy": bool, "torque": bool, "joints": {...}}
```

- `run` blocks until the skill ends and returns `{"ok": bool, "result": str, "detail": str}`, plus `"auto_rest": true` when the arm went to rest by itself first.
- Results: `done`; fetch_bottle: `handed_over`, `put_back` (nobody took it), `missed`, `no_bottle` (no camera can see it), `out_of_reach`, `turn_failed`; and for every skill `stopped`, `error`, `busy`. `ok` is true for `done`, `handed_over` and `put_back`.
- `on_phase` is called with: `auto_rest`, `start`, `seen`, `at_target`, `gripped`, `lifted`, `carrying`, `offering`, `handed_over`, `put_back`, `missed`, `returning`, `done`, `error`.
  - fetch_bottle, in order: `start`, `seen`, `at_target`, `gripped`, `lifted`, `carrying`, `offering`, then `handed_over` or `put_back`, `returning`, `done`. Missed: `start`, `seen`, `at_target`, `missed`, `returning`, `done`. Turn failed: ... `carrying`, `put_back`, `returning`, `done`. Not seen: `start`, `done` (or `start`, `returning`, `done` after a wrist sweep). Out of reach: `start`, `seen`, `done`.
  - rest: `start`, `returning`, `done`, with `put_back` after `start` when it first puts a held bottle down.
- `stop()` holds every arm joint where it is at any stage, never switches torque off, and leaves the gripper's squeeze alone (a held bottle stays held).
- After `stopped` or `error` the arm is out of rest. The next skill sends it to rest first by itself: the pipeline runs `rest` (its `arm` events carry `"auto": true`, starting with phase `auto_rest`, and the skill's result carries `auto_rest` with the rest result); a `gesture` that would follow is skipped. The arm does the same as a backstop when used directly (phase `auto_rest`, then the skill).
- `status()` also gives `at_rest`, `holding` (the gripper squeezing something) and `held` (the item a `pick` left in the carry pose, as in a scene entry, or null).
- While holding (after `pick`, before `place`, `hand_over` or `rest`), a `gesture` is skipped (result `done`, `skipped: true`, nothing moves) and any skill other than `look`, `place`, `hand_over` and `rest` returns `hands_full` without moving. The pipeline does not send the arm to rest before a skill while `held` is set, so a gesture never puts a carried item down.
- General skills (`look`, `point_at`, `pick`, `place`, `hand_over`): see `backend/arm/README.md`. Their extra results: `holding`, `placed`, `not_found`, `no_picture`, `not_taught`, `not_graspable`, `nothing_held`, `not_taken`, `hands_full`; `ok` is also true for `holding` and `placed`.
- Two classes: `RealArm` (serial port COM3) and `FakeArm` (sleeps for realistic durations, no hardware). The server uses `FakeArm` unless the environment variable `PHANTOM_ARM=real` is set. FakeArm switches for fetch_bottle: `PHANTOM_FAKE_FETCH` (an outcome name), `PHANTOM_FAKE_OVERHEAD=0` (pretend the phone is off), `PHANTOM_FAKE_MISS=1`.
- Only one skill runs at a time. A second request while busy is refused with result `busy`.

## HTTP and WebSocket API (backend/server.py, 127.0.0.1:8765)

- `POST /api/request` body `{"text": str, "agent": str | null}`. Runs the whole pipeline. Returns `{"agent", "say", "plan", "audio_url", "arm_result"}`, plus `tools` (each tool's result), `followups` (the result lines), `group` (chip-in lines) and `routed` (`{"why", "by"}`). When Merlin answers herself, `agent` is `merlin`.
- `POST /api/stt` multipart audio file. Returns `{"text"}` (ElevenLabs speech to text).
- `POST /api/stop`. Stops the arm, and a running desk plan runs no further steps.
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
| `arm` | `agent`, `skill`, `phase`, and `auto: true` for an automatic rest; in a desk plan also `step` (0 based), `steps` (plan length), and `retry: true` or `recovery: true` on the extra look and put-back steps |
| `scene` | `agent` (null while Merlin is still routing), `result` (`done`, `no_picture`, `not_found`, `no_arm`, ...), `items` (`name`, `x_cm`, `y_cm`, `dist_cm`, `tier`, `graspable`), `ms`: the look before a desk plan or a desk question |
| `arm_plan` | `agent`, `phase`: `start` (with `steps`, the checked list) or `done` (with `result`, `ok` and `steps`, what each step returned) |
| `arm_step` | `agent`, `step`, `skill`, `params`, `result`, `ok`, `detail`, and `cached`, `retry`, `recovery`, `auto_rest` when they apply; one per step run |
| `note` | `agent`, `text`, `kind`: `note` or `todo` |
| `timer_done` | `agent`, `label` |
| `error` | `where`, `message` |

Pipeline for one request:
1. Send `request`.
2. If the text names an agent (tolerate near misses such as "Gauther", "Escanoor"), that agent takes it and Merlin says nothing. "Merlin, ask Ban to ..." goes to Ban with a line from Merlin. Spoken to directly ("Merlin, how are you?"), Merlin's own LLM either answers in one line (a `merlin` event with `chosen: null`, no agent speaks) or hands the job on. Otherwise Merlin picks by the kind of job and says one short line:
   - research, facts and hard questions: Escanor
   - timers, reminders, breaks, the owner's wellbeing: Elizabeth
   - opening apps, files, folders, pages and music, media keys, setting things up: Ban
   - notes, to-dos, dictation: King
   - the screen, the clipboard, exact facts about the machine, code and commands: Gowther
   - the bottle: whoever is named, otherwise Ban
   Unnamed requests are routed by Claude (`claude-haiku-5-5` through `providers.get_json("anthropic", ...)`, a small JSON answer `{"agent", "kind", "why"}`, 3 s cap, 4 s for a desk job since its look runs alongside). The answer is checked against the cast; on a failure, a timeout or a bad name the fixed rules in `merlin.job_kind` decide (an `error` event with `where: "route"` says so). A request asked again goes back to whoever dodged it, and the bottle goes to Ban, without asking Claude. Safety refusals are decided in code, never by the model.
   "Everyone" or "all of you": the one picked does the job and one or two others chip in a brief line each (no tools, no arm).
   When the kind of job isn't clear, about one request in four, two agents each say one line wanting or dodging the job before Merlin decides.
3. The chosen agent's LLM returns the plan. If its provider fails or takes more than 8 s, fall through: gemini to openai, openai to gemini.
4. Merlin's check validates the plan against this file and clamps the numbers. A bad plan is replaced by a safe one: the agent says it cannot do that, no arm skill.
5. Speech is generated (cached on disk by voice id and text), `say` is sent, laptop tools run (the ones that answer start while the line is being made), the line with the result is sent as another `say`, any chip-in lines follow, then the arm skill runs with `arm` events.
6. Agent states are sent throughout. Everyone returns to `idle` at the end.

Desk jobs and desk questions (not the plain bottle jobs, which keep fetch_bottle and grip_bottle):
- The arm's `look` runs first, at the same time as Merlin's routing (a `scene` event follows). The scene list (names, positions in cm, tier, graspable), the person and `status()["held"]` go into the character's prompt, so the plan names real items. A look is kept for 6 s (`PHANTOM_SCENE_TTL`), so a `look` step in the same plan reuses it.
- No picture: the character says so in a fixed plain line, no LLM call, nothing moves.
- A desk question goes to Gowther (unless someone is named) with no Claude call. The first line is a quick one; the answer is built from the scene list in code and spoken as the second line.
- Unnamed desk jobs are routed by Claude by the kind of job (showy fetch: Escanor; careful or fragile, tidy: Elizabeth; quick grab: Ban; pointing, "where": King; exact placing, distances: Gowther), told who had the arm last so it is shared around. The fixed-rule fallback (`merlin.desk_rule`) does the same with a rotation.
- The runner runs the steps in order through `run_arm` (same `arm` events, stop handling and auto-rest), checks `/api/stop` between steps, and reacts: `pick` missed, one retry after a fresh `look`, then it gives up; `not_found`, `out_of_reach`, `not_taught`, `not_graspable`, `no_picture` stop the plan; `hand_over` `not_taken` is followed by `place` `"back"`; any other failure while holding is followed by `place` `"back"` (and `rest` if that fails); `stopped` moves nothing more. A last `say` from that character's fixed lines tells what really happened (handed over, placed, not taken and put back, missed twice, not taught, and so on).
- `arm_result` for a desk plan is `{"ok", "result", "skill": "plan", "steps", "item", "say", "detail"}`; `result` is `handed_over`, `placed`, `pointed`, `done`, `not_taken`, `put_back`, `stopped`, `failed` or the refusal. The response also carries `timing` (`route_ms`, `look_ms`, `look_ready_ms`, `plan_ready_ms`, `first_say_ms`, `arm_ms`).

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
