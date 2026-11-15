# Arm skills, for whoever writes the planner

Every skill runs through `arm.run(skill, params, on_phase)` and returns
`{"ok": bool, "result": str, "detail": str, ...}`. One skill at a time (`busy` otherwise), `stop()`
holds where it is (`stopped`), and after `stopped` or `error` the next skill goes to rest first by
itself (`auto_rest: true`). Unknown parameters are dropped and numbers are clamped, as in
`docs/contract.md`. FakeArm (the default) and RealArm (`PHANTOM_ARM=real`) behave the same.

The old skills (`fetch_bottle`, `grip_bottle`, `point`, `half_reach`, `wave`, `rest`, `gesture`)
are unchanged; see `docs/contract.md`.

## Desk positions

Every position is in centimetres in the arm frame: from the arm base, `x` to the right in the
overhead picture, `y` straight away from the arm. `dist_cm` is the distance from the base,
`bearing` the angle in degrees from straight ahead (positive to the right). A lying item's
`axis_deg` is the direction of its long side in the same frame: 0 points straight away from the
base, positive turns toward `+x`, kept between -90 and 90.

## Tiers

Each item gets a tier, which picks the taught grips used for it:

| tier | what | where the grips come from |
|---|---|---|
| `tall` | stands about as high as the water bottle; gripped at the cap | `setup/bottle_teach.json` (the fetch_bottle grips), plus any tall spots in `setup/grip_teach.json` |
| `can` | a standing can (gripped round the body from above) or a cup (pinched at the wall nearest the arm) | `setup/grip_teach.json` |
| `low` | lies on the desk; the wrist turns so the jaws close across it | `setup/grip_teach.json` |

Known names decide the tier (bottle: tall; can, red bull, juice, cup, mug, glass: can; sharpie,
marker, pen, scissors: low); anything else uses Gemini's height guess. A tier is usable once at
least one spot (near, middle, far, farthest) has been taught with `setup/tools/teach_grip.py`.
Its reach is the nearest taught distance minus 4 cm to the farthest plus 2 cm. Taught on 2026-11-15:
`tall` 19.2 to 41.3 cm and `low` (Sharpie) 23.2 to 46.2 cm.

## Manner

`pace` (0 to 1) stretches every leg exactly as in fetch_bottle (1.8 times longer at 0).
`approach`: `direct`, or `look_first` (pause 1 s above the item before going down; also accepted
as `look_first: true`). `lift_cm_style`: `low`, `normal` or `high`, how high the item is carried
(shoulder 190, 230 or 320 ticks above the nearest taught grip). `wait_s` (2 to 6) for hand_over.
The ego numbers in the contract carry over: use the agent's fetch_bottle `pace` and `wait_s`,
and Elizabeth's `look_first` as `approach: "look_first"`.

## The general skills

| skill | params | results | phases |
|---|---|---|---|
| `look` | `target` (optional name) | `done` (plus `scene`: list of items, `person`, and `match` when a target was given), `no_picture`, `not_found` (Gemini failed) | start, seen, done |
| `point_at` | `target` (name or `{"x_cm","y_cm"}`), `pace` | `done` (plus `target`), `not_found`, `no_picture`, `out_of_reach` | start, seen, at_target, returning, done |
| `pick` | `target` (name, or `{"x_cm","y_cm"}` for the item nearest that spot), `pace`, `approach`, `lift_cm_style` | `holding` (plus `held`, and `aim` when it lined up with the wrist camera), `missed`, `not_found`, `no_picture`, `out_of_reach`, `not_taught`, `not_graspable` | start, seen, (aiming ...), at_target, gripped, lifted, done |
| `place` | `where`, `pace`, `approach` | `placed` (plus `placed_at`), `out_of_reach`, `not_found`, `nothing_held`, `turn_failed` | start, seen, carrying, at_target, put_back, returning, done |
| `hand_over` | `wait_s`, `pace` | `handed_over`, `not_taken`, `nothing_held`, `missed`, `turn_failed` | start, seen, carrying, offering, then handed_over, returning, done, or just done |

Each item in a `scene` list: `name`, `box`, `x_cm`, `y_cm`, `dist_cm`, `bearing`, `tier`,
`graspable`, `axis_deg` (lying items only). `person`: `x_cm`, `y_cm`, `bearing`, `pan`, or null.

`where` for place: `"back"` (where it was picked up, the default), `{"x_cm", "y_cm"}` (where the
item's centre goes), or `{"near": item name, "side": "left" | "right" | "front" | "behind"}`
(10 cm from that item's centre; sides as seen from the arm, front is toward the arm). A plain
name means in front of that item.

`ok` is true for `done`, `holding`, `placed` and `handed_over`.

### Last aim with the wrist camera (pick)

The overhead camera alone can be off by a couple of centimetres and badly wrong about a pen's angle.
So after pick reaches the hover pose above the item (the `via` pose, jaws open) and before it goes
down, it looks through the wrist camera and lines the jaws up, for `low` items by default:

1. Take one upright wrist picture (scaled to 640 x 360) and read it
   (`backend/vision/wrist_aim.py`): the angle of the item's long side clockwise from vertical, and
   the x where its centre line crosses the jaw row. Plain OpenCV first (the item's two long edges;
   instant); Gemini only when OpenCV is not sure (one call, about 10 s); neither sure: no reading.
2. Angle off by more than 6 degrees: turn the wrist roll only (turning the wrist turns the picture).
   Otherwise, offset more than 25 px: turn the base. Within both: lined up.
3. Wait for the move, look again. At most 3 looks; then down, close and lift as before.

A picture that cannot be read ends the loop and pick carries on with the aim it has (the overhead
one, or what the earlier looks already corrected). `stop()` works throughout, also while a picture
is being read. Each look sends a phase that starts with `aiming` and carries the numbers, for
example `aiming 1: turned +35 deg from the jaws, wrist roll -398 to 1618` or
`aiming 3: lined up (within 1 deg, 6 px)`. The result's `detail` ends with a `wrist aim:` summary
and the result carries `aim`, one entry per look (`angle_deg`, `offset_px`, `source`, `d_roll`,
`d_pan`, `capped`, `done`, or `unread`). Each look's picture is saved as
`setup/last_wrist_aim_N.jpg`, with the reading drawn on it in `setup/last_wrist_aim_N_read.jpg`.

Which tiers aim: `PHANTOM_WRIST_AIM=low` (default), `all` or `off` (or a list such as `can,low`),
or `RealArm(wrist_aim=...)`, or `try_real ... --aim all`. `place` does not aim.

Constants, measured on the arm on 2026-11-15 with the king size Sharpie (adjust after the next run):

| what | value | where |
|---|---|---|
| jaw gap centre in the picture | x 452 of 640 (0.706 of the width) | `backend/vision/wrist_aim.py` `AIM["jaw_x"]` |
| open jaw tips | about x 300 and 600 (`jaw_half_gap` 150) | `AIM["jaw_half_gap"]` |
| jaw row (where the centre line is read) | y 290 of 360 | `AIM["jaw_row"]` |
| item looked for above | y 240 (upper two thirds) | `AIM["item_rows"]` |
| OpenCV / Gemini sure enough | 0.6 / 0.5 | `AIM["cv_min_conf"]`, `AIM["gemini_min_conf"]` |
| Gemini time limit | 10 s | `AIM["gemini_timeout_s"]` |
| roll | roll_new = roll_now - angle x 11.38 (400 ticks turned 35 deg to vertical) | `backend/arm/desk.py` `WRIST_AIM["roll_ticks_per_deg"]`, `["roll_sign"]` |
| pan | pan_new = pan_now + offset_px / 1.4 (35 ticks lower moved the pen 50 px right at 29.5 cm) | `WRIST_AIM["pan_px_per_tick"]`, `["pan_sign"]` |
| pan gain with reach | 1.4 x dist_cm / 29.5 | `WRIST_AIM["pan_gain_scales"]`, `["pan_gain_ref_cm"]` |
| tolerance | 6 deg, 25 px | `WRIST_AIM["angle_tol_deg"]`, `["offset_tol_px"]` |
| largest single correction | 500 ticks roll, 80 ticks pan | `WRIST_AIM["max_roll_step"]`, `["max_pan_step"]` |
| looks | 3 | `WRIST_AIM["rounds"]` |

To check the reader on saved pictures:

    python -m backend.vision.wrist_aim "setup/wrist_samples/*.jpg" --name sharpie --gemini --draw

FakeArm aims too: by default a pretend item a little turned and to the left that answers the
corrections; `arm.wrist_reader` takes a list of scripted readings or a function instead.

### What holding means

`pick` ends with the item tucked in close (the carry pose), not at rest. While holding, only
`look`, `place`, `hand_over` and `rest` run; a `gesture` is skipped (result `done`,
`skipped: true`, nothing moves) and anything else returns `hands_full` without moving. `rest`
while holding stands the item on the desk where the arm is, lets go and folds back.
`status()["held"]` shows the item while holding (null otherwise); `status()["holding"]` is the
gripper's own reading.

`hand_over` turns to the person, holds the item out and waits for a tug or a hand. `not_taken`
means it tucked the item back in and is still holding it: the planner then chooses `place` (for
example `"back"`) or tries again.

Refusals (`not_found`, `out_of_reach`, `not_taught`, `not_graspable`, `nothing_held`,
`hands_full`) happen before anything moves. A `place` that refuses keeps holding the item.

### A plan, for example

```json
[{"skill": "look"},
 {"skill": "pick", "target": "red bull can", "pace": 0.8},
 {"skill": "hand_over", "wait_s": 4},
 {"skill": "place", "where": "back"}]
```
(the last step only when hand_over said `not_taken`).

## Trying it

FakeArm switches: `PHANTOM_FAKE_TIERS=tall,can,low` (pretend those tiers are taught),
`PHANTOM_FAKE_TAKEN=0` (nobody takes a handover), `PHANTOM_FAKE_OVERHEAD=0` (no picture),
`PHANTOM_FAKE_MISS=1` (every grip misses). The pretend desk holds the water bottle, a juice can,
a Red Bull can, a cup, a Sharpie and scissors.

On the real arm, with `--plan` first (prints, opens no port; `--look` also asks the camera):

    python -m backend.arm.try_real pick --target "red bull" --plan --look
    python -m backend.arm.try_real pick place --target "water bottle" --where back --pace 0
