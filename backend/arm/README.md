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
   Otherwise, offset more than 25 px: turn the base. Otherwise, for a compact item only (a case, a
   cap: the reader says `compact` and gives its centre row, width and whether the bottom edge cuts
   it off), centre row more than 30 px from y 235: move the hover in or out along the taught blend
   (the tier poses for the distance plus or minus the step); cut off by the bottom edge: 3 cm back
   toward the base. Within all: lined up. One correction per look, in that order.
3. Wait for the move, look again. At most 4 looks.
4. A compact item wider than 60 percent of the jaw gap at the present opening: open wider before
   going down (never above 2600 ticks). Then down, close and lift as before.

Long items are aimed exactly as before (they cross the jaw row wherever the hover is, so the reach
is not corrected for them). A compact item (`earbuds`, `case`, `cap` and the other round words in
`backend/vision/wrist_aim.py`) does not turn the wrist for the overhead angle: it starts at the
taught roll. For a compact OpenCV blob the confidence is capped at 0.45, so Gemini (asked with a
prompt that allows the item to be cut off between the jaw tips) decides; on 2026-11-15 OpenCV took
desk glare for the earbuds case and lined up on nothing. A Gemini box wider than 330 px is taken
for the jaws and not used. The miss check after closing uses the thin item rule (jaws stopping
within 50 ticks of the taught pen grip count as empty) only for long thin items; for a compact or
widened grip only nearly fully closed (100 ticks from 1310) is a miss.

A picture that cannot be read ends the loop and pick carries on with the aim it has (the overhead
one, or what the earlier looks already corrected). `stop()` works throughout, also while a picture
is being read. Each look sends a phase that starts with `aiming` and carries the numbers, for
example `aiming 1: turned +35 deg from the jaws, wrist roll -398 to 1618` or
`aiming 3: lined up (within 1 deg, 6 px)`. The result's `detail` ends with a `wrist aim:` summary
and the result carries `aim`, one entry per look (`angle_deg`, `offset_px`, `source`, `d_roll`,
`d_pan`, `capped`, `done`, or `unread`). Each look's picture is saved as
`setup/last_wrist_aim_N.jpg`, with the reading drawn on it in `setup/last_wrist_aim_N_read.jpg`.
A reach correction reads like `aiming 1: cut off by the bottom edge: too near, moving back, reach
-3.0 cm to 26.5`, and a wider opening adds a last `aim` entry with `opened` and `width_px`.

Which tiers aim: `PHANTOM_WRIST_AIM=low` (default), `all` or `off` (or a list such as `can,low`),
or `RealArm(wrist_aim=...)`, or `try_real ... --aim all`. `place` does not aim.

Constants, measured on the arm on 2026-11-15 with the king size Sharpie (adjust after the next run):

| what | value | where |
|---|---|---|
| jaw gap centre in the picture | x 452 of 640 (0.706 of the width) | `backend/vision/wrist_aim.py` `AIM["jaw_x"]` |
| jaw tips at the jaw row | x 300 (fixed jaw) and 520 at 1750 ticks; the right tip moves 0.5 px per tick of opening (first guess from two pictures) | `AIM["jaw_tips_ref"]`, `["jaw_open_ref"]`, `["jaw_px_per_tick"]` |
| cut off by the bottom edge | item reaches within 6 px of y 360 | `AIM["bottom_edge_px"]` |
| compact OpenCV blob, most it is trusted | 0.45 (below 0.6, so Gemini decides) | `AIM["compact_cv_conf"]` |
| widest compact Gemini box used | 330 px | `AIM["compact_max_width_px"]` |
| jaw row (where the centre line is read) | y 290 of 360 | `AIM["jaw_row"]` |
| item looked for above | y 240 (upper two thirds) | `AIM["item_rows"]` |
| OpenCV / Gemini sure enough | 0.6 / 0.5 | `AIM["cv_min_conf"]`, `AIM["gemini_min_conf"]` |
| Gemini time limit | 10 s | `AIM["gemini_timeout_s"]` |
| roll | roll_new = roll_now - angle x 11.38 (400 ticks turned 35 deg to vertical) | `backend/arm/desk.py` `WRIST_AIM["roll_ticks_per_deg"]`, `["roll_sign"]` |
| pan | pan_new = pan_now + offset_px / 1.4 (35 ticks lower moved the pen 50 px right at 29.5 cm) | `WRIST_AIM["pan_px_per_tick"]`, `["pan_sign"]` |
| pan gain with reach | 1.4 x dist_cm / 29.5 | `WRIST_AIM["pan_gain_scales"]`, `["pan_gain_ref_cm"]` |
| tolerance | 6 deg, 25 px | `WRIST_AIM["angle_tol_deg"]`, `["offset_tol_px"]` |
| largest single correction | 500 ticks roll, 80 ticks pan | `WRIST_AIM["max_roll_step"]`, `["max_pan_step"]` |
| looks | 4 | `WRIST_AIM["rounds"]` |
| reach target row (compact items) | y 235 of 360, just above the jaw row (first guess) | `WRIST_AIM["reach_target_y"]` |
| reach gain | 0.035 cm per px (jaw gap about 220 px for roughly 5 cm; a guess) | `WRIST_AIM["reach_cm_per_px"]` |
| reach tolerance, largest step | 30 px, 3 cm (also the step back when cut off) | `WRIST_AIM["reach_tol_px"]`, `["max_reach_step_cm"]` |
| open wider when | item width over 60 percent of the jaw gap | `WRIST_AIM["wide_ratio"]` |
| opening | 1750 + (width_px - 220) / 0.5 + 120 ticks, at most 2600 | `WRIST_AIM["open_margin_ticks"]`, `["open_max"]` |

To check the reader on saved pictures:

    python -m backend.vision.wrist_aim "setup/wrist_samples/*.jpg" --name sharpie --gemini --draw
    python -m backend.vision.wrist_aim setup/last_wrist_aim_1.jpg --name "earbuds case" --gemini --opening 1750

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
