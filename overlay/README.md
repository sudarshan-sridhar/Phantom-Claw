# Phantom Claw overlay

Six pixel pets live across the whole primary screen: Escanor, Elizabeth, Ban, King, Gowther and Merlin. Each has a home in a different part of the screen and roams from it: walking along the bottom edge, perching on the top and side edges, drifting or leaping between spots, resting, then moving on. They react live to the Phantom Claw server. Tags, speech, notes and the input bar use the Hack Dearborn: Conjure Reality look (brand palette, fonts, and the artist's scroll, crystal ball, sparkle and wand).

## Install and run

From this folder (`third-hand/overlay`):

```
npm install
npm start          # live mode, connects to ws://127.0.0.1:8765/ws
npm run demo       # scripted show, no server needed
npm run sprites    # make the base sprites with the OpenAI image API (see below)
```

Other scripts:

```
npm run snap       # re-snap the saved raw sprites to the pixel grid, no API calls
npm run shot       # demo plus four screenshots in test/out/, then quits
npm run selftest   # checks click-through and the click-to-ask bar, prints SELFTEST lines, then quits
```

Add `--fallback` (for example `npx electron . --demo --fallback`) to see the code-drawn sprites instead of the generated ones.

## Using it

- The window covers the screen's work area (everything except the taskbar), is transparent and stays on top. Clicks pass through everywhere except on a pet or on the input bar.
- Click a pet to ask it directly. Press Ctrl+Shift+Space (or click the tray icon) to ask the whole team and let Merlin choose.
- The bar has a text field (Enter sends), Hold to talk (records the microphone, sends it to `/api/stt` as the form field `file`, then submits the words) and Stop (calls `/api/stop`). Esc closes it.
- While someone has the arm, a Stop the arm button also shows at the top left under the arm status.
- Drag a pet to move it. Walkers drop to the bottom edge; Elizabeth and Merlin stay where you leave them. Right-click a pet, or the tray icon, for the menu with Quit.
- The small mark at the bottom left reads linked, linking, offline or demo. The pets keep wandering when the server is down, and the link retries every few seconds.

## Habits

| pet | home | how it moves |
|---|---|---|
| Escanor | bottom left | struts slowly along an edge, leaps to a new one now and then |
| Elizabeth | top left | drifts gently anywhere |
| Ban | right edge | darts across the screen in quick leaps, short rests |
| King | bottom right | finds a spot and sleeps for long stretches |
| Gowther | centre, low | walks in straight lines, then stands still |
| Merlin | top right | floats in the upper part of the screen |

Pets keep a minimum distance from each other. When one thinks, speaks or takes the arm, it comes forward to a clear spot with room for its scroll, and the scroll is kept fully on screen and clear of other bubbles. A pet holding the arm grows by one pixel step.

## What the server events do

| event | on screen |
|---|---|
| `request` | a small purple note at top right with the request |
| `merlin` | the argument lines, then Merlin's line, each on a scroll over the speaker, in order |
| `agent_state` | idle roams, listening perks up with a sparkle, thinking shows a small scroll with dots, speaking bounces, has_arm comes forward and grows |
| `say` | the line on a scroll over the speaker, and its `audio_url` plays |
| `arm` | top left status with the crystal ball, for example "Ban has the arm: lifting" |
| `note`, `timer_done` | a short beige note at top right |
| `error` | a short black note at top right |

Speech lines play one at a time in arrival order. Audio URLs are relative to the server (`/audio/...`).

## Performance

Each pet is a small layer with its sprite drawn once at a whole-number scale; moving it only changes a CSS transform, so nothing is repainted. Breathing, floating and bouncing step in whole art pixels from the update loop, which runs about 30 times a second while someone travels and 5 times a second when all are resting. Measured on the dev laptop: about 11 percent of one CPU core for all Electron processes while the demo runs.

## Sprites

`tools/make_sprites.mjs` asks `gpt-image-1-mini` for one original pixel art sprite per character (1024x1024, transparent background, quality medium) and saves the raw picture in `assets/sprites/raw/`. It then snaps the picture to a true pixel grid: it finds the picture's own pixel size from where colour edges line up (or uses a fixed 40 pixel grid when none is clear), reduces it to 12 colours, takes the most common colour of each cell with alpha fully on or off, and adds a dark outline where one is missing. The result is `assets/sprites/<id>.png`, about 32 to 48 pixels tall, shown at 3x or 4x. Prompts are kept in `assets/sprites/prompts.json`. Every API call is counted in `assets/sprites/calls.json` and the script refuses to pass 12 calls. The key is read from `../.env` at run time and is never printed. Existing raw files are skipped unless `--force` is given; `--only ban` makes just one. Earlier raw versions are kept in `assets/sprites/raw/v1/`.

If a sprite file is missing, the app draws that character in code instead.

## Files

- `main.js` window, click-through, global shortcut, tray, screenshot and self test flags
- `preload.js` the small bridge to the page
- `renderer/app.js` pets, habits, movement, bubbles, events, input bar
- `renderer/sprites.js` sprite loading and the code-drawn fallbacks
- `renderer/net.js` WebSocket link with reconnect
- `renderer/demo.js` the scripted demo
- `assets/hd/` brand fonts and artwork copied from the Hack Dearborn: Conjure Reality set
