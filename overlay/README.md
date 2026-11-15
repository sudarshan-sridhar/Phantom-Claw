# Phantom Claw overlay

Six pixel pets live across the whole primary screen: Escanor, Elizabeth, Ban, King, Gowther and Merlin. Each has a home in a different part of the screen and roams from it: walking along the bottom edge, perching on the top and side edges, drifting or leaping between spots, resting, then moving on. They react live to the Phantom Claw server. The screen holds only the pets, their names in plain lettering, and the artist's scroll with whatever line a pet is saying. There are no panels, notes, status marks or buttons on screen. Everything follows the Hack Dearborn: Conjure Reality look (brand palette, fonts, the scroll and the sparkle).

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
npm run shot       # demo plus three screenshots in test/out/, then quits
npm run selftest   # checks click-through, the click-to-ask bar, that nothing else is on screen, the stop shortcut and the tray Stop item
```

Add `--fallback` (for example `npx electron . --demo --fallback`) to see the code-drawn sprites instead of the generated ones.

## Using it

- The window covers the screen's work area (everything except the taskbar), is transparent and stays on top. Clicks pass through everywhere except on a pet or on the input bar.
- Voice: with the listener running (`python -m backend.listen`), the pets stay off screen while it is asleep. "Phantom Claw, start", a quick Ctrl+Shift tap, or Ctrl+Shift+L brings them in; "Phantom Claw, stop" or the same keys send them away. Add `--show` (`npm start -- --show`) to keep them on screen all the time.
- Click a pet to ask it directly. Press Ctrl+Shift+Space (or click the tray icon) to ask the whole team and let Merlin choose.
- The bar is one line of text on a thin underline (Enter sends and closes it) and a hold to talk control (records the microphone, sends it to `/api/stt` as the form field `file`, then submits the words). Esc or clicking into another window closes it.
- Stopping the arm: press Ctrl+Shift+X from anywhere, or right-click the tray icon and pick Stop the arm. Both call `POST /api/stop`. There is no Stop button on screen.
- Drag a pet to move it. Walkers drop to the bottom edge; Elizabeth and Merlin stay where you leave them. Right-click a pet, or the tray icon, for the menu with Quit.
- The pets keep wandering when the server is down, and the link retries every few seconds. Link problems and server errors go to the developer console only.

## Habits

| pet | home | how it moves |
|---|---|---|
| Escanor | bottom left | struts slowly along an edge, leaps to a new one now and then |
| Elizabeth | top left | drifts gently anywhere |
| Ban | right edge | darts across the screen in quick leaps, short rests |
| King | bottom right | finds a spot and sleeps for long stretches |
| Gowther | centre, low | walks in straight lines, then stands still |
| Merlin | top right | floats in the upper part of the screen |

Pets keep a minimum distance from each other. When one thinks, speaks or takes the arm, it comes forward to a clear spot with room for its scroll. Only one scroll is on screen at a time, kept fully on screen, and it goes away when the line ends. A pet holding the arm grows by one pixel step.

## What the server events do

Only what a pet says is ever written on screen. Every other event shows in how the pets move, or not at all.

| event | on screen |
|---|---|
| `merlin` | the argument lines, then Merlin's line, each on a scroll over the speaker, in order |
| `say` | the line on a scroll over the speaker, and its `audio_url` plays |
| `agent_state` | idle roams, listening perks up with a sparkle, thinking shows a small pixel "..." over the head, speaking bounces, has_arm comes forward, sparkles and grows |
| `hear` | the named pet hops, perks up and comes forward (the owner's words are not shown) |
| `arm` | the pet holding the arm sparkles and grows; it lets go when the run ends |
| `request`, `timer_done` | the named pet gives a small hop |
| `listen` | the pets come in or leave |
| `note`, `error` | nothing on screen |

Speech lines play one at a time in arrival order. Audio URLs are relative to the server (`/audio/...`).

## Performance

Each pet is a small layer with its sprite drawn once at a whole-number scale; moving it only changes a CSS transform, so nothing is repainted. Breathing, floating and bouncing step in whole art pixels from the update loop, which runs about 30 times a second while someone travels and 5 times a second when all are resting. Measured on the dev laptop: about 11 percent of one CPU core for all Electron processes while the demo runs.

## Sprites

`tools/make_sprites.mjs` asks `gpt-image-1-mini` for one original pixel art sprite per character (1024x1024, transparent background, quality medium) and saves the raw picture in `assets/sprites/raw/`. It then snaps the picture to a true pixel grid: it finds the picture's own pixel size from where colour edges line up (or uses a fixed 40 pixel grid when none is clear), reduces it to 12 colours, takes the most common colour of each cell with alpha fully on or off, and adds a dark outline where one is missing. The result is `assets/sprites/<id>.png`, about 32 to 48 pixels tall, shown at 3x or 4x. Prompts are kept in `assets/sprites/prompts.json`. Every API call is counted in `assets/sprites/calls.json` and the script refuses to pass 12 calls. The key is read from `../.env` at run time and is never printed. Existing raw files are skipped unless `--force` is given; `--only ban` makes just one. Earlier raw versions are kept in `assets/sprites/raw/v1/`.

If a sprite file is missing, the app draws that character in code instead.

## Files

- `main.js` window, click-through, global shortcuts (including Ctrl+Shift+X to stop the arm), tray menu, screenshot and self test flags
- `preload.js` the small bridge to the page
- `renderer/app.js` pets, habits, movement, bubbles, events, input bar
- `renderer/sprites.js` sprite loading and the code-drawn fallbacks
- `renderer/net.js` WebSocket link with reconnect
- `renderer/demo.js` the scripted demo
- `assets/hd/` brand fonts and artwork copied from the Hack Dearborn: Conjure Reality set
