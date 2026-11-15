// Make the six base sprites for the overlay.
//
// 1. Ask the OpenAI image API for one original pixel art sprite per character
//    (gpt-image-1-mini, 1024x1024, transparent background, quality medium).
// 2. Snap each picture to a true pixel grid: find the picture's own pixel size,
//    reduce to a 12 colour palette, take the most common colour of each cell
//    (alpha is fully on or fully off), and add a dark outline where it is
//    missing. If no clear grid is found, a fixed grid is used instead.
//
// Usage (from the overlay folder):
//   node tools/make_sprites.mjs                    generate every missing sprite, then snap
//   node tools/make_sprites.mjs --only ban         generate just one (or a comma list)
//   node tools/make_sprites.mjs --only ban --force generate again even if it exists
//   node tools/make_sprites.mjs --snap-only        no API calls, only re-snap the raw files
//   node tools/make_sprites.mjs --grid 40          fixed grid size when no native grid is found (default 40)
//
// The key is read from ../.env at run time and is never printed.
// Every API call is counted in assets/sprites/calls.json and the script refuses
// to go past the hard cap of 12 calls.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { PNG } from 'pngjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const OVERLAY = path.resolve(HERE, '..');
const ROOT = path.resolve(OVERLAY, '..');
const SPRITES = path.join(OVERLAY, 'assets', 'sprites');
const RAW = path.join(SPRITES, 'raw');
const PREVIEW = path.join(OVERLAY, 'test', 'out');
const LEDGER = path.join(SPRITES, 'calls.json');
const PROMPTS = path.join(SPRITES, 'prompts.json');
const CALL_CAP = 12;

const MODEL = 'gpt-image-1-mini';
const ENDPOINT = 'https://api.openai.com/v1/images/generations';

// Second version of the style, used for the sprites made again after review:
// fewer, bigger pixels, a thick black outline and clear open eyes.
const STYLE =
  'Low resolution pixel art sprite, about 32 by 32 pixels, shown scaled up so every pixel is a big crisp square. ' +
  'Cute chibi mascot with a big head and a small round body, full body, front three-quarter view, standing. ' +
  'Big round shiny eyes with a white highlight and a clear simple face. ' +
  'Thick solid black outline around the whole character. Flat colours, at most 10 colours: {palette}. ' +
  'Plain fully transparent background, the character centered with empty space around it. ' +
  'Hard pixel edges only: no glow, no blur, no gradients, no anti-aliasing. ' +
  'No text, no letters, no frame, no floor, no cast shadow, no scenery. ' +
  'An original character design, not based on any existing character, logo or artwork.';

const CAST = {
  escanor: {
    subject: 'a lion cub: round golden face, a big fluffy deep orange mane all around the face, small round ears, open eyes and a proud happy smile with the chin up, a short tail with an orange tuft, and a tiny yellow sun with short rays floating just above the mane',
    palette: 'gold #F5A623, deep orange #C1440E, warm yellow, white and black',
  },
  elizabeth: {
    subject: 'a small gentle winged sprite creature with a round soft silver white body, little soft blue wings like a butterfly, pink cheeks, a small pink flower on its head and big kind eyes',
    palette: 'silver white #E8F1F8, soft blue #8ECAE6, pink #F4A6C0 and a dark navy outline',
    // The model cut her white body out together with the background, so the
    // snap step fills enclosed see-through areas with her silver white.
    fillHoles: '#E8F1F8',
  },
  ban: {
    subject: 'a fox kit sitting and facing the viewer: a round crimson red head with two big pointed ears, a simple cute face with two big round black eyes with white highlights, a small black nose and a confident little smirk, a pale silver muzzle and chest, a small pale silver tuft of hair on top of the head, and a big bushy crimson tail with a pale silver tip curled beside it. Simple shapes and few details',
    palette: 'crimson #C1121F, dark red, pale silver #DCE3EA, white and black',
  },
  king: {
    subject: 'a sleepy chubby bear cub with orange fur wearing a small green hoodie, eyes closed, sitting lazily on a small soft green pillow',
    palette: 'orange #F4A261, green #2D6A4F, light green and a dark brown outline',
  },
  gowther: {
    subject: 'a small goat kid with magenta pink fur, two small cream horns, big round glasses and a flat expressionless face',
    palette: 'magenta pink #D63384, light pink, cream, grey glass and a dark plum outline',
  },
  merlin: {
    subject: 'a baby wild boar (a piglet): dark violet fur, a round body, a pig snout with two nostrils, two small cream tusks, pointed ears, a little bristly mane on the back, calm gold eyes, and a small gold orb floating beside it',
    palette: 'dark violet #3C096C, lighter purple, gold #F5C400, cream, white and black',
  },
};

// ---------- small helpers ----------

function args() {
  const a = process.argv.slice(2);
  const out = { only: null, force: false, snapOnly: false, grid: 40 };
  for (let i = 0; i < a.length; i++) {
    if (a[i] === '--only') out.only = a[++i].split(',').map((s) => s.trim());
    else if (a[i] === '--force') out.force = true;
    else if (a[i] === '--snap-only') out.snapOnly = true;
    else if (a[i] === '--grid') out.grid = parseInt(a[++i], 10) || 48;
  }
  return out;
}

function readJson(file, fallback) {
  try { return JSON.parse(fs.readFileSync(file, 'utf8')); } catch { return fallback; }
}

function writeJson(file, obj) {
  fs.writeFileSync(file, JSON.stringify(obj, null, 2) + '\n');
}

function readKey() {
  const text = fs.readFileSync(path.join(ROOT, '.env'), 'utf8');
  for (const line of text.split(/\r?\n/)) {
    const m = line.match(/^\s*OPENAI_API_KEY\s*=\s*(.*)\s*$/);
    if (m) return m[1].replace(/^["']|["']$/g, '').trim();
  }
  return '';
}

function promptFor(id) {
  const c = CAST[id];
  return 'Original pixel art game sprite of ' + c.subject + '. ' + STYLE.replace('{palette}', c.palette);
}

// ---------- image generation ----------

async function generate(id, key) {
  const ledger = readJson(LEDGER, { cap: CALL_CAP, count: 0, log: [] });
  if (ledger.count >= CALL_CAP) {
    throw new Error('image call cap reached (' + ledger.count + ' of ' + CALL_CAP + '), not calling the API');
  }
  const prompt = promptFor(id);
  ledger.count += 1;
  ledger.log.push({ id, model: MODEL, at: new Date().toISOString() });
  writeJson(LEDGER, ledger); // count the call before it is made, so a crash still counts it

  console.log('[' + id + '] calling ' + MODEL + ' (call ' + ledger.count + ' of ' + CALL_CAP + ')');
  const res = await fetch(ENDPOINT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + key },
    body: JSON.stringify({
      model: MODEL,
      prompt,
      n: 1,
      size: '1024x1024',
      quality: 'medium',
      background: 'transparent',
      output_format: 'png',
    }),
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = (body && body.error && body.error.message) || ('HTTP ' + res.status);
    throw new Error('[' + id + '] image API error: ' + msg);
  }
  const b64 = body.data && body.data[0] && body.data[0].b64_json;
  if (!b64) throw new Error('[' + id + '] no image in the response');
  fs.writeFileSync(path.join(RAW, id + '.png'), Buffer.from(b64, 'base64'));

  const prompts = readJson(PROMPTS, {});
  prompts[id] = {
    model: MODEL,
    size: '1024x1024',
    quality: 'medium',
    background: 'transparent',
    faces: 'right',
    prompt,
    raw: 'raw/' + id + '.png',
    sprite: id + '.png',
    generated_at: new Date().toISOString(),
  };
  writeJson(PROMPTS, prompts);
  console.log('[' + id + '] saved raw/' + id + '.png');
}

// ---------- pixel grid snapping ----------

function removeFlatBackground(png) {
  // If the model ignored the transparent background, flood fill from the
  // corners and clear everything close to the corner colour.
  const { width: w, height: h, data } = png;
  const corner = (x, y) => data[(y * w + x) * 4 + 3];
  if (corner(0, 0) < 250 && corner(w - 1, h - 1) < 250) return;
  const i0 = 0;
  const br = data[i0], bg = data[i0 + 1], bb = data[i0 + 2];
  const seen = new Uint8Array(w * h);
  const stack = [0, w - 1, (h - 1) * w, h * w - 1];
  while (stack.length) {
    const p = stack.pop();
    if (seen[p]) continue;
    seen[p] = 1;
    const i = p * 4;
    const d = Math.abs(data[i] - br) + Math.abs(data[i + 1] - bg) + Math.abs(data[i + 2] - bb);
    if (d > 60) continue;
    data[i + 3] = 0;
    const x = p % w, y = (p / w) | 0;
    if (x > 0) stack.push(p - 1);
    if (x < w - 1) stack.push(p + 1);
    if (y > 0) stack.push(p - w);
    if (y < h - 1) stack.push(p + w);
  }
}

function kmeans(samples, k, iters) {
  // samples: flat array [r,g,b,r,g,b,...]; deterministic farthest point start.
  const n = samples.length / 3;
  const cent = [];
  cent.push([samples[0], samples[1], samples[2]]);
  const dist = new Float64Array(n).fill(Infinity);
  while (cent.length < k) {
    const c = cent[cent.length - 1];
    let best = -1, bestD = -1;
    for (let i = 0; i < n; i++) {
      const dr = samples[i * 3] - c[0], dg = samples[i * 3 + 1] - c[1], db = samples[i * 3 + 2] - c[2];
      const d = dr * dr + dg * dg + db * db;
      if (d < dist[i]) dist[i] = d;
      if (dist[i] > bestD) { bestD = dist[i]; best = i; }
    }
    if (bestD <= 0) break;
    cent.push([samples[best * 3], samples[best * 3 + 1], samples[best * 3 + 2]]);
  }
  const assign = new Int32Array(n);
  for (let it = 0; it < iters; it++) {
    const sum = cent.map(() => [0, 0, 0, 0]);
    for (let i = 0; i < n; i++) {
      let bi = 0, bd = Infinity;
      for (let j = 0; j < cent.length; j++) {
        const dr = samples[i * 3] - cent[j][0], dg = samples[i * 3 + 1] - cent[j][1], db = samples[i * 3 + 2] - cent[j][2];
        const d = dr * dr + dg * dg + db * db;
        if (d < bd) { bd = d; bi = j; }
      }
      assign[i] = bi;
      const s = sum[bi];
      s[0] += samples[i * 3]; s[1] += samples[i * 3 + 1]; s[2] += samples[i * 3 + 2]; s[3]++;
    }
    for (let j = 0; j < cent.length; j++) {
      if (sum[j][3]) cent[j] = [sum[j][0] / sum[j][3], sum[j][1] / sum[j][3], sum[j][2] / sum[j][3]];
    }
  }
  return cent.map((c) => c.map((v) => Math.round(v)));
}

function nearest(pal, r, g, b) {
  let bi = 0, bd = Infinity;
  for (let j = 0; j < pal.length; j++) {
    const dr = r - pal[j][0], dg = g - pal[j][1], db = b - pal[j][2];
    const d = dr * dr + dg * dg + db * db;
    if (d < bd) { bd = d; bi = j; }
  }
  return bi;
}

// ---------- native grid snapping ----------
// The model draws "pixel art" with its own big fake pixels. Find that grid
// (cell size and offset) from where colour edges line up, then sample one
// colour per real cell. This keeps the artist's pixels instead of resampling
// them onto a grid that does not match.

function detectGrid(png, x0, y0, x1, y1) {
  const { width: w, data } = png;
  const ex = new Float64Array(png.width + 1), ey = new Float64Array(png.height + 1);
  const differs = (i, j) => {
    const ai = data[i + 3] >= 128, aj = data[j + 3] >= 128;
    if (ai !== aj) return true;
    if (!ai) return false;
    return Math.abs(data[i] - data[j]) + Math.abs(data[i + 1] - data[j + 1]) + Math.abs(data[i + 2] - data[j + 2]) > 60;
  };
  for (let y = y0; y <= y1; y++) for (let x = x0 + 1; x <= x1 + 1; x++) if (differs((y * w + x - 1) * 4, (y * w + x) * 4)) ex[x]++;
  for (let y = y0 + 1; y <= y1 + 1; y++) for (let x = x0; x <= x1; x++) if (differs(((y - 1) * w + x) * 4, (y * w + x) * 4)) ey[y]++;

  const fit = (edges, lo, hi, p) => {
    const bins = Math.ceil(p * 4);
    const hist = new Float64Array(bins);
    let tot = 0;
    for (let x = lo; x <= hi; x++) {
      if (!edges[x]) continue;
      hist[Math.floor((x % p) * 4) % bins] += edges[x];
      tot += edges[x];
    }
    let bs = -1, bp = 0;
    for (let b = 0; b < bins; b++) {
      let s = 0;
      for (let k = -4; k <= 4; k++) s += hist[(b + k + bins) % bins];
      if (s > bs) { bs = s; bp = b; }
    }
    return { frac: tot ? bs / tot : 0, phase: bp / 4 };
  };

  const results = [];
  for (let p = 10; p <= 64; p += 0.05) {
    const a = fit(ex, x0, x1 + 1, p), b = fit(ey, y0, y1 + 1, p);
    const frac = (a.frac + b.frac) / 2;
    results.push({ p, frac, ratio: frac / Math.min(1, 2.25 / p), px: a.phase, py: b.phase });
  }
  let best = results.reduce((m, r) => (r.ratio > m.ratio ? r : m));
  // If half or a third of the winner fits nearly as well, the winner was a multiple.
  for (const div of [2, 3]) {
    const sub = results.filter((r) => Math.abs(r.p - best.p / div) < 0.6).reduce((m, r) => (!m || r.ratio > m.ratio ? r : m), null);
    if (sub && sub.ratio >= 0.8 * best.ratio && sub.p >= 10) best = sub;
  }
  return best;
}

function snapNative(id) {
  const rawFile = path.join(RAW, id + '.png');
  if (!fs.existsSync(rawFile)) return false;
  const png = PNG.sync.read(fs.readFileSync(rawFile));
  removeFlatBackground(png);
  const { width: w, height: h, data } = png;

  let x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (data[(y * w + x) * 4 + 3] >= 128) {
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
  }
  if (x1 < 0) return false;

  const g = detectGrid(png, x0, y0, x1, y1);
  const p = g.p;
  const cellsTall = (y1 - y0 + 1) / p;
  if (g.frac < 0.4 || cellsTall < 24 || cellsTall > 72) {
    console.log('[' + id + '] no clear native grid (cell ' + p.toFixed(1) + ', fit ' + g.frac.toFixed(2) + '), using a fixed grid');
    return false;
  }

  // Palette from the opaque pixels.
  const samples = [];
  const step = Math.max(1, Math.floor(Math.sqrt(((x1 - x0) * (y1 - y0)) / 40000)));
  for (let y = y0; y <= y1; y += step) for (let x = x0; x <= x1; x += step) {
    const i = (y * w + x) * 4;
    if (data[i + 3] >= 128) samples.push(data[i], data[i + 1], data[i + 2]);
  }
  const pal = kmeans(samples, 12, 10);

  const kx0 = Math.floor((x0 - g.px) / p), kx1 = Math.ceil((x1 + 1 - g.px) / p) - 1;
  const ky0 = Math.floor((y0 - g.py) / p), ky1 = Math.ceil((y1 + 1 - g.py) / p) - 1;
  const gw = kx1 - kx0 + 1, gh = ky1 - ky0 + 1;
  const OW = gw + 2, OH = gh + 2; // one empty pixel of room on each side for the outline
  const out = new PNG({ width: OW, height: OH });
  out.data.fill(0);

  for (let gy = 0; gy < gh; gy++) for (let gx = 0; gx < gw; gx++) {
    const sx0 = g.px + (kx0 + gx) * p, sy0 = g.py + (ky0 + gy) * p;
    let total = 0, solid = 0;
    const votes = new Map();
    for (let y = Math.floor(sy0); y < Math.ceil(sy0 + p); y++) {
      for (let x = Math.floor(sx0); x < Math.ceil(sx0 + p); x++) {
        total++;
        if (x < 0 || y < 0 || x >= w || y >= h) continue;
        const i = (y * w + x) * 4;
        if (data[i + 3] < 128) continue;
        solid++;
        const fx = (x - sx0) / p, fy = (y - sy0) / p;
        if (fx < 0.25 || fx > 0.75 || fy < 0.25 || fy > 0.75) continue;
        const c = nearest(pal, data[i], data[i + 1], data[i + 2]);
        votes.set(c, (votes.get(c) || 0) + 1);
      }
    }
    if (!total || solid / total < 0.5 || !votes.size) continue;
    let bc = 0, bv = -1;
    for (const [c, v] of votes) if (v > bv) { bv = v; bc = c; }
    const o = ((gy + 1) * OW + gx + 1) * 4;
    out.data[o] = pal[bc][0]; out.data[o + 1] = pal[bc][1]; out.data[o + 2] = pal[bc][2]; out.data[o + 3] = 255;
  }

  finishSprite(id, out, pal);
  console.log('[' + id + '] snapped to its own grid: cell ' + p.toFixed(2) + ' px, ' + OW + 'x' + OH + ' pixels, fit ' + g.frac.toFixed(2));
  return true;
}

// Shared clean-up: fill enclosed holes (if asked), drop stray pixels, make sure
// the whole silhouette has a dark outline, then save the sprite and a preview.
function finishSprite(id, out, pal) {
  const W = out.width, H = out.height, d = out.data;
  const alpha = (x, y) => (x < 0 || y < 0 || x >= W || y >= H) ? 0 : d[(y * W + x) * 4 + 3];
  const lum = (i) => 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
  const setPx = (x, y, c) => { const i = (y * W + x) * 4; d[i] = c[0]; d[i + 1] = c[1]; d[i + 2] = c[2]; d[i + 3] = 255; };

  if (CAST[id].fillHoles) {
    const hex = CAST[id].fillHoles.replace('#', '');
    const fill = [0, 2, 4].map((k) => parseInt(hex.slice(k, k + 2), 16));
    const outside = new Uint8Array(W * H);
    const stack = [];
    for (let x = 0; x < W; x++) stack.push([x, 0], [x, H - 1]);
    for (let y = 0; y < H; y++) stack.push([0, y], [W - 1, y]);
    while (stack.length) {
      const [x, y] = stack.pop();
      if (x < 0 || y < 0 || x >= W || y >= H || outside[y * W + x] || alpha(x, y)) continue;
      outside[y * W + x] = 1;
      stack.push([x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]);
    }
    let filled = 0;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (!outside[y * W + x] && !alpha(x, y)) { setPx(x, y, fill); filled++; }
    console.log('[' + id + '] filled ' + filled + ' enclosed see-through pixels');
  }

  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if (alpha(x, y) && !alpha(x - 1, y) && !alpha(x + 1, y) && !alpha(x, y - 1) && !alpha(x, y + 1)) d[(y * W + x) * 4 + 3] = 0;
  }

  // Outline: wherever an edge pixel of the silhouette is not already dark, add
  // a dark pixel just outside it.
  const darkest = pal.reduce((m, c) => (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2] < 0.299 * m[0] + 0.587 * m[1] + 0.114 * m[2] ? c : m));
  const ink = (0.299 * darkest[0] + 0.587 * darkest[1] + 0.114 * darkest[2]) < 60 ? darkest : [24, 14, 28];
  const add = [];
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if (alpha(x, y)) continue;
    for (const [nx, ny] of [[x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]]) {
      if (alpha(nx, ny) && lum((ny * W + nx) * 4) > 70) { add.push([x, y]); break; }
    }
  }
  add.forEach(([x, y]) => setPx(x, y, ink));

  fs.writeFileSync(path.join(SPRITES, id + '.png'), PNG.sync.write(out));

  const k = 8;
  const big = new PNG({ width: W * k, height: H * k });
  for (let y = 0; y < H * k; y++) for (let x = 0; x < W * k; x++) {
    const s = (((y / k) | 0) * W + ((x / k) | 0)) * 4, o = (y * W * k + x) * 4;
    const checker = (((x >> 4) + (y >> 4)) & 1) ? 235 : 210;
    const a = d[s + 3] / 255;
    big.data[o] = d[s] * a + checker * (1 - a);
    big.data[o + 1] = d[s + 1] * a + checker * (1 - a);
    big.data[o + 2] = d[s + 2] * a + checker * (1 - a);
    big.data[o + 3] = 255;
  }
  fs.mkdirSync(PREVIEW, { recursive: true });
  fs.writeFileSync(path.join(PREVIEW, 'preview_' + id + '.png'), PNG.sync.write(big));
}

function snap(id, grid) {
  const rawFile = path.join(RAW, id + '.png');
  if (!fs.existsSync(rawFile)) return false;
  const png = PNG.sync.read(fs.readFileSync(rawFile));
  removeFlatBackground(png);
  const { width: w, height: h, data } = png;

  // Bounding box of the character.
  let x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (data[(y * w + x) * 4 + 3] >= 128) {
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
  }
  if (x1 < 0) { console.log('[' + id + '] raw image is empty'); return false; }

  // Square box around the character with a small margin, feet on the bottom.
  const side = Math.ceil(Math.max(x1 - x0 + 1, y1 - y0 + 1) * 1.04);
  const cx = (x0 + x1) / 2;
  const bx = Math.round(cx - side / 2);
  const by = Math.round(y1 + 1 - side + side * 0.01);
  const cell = side / grid;

  // Palette from the opaque pixels.
  const samples = [];
  const step = Math.max(1, Math.floor(Math.sqrt(((x1 - x0) * (y1 - y0)) / 40000)));
  for (let y = y0; y <= y1; y += step) for (let x = x0; x <= x1; x += step) {
    const i = (y * w + x) * 4;
    if (data[i + 3] >= 128) samples.push(data[i], data[i + 1], data[i + 2]);
  }
  const pal = kmeans(samples, 12, 10);

  const out = new PNG({ width: grid, height: grid });
  for (let gy = 0; gy < grid; gy++) for (let gx = 0; gx < grid; gx++) {
    const sx0 = bx + gx * cell, sy0 = by + gy * cell;
    let total = 0, solid = 0;
    const votes = new Map();
    for (let y = Math.floor(sy0); y < Math.ceil(sy0 + cell); y++) {
      for (let x = Math.floor(sx0); x < Math.ceil(sx0 + cell); x++) {
        total++;
        if (x < 0 || y < 0 || x >= w || y >= h) continue;
        const i = (y * w + x) * 4;
        if (data[i + 3] < 128) continue;
        solid++;
        // Only the inner part of a cell votes on its colour, so edges between
        // the source's own fake pixels do not smear into each other.
        const fx = (x - sx0) / cell, fy = (y - sy0) / cell;
        if (fx < 0.2 || fx > 0.8 || fy < 0.2 || fy > 0.8) continue;
        const p = nearest(pal, data[i], data[i + 1], data[i + 2]);
        votes.set(p, (votes.get(p) || 0) + 1);
      }
    }
    const o = (gy * grid + gx) * 4;
    if (!total || solid / total < 0.5 || votes.size === 0) { out.data[o + 3] = 0; continue; }
    let bestP = 0, bestV = -1;
    for (const [p, v] of votes) if (v > bestV) { bestV = v; bestP = p; }
    out.data[o] = pal[bestP][0]; out.data[o + 1] = pal[bestP][1]; out.data[o + 2] = pal[bestP][2]; out.data[o + 3] = 255;
  }

  // Pad by one pixel for the outline, then the shared clean-up.
  const padded = new PNG({ width: grid + 2, height: grid + 2 });
  padded.data.fill(0);
  for (let y = 0; y < grid; y++) out.data.copy(padded.data, ((y + 1) * (grid + 2) + 1) * 4, y * grid * 4, (y + 1) * grid * 4);
  finishSprite(id, padded, pal);
  console.log('[' + id + '] snapped to a fixed ' + grid + 'x' + grid + ' grid with ' + pal.length + ' colours');
  return true;
}

// ---------- main ----------

async function main() {
  const opt = args();
  fs.mkdirSync(RAW, { recursive: true });
  const ids = opt.only || Object.keys(CAST);
  for (const id of ids) if (!CAST[id]) throw new Error('unknown character: ' + id);

  if (!opt.snapOnly) {
    const key = readKey();
    if (!key) throw new Error('OPENAI_API_KEY is missing from .env');
    for (const id of ids) {
      if (!opt.force && fs.existsSync(path.join(RAW, id + '.png'))) {
        console.log('[' + id + '] raw sprite exists, skipping the API (use --force to make it again)');
        continue;
      }
      try { await generate(id, key); } catch (e) { console.log(String(e.message || e)); }
    }
  }
  for (const id of ids) {
    if (!snapNative(id) && !snap(id, opt.grid)) console.log('[' + id + '] no raw sprite yet; the app will use its drawn fallback');
  }
  const ledger = readJson(LEDGER, { count: 0 });
  console.log('image calls used so far: ' + ledger.count + ' of ' + CALL_CAP);
}

main().catch((e) => { console.error(String(e.message || e)); process.exit(1); });
