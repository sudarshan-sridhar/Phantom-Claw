// Sprites: load the generated pixel art, or draw a fallback in code.
// Every sprite ends up as a small canvas (about 48x48) with one canvas pixel
// per art pixel. The app scales it up by whole numbers with smoothing off.
'use strict';

const Sprites = (() => {
  const GRID = 48;

  // ---------- tiny pixel painter for the fallbacks ----------
  class Pix {
    constructor(w, h) { this.w = w; this.h = h; this.c = new Array(w * h).fill(null); }
    set(x, y, col) {
      x = Math.round(x); y = Math.round(y);
      if (x >= 0 && y >= 0 && x < this.w && y < this.h) this.c[y * this.w + x] = col;
    }
    get(x, y) { return (x < 0 || y < 0 || x >= this.w || y >= this.h) ? null : this.c[y * this.w + x]; }
    rect(x, y, w, h, col) { for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) this.set(x + i, y + j, col); }
    ellipse(cx, cy, rx, ry, col) {
      for (let y = Math.floor(cy - ry); y <= Math.ceil(cy + ry); y++) {
        for (let x = Math.floor(cx - rx); x <= Math.ceil(cx + rx); x++) {
          const dx = (x - cx) / (rx + 0.5), dy = (y - cy) / (ry + 0.5);
          if (dx * dx + dy * dy <= 1) this.set(x, y, col);
        }
      }
    }
    disc(cx, cy, r, col) { this.ellipse(cx, cy, r, r, col); }
    tri(ax, ay, bx, by, cx, cy, col) {
      const minX = Math.floor(Math.min(ax, bx, cx)), maxX = Math.ceil(Math.max(ax, bx, cx));
      const minY = Math.floor(Math.min(ay, by, cy)), maxY = Math.ceil(Math.max(ay, by, cy));
      const area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay);
      for (let y = minY; y <= maxY; y++) for (let x = minX; x <= maxX; x++) {
        const w0 = ((bx - x) * (cy - y) - (cx - x) * (by - y)) / area;
        const w1 = ((cx - x) * (ay - y) - (ax - x) * (cy - y)) / area;
        const w2 = 1 - w0 - w1;
        if (w0 >= -0.02 && w1 >= -0.02 && w2 >= -0.02) this.set(x, y, col);
      }
    }
    line(x0, y0, x1, y1, col) {
      x0 = Math.round(x0); y0 = Math.round(y0); x1 = Math.round(x1); y1 = Math.round(y1);
      const dx = Math.abs(x1 - x0), dy = -Math.abs(y1 - y0);
      const sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
      let err = dx + dy;
      for (;;) {
        this.set(x0, y0, col);
        if (x0 === x1 && y0 === y1) break;
        const e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
      }
    }
    outline(col) {
      const add = [];
      for (let y = 0; y < this.h; y++) for (let x = 0; x < this.w; x++) {
        if (this.get(x, y)) continue;
        if (this.get(x - 1, y) || this.get(x + 1, y) || this.get(x, y - 1) || this.get(x, y + 1)) add.push([x, y]);
      }
      add.forEach(([x, y]) => this.set(x, y, col));
    }
    toCanvas() {
      const cv = document.createElement('canvas');
      cv.width = this.w; cv.height = this.h;
      const g = cv.getContext('2d');
      for (let y = 0; y < this.h; y++) for (let x = 0; x < this.w; x++) {
        const col = this.get(x, y);
        if (col) { g.fillStyle = col; g.fillRect(x, y, 1, 1); }
      }
      return cv;
    }
  }

  // ---------- fallback designs (original, simple chibi shapes) ----------
  const FALLBACK = {
    escanor(p) {
      const gold = '#F5A623', orange = '#C1440E', light = '#FFD27A', dark = '#3a1500';
      p.line(32, 38, 38, 33, gold); p.line(33, 38, 39, 33, gold); p.disc(40, 31, 2, orange); // tail
      p.ellipse(24, 37, 9, 7, gold); p.ellipse(24, 38, 5, 4, light);                       // body
      p.rect(17, 41, 4, 5, gold); p.rect(27, 41, 4, 5, gold);                               // legs
      for (let a = 0; a < 12; a++) {                                                        // mane spikes
        const t = (a / 12) * Math.PI * 2;
        p.disc(24 + Math.cos(t) * 12, 20 + Math.sin(t) * 11, 3, orange);
      }
      p.disc(24, 20, 12, orange);
      p.disc(17, 12, 3, gold); p.disc(31, 12, 3, gold);                                     // ears
      p.disc(24, 21, 8, gold); p.ellipse(24, 25, 4, 3, light);                              // face, muzzle
      p.rect(19, 20, 3, 1, dark); p.rect(27, 20, 3, 1, dark);                               // smug eyes
      p.set(24, 24, dark); p.set(23, 26, dark); p.set(24, 27, dark); p.set(25, 27, dark); p.set(26, 26, dark);
      p.disc(38, 5, 3, '#FFE14D');                                                          // tiny sun
      [[38, 0], [38, 10], [33, 5], [43, 5], [35, 2], [41, 2], [35, 8], [41, 8]].forEach(([x, y]) => p.set(x, y, '#FFE14D'));
      p.outline(dark);
    },
    elizabeth(p) {
      const white = '#E8F1F8', blue = '#8ECAE6', pink = '#F4A6C0', deep = '#5d8fb0', dark = '#1d2b4a';
      p.ellipse(12, 22, 6, 9, blue); p.ellipse(36, 22, 6, 9, blue);                         // wings
      p.ellipse(12, 23, 3, 5, '#BFE3F2'); p.ellipse(36, 23, 3, 5, '#BFE3F2');
      p.ellipse(19, 41, 3, 2, white); p.ellipse(29, 41, 3, 2, white);                       // feet
      p.disc(24, 27, 12, white);                                                            // round body
      p.ellipse(24, 33, 7, 4, '#D6E6F2');
      p.disc(24, 13, 3, pink); p.disc(20, 14, 2, pink); p.disc(28, 14, 2, pink); p.set(24, 13, '#FFE14D'); // flower
      p.rect(19, 24, 2, 3, dark); p.rect(28, 24, 2, 3, dark); p.set(19, 24, '#ffffff'); p.set(28, 24, '#ffffff');
      p.rect(16, 28, 2, 1, pink); p.rect(31, 28, 2, 1, pink);                               // cheeks
      p.set(23, 29, deep); p.set(24, 30, deep); p.set(25, 29, deep);
      p.outline(dark);
    },
    ban(p) {
      const red = '#C1121F', silver = '#DCE3EA', deep = '#7a0a14', dark = '#1a0508';
      p.ellipse(37, 35, 7, 5, red); p.disc(42, 31, 3, silver);                              // tail
      p.ellipse(22, 37, 8, 7, red); p.ellipse(22, 38, 4, 5, silver);                        // body, chest
      p.rect(16, 41, 3, 5, red); p.rect(25, 41, 3, 5, red);
      p.tri(11, 5, 13, 15, 20, 11, red); p.tri(33, 5, 31, 15, 24, 11, red);                 // ears
      p.tri(13, 8, 14, 13, 18, 11, deep); p.tri(31, 8, 30, 13, 26, 11, deep);
      p.disc(22, 20, 10, red);                                                              // head
      p.rect(19, 9, 6, 2, silver); p.rect(20, 7, 3, 2, silver); p.set(24, 8, silver);      // silver tuft
      p.ellipse(22, 25, 6, 4, silver); p.set(22, 22, dark);                                 // muzzle, nose
      p.line(16, 18, 19, 19, dark); p.line(25, 19, 28, 18, dark);                           // sly eyes
      p.line(18, 27, 24, 27, dark); p.set(25, 26, dark); p.set(26, 25, dark);               // grin
      p.outline(dark);
    },
    king(p) {
      const orange = '#F4A261', green = '#2D6A4F', light = '#52B788', muzzle = '#FFD6A5', dark = '#2a1a0c';
      p.ellipse(24, 43, 17, 4, green); p.rect(9, 41, 30, 1, light);                         // pillow
      p.ellipse(24, 34, 11, 8, green); p.rect(20, 30, 8, 2, '#23573f');                     // hoodie body
      p.disc(14, 37, 3, orange); p.disc(34, 37, 3, orange);                                 // paws
      p.disc(15, 11, 3, orange); p.disc(33, 11, 3, orange);                                 // ears
      p.set(15, 11, '#c9773b'); p.set(33, 11, '#c9773b');
      p.disc(24, 19, 10, orange); p.ellipse(24, 23, 4, 3, muzzle); p.set(24, 22, dark);     // head
      p.rect(18, 19, 3, 1, dark); p.rect(27, 19, 3, 1, dark);                               // sleepy eyes
      p.set(23, 25, dark); p.set(24, 25, dark); p.set(25, 25, dark);
      p.outline(dark);
    },
    gowther(p) {
      const pink = '#D63384', light = '#F28DBA', cream = '#F1E3C8', glass = '#CFE3F0', dark = '#3a0a22';
      p.line(18, 9, 16, 4, cream); p.line(19, 9, 17, 4, cream);                             // horns
      p.line(30, 9, 32, 4, cream); p.line(29, 9, 31, 4, cream);
      p.ellipse(12, 17, 4, 2, pink); p.ellipse(36, 17, 4, 2, pink);                         // ears
      p.ellipse(24, 37, 8, 7, pink);                                                        // body
      p.rect(18, 41, 3, 5, pink); p.rect(27, 41, 3, 5, pink);
      p.rect(18, 45, 3, 1, dark); p.rect(27, 45, 3, 1, dark);
      p.disc(24, 19, 10, pink); p.ellipse(24, 25, 5, 3, light);                             // head, muzzle
      p.disc(20, 19, 3, '#333344'); p.disc(28, 19, 3, '#333344');                           // glasses
      p.disc(20, 19, 2, glass); p.disc(28, 19, 2, glass); p.set(24, 19, '#333344');
      p.set(20, 19, dark); p.set(28, 19, dark);
      p.rect(22, 27, 5, 1, dark);                                                           // flat mouth
      p.outline(dark);
    },
    merlin(p) {
      const violet = '#3C096C', mid = '#5A189A', snout = '#7B2CBF', gold = '#F5C400', cream = '#F1E3C8', dark = '#120020';
      p.disc(40, 28, 4, '#FFE680'); p.disc(40, 28, 3, gold); p.set(39, 27, '#fffbe0');      // orb
      p.ellipse(22, 34, 13, 9, violet); p.ellipse(22, 36, 8, 5, mid);                       // body
      p.rect(13, 41, 4, 5, violet); p.rect(27, 41, 4, 5, violet);
      p.tri(12, 9, 14, 17, 19, 13, violet); p.tri(32, 9, 30, 17, 25, 13, violet);           // ears
      p.disc(22, 23, 10, violet);                                                           // head
      p.ellipse(22, 28, 5, 3, snout); p.set(20, 28, dark); p.set(24, 28, dark);             // snout
      p.set(16, 30, cream); p.set(16, 31, cream); p.set(28, 30, cream); p.set(28, 31, cream); // tusks
      p.rect(17, 21, 2, 2, gold); p.rect(26, 21, 2, 2, gold);                               // gold eyes
      p.outline(dark);
    },
  };

  function measure(canvas) {
    const g = canvas.getContext('2d', { willReadFrequently: true });
    const { width: w, height: h } = canvas;
    const d = g.getImageData(0, 0, w, h).data;
    let left = w, right = -1, top = h, bottom = -1;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      if (d[(y * w + x) * 4 + 3] > 100) {
        if (x < left) left = x; if (x > right) right = x; if (y < top) top = y; if (y > bottom) bottom = y;
      }
    }
    if (right < 0) { left = 0; right = w - 1; top = 0; bottom = h - 1; }
    return { canvas, w, h, left, right, top, bottom };
  }

  function fallback(id) {
    const p = new Pix(GRID, GRID);
    (FALLBACK[id] || FALLBACK.gowther)(p);
    return Object.assign(measure(p.toCanvas()), { source: 'fallback' });
  }

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error('missing ' + src));
      img.src = src;
    });
  }

  // Load the generated sprite if it is there; otherwise use the drawn one.
  async function load(id, useImages) {
    if (!useImages) return fallback(id);
    try {
      const img = await loadImage('../assets/sprites/' + id + '.png');
      let w = img.naturalWidth, h = img.naturalHeight;
      // The snapped files are 48x48. If a big raw file ever lands here, shrink
      // it to the grid with nearest neighbour so it still reads as pixels.
      if (w > 96 || h > 96) { const s = GRID / Math.max(w, h); w = Math.round(w * s); h = Math.round(h * s); }
      const cv = document.createElement('canvas');
      cv.width = w; cv.height = h;
      const g = cv.getContext('2d', { willReadFrequently: true });
      g.imageSmoothingEnabled = false;
      g.drawImage(img, 0, 0, w, h);
      return Object.assign(measure(cv), { source: 'image' });
    } catch (e) {
      return fallback(id);
    }
  }

  return { load, fallback, GRID };
})();
