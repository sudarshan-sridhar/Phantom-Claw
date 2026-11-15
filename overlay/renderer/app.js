// Phantom Claw overlay: six pixel pets living across the whole screen.
//
// Performance: each pet is a small DOM layer holding two tiny canvases (the top
// and bottom half of its sprite, drawn once at a whole-number scale). Moving a
// pet only changes its CSS transform, so the compositor does the work and
// nothing is repainted. Breathing, floating and bouncing are CSS animations in
// whole art pixels. When no pet is travelling, the update loop drops to a few
// ticks per second.
'use strict';

const CAST = [
  { id: 'escanor',   name: 'Escanor',   tag: '#F5A623', tagInk: '#000000', habit: 'strut' },
  { id: 'elizabeth', name: 'Elizabeth', tag: '#E8F1F8', tagInk: '#000000', habit: 'drift' },
  { id: 'ban',       name: 'Ban',       tag: '#C1121F', tagInk: '#ffffff', habit: 'dart' },
  { id: 'king',      name: 'King',      tag: '#2D6A4F', tagInk: '#F4A261', habit: 'sleep' },
  { id: 'gowther',   name: 'Gowther',   tag: '#D63384', tagInk: '#ffffff', habit: 'line' },
  { id: 'merlin',    name: 'Merlin',    tag: '#3C096C', tagInk: '#F5C400', habit: 'float' },
];
const BY_ID = Object.fromEntries(CAST.map((c) => [c.id, c]));

const PHASE_WORDS = {
  start: 'starting', at_target: 'at the bottle', gripped: 'gripped it', lifted: 'lifting',
  put_back: 'putting it back', missed: 'missed', returning: 'returning', done: 'done', error: 'stopped on an error',
};
const SKILL_WORDS = { grip_bottle: 'the bottle', point: 'a point', half_reach: 'a half reach', wave: 'a wave', rest: 'rest' };

window.addEventListener('error', (e) => { window.__lastError = String(e.message); });
window.addEventListener('unhandledrejection', (e) => { window.__lastError = String(e.reason && e.reason.stack || e.reason); });

const App = (() => {
  const world = document.getElementById('world');
  let W = 0, H = 0, dpr = 1;            // window size in CSS pixels
  let flags = { demo: false, server: 'http://127.0.0.1:8765' };
  let pets = [];
  let armStatus = null;
  // Listening: asleep means the pets are off screen until "Phantom Claw, start" or the shortcut.
  let listenAwake = false;
  let presence = 'shown';               // shown, entering, leaving, hidden
  let presenceAt = 0;                   // when entering or leaving finishes
  let activeUntil = 0;                  // stay on screen a moment after the last activity
  let ticks = 0, tickStart = performance.now(), tickRate = 0;

  const now = () => performance.now() / 1000;
  const rand = (a, b) => a + Math.random() * (b - a);
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const snap = (v) => Math.round(v * dpr) / dpr;   // whole device pixels
  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

  // ---------- where pets may stand ----------
  // Positions are the point between the pet's feet, in CSS pixels.
  const TAG_ROOM = 30;
  function bounds(p) {
    return {
      left: p.w / 2 + 12,
      right: W - p.w / 2 - 12,
      top: p.h + 14,                 // standing on the top edge of the screen
      floor: H - TAG_ROOM,           // standing on the bottom edge (top of the taskbar)
    };
  }

  const SURFACES = {
    floor: (p, x0 = 0, x1 = 1) => { const b = bounds(p); return { x: lerpX(b, x0, x1), y: b.floor, kind: 'floor' }; },
    top: (p, x0 = 0, x1 = 1) => { const b = bounds(p); return { x: lerpX(b, x0, x1), y: b.top, kind: 'top' }; },
    left: (p, y0 = 0.25, y1 = 0.8) => { const b = bounds(p); return { x: b.left, y: lerpY(b, y0, y1), kind: 'side' }; },
    right: (p, y0 = 0.25, y1 = 0.8) => { const b = bounds(p); return { x: b.right, y: lerpY(b, y0, y1), kind: 'side' }; },
    free: (p, x0, x1, y0, y1) => { const b = bounds(p); return { x: lerpX(b, x0, x1), y: lerpY(b, y0, y1), kind: 'free' }; },
  };
  function lerpX(b, x0, x1) { return clamp(W * rand(x0, x1), b.left, b.right); }
  function lerpY(b, y0, y1) { return clamp(H * rand(y0, y1), b.top, b.floor); }

  // Each pet has a home in a different part of the screen, and roams from it.
  const HOME = {
    escanor: (p) => SURFACES.floor(p, 0.04, 0.34),
    elizabeth: (p) => SURFACES.free(p, 0.05, 0.30, 0.14, 0.40),
    ban: (p) => SURFACES.right(p, 0.30, 0.70),
    king: (p) => SURFACES.floor(p, 0.68, 0.96),
    gowther: (p) => SURFACES.free(p, 0.40, 0.60, 0.58, 0.80),
    merlin: (p) => SURFACES.free(p, 0.60, 0.90, 0.14, 0.36),
  };
  const ROAM = {
    escanor: (p) => pick([() => SURFACES.floor(p), () => SURFACES.floor(p), () => SURFACES.top(p, 0.05, 0.6), () => SURFACES.left(p)]),
    elizabeth: (p) => SURFACES.free(p, 0.04, 0.96, 0.10, 0.85),
    ban: (p) => pick([() => SURFACES.floor(p), () => SURFACES.top(p), () => SURFACES.left(p), () => SURFACES.right(p), () => SURFACES.free(p, 0.1, 0.9, 0.2, 0.8)]),
    king: (p) => pick([() => SURFACES.floor(p, 0.3, 1), () => SURFACES.right(p, 0.5, 0.9), () => SURFACES.top(p, 0.6, 1)]),
    gowther: (p) => SURFACES.free(p, 0.15, 0.85, 0.35, 0.88),
    merlin: (p) => SURFACES.free(p, 0.10, 0.92, 0.10, 0.45),
  };
  function pick(fns) { return fns[Math.floor(Math.random() * fns.length)](); }

  // How each habit travels, and how long it rests.
  const MOTION = {
    strut: { walk: 46, leap: 300, hop: 2, rest: [2.5, 6], roam: 0.3 },
    drift: { drift: 70, rest: [2, 5], roam: 0.35 },
    dart: { walk: 260, leap: 1100, hop: 1.5, rest: [0.6, 2.2], roam: 0.55 },
    sleep: { walk: 24, leap: 220, hop: 1, rest: [18, 36], roam: 0.25 },
    line: { line: 62, hop: 1, rest: [6, 12], roam: 0.3 },
    float: { drift: 48, rest: [3, 7], roam: 0.35 },
  };

  // Pick a spot from a generator that keeps clear of the other pets.
  function clearSpot(p, gen, minGap) {
    let best = null, bestScore = -Infinity;
    for (let i = 0; i < 16; i++) {
      const s = gen();
      let near = Infinity;
      for (const o of pets) {
        if (o === p) continue;
        near = Math.min(near, dist(s, o), o.move ? dist(s, o.move.to) : Infinity);
      }
      if (near >= minGap) return s;
      if (near > bestScore) { bestScore = near; best = s; }
    }
    return best;
  }

  // ---------- speech queue (one line at a time, in order) ----------
  const speech = { queue: [], current: null, audio: null, el: null };

  function enqueueLine(agent, text, audioUrl) {
    if (!text) return;
    speech.queue.push({ agent, text: String(text).slice(0, 200), audioUrl: audioUrl || null });
    while (speech.queue.length > 6) speech.queue.shift();
  }
  const textDuration = (text) => clamp(1.6 + text.length * 0.055, 2.2, 9);

  function pumpSpeech(t) {
    if (speech.current && t >= speech.current.until) {
      const done = find(speech.current.agent);
      speech.current = null;
      if (speech.audio) { speech.audio.pause(); speech.audio = null; }
      hideBubble();
      if (done) {
        // Back to wandering once its line is over, unless it is still working.
        if (done.state === 'idle' || done.state === 'listening') { done.forward = false; done.restUntil = t + rand(2, 4); }
        done.refreshClasses();
      }
    }
    if (speech.current || !speech.queue.length) return;
    const line = speech.queue.shift();
    line.start = t;
    line.until = t + textDuration(line.text);
    speech.current = line;
    const p = find(line.agent);
    if (p) { p.wake(); p.comeForward(); p.refreshClasses(); }
    showBubble(line);
    if (line.audioUrl && !flags.demo) {
      const url = /^https?:/.test(line.audioUrl) ? line.audioUrl : flags.server + line.audioUrl;
      const audio = new Audio(url);
      speech.audio = audio;
      line.until = t + 30;
      audio.addEventListener('loadedmetadata', () => {
        if (speech.current === line && isFinite(audio.duration)) line.until = line.start + Math.max(audio.duration + 0.4, 1.6);
      });
      audio.addEventListener('ended', () => { if (speech.current === line) line.until = Math.min(line.until, now() + 0.4); });
      audio.addEventListener('error', () => { if (speech.current === line) line.until = line.start + textDuration(line.text); });
      audio.play().catch(() => { if (speech.current === line) line.until = line.start + textDuration(line.text); });
    }
  }

  // ---------- bubbles ----------
  function makeBubble(cls) {
    const el = document.createElement('div');
    el.className = 'bubble hidden ' + (cls || '');
    el.innerHTML = '<span class="text"></span><i class="tail"></i>';
    world.appendChild(el);
    return el;
  }
  const speechEl = makeBubble('');

  function showBubble(line) {
    speechEl.querySelector('.text').textContent = line.text;
    speechEl.classList.remove('hidden');
    speechEl._agent = line.agent;
    speechEl._sig = '';
  }
  function hideBubble() { speechEl.classList.add('hidden'); speechEl._agent = null; }

  // Place a bubble over its pet, fully on screen, avoiding rects already taken.
  function placeBubble(el, p, taken) {
    const bw = el.offsetWidth, bh = el.offsetHeight;
    const head = p.headRect();
    const gap = 14;
    const cands = [
      { x: head.cx - bw / 2, y: head.top - gap - bh, cls: '' },
      { x: head.cx - bw / 2, y: p.y + TAG_ROOM + gap, cls: 'below' },
      { x: head.right + gap, y: head.top, cls: 'side' },
      { x: head.left - gap - bw, y: head.top, cls: 'side' },
    ];
    let chosen = null;
    for (const c of cands) {
      c.x = clamp(c.x, 8, W - bw - 8);
      const fits = c.y >= 8 && c.y + bh <= H - 8;
      c.y = clamp(c.y, 8, H - bh - 8);
      const r = { x: c.x, y: c.y, w: bw, h: bh };
      const overlap = taken.some((o) => r.x < o.x + o.w && r.x + r.w > o.x && r.y < o.y + o.h && r.y + r.h > o.y);
      if (fits && !overlap) { chosen = c; break; }
      if (!chosen) chosen = c;
    }
    const sig = Math.round(chosen.x) + ',' + Math.round(chosen.y) + chosen.cls;
    if (el._sig !== sig) {
      el._sig = sig;
      el.classList.toggle('below', chosen.cls === 'below');
      el.classList.toggle('side', chosen.cls === 'side');
      el.style.setProperty('--tail-x', clamp(head.cx - chosen.x - 34, 6, bw - 68 - 6) + 'px');
      el.style.transform = 'translate3d(' + snap(chosen.x) + 'px,' + snap(chosen.y) + 'px,0)';
    }
    return { x: chosen.x, y: chosen.y, w: bw, h: bh };
  }

  function layoutBubbles() {
    const taken = [];
    if (speech.current && speechEl._agent) {
      const p = find(speechEl._agent);
      if (p) taken.push(placeBubble(speechEl, p, taken));
    }
    for (const p of pets) {
      if (p.heard && now() >= p.heard.until) p.heard = null;
      const heard = p.heard;
      const thinking = p.state === 'thinking';
      p.thinkEl.classList.toggle('hidden', !(thinking || heard));
      // What the listener heard stays in the pet's bubble until its reply comes.
      p.thinkEl.classList.toggle('thinking', !heard);
      p.thinkEl.classList.toggle('heard', !!heard);
      if (heard) {
        const txt = '\u201c' + heard.text + '\u201d';
        if (p.thinkEl._dots !== txt) { p.thinkEl.querySelector('.text').textContent = txt; p.thinkEl._dots = txt; p.thinkEl._sig = ''; }
        taken.push(placeBubble(p.thinkEl, p, taken));
      } else if (thinking) {
        const n = 1 + (Math.floor(now() * 3) % 3);
        const dots = '.'.repeat(n).padEnd(3, ' ');
        if (p.thinkEl._dots !== dots) { p.thinkEl.querySelector('.text').textContent = dots; p.thinkEl._dots = dots; }
        taken.push(placeBubble(p.thinkEl, p, taken));
      }
    }
  }

  // ---------- notes and status ----------
  const toastsEl = document.getElementById('toasts');
  function toast(text, kind) {
    const el = document.createElement('div');
    el.className = 'plaque ' + (kind || '');
    el.textContent = text;
    toastsEl.appendChild(el);
    while (toastsEl.children.length > 3) toastsEl.firstChild.remove();
    setTimeout(() => el.remove(), kind === 'error' ? 7000 : 5000);
  }

  const statusEl = document.getElementById('status');
  const stopPill = document.getElementById('stop-pill');
  function renderStatus(t) {
    if (armStatus && t - armStatus.at > (['done', 'missed', 'error'].includes(armStatus.phase) ? 4 : 60)) armStatus = null;
    const active = !!armStatus || pets.some((p) => p.state === 'has_arm');
    stopPill.classList.toggle('hidden', !active);
    statusEl.classList.toggle('hidden', !armStatus);
    if (!armStatus) return;
    const def = BY_ID[armStatus.agent];
    const word = armStatus.phase === 'start' ? 'starting ' + (SKILL_WORDS[armStatus.skill] || armStatus.skill || '') : (PHASE_WORDS[armStatus.phase] || armStatus.phase);
    const text = (def ? def.name : armStatus.agent) + ' has the arm: ' + word;
    const span = statusEl.querySelector('span');
    if (span.textContent !== text) span.textContent = text;
  }

  const linkEl = document.getElementById('link');
  function setLink(state) {
    linkEl.className = 'plaque ' + state;
    linkEl.querySelector('span').textContent = { live: 'linked', offline: 'offline', connecting: 'linking', demo: 'demo' }[state] || state;
    if (state === 'offline') setListen(false);   // no server, nobody is listening
    renderEar();
  }

  // ---------- listening: the small mark next to the link mark, and the pets coming and going ----------
  const earEl = document.getElementById('ear');
  function renderEar() {
    if (!earEl) return;
    earEl.classList.toggle('hidden', !!flags.demo);
    earEl.classList.toggle('on', listenAwake);
    const text = listenAwake ? 'listening' : 'asleep';
    const span = earEl.querySelector('span');
    if (span.textContent !== text) span.textContent = text;
    earEl.style.left = (linkEl.offsetLeft + linkEl.offsetWidth + 6) + 'px';
  }

  function setListen(awake) {
    const was = listenAwake;
    listenAwake = !!awake;
    if (was && !listenAwake) activeUntil = Math.max(activeUntil, now() + 1.5);  // room for Merlin's goodbye line
    if (!was && listenAwake) for (const p of pets) p.perkAt = now();
    renderEar();
  }

  async function toggleListen() {
    if (flags.demo) { setListen(!listenAwake); return; }
    try {
      await fetch(flags.server + '/api/listen', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ awake: !listenAwake, source: 'overlay' }),
      });
    } catch (e) {
      toast('The server is not reachable', 'error');
    }
  }

  const alwaysShow = () => !!(flags.demo || flags.show);
  const barOpen = () => !bar.classList.contains('hidden');
  const busyNow = () => barOpen() || !!speech.current || speech.queue.length > 0 || pets.some((p) => p.state !== 'idle' || p.heard);
  const floats = (p) => p.def.habit === 'float' || p.def.habit === 'drift';

  // Walkers come in from the nearest side, Elizabeth and Merlin drift down from the top.
  function edgeSpot(p, near) {
    if (floats(p)) return { x: near.x, y: -12, kind: 'free' };
    return { x: near.x < W / 2 ? -p.w : W + p.w, y: near.y, kind: near.kind || p.surface };
  }

  function trip(p, from, to, t, dur) {
    p.move = { from, to, style: floats(p) ? 'drift' : 'walk', t0: t, dur, apex: 0 };
    if (Math.abs(to.x - from.x) > 4) p.facing = to.x > from.x ? 1 : -1;
    p.sleeping = false;
    p.refreshClasses();
  }

  function enterAll(t) {
    const fromHidden = presence === 'hidden';
    world.classList.remove('away');
    for (const p of pets) {
      const home = clearSpot(p, () => HOME[p.id](p), 160);
      if (fromHidden) { const s = edgeSpot(p, home); p.x = s.x; p.y = s.y; }
      p.fall = null; p.dragging = false; p.forward = false;
      const dur = rand(0.9, 1.2);
      trip(p, { x: p.x, y: p.y }, home, t, dur);
      p.restUntil = t + dur + rand(1, 3);
    }
    presence = 'entering';
    presenceAt = t + 1.3;
  }

  function leaveAll(t) {
    for (const p of pets) {
      p.forward = false;
      trip(p, { x: p.x, y: p.y }, edgeSpot(p, p), t, rand(0.8, 1.1));
    }
    presence = 'leaving';
    presenceAt = t + 1.25;
  }

  function updatePresence(t) {
    if (busyNow()) activeUntil = Math.max(activeUntil, t + 1.0);
    const want = alwaysShow() || listenAwake || t < activeUntil;
    if (want && (presence === 'hidden' || presence === 'leaving')) enterAll(t);
    else if (!want && (presence === 'shown' || presence === 'entering')) leaveAll(t);
    if (presence === 'entering' && t >= presenceAt) presence = 'shown';
    if (presence === 'leaving' && t >= presenceAt) {
      presence = 'hidden';
      world.classList.add('away');
      for (const p of pets) p.move = null;
    }
  }

  // ---------- pets ----------
  class Pet {
    constructor(def, sprite) {
      Object.assign(this, { def, id: def.id, sprite });
      this.motion = MOTION[def.habit];
      this.state = 'idle';
      this.facing = 1;
      this.move = null;           // current trip
      this.restUntil = now() + rand(0.5, 2.5);
      this.forward = false;       // has come forward to talk or work
      this.sleeping = false;
      this.dragging = false;
      this.fall = null;
      this.perkAt = -10;
      this.lastSig = '';
      this.build();
      const s = HOME[this.id](this);
      this.x = s.x; this.y = s.y; this.surface = s.kind;
    }

    build() {
      const el = document.createElement('div');
      el.className = 'pet idle ' + (this.def.habit === 'float' ? 'float' : this.def.habit === 'drift' ? 'hover' : '');
      el.innerHTML = '<div class="body"><div class="flip"></div></div>';
      this.el = el;
      this.flipEl = el.querySelector('.flip');
      this.bodyEl = el.querySelector('.body');
      this.seed = Math.random() * 5;
      this.animSig = '';
      this.upper = document.createElement('canvas'); this.upper.className = 'art upper';
      this.lower = document.createElement('canvas'); this.lower.className = 'art lower';
      this.flipEl.append(this.upper, this.lower);
      const tag = document.createElement('div');
      tag.className = 'tag';
      tag.textContent = this.def.name;
      tag.style.background = this.def.tag;
      tag.style.color = this.def.tagInk;
      el.appendChild(tag);
      this.sparkA = sparkleImg('a'); this.sparkB = sparkleImg('b');
      el.append(this.sparkA, this.sparkB);
      this.zA = zCanvas('a'); this.zB = zCanvas('b');
      el.append(this.zA, this.zB);
      world.appendChild(el);
      this.thinkEl = makeBubble('thinking');
      this.render(false);
    }

    // Draw the sprite at a whole-number scale into the two half canvases.
    render(big) {
      const sp = this.sprite;
      const artW = sp.right - sp.left + 1, artH = sp.bottom - sp.top + 1;
      let u = Math.max(2, Math.round(112 * dpr / artH));
      while (u > 2 && artH * u / dpr > 128) u--;
      if (big) u += 1;
      this.unit = u;
      const split = Math.round(artH * 0.5);
      const draw = (cv, y0, rows) => {
        cv.width = artW * u; cv.height = rows * u;
        cv.style.width = (artW * u / dpr) + 'px';
        cv.style.height = (rows * u / dpr) + 'px';
        const g = cv.getContext('2d');
        g.imageSmoothingEnabled = false;
        g.drawImage(sp.canvas, sp.left, sp.top + y0, artW, rows, 0, 0, artW * u, rows * u);
      };
      draw(this.upper, 0, split);
      draw(this.lower, split, artH - split);
      this.w = artW * u / dpr;
      this.h = artH * u / dpr;
      this.upper.style.top = '0px';
      this.lower.style.top = (split * u / dpr) + 'px';
      this.el.style.width = this.w + 'px';
      this.el.style.height = this.h + 'px';
      this.el.style.setProperty('--u', (u / dpr) + 'px');
      this.el.style.setProperty('--breath', this.def.habit === 'sleep' ? '2.6s' : '1.4s');
      this.sparkA.style.left = '-14px'; this.sparkA.style.top = '6px';
      this.sparkB.style.left = (this.w + 2) + 'px'; this.sparkB.style.top = (this.h * 0.35) + 'px';
      this.zA.style.left = (this.w * 0.7) + 'px'; this.zA.style.top = '-6px';
      this.zB.style.left = (this.w * 0.8) + 'px'; this.zB.style.top = '-16px';
      this.lastSig = '';
    }

    headRect() {
      return { cx: this.x, top: this.y - this.h - this.lift(), left: this.x - this.w / 2, right: this.x + this.w / 2 };
    }

    speaking() { return !!(speech.current && speech.current.agent === this.id); }
    active() { return this.state !== 'idle' || this.speaking(); }

    wake() { this.sleeping = false; }

    setState(s) {
      if (!['idle', 'listening', 'thinking', 'speaking', 'has_arm'].includes(s) || s === this.state) return;
      const wasBig = this.state === 'has_arm';
      this.state = s;
      if (s !== 'idle') this.wake();
      if (s === 'listening') { this.perkAt = now(); this.facing = this.x < W / 2 ? 1 : -1; }
      if (s === 'thinking' || s === 'speaking' || s === 'has_arm') this.comeForward();
      if (s === 'idle' && !this.speaking()) { this.forward = false; this.restUntil = now() + rand(2, 4); }
      if ((s === 'has_arm') !== wasBig) this.render(s === 'has_arm');
      this.refreshClasses();
    }

    refreshClasses() {
      const speakingNow = this.speaking() || this.state === 'speaking';
      const cl = this.el.classList;
      cl.toggle('idle', this.state === 'idle' && !speakingNow && !this.move);
      cl.toggle('speaking', speakingNow);
      cl.toggle('listening', this.state === 'listening');
      const spark = this.state === 'has_arm' || this.state === 'listening';
      this.sparkA.classList.toggle('hidden', !spark);
      this.sparkB.classList.toggle('hidden', this.state !== 'has_arm');
      const zz = this.sleeping && !this.move && !this.active();
      this.zA.classList.toggle('hidden', !zz);
      this.zB.classList.toggle('hidden', !zz);
    }

    // Come forward to a clear spot with room for a bubble.
    comeForward() {
      if (this.forward && !this.move) return;
      this.forward = true;
      const upper = this.def.habit === 'float';
      const minY = this.h + 190;
      const spot = clearSpot(this, () => {
        const b = bounds(this);
        const x = clamp(W * rand(0.16, 0.84), b.left, b.right);
        const y = upper ? clamp(H * rand(0.34, 0.52), minY, b.floor) : clamp(H * rand(0.52, 0.86), minY, b.floor);
        return { x, y, kind: 'free' };
      }, 280);
      // Prefer a spot not too far away.
      this.travel(spot, true);
    }

    // Start a trip to a spot, in this pet's own way of moving.
    travel(to, quick) {
      const from = { x: this.x, y: this.y };
      const d = dist(from, to);
      if (d < 2) { this.move = null; return; }
      const m = this.motion;
      let style, speed;
      if (m.drift) { style = 'drift'; speed = m.drift; }
      else if (m.line) { style = 'line'; speed = m.line; }
      else if (Math.abs(to.y - from.y) < 2 && to.kind === this.surface) { style = 'walk'; speed = m.walk; }
      else { style = 'leap'; speed = m.leap; }
      let dur = d / speed;
      if (quick) dur = Math.min(dur, style === 'leap' ? 0.9 : 1.4);
      dur = Math.max(dur, style === 'leap' ? 0.35 : 0.3);
      this.move = { from, to, style, t0: now(), dur, apex: style === 'leap' ? clamp(d * 0.28, 40, 220) : 0 };
      if (Math.abs(to.x - from.x) > 4) this.facing = to.x > from.x ? 1 : -1;
      this.sleeping = false;
      this.refreshClasses();
    }

    planNext(t) {
      const m = this.motion;
      if (this.def.habit === 'line' && Math.random() < 0.3) { this.facing = -this.facing; this.restUntil = t + rand(3, 6); return; }
      const gen = Math.random() < m.roam ? () => ROAM[this.id](this) : () => HOME[this.id](this);
      this.travel(clearSpot(this, gen, 220));
    }

    lift() {
      if (this.fall) return 0;
      if (!this.move) return 0;
      const k = clamp((now() - this.move.t0) / this.move.dur, 0, 1);
      if (this.move.style === 'leap') return this.move.apex * 4 * k * (1 - k);
      const hop = this.motion.hop || 0;
      if (!hop) return 0;
      const rate = this.def.habit === 'dart' ? 16 : 8;
      return Math.abs(Math.sin((now() - this.move.t0) * rate)) * hop * this.unit / dpr;
    }

    update(t, dt) {
      if (this.dragging) return true;
      if (this.fall) {
        this.fall.vy += 2600 * dt;
        this.y += this.fall.vy * dt;
        const floor = bounds(this).floor;
        if (this.y >= floor) { this.y = floor; this.surface = 'floor'; this.fall = null; this.restUntil = t + rand(1, 2); this.refreshClasses(); }
        return true;
      }
      if (this.move) {
        const mv = this.move;
        let k = clamp((t - mv.t0) / mv.dur, 0, 1);
        let e = k;
        if (mv.style === 'drift') e = k * k * (3 - 2 * k);
        if (mv.style === 'leap') e = k;
        this.x = mv.from.x + (mv.to.x - mv.from.x) * e;
        this.y = mv.from.y + (mv.to.y - mv.from.y) * e;
        if (mv.style === 'drift') this.x += Math.sin(k * Math.PI * 2) * 10 * (1 - k);
        if (k >= 1) {
          this.x = mv.to.x; this.y = mv.to.y; this.surface = mv.to.kind;
          this.move = null;
          const [r0, r1] = this.motion.rest;
          this.restUntil = t + rand(r0, r1);
          if (this.def.habit === 'sleep' && !this.active()) this.sleeping = true;
          this.refreshClasses();
        }
        return true;
      }
      if (presence === 'hidden' || presence === 'leaving') return false;
      if (this.active() || this.forward) return false;
      if (t >= this.restUntil) this.planNext(t);
      return !!this.move;
    }

    // Write the transform only when it changed.
    place() {
      const x = this.x - this.w / 2;
      const y = this.y - this.h - this.lift();
      const sig = Math.round(x * dpr) + ',' + Math.round(y * dpr) + ',' + this.facing;
      if (sig === this.lastSig) return;
      this.lastSig = sig;
      this.el.style.transform = 'translate3d(' + snap(x) + 'px,' + snap(y) + 'px,0)';
      this.flipEl.classList.toggle('left', this.facing < 0);
      this.rect = { x: snap(x), y: snap(y), w: this.w, h: this.h };
    }

    // Small idle motions, stepped in whole art pixels; styles change only when a step changes.
    animate(t) {
      const u = this.unit / dpr;
      const talking = this.speaking() || this.state === 'speaking';
      let body = 0;
      if (this.def.habit === 'float') body -= [0, 1, 2, 1][Math.floor((t + this.seed) / 0.7) % 4];
      if (this.def.habit === 'drift') body -= [0, 1, 2, 1][Math.floor((t + this.seed) / 0.4) % 4];
      if (talking) body -= (Math.floor(t / 0.16) % 2) * 2;
      if (this.state === 'listening') body -= 1;
      const period = this.def.habit === 'sleep' ? 2.6 : 1.4;
      const breath = (this.state === 'idle' && !talking && !this.move && ((t + this.seed) % period) < period / 2) ? 1 : 0;
      const sparkOn = this.state === 'has_arm' || this.state === 'listening';
      const sk = sparkOn ? Math.floor(t / 0.4) % 3 : -1;
      const zOn = this.sleeping && !this.move && !this.active();
      const zk = zOn ? Math.floor(((t * 0.42) % 1) * 8) : -1;
      const sig = body + '|' + breath + '|' + sk + '|' + zk;
      if (sig === this.animSig) return;
      this.animSig = sig;
      this.bodyEl.style.transform = body ? 'translateY(' + body * u + 'px)' : '';
      this.upper.style.transform = breath ? 'translateY(' + u + 'px)' : '';
      if (sk >= 0) {
        this.sparkA.style.transform = 'scale(' + [0.6, 1, 0.8][sk] + ')';
        this.sparkB.style.transform = 'scale(' + [1, 0.7, 0.6][sk] + ')';
      }
      if (zk >= 0) {
        const zb = (zk + 4) % 8;
        this.zA.style.transform = 'translate(' + zk * 2 + 'px,' + (-zk * 5) + 'px)';
        this.zA.style.opacity = String(1 - zk / 8);
        this.zB.style.transform = 'translate(' + zb * 2 + 'px,' + (-zb * 5) + 'px)';
        this.zB.style.opacity = String(1 - zb / 8);
      }
    }
  }

  function sparkleImg(cls) {
    const img = document.createElement('img');
    img.src = '../assets/hd/sparkle.png';
    img.className = 'sparkle hidden ' + cls;
    return img;
  }

  function zCanvas(cls) {
    // A pixel Z in beige with a black outline, so it reads on light and dark desktops.
    const Z = ['11111', '00010', '00100', '01000', '11111'];
    const s = Math.max(3, Math.round(3 * dpr));
    const n = 7; // 5 pixels plus one pixel of outline on each side
    const cv = document.createElement('canvas');
    cv.width = n * s; cv.height = n * s;
    cv.style.width = (n * s / dpr) + 'px'; cv.style.height = (n * s / dpr) + 'px';
    cv.className = 'zz hidden ' + cls;
    const g = cv.getContext('2d');
    const on = (x, y) => Z[y] && Z[y][x] === '1';
    for (let y = -1; y <= 5; y++) for (let x = -1; x <= 5; x++) {
      if (on(x, y)) { g.fillStyle = '#efe4d0'; g.fillRect((x + 1) * s, (y + 1) * s, s, s); }
      else if (on(x - 1, y) || on(x + 1, y) || on(x, y - 1) || on(x, y + 1)) { g.fillStyle = '#000'; g.fillRect((x + 1) * s, (y + 1) * s, s, s); }
    }
    return cv;
  }

  function find(id) { return pets.find((p) => p.id === id); }

  // Keep pets from settling on top of each other.
  function spreadOut(t) {
    for (const p of pets) {
      if (p.move || p.active() || p.forward || p.dragging || p.fall) continue;
      for (const o of pets) {
        if (o === p || o.move) continue;
        if (dist(p, o) < 150) { p.restUntil = Math.min(p.restUntil, t + 0.3); break; }
      }
    }
  }

  // ---------- main loop ----------
  let lastT = now(), lastSpread = 0;
  function frame() {
    const t = now();
    const dt = Math.min(0.1, t - lastT);
    lastT = t;
    pumpSpeech(t);
    updatePresence(t);
    let busy = false;
    for (const p of pets) busy = p.update(t, dt) || busy;
    for (const p of pets) { p.place(); p.animate(t); }
    layoutBubbles();
    renderStatus(t);
    if (presence === 'shown' && t - lastSpread > 1.5) { spreadOut(t); lastSpread = t; }
    busy = busy || pets.some((p) => p.dragging || p.fall || p.speaking() || p.state === 'speaking');
    ticks++;
    const ms = performance.now();
    if (ms - tickStart > 1000) { tickRate = ticks * 1000 / (ms - tickStart); ticks = 0; tickStart = ms; }
    // Smooth while someone travels; a few ticks a second when all are resting.
    // About 30 updates a second while moving (plenty for pixel art), whatever the display rate.
    if (busy) requestAnimationFrame(gate);
    else setTimeout(frame, 200);
  }
  let lastFrameMs = 0;
  function gate(ms) {
    if (ms - lastFrameMs < 30) { requestAnimationFrame(gate); return; }
    lastFrameMs = ms;
    frame();
  }

  // ---------- events from the server (or the demo) ----------
  function handleEvent(ev) {
    if (!ev || typeof ev !== 'object') return;
    const nameOf = (id) => (BY_ID[id] ? BY_ID[id].name : id || 'Someone');
    switch (ev.type) {
      case 'request':
        toast('Asked: ' + String(ev.text || '').slice(0, 80), 'request');
        if (ev.named_agent && find(ev.named_agent)) find(ev.named_agent).perkAt = now();
        break;
      case 'merlin':
        if (find('merlin')) find('merlin').heard = null;
        (ev.argued || []).forEach((a) => enqueueLine(a.agent, a.say, a.audio_url));
        if (ev.say) enqueueLine('merlin', ev.say, ev.audio_url);
        break;
      case 'agent_state': {
        const p = find(ev.agent);
        if (p) {
          if (ev.state === 'idle') p.heard = null;
          p.setState(ev.state);
        }
        break;
      }
      case 'say':
        if (find(ev.agent)) find(ev.agent).heard = null;
        enqueueLine(ev.agent, ev.text, ev.audio_url);
        break;
      case 'hear': {
        // The listener caught a name: perk up now, before the LLM has even started.
        const p = find(ev.agent);
        if (p) {
          p.heard = { text: String(ev.text || '').slice(0, 140), until: now() + 25 };
          p.wake();
          p.perkAt = now();
          if (p.state === 'idle') p.setState('listening'); else p.refreshClasses();
          p.comeForward();
        }
        break;
      }
      case 'listen':
        if (typeof ev.awake === 'boolean') setListen(ev.awake);
        break;
      case 'arm':
        armStatus = { agent: ev.agent, skill: ev.skill, phase: ev.phase, at: now() };
        break;
      case 'note':
        toast(nameOf(ev.agent) + (ev.kind === 'todo' ? ' added a to-do: ' : ' noted: ') + String(ev.text || '').slice(0, 80));
        break;
      case 'timer_done':
        toast('Timer done: ' + (ev.label || 'timer') + ' (' + nameOf(ev.agent) + ')');
        break;
      case 'error':
        toast('Trouble in ' + (ev.where || 'the server') + ': ' + String(ev.message || '').slice(0, 90), 'error');
        if (ev.where === 'pipeline') pets.forEach((p) => { p.heard = null; });
        break;
      default:
        break;
    }
  }

  // ---------- pointer: click-through, clicks, drags ----------
  const bar = document.getElementById('bar');
  let ignoring = true;
  let press = null;

  function petAt(x, y) {
    if (presence === 'hidden') return null;
    const pad = 4;
    for (let i = pets.length - 1; i >= 0; i--) {
      const r = pets[i].rect;
      if (r && x >= r.x - pad && x <= r.x + r.w + pad && y >= r.y - pad && y <= r.y + r.h + pad) return pets[i];
    }
    return null;
  }

  function setIgnore(v) {
    if (v === ignoring) return;
    ignoring = v;
    window.overlay.setIgnore(v);
  }

  document.addEventListener('mousemove', (e) => {
    if (press) {
      const p = press.p;
      if (!p.dragging && Math.hypot(e.clientX - press.cx, e.clientY - press.cy) > 5) {
        p.dragging = true; p.move = null; p.forward = false; p.wake(); p.refreshClasses();
      }
      if (p.dragging) {
        p.x = clamp(e.clientX, p.w / 2, W - p.w / 2);
        p.y = clamp(e.clientY + p.h * 0.5, p.h, H - TAG_ROOM);
      }
      return;
    }
    const el = document.elementFromPoint(e.clientX, e.clientY);
    const overUi = !!(el && el.closest('.interactive'));
    const p = petAt(e.clientX, e.clientY);
    setIgnore(!(overUi || p));
    document.body.style.cursor = p ? 'grab' : '';
  });

  document.addEventListener('mousedown', (e) => {
    const p = petAt(e.clientX, e.clientY);
    if (!p) return;
    if (e.button === 2) { window.overlay.showMenu(); return; }
    press = { p, cx: e.clientX, cy: e.clientY };
    document.body.style.cursor = 'grabbing';
  });

  document.addEventListener('mouseup', () => {
    if (!press) return;
    const p = press.p;
    press = null;
    document.body.style.cursor = '';
    if (p.dragging) {
      p.dragging = false;
      // Floaters stay where they are put; the others drop to the bottom edge.
      if (p.motion.drift) { p.surface = 'free'; p.restUntil = now() + rand(2, 4); }
      else p.fall = { vy: 0 };
      p.refreshClasses();
    } else {
      p.wake();
      p.perkAt = now();
      openBar(p.id);
    }
  });

  document.addEventListener('contextmenu', (e) => e.preventDefault());

  // ---------- input bar ----------
  const barText = document.getElementById('bar-text');
  const barWho = document.getElementById('bar-who');
  const barTalk = document.getElementById('bar-talk');
  const barSend = document.getElementById('bar-send');
  let barAgent = null;

  function openBar(agent) {
    barAgent = agent && BY_ID[agent] ? agent : null;
    barWho.textContent = barAgent ? 'Ask ' + BY_ID[barAgent].name : 'Ask the team';
    bar.classList.remove('hidden');
    const bw = bar.offsetWidth, bh = bar.offsetHeight;
    const p = barAgent && find(barAgent);
    let x = W / 2 - bw / 2, y = H * 0.62;
    if (p) {
      const head = p.headRect();
      x = head.cx - bw / 2;
      y = head.top - bh - 16;
      if (y < 8) y = p.y + TAG_ROOM + 10;
    }
    bar.style.left = Math.round(clamp(x, 8, W - bw - 8)) + 'px';
    bar.style.top = Math.round(clamp(y, 8, H - bh - 8)) + 'px';
    window.overlay.focusWindow();
    setTimeout(() => barText.focus(), 30);
  }

  function closeBar() {
    bar.classList.add('hidden');
    barText.value = '';
    barAgent = null;
    setIgnore(true);
  }

  async function submit(text, agent) {
    text = String(text || '').trim();
    if (!text) return;
    closeBar();
    if (flags.demo) { Demo.respond(text, agent); return; }
    try {
      const res = await fetch(flags.server + '/api/request', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, agent: agent || null }),
      });
      if (!res.ok) toast('The server refused that request (' + res.status + ')', 'error');
    } catch (e) {
      toast('The server is not reachable', 'error');
    }
  }

  async function stopArm() {
    if (flags.demo) {
      armStatus = armStatus && { ...armStatus, phase: 'done', at: now() };
      pets.forEach((p) => { if (p.state === 'has_arm') p.setState('idle'); });
      toast('Arm stopped');
      return;
    }
    try {
      await fetch(flags.server + '/api/stop', { method: 'POST' });
      toast('Arm stopped');
    } catch (e) {
      toast('Could not reach the server to stop the arm', 'error');
    }
  }

  barSend.addEventListener('click', () => submit(barText.value, barAgent));
  barText.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') submit(barText.value, barAgent);
    if (e.key === 'Escape') closeBar();
  });
  document.getElementById('bar-close').addEventListener('click', closeBar);
  document.getElementById('bar-stop').addEventListener('click', stopArm);
  stopPill.addEventListener('click', stopArm);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !bar.classList.contains('hidden')) closeBar(); });

  // Hold to talk: record while the button is held, then speech to text.
  let recorder = null, chunks = [];
  async function startTalk() {
    if (recorder) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      chunks = [];
      recorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
      recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
      recorder.onstop = async () => {
        stream.getTracks().forEach((tr) => tr.stop());
        const blob = new Blob(chunks, { type: 'audio/webm' });
        recorder = null;
        barTalk.classList.remove('recording');
        barTalk.textContent = 'Hold to talk';
        if (blob.size < 2000) return;
        if (flags.demo) { barText.value = 'Can someone lift the bottle'; return; }
        barTalk.classList.add('busy');
        try {
          const form = new FormData();
          form.append('file', blob, 'speech.webm');
          const res = await fetch(flags.server + '/api/stt', { method: 'POST', body: form });
          const data = await res.json();
          if (data && data.text) submit(data.text, barAgent);
          else toast('No words heard');
        } catch (e) {
          toast('Speech to text is not reachable', 'error');
        } finally {
          barTalk.classList.remove('busy');
        }
      };
      recorder.start();
      barTalk.classList.add('recording');
      barTalk.textContent = 'Listening';
    } catch (e) {
      recorder = null;
      toast('Microphone not available', 'error');
    }
  }
  function endTalk() { if (recorder && recorder.state === 'recording') recorder.stop(); }
  barTalk.addEventListener('pointerdown', (e) => { e.preventDefault(); startTalk(); });
  barTalk.addEventListener('pointerup', endTalk);
  barTalk.addEventListener('pointerleave', endTalk);

  // ---------- boot ----------
  function resize() {
    const oldW = W || window.innerWidth, oldH = H || window.innerHeight;
    W = window.innerWidth; H = window.innerHeight;
    const newDpr = window.devicePixelRatio || 1;
    const dprChanged = newDpr !== dpr;
    dpr = newDpr;
    for (const p of pets) {
      if (dprChanged) p.render(p.state === 'has_arm');
      const b = bounds(p);
      p.x = clamp(p.x * W / oldW, b.left, b.right);
      p.y = clamp(p.y * H / oldH, b.top, b.floor);
      p.move = null;
    }
  }

  async function boot() {
    flags = Object.assign(flags, await window.overlay.flags());
    dpr = window.devicePixelRatio || 1;
    W = window.innerWidth; H = window.innerHeight;
    window.addEventListener('resize', resize);
    await Promise.all([
      document.fonts.load('16px "HD Obra"'), document.fonts.load('16px "HD Fell"'), document.fonts.load('16px "HD Aseina"'),
    ].map((p) => p.catch(() => {})));
    const sprites = await Promise.all(CAST.map((c) => Sprites.load(c.id, !flags.fallback)));
    pets = CAST.map((c, i) => new Pet(c, sprites[i]));
    window.overlay.onOpenInput((agent) => openBar(agent));
    window.overlay.onStopArm(stopArm);
    if (window.overlay.onToggleListen) window.overlay.onToggleListen(toggleListen);
    if (!alwaysShow()) {
      // Start asleep and off screen; the server tells us if the listener is already awake.
      presence = 'hidden';
      world.classList.add('away');
    }
    renderEar();
    frame();
    if (flags.shots && flags.shots.length) {
      // Screenshot runs only: keep the compositor producing frames so
      // capturePage never returns an old frame of a mostly idle page.
      const k = document.createElement('div');
      k.className = 'keepalive';
      document.body.appendChild(k);
    }
    if (flags.demo) { setLink('demo'); Demo.start(); }
    else Net.start(flags.server);
  }

  // Hooks for the self test in main.js.
  window.__overlayTest = {
    info: () => {
      // An empty point: far from every pet and not on any control.
      let empty = { x: W / 2, y: H / 2 }, bestD = -1;
      for (let i = 0; i < 50; i++) {
        const c = { x: rand(40, W - 40), y: rand(80, H - 80) };
        const d = Math.min(...pets.map((p) => dist(c, { x: p.x, y: p.y - p.h / 2 })));
        if (d > bestD) { bestD = d; empty = c; }
      }
      return {
        avatars: pets.map((p) => ({ id: p.id, source: p.sprite.source, cx: p.rect.x + p.rect.w / 2, cy: p.rect.y + p.rect.h / 2 })),
        empty,
      };
    },
    bar: () => ({ open: !bar.classList.contains('hidden'), agent: barAgent }),
    fps: () => tickRate,
  };

  boot();
  return { handleEvent, setLink, toast, find, enqueueLine, CAST, BY_ID, presence: () => presence, listening: () => listenAwake };
})();
