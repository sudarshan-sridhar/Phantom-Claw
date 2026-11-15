// Phantom Claw overlay: main process.
// A frameless, transparent, always-on-top window over the primary display.
// Clicks pass through everywhere except avatars and the input bar; the
// renderer tells us when the pointer is over something interactive.

const { app, BrowserWindow, ipcMain, screen, globalShortcut, session, Tray, Menu, nativeImage } = require('electron');
const fs = require('fs');
const path = require('path');

const SHORTCUT = 'CommandOrControl+Shift+Space';
// Backup for the listener's own Ctrl+Shift tap: toggles listening through the server.
const LISTEN_SHORTCUT = 'CommandOrControl+Shift+L';
// Stops the robot arm from anywhere. There is no Stop button on screen, so this
// and the tray menu are the ways to stop it.
const STOP_SHORTCUT = 'CommandOrControl+Shift+X';

const argv = process.argv.slice(1);
const flag = (name) => argv.includes('--' + name);
const value = (name) => {
  const hit = argv.find((a) => a.startsWith('--' + name + '='));
  return hit ? hit.slice(name.length + 3) : null;
};

const FLAGS = {
  demo: flag('demo'),
  show: flag('show'),   // keep the avatars on screen even while the listener is asleep
  selftest: flag('selftest'),
  fallback: flag('fallback'),
  shots: (value('shots') || '').split(',').map((s) => parseInt(s, 10)).filter((n) => n > 0),
  quitAfter: parseInt(value('quit-after') || '0', 10),
  server: value('server') || 'http://127.0.0.1:8765',
};

let win = null;
let tray = null;
let ignoring = null;
let topTimer = null;

if (!app.requestSingleInstanceLock()) {
  app.quit();
}

// The window covers the primary display's work area: the whole screen except
// the taskbar, so the pets can walk along the top of the taskbar and Windows
// never mistakes the overlay for a full-screen app.
function windowBounds() {
  return { ...screen.getPrimaryDisplay().workArea };
}

function setIgnore(ignore) {
  if (!win || win.isDestroyed() || ignoring === ignore) return;
  ignoring = ignore;
  // forward: true keeps mousemove events coming while clicks pass through,
  // so the renderer can notice the pointer arriving over an avatar.
  win.setIgnoreMouseEvents(ignore, { forward: true });
}

function keepOnTop() {
  if (!win || win.isDestroyed()) return;
  // 'screen-saver' is the highest level Electron offers; it sits above most
  // full-screen windows on Windows. Re-assert now and then because some apps
  // push themselves to the top when they go full screen.
  win.setAlwaysOnTop(true, 'screen-saver');
}

function makeTrayIcon() {
  const s = 16;
  const buf = Buffer.alloc(s * s * 4);
  for (let y = 0; y < s; y++) for (let x = 0; x < s; x++) {
    const i = (y * s + x) * 4;
    const dx = x - 7.5, dy = y - 8.5;
    const inside = dx * dx + dy * dy < 42;
    const ear = (y < 5 && (x === 3 || x === 4 || x === 11 || x === 12));
    if (inside || ear) {
      const eye = (y === 8 && (x === 5 || x === 10));
      buf[i] = eye ? 30 : 35;      // B
      buf[i + 1] = eye ? 20 : 166; // G
      buf[i + 2] = eye ? 20 : 245; // R
      buf[i + 3] = 255;
    }
  }
  return nativeImage.createFromBitmap(buf, { width: s, height: s });
}

function createWindow() {
  const b = windowBounds();
  win = new BrowserWindow({
    ...b,
    frame: false,
    transparent: true,
    backgroundColor: '#00000000',
    resizable: false,
    movable: false,
    minimizable: false,
    maximizable: false,
    fullscreenable: false,
    hasShadow: false,
    skipTaskbar: true,
    alwaysOnTop: true,
    focusable: true,
    show: false,
    title: 'Phantom Claw',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      backgroundThrottling: false,
      spellcheck: false,
    },
  });
  win.setMenu(null);
  keepOnTop();
  win.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });
  setIgnore(true);
  win.loadFile(path.join(__dirname, 'renderer', 'index.html'));
  win.once('ready-to-show', () => win.showInactive());
  topTimer = setInterval(keepOnTop, 4000);

  win.on('closed', () => { win = null; });

  const refit = () => { if (win && !win.isDestroyed()) win.setBounds(windowBounds()); };
  screen.on('display-metrics-changed', refit);
  screen.on('display-added', refit);
  screen.on('display-removed', refit);
}

function quitApp() {
  if (topTimer) clearInterval(topTimer);
  app.quit();
}

// Ask the server to stop the arm, then tell the page so the pets let go of it.
// In demo mode there is no server, so only the page is told.
async function stopArm() {
  if (win && !win.isDestroyed()) win.webContents.send('stop-arm');
  if (FLAGS.demo) { console.log('stop (demo, no server call)'); return; }
  try {
    const res = await fetch(FLAGS.server + '/api/stop', { method: 'POST' });
    console.log('stop sent, server answered ' + res.status);
  } catch (e) {
    console.log('could not reach the server to stop the arm: ' + e.message);
  }
}

function menuTemplate() {
  return [
    { label: 'Stop the arm (Ctrl+Shift+X)', click: stopArm },
    { type: 'separator' },
    { label: 'Talk to the team (Ctrl+Shift+Space)', click: () => openInput(null) },
    { label: 'Start or stop listening (Ctrl+Shift)', click: () => win && win.webContents.send('toggle-listen') },
    { type: 'separator' },
    { label: 'Quit Phantom Claw', click: quitApp },
  ];
}

function showMenu() {
  Menu.buildFromTemplate(menuTemplate()).popup({ window: win || undefined });
}

function openInput(agent) {
  if (!win || win.isDestroyed()) return;
  setIgnore(false);
  win.focus();
  win.webContents.focus();
  win.webContents.send('open-input', agent || null);
}

// ---------- IPC from the renderer ----------

ipcMain.handle('get-flags', () => FLAGS);
ipcMain.on('set-ignore', (_e, ignore) => setIgnore(!!ignore));
ipcMain.on('focus-window', () => {
  if (!win) return;
  setIgnore(false);
  win.focus();
  win.webContents.focus();
});
ipcMain.on('show-menu', showMenu);
ipcMain.on('quit', quitApp);

// ---------- test helpers (screenshots and self test) ----------

async function capture(file) {
  // Force a fresh frame first; a mostly idle page may not have painted lately.
  win.webContents.invalidate();
  await win.webContents.executeJavaScript(
    'new Promise((r) => { document.body.style.outline = document.body.style.outline ? "" : "0px solid transparent"; requestAnimationFrame(() => requestAnimationFrame(r)); })');
  await new Promise((r) => setTimeout(r, 150));
  const img = await win.webContents.capturePage();
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, img.toPNG());
  console.log('SHOT ' + file + ' ' + img.getSize().width + 'x' + img.getSize().height);
}

function scheduleTests() {
  if (FLAGS.shots.length || FLAGS.selftest) {
    // Show page errors in the terminal during test runs.
    win.webContents.on('console-message', (e, legacyLevel, legacyMessage, line, source) => {
      const level = e && e.level !== undefined ? e.level : legacyLevel;
      const message = (e && e.message) || legacyMessage || '';
      if (level === 'error' || level === 'warning' || level >= 2) console.log('PAGE ' + message + (line ? ' (' + source + ':' + line + ')' : ''));
    });
  }
  const outDir = path.join(__dirname, 'test', 'out');
  FLAGS.shots.forEach((ms, i) => {
    setTimeout(() => capture(path.join(outDir, 'shot_' + (i + 1) + '.png')).catch((e) => console.log('SHOT FAILED ' + e.message)), ms);
  });
  if (FLAGS.quitAfter > 0) setTimeout(quitApp, FLAGS.quitAfter);
  if (FLAGS.selftest) setTimeout(selfTest, 2500);
  if (flag('probe')) {
    [2000, 4000, 8000, 12000, 16000, 20000, 22500].forEach((ms) => setTimeout(async () => {
      const r = await win.webContents.executeJavaScript('JSON.stringify({bubble: !document.querySelector(".bubble").classList.contains("hidden"), furniture: window.__overlayTest.furniture(), states: window.__overlayTest.states(), err: window.__lastError || null})');
      console.log('PROBE ' + ms + ' ' + r);
    }, ms));
  }
}

async function selfTest() {
  const wc = win.webContents;
  const results = [];
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  try {
    const info = await wc.executeJavaScript('window.__overlayTest && window.__overlayTest.info()');
    results.push('avatars=' + info.avatars.length + ' sprites=' + info.avatars.map((a) => a.id + ':' + a.source).join(','));
    const a = info.avatars.find((v) => v.id === 'gowther') || info.avatars[0];
    // Pointer over an avatar: the window must stop ignoring the mouse.
    wc.sendInputEvent({ type: 'mouseMove', x: Math.round(a.cx), y: Math.round(a.cy) });
    await wait(300);
    results.push('over avatar -> ignoring=' + ignoring + (ignoring === false ? ' PASS' : ' FAIL'));
    // Pointer over empty space: clicks must pass through again.
    wc.sendInputEvent({ type: 'mouseMove', x: Math.round(info.empty.x), y: Math.round(info.empty.y) });
    await wait(300);
    results.push('over empty space -> ignoring=' + ignoring + (ignoring === true ? ' PASS' : ' FAIL'));
    // Click the avatar: the input bar opens with that agent.
    wc.sendInputEvent({ type: 'mouseMove', x: Math.round(a.cx), y: Math.round(a.cy) });
    await wait(150);
    wc.sendInputEvent({ type: 'mouseDown', x: Math.round(a.cx), y: Math.round(a.cy), button: 'left', clickCount: 1 });
    wc.sendInputEvent({ type: 'mouseUp', x: Math.round(a.cx), y: Math.round(a.cy), button: 'left', clickCount: 1 });
    await wait(300);
    const bar = await wc.executeJavaScript('window.__overlayTest.bar()');
    results.push('click avatar -> bar open=' + bar.open + ' agent=' + bar.agent + (bar.open && bar.agent === a.id ? ' PASS' : ' FAIL'));
    await capture(path.join(__dirname, 'test', 'out', 'selftest_bar.png'));
    // Nothing on screen except pets, names, the scroll and the bar.
    const extra = await wc.executeJavaScript('window.__overlayTest.furniture()');
    results.push('other things on screen=' + JSON.stringify(extra) + (extra.length === 0 ? ' PASS' : ' FAIL'));
    // The arm can still be stopped without a button on screen.
    const stopKey = globalShortcut.isRegistered(STOP_SHORTCUT);
    results.push('stop shortcut ' + STOP_SHORTCUT + ' registered=' + stopKey + (stopKey ? ' PASS' : ' FAIL'));
    const hasStop = menuTemplate().some((m) => /stop the arm/i.test(m.label || ''));
    results.push('tray menu has Stop=' + hasStop + (hasStop ? ' PASS' : ' FAIL'));
    const fps = await wc.executeJavaScript('window.__overlayTest.fps()');
    results.push('render fps=' + fps.toFixed(1));
  } catch (e) {
    results.push('ERROR ' + e.message);
  }
  results.forEach((r) => console.log('SELFTEST ' + r));
  quitApp();
}

// ---------- app lifecycle ----------

app.whenReady().then(() => {
  // Allow the microphone for hold-to-talk; nothing else.
  session.defaultSession.setPermissionRequestHandler((_wc, permission, cb) => cb(permission === 'media'));
  session.defaultSession.setPermissionCheckHandler((_wc, permission) => permission === 'media');

  createWindow();

  if (!globalShortcut.register(SHORTCUT, () => openInput(null))) {
    console.log('could not register ' + SHORTCUT + ' (another app may own it)');
  }
  if (!globalShortcut.register(LISTEN_SHORTCUT, () => win && win.webContents.send('toggle-listen'))) {
    console.log('could not register ' + LISTEN_SHORTCUT + ' (another app may own it)');
  }
  if (!globalShortcut.register(STOP_SHORTCUT, stopArm)) {
    console.log('could not register ' + STOP_SHORTCUT + ' (another app may own it)');
  }

  try {
    tray = new Tray(makeTrayIcon());
    tray.setToolTip('Phantom Claw (right-click to stop the arm)');
    tray.on('click', () => openInput(null));
    tray.on('right-click', showMenu);
  } catch (e) {
    console.log('tray unavailable: ' + e.message);
  }

  win.webContents.once('did-finish-load', scheduleTests);
});

app.on('second-instance', () => openInput(null));
app.on('will-quit', () => {
  globalShortcut.unregisterAll();
  if (tray) { tray.destroy(); tray = null; }
});
app.on('window-all-closed', () => app.quit());
