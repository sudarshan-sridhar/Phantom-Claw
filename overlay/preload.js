// Small, explicit bridge between the page and the main process.
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('overlay', {
  flags: () => ipcRenderer.invoke('get-flags'),
  setIgnore: (ignore) => ipcRenderer.send('set-ignore', !!ignore),
  focusWindow: () => ipcRenderer.send('focus-window'),
  showMenu: () => ipcRenderer.send('show-menu'),
  quit: () => ipcRenderer.send('quit'),
  onOpenInput: (fn) => ipcRenderer.on('open-input', (_e, agent) => fn(agent)),
  onStopArm: (fn) => ipcRenderer.on('stop-arm', () => fn()),
  onToggleListen: (fn) => ipcRenderer.on('toggle-listen', () => fn()),
});
