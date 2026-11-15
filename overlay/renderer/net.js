// Live link to the Phantom Claw server: one WebSocket that reconnects by itself.
'use strict';

const Net = (() => {
  let ws = null;
  let base = 'http://127.0.0.1:8765';
  let delay = 1000;
  let timer = null;

  function start(server) {
    base = server;
    connect();
  }

  function connect() {
    clearTimeout(timer);
    App.setLink(ws ? 'offline' : 'connecting');
    let sock;
    try {
      sock = new WebSocket(base.replace(/^http/, 'ws') + '/ws');
    } catch (e) {
      retry();
      return;
    }
    ws = sock;
    sock.onopen = () => {
      delay = 1000;
      App.setLink('live');
      syncCast();
    };
    sock.onmessage = (msg) => {
      let ev = null;
      try { ev = JSON.parse(msg.data); } catch (e) { return; }
      App.handleEvent(ev);
    };
    sock.onclose = () => { App.setLink('offline'); retry(); };
    sock.onerror = () => { /* onclose follows and schedules the retry */ };
  }

  function retry() {
    clearTimeout(timer);
    timer = setTimeout(connect, delay);
    delay = Math.min(delay * 1.6, 5000);
  }

  // Pick up the current states after a (re)connect, if the server offers them.
  async function syncCast() {
    try {
      const res = await fetch(base + '/api/cast');
      if (!res.ok) return;
      const data = await res.json();
      const list = Array.isArray(data) ? data : (data.cast || Object.entries(data).map(([id, v]) => ({ id, ...v })));
      for (const c of list) {
        if (c && c.id && c.state) App.handleEvent({ type: 'agent_state', agent: c.id, state: c.state });
      }
    } catch (e) { /* the overlay works without it */ }
  }

  return { start };
})();
