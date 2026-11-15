// Demo mode: plays the contract's events on a loop so the overlay can be judged
// with no server running. Started by `npm run demo`.
'use strict';

const Demo = (() => {
  const ALL = ['escanor', 'elizabeth', 'ban', 'king', 'gowther', 'merlin'];
  const states = (state, ids) => (ids || ALL).map((agent) => ({ type: 'agent_state', agent, state }));
  const armRun = (agent, skill, phases) => phases.map(([wait, phase]) => [wait, { type: 'arm', agent, skill, phase }]);

  // Each step: [seconds to wait before it, event or list of events].
  const SCRIPT = [
    [1.5, { type: 'request', text: 'Can someone lift the water bottle for me?', named_agent: null }],
    [0.1, states('listening', ['escanor', 'elizabeth', 'ban', 'king', 'gowther'])],
    [1.0, { type: 'merlin', chosen: 'ban', say: 'Ban asked first and means it. The bottle is yours, Ban. Gently.', audio_url: null,
      argued: [
        { agent: 'ban', say: 'Mine. I will have it up before you blink.', audio_url: null },
        { agent: 'king', say: 'Not me. Too heavy. Too far.', audio_url: null },
      ] }],
    [0.2, states('idle', ['escanor', 'elizabeth', 'king', 'gowther'])],
    [9.5, states('thinking', ['ban'])],
    [2.0, [states('speaking', ['ban'])[0], { type: 'say', agent: 'ban', text: 'Easy. Watch this, and I will throw in a twirl for free.', audio_url: null }]],
    [3.6, states('has_arm', ['ban'])],
    ...armRun('ban', 'grip_bottle', [[0.4, 'start'], [1.4, 'at_target'], [1.2, 'gripped']]),
    // King gets asked something while Ban is still holding the arm.
    [0.0, [{ type: 'request', text: 'King, set a tea timer and add tea to my list', named_agent: 'king' }, states('listening', ['king'])[0]]],
    ...armRun('ban', 'grip_bottle', [[1.0, 'lifted']]),
    [0.0, states('thinking', ['king'])],
    ...armRun('ban', 'grip_bottle', [[3.0, 'put_back'], [1.4, 'returning'], [1.3, 'done']]),
    [0.6, [states('idle', ['ban'])[0], { type: 'note', agent: 'ban', text: 'bottle lifted and put back', kind: 'note' }]],

    [1.0, [states('speaking', ['king'])[0], { type: 'say', agent: 'king', text: 'Fine. Five minutes. Wake me after.', audio_url: null }]],
    [3.2, [states('idle', ['king'])[0], { type: 'note', agent: 'king', text: 'make tea', kind: 'todo' }]],
    [2.0, { type: 'timer_done', agent: 'king', label: 'tea' }],

    [2.5, { type: 'request', text: 'Gowther, lift the bottle', named_agent: 'gowther' }],
    [0.1, states('listening', ['gowther'])],
    [1.4, states('thinking', ['gowther'])],
    [1.5, [states('speaking', ['gowther'])[0], { type: 'say', agent: 'gowther', text: 'Lifting the bottle. Holding five seconds.', audio_url: null }]],
    [3.0, states('has_arm', ['gowther'])],
    ...armRun('gowther', 'grip_bottle', [[0.3, 'start'], [1.6, 'at_target'], [1.3, 'gripped'], [1.0, 'lifted'], [5.0, 'put_back'], [1.5, 'returning'], [1.2, 'done']]),
    [0.5, [states('idle', ['gowther'])[0], { type: 'say', agent: 'gowther', text: 'Done.', audio_url: null }]],
    [6.0, null], // pause, then loop
  ];

  let running = false;
  let step = 0;
  let timer = null;

  function emit(ev) {
    if (!ev) return;
    (Array.isArray(ev) ? ev : [ev]).forEach((e) => App.handleEvent(e));
  }

  function next() {
    if (!running) return;
    const [wait, ev] = SCRIPT[step];
    timer = setTimeout(() => {
      emit(ev);
      step = (step + 1) % SCRIPT.length;
      next();
    }, wait * 1000);
  }

  function start() {
    running = true;
    step = 0;
    next();
  }

  // Typed requests in demo mode get a short scripted answer.
  function respond(text, agent) {
    const pick = agent || ['escanor', 'elizabeth', 'ban', 'king', 'gowther'][Math.floor(Math.random() * 5)];
    const name = App.BY_ID[pick].name;
    const lines = {
      escanor: 'Beneath me. Ask me nicely and we will see.',
      elizabeth: 'Of course. Is there anything else you need?',
      ban: 'On it. Faster than you can say please.',
      king: 'Sigh. Fine.',
      gowther: 'Request received. Done.',
      merlin: 'Interesting. Let me find the right one for that.',
    };
    const seq = [
      [0, { type: 'request', text, named_agent: agent || null }],
      [100, { type: 'agent_state', agent: pick, state: 'listening' }],
    ];
    if (!agent) seq.push([900, { type: 'merlin', chosen: pick, say: name + ', this one is yours.', audio_url: null, argued: [] }]);
    seq.push([agent ? 1400 : 3600, { type: 'agent_state', agent: pick, state: 'thinking' }]);
    seq.push([agent ? 2800 : 5000, [{ type: 'agent_state', agent: pick, state: 'speaking' }, { type: 'say', agent: pick, text: lines[pick], audio_url: null }]]);
    seq.push([agent ? 6000 : 8200, { type: 'agent_state', agent: pick, state: 'idle' }]);
    seq.forEach(([ms, ev]) => setTimeout(() => emit(ev), ms));
  }

  return { start, respond };
})();
