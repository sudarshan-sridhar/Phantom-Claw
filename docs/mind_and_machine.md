# Mind and Machine: brain signals choosing what the arm does

## Thirty seconds, spoken

Phantom Claw is a robot arm shared by six characters on my laptop. For this track the request comes from a brain signal. The g.tec headsets were lost in transit, so we use real brain recordings made with g.tec amplifiers: a person looking at one of four flickering lights. We replay a recording as a live stream. Our own decoder works out which light they are looking at. Three lights mean bottle, Sharpie and scissors, and the fourth means stop. The choice goes to the same characters that take my voice, and the arm fetches the item or stops. The arm only moves when the decoder is sure. On the first person in the data it picked the right target 39 times out of 40 with one wrong move. It is not my own brain today. The same code takes a g.tec Unicorn headset the moment one is in the room.

## For Devpost

### What it does

A person looks at one of four lights, each flickering at its own rate: 9, 10, 12 or 15 times a second. The visual part of the brain picks up the rate of the light being looked at. Eight electrodes over the back of the head record it. We read which rate is there and turn it into a job for the robot:

| Light | Job |
|---|---|
| 9 Hz | give me the bottle |
| 10 Hz | give me the Sharpie |
| 12 Hz | give me the scissors |
| 15 Hz | stop the arm |

The job goes to the agent server as a plain request, the same way a spoken request does. One of the characters takes it, plans the arm steps from what the camera sees on the desk, and the SO-101 arm carries them out. The stop goes straight to the arm and works in the middle of a job.

### What is real

- The brain data is real. It is the SSVEP set g.tec gives out for its BR41N.IO hackathons, recorded with g.tec amplifiers: two people, two sessions each, 8 channels, 256 samples a second, 20 trials per session.
- The decoding is ours and runs live. The recording is played back as a stream at its real speed, and our decoder (canonical correlation analysis on a 3 second window) reads it as it arrives. The file also says which light was on, so every choice is checked on screen.
- The robot is real. The request goes through the same agents, plan check and arm skills as a voice command.
- It is not the owner's own brain. The headsets for the track were lost in transit. We also tried the PiEEG Octopus 16 on the owner's head. Its dry pins gave no usable signal: closing the eyes, looking at a flicker and clenching the jaw all changed nothing in the recording. So we did not build the demo on it.
- It is ready for a headset. A g.tec Unicorn streams over LSL, which is what our reader takes. One switch changes the source from the file to the headset. That path has not been run, because no headset was available.

### The safety part: a gate and a stop

A robot arm next to a person should not act on a weak guess. The decoder gives each target a score. The arm moves only when the best score and its lead over the second best both pass a limit for one second. The limits are set from the same person's other session, never from the one being played. Everything else is held back and counted on the page.

Measured over every session, 40 lights per person. A wrong move means the wrong target, or a move while no light was on.

| | Gate off | Gate on |
|---|---|---|
| Person 1, right target | 29 | 39 |
| Person 1, wrong moves | 21 | 1 |
| Person 1, lights with no move | 2 | 1 |
| Person 2, right target | 21 | 6 |
| Person 2, wrong moves | 29 | 6 |
| Person 2, lights with no move | 3 | 31 |

Person 1 decodes well, and the gate removes almost every wrong move. Person 2 decodes poorly. With the gate on, the arm mostly stays still for that person. With a stricter gate it makes no wrong move at all for either person, and moves less. Guessing would be right one time in four.

The fourth light is the stop. It is sent at once, also while the arm is carrying something, and it always wins over a job.

### End to end

We ran a whole session at real speed into a copy of the agent server with a simulated arm: 20 lights, 19 decoded right, none wrong, one with no choice. Five stops and nine requests went out. Six choices were dropped because the arm was still on a job, since only one job runs at a time. Three jobs were stopped part way by a stop light. Each request came back with the character that took it, the arm steps and the result, and the page shows all of that.

For a live demo we play chosen trials one at a time, for example a Sharpie trial, or a scissors trial followed by a stop trial. They are real trials from the recording, played in an order we choose.

### How we built it

- Python. The recording is read with SciPy and published with Lab Streaming Layer, the same interface a g.tec Unicorn uses.
- Decoder: band-pass, then canonical correlation against sine and cosine references at each rate and its second harmonic, every quarter second.
- A small status page shows the four targets, the decoded one, the confidence against the gate, the count held back, and what the agents and the arm did.
- Tests cover the file reader, the decoder, the gate, the one-job rule and the stop.

### What we learned

- A calibrated gate matters more than a better score. Without it the same decoder made 21 wrong moves on the good subject.
- Brain decoding varies a lot between people. One of the two people in the data cannot drive the arm safely with this method, and the system shows that by staying still.
- Dry electrodes through hair gave us nothing in the time we had. Recorded data from proper amplifiers let us build and measure the rest.

### What is next

- Run it on a g.tec Unicorn headset over LSL.
- Train on a short calibration for each person so more people pass the gate.
- Use g.tec's other public sets (P300, motor imagery) through the same replay path.
