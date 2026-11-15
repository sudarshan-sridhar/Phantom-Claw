# Phantom Claw

A third hand you can read, so you can trust it next to you.

## Inspiration

A robot arm on your desk is only useful if you can tell what it is about to do. We wanted an arm with a character you can read, the way you read a coworker. So we gave one arm to six characters with different natures and let you pick who does the job.

## What it does

Six pixel animals live on the edge of your screen. Each is a character named after the Seven Deadly Sins: Escanor (proud), Elizabeth (caring), Ban (eager), King (lazy), Gowther (plain and exact), and Merlin, who decides who takes a job when you name nobody.

You talk to them. Say "Phantom Claw, start" and then call one by name.

On the laptop they open apps and folders, search the web, play music, set timers, take notes, take a screenshot and read the screen. They refuse to delete files, send messages, pay, type passwords or change settings.

On the desk they share one SO-101 robot arm. Ask for the water bottle, the Sharpie or the scissors and the arm finds it with an overhead phone camera, lines its jaws up with a camera on the gripper, picks it up, turns to where you are standing and holds it out. It lets go when it sees your hand or feels you pull. If nobody takes it, it puts the item back where it was.

The same request looks different with each character. Escanor is fast and carries high. Elizabeth looks first and moves gently. King points at the thing and only fetches it if you insist.

A record page shows which AI model answered for each character, how long it took, and every arm job, stored in Tiger Data.

## How we built it

- Arm: SO-101 follower driven straight from the laptop in Python. Grips were taught by hand at two distances and are blended by how far away the item is.
- Seeing: four ArUco markers map the overhead picture to centimetres on the desk. Gemini lists what is on the desk, Claude pins down the exact spot of the item asked for, and OpenCV on the gripper camera corrects the wrist angle and aim before the jaws go down. If the overhead camera drops, the arm surveys the desk with the gripper camera and says so.
- Characters: Merlin and Gowther run on Claude. Escanor and King run on OpenAI. Elizabeth and Ban run on Gemini. Each falls back to another provider if its own fails. Merlin checks every plan before it runs, and the safety rules are in code, not in a prompt.
- Voice: local Whisper on the GPU for hearing, ElevenLabs for the six voices.
- Screen: an Electron overlay with original pixel art.
- Record: every request, reply and joint reading goes to Tiger Data hypertables.

## Challenges we ran into

- A general reach calculation was 6 cm off, so we dropped it and blended hand-taught grips instead.
- One overhead reading is not accurate enough to grip a pen. Adding the gripper camera look before closing fixed that.
- The laptop's built-in microphone turned out to be silent. Speech worked once we moved to the camera's microphone.
- Brain control: the g.tec headsets for the Mind and Machine track did not arrive, and our PiEEG board gave no usable signal on a real head with dry pins. We built the software side anyway. Recorded brain signals from g.tec amplifiers are decoded live by our code and choose what the arm fetches, with a confidence gate so the arm stays still when the signal is unclear. It is recorded data, not our own brains.

## Accomplishments that we're proud of

- The arm picks up three different things from anywhere inside the markers, at any angle, and hands them to the right person.
- You can tell the characters apart by how the arm moves.
- It says plainly what it could not do and puts things back.

## What we learned

Teaching a few good poses by hand and blending them beat the maths we did not have time to get right. Two cheap cameras that check each other beat one good guess. And a robot that tells you when it failed is easier to trust than one that is right most of the time.

## What's next

More objects and grip heights, live brain control with a working headset, and a second arm.

## Built with

Python, FastAPI, OpenCV, ArUco, Claude, Gemini, OpenAI, ElevenLabs, faster-whisper, Tiger Data (TimescaleDB), Electron, SO-101, LeRobot servos.

## Links

- Code: https://github.com/sudarshan-sridhar/Phantom-Claw
- Demo video: https://youtu.be/MHnvnfpzpcg
