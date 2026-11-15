# Phantom Claw

A third hand you can read, so you can trust it next to you.

![Phantom Claw picking up a bottle](assets/demo.gif)

Watch the demo: https://www.youtube.com/watch?v=O80itQ5YU0w

Built at Hack The Nest 2026 (Eastern Michigan University) by team The Third Hand.

## The problem

You can't tell what a robot arm or an AI assistant is about to do, or how sure it is. So you either watch it the whole time or you keep it away from your work.

People read each other's intent from manner. You know a careful colleague will double-check and an impatient one will rush. Phantom Claw gives a machine a manner you can read.

## What it is

A cast of AI agents shares one SO-101 robot arm on your desk. Each agent has a role, an ego, its own LLM and its own voice. The ego changes how the arm moves as well as what the agent says: its speed, how far it reaches and how careful it is.

The cast is a fan tribute to the Seven Deadly Sins anime, where each knight carries one sin.

| Agent | Ego | On the arm |
|---|---|---|
| Escanor | Proud | Fetches when the job is worthy, points at it when it isn't |
| Elizabeth | Caring | Slow, gentle, checks twice before gripping |
| Ban | Greedy and eager | Fast grabs, often before you finish asking |
| King | Lazy | The least movement that counts |
| Gowther | Neutral | Exactly what was asked, nothing more |
| Merlin | The Conductor | Decides who takes each job and checks the plan |

## What works today

- The arm picks up a bottle by the cap, holds it in the air, puts it back in the same place and folds away. It has done this run after run with the grip landing in the same spot.
- Skills are taught by hand. With the motors off, you move the arm through the job once. The arm records the joints, finds the still moments and replays them as key poses.
- Motion is one smooth blended move, with the elbow leading the shoulder so the forearm never swings into the desk.
- The wrist camera sweeps the desk to find an object and turns the arm to face it.
- Safety is the same for every agent: joint limits, a current cutoff, and a check that refuses to power a joint that reads out of range.

## What we are building next

- The agent layer: OpenAI, Gemini and Claude behind one interface, with Merlin routing each request
- ElevenLabs voices, one per agent, and push-to-talk speech in
- Pixel avatars that sit on the laptop screen and show who has the arm
- Laptop work for the same agents: notes, lookups, timers, reading the screen
- Brain control with a PiEEG board: focus level and a stop signal
- Reach maths, so the arm can pick up an object anywhere in view

## How it works

Everything runs on one laptop. The arm plugs in over USB.

1. A request comes in by voice or text.
2. Merlin picks the agent for the job and checks its plan against the arm's limits.
3. The agent answers with one small structured plan: which skill, which target, and its ego's settings. Free text never drives the arm.
4. The skill runs on the arm with those settings, and the agent speaks in its own voice.

## Hardware

- SO-101 follower arm with six STS3215 servos, 12 V
- Wrist camera on the gripper
- A laptop

There is no leader arm, which is why every skill is taught by hand.

## Built with

Python, LeRobot, OpenCV, OpenAI, Gemini, Claude, ElevenLabs, Tiger Data.

AI tools used: Claude Code for planning, writing code and driving the arm during the build.

## Team

- Sudarshan Sridhar
- Aditi Dharmadhikari

## Note

The agent names come from The Seven Deadly Sins. The avatars, lines and voices are our own. This project is a fan tribute and is not affiliated with the series.
