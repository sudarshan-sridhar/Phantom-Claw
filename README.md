<div align="center">

# Phantom Claw

### A third hand you can read, so you can trust it next to you.

![Hack The Nest 2026](https://img.shields.io/badge/Hack%20The%20Nest-2026-0B6E4F?style=for-the-badge)
![Team](https://img.shields.io/badge/Team-The%20Third%20Hand-C9A227?style=for-the-badge)
![Arm](https://img.shields.io/badge/Arm-SO--101-7B2CBF?style=for-the-badge)

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![LeRobot](https://img.shields.io/badge/LeRobot-0.6-FFB000?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-vision-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-agents-412991?style=flat-square)
![Gemini](https://img.shields.io/badge/Gemini-vision-4285F4?style=flat-square&logo=googlegemini&logoColor=white)
![Claude](https://img.shields.io/badge/Claude-conductor-D97757?style=flat-square)
![ElevenLabs](https://img.shields.io/badge/ElevenLabs-voices-111111?style=flat-square)
![Tiger Data](https://img.shields.io/badge/Tiger%20Data-telemetry-F5C400?style=flat-square)

<img src="assets/demo.gif" alt="Phantom Claw picking up a bottle" width="420">

[![Watch the demo](https://img.shields.io/badge/Watch%20the%20demo-YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=O80itQ5YU0w)

</div>

## The problem

You can't tell what a robot arm or an AI assistant is about to do, or how sure it is. So you either watch it the whole time or you keep it away from your work.

People read each other's intent from manner. You know a careful colleague will double-check and an impatient one will rush. Phantom Claw gives a machine a manner you can read.

> [!TIP]
> Who it is for: anyone working with both hands busy or one hand unavailable. Soldering, lab work, crafting, limited mobility.

## What it is

A cast of AI agents shares one SO-101 robot arm on your desk. Each agent has a role, an ego, its own LLM and its own voice. The ego changes how the arm moves as well as what the agent says: its speed, how far it reaches and how careful it is.

The cast is a fan tribute to the Seven Deadly Sins anime, where each knight carries one sin.

| Agent | Ego | On the arm |
|---|---|---|
| ![Escanor](https://img.shields.io/badge/Escanor-F5A623?style=for-the-badge) | Proud | Fetches when the job is worthy, points at it when it isn't |
| ![Elizabeth](https://img.shields.io/badge/Elizabeth-8ECAE6?style=for-the-badge) | Caring | Slow, gentle, checks twice before gripping |
| ![Ban](https://img.shields.io/badge/Ban-C1121F?style=for-the-badge) | Greedy and eager | Fast grabs, often before you finish asking |
| ![King](https://img.shields.io/badge/King-2D6A4F?style=for-the-badge) | Lazy | The least movement that counts |
| ![Gowther](https://img.shields.io/badge/Gowther-D63384?style=for-the-badge) | Neutral | Exactly what was asked, nothing more |
| ![Merlin](https://img.shields.io/badge/Merlin-3C096C?style=for-the-badge) | The Conductor | Decides who takes each job and checks the plan |

## How it works

Everything runs on one laptop. The arm plugs in over USB.

```mermaid
flowchart TD
    A[Voice or text request] --> B[Merlin, the Conductor]
    B --> C{Which agent?}
    C --> E1[Escanor]
    C --> E2[Elizabeth]
    C --> E3[Ban]
    C --> E4[King]
    C --> E5[Gowther]
    E1 & E2 & E3 & E4 & E5 --> P[One structured plan: skill, target, ego settings]
    P --> S[Safety check: joint limits, speed cap, range check]
    S --> R[SO-101 arm runs the skill]
    S --> V[Agent speaks in its own voice]

    classDef conductor fill:#3C096C,stroke:#240046,color:#ffffff
    classDef escanor fill:#F5A623,stroke:#B9770E,color:#000000
    classDef elizabeth fill:#8ECAE6,stroke:#219EBC,color:#000000
    classDef ban fill:#C1121F,stroke:#780000,color:#ffffff
    classDef king fill:#2D6A4F,stroke:#1B4332,color:#ffffff
    classDef gowther fill:#D63384,stroke:#A61E6A,color:#ffffff
    classDef safe fill:#0B6E4F,stroke:#073B3A,color:#ffffff
    classDef arm fill:#7B2CBF,stroke:#5A189A,color:#ffffff
    class B conductor
    class E1 escanor
    class E2 elizabeth
    class E3 ban
    class E4 king
    class E5 gowther
    class S safe
    class R,V arm
```

Free text never drives the arm. Every agent answers with one small structured plan, and that plan is checked before anything moves.

## What works today

![Working](https://img.shields.io/badge/status-working-2EA44F?style=flat-square)

- The arm picks up a bottle by the cap, holds it in the air, puts it back in the same place and folds away. It has done this run after run with the grip landing in the same spot.
- Skills are taught by hand. With the motors off, you move the arm through the job once. The arm records the joints, finds the still moments and replays them as key poses.
- Motion is one smooth blended move, with the elbow leading the shoulder so the forearm never swings into the desk.
- The wrist camera sweeps the desk to find an object and turns the arm to face it.

> [!IMPORTANT]
> Safety is the same for every agent: joint limits, a current cutoff, and a check that refuses to power a joint that reads out of range. No ego can change them.

## What we are building next

![In progress](https://img.shields.io/badge/status-in%20progress-F5A623?style=flat-square)

- The agent layer: OpenAI, Gemini and Claude behind one interface, with Merlin routing each request
- ElevenLabs voices, one per agent, and push-to-talk speech in
- Pixel avatars that sit on the laptop screen and show who has the arm
- Laptop work for the same agents: notes, lookups, timers, reading the screen
- Brain control with a PiEEG board: focus level and a stop signal
- Reach maths, so the arm can pick up an object anywhere in view

## Hardware

- SO-101 follower arm with six STS3215 servos, 12 V
- Wrist camera on the gripper
- A laptop

There is no leader arm, which is why every skill is taught by hand.

## AI tools used

Claude Code for planning, writing code and driving the arm during the build.

## Team

![Sudarshan Sridhar](https://img.shields.io/badge/Sudarshan%20Sridhar-0B6E4F?style=for-the-badge)
![Aditi Dharmadhikari](https://img.shields.io/badge/Aditi%20Dharmadhikari-C9A227?style=for-the-badge)

> [!NOTE]
> The agent names come from The Seven Deadly Sins. The avatars, lines and voices are our own. This project is a fan tribute and is not affiliated with the series.
