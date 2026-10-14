# Physyk — Agentic Pick-and-Place

> **New to the project?** Start with `HOW_IT_WORKS.md` — a plain-English walkthrough with no code.

## Problem Statement

Most robots in a factory today can only do one thing, one way, forever. You program a robot arm to pick up a part from exactly position X and put it down at exactly position Y, and that is all it will ever do — the moment the part moves, the layout changes, or someone gives it a new instruction in plain English, it breaks. Re-programming a robot for a new task usually means an engineer sitting down and hand-coding new motion paths, which can take days.

**Physyk asks a simple question:** what if you could just *tell* the robot what to do — in plain English, with the layout allowed to change — and have it figure out the rest, verify its own work, and recover when something goes wrong?

## Objectives

1. **Understand natural-language instructions** ("pick up the red cube and place it in the tray", "stack the cubes") without hand-coded, per-task programming.
2. **Adapt to a changing scene**: object positions are not fixed or hardcoded; the robot perceives where things actually are before acting.
3. **Verify real outcomes, not assumptions**: confirm a task actually succeeded using live sensor or position data, not just "the motion finished so it must have worked."
4. **Recover from failure automatically**: retry, re-plan or escalate when a pick or placement does not go as expected, instead of silently reporting success.

## Core Capabilities

- **Instruction parsing**: extracts the source object and destination from natural-language commands.
- **Stack-aware planning**: "stack the cubes" is a vertical pile planned by a local, deterministic planner from the cubes' positions (never from colour), on the tray or on the table, from any starting arrangement: piles in the wrong place are taken apart from the top, partial piles are extended, covered cubes are never picked, nearby cubes are ordered to keep the gripper clear, and a cube out of the arm's reach is reported instead of attempted. The plan is rewritten from the live scene after every step.
- **Safe placement**: Randomize and "a free spot on the table" share one safe zone: away from the robot base, inside reach, clear of the tray and apart from the other cubes.
- **Physics simulation**: a Franka Panda arm in NVIDIA Isaac Sim, with real gravity, contact and friction, in a factory-floor scene.
- **Deterministic control**: a validated 10-phase grasp-and-place state machine (NVIDIA's `PickPlaceController`) on RMPFlow, with grasp confirmation by gripper width.
- **Pure perception mode**: a GUI switch that finds the cubes *by looking* through the overview camera (colour segmentation + depth) instead of reading simulator coordinates. The arm, the agent and the outcome check then use only camera positions. See `PERCEPTION.md`.
- **Outcome verification**: compares a live position (simulator telemetry, or a fresh camera look) with the target to determine actual task success, plus an independent Cosmos visual check that runs beside the arm.

> **Scene:** a work table with a yellow tray and three cubes, inside a visual-only factory backdrop (conveyor, yellow robot cell, racking, tool cabinet, ceiling). `PHYSYK_BACKGROUND=grid` restores the original blue-grid ground.

> **Running locally:** everything runs on one laptop (Isaac Sim in Docker, Cosmos as a local NIM, Nemotron through NVIDIA's hosted API). Setup and commands are in `instructions.md`.

---

## 1. Pick-and-Place — the physical layer

At the bottom of everything is one simple, reliable capability: **a Franka Panda robot arm that can pick up an object from wherever it actually is, and place it wherever it is told to.** This runs inside NVIDIA Isaac Sim, so every grasp, lift and placement is real simulated physics, not animation.

- The arm works out *what* to pick up and *where* to put it down from the instruction text.
- Motion is a validated 10-phase sequence (approach, descend, settle, grasp, lift, transport, place, release, retreat, return).
- After every task, the system checks the object's **real final position** against where it was supposed to land — success or failure is measured, not assumed.

## 2. Agentic Orchestration — planning, Nemotron, Cosmos and pure perception

*(Nemotron runs on NVIDIA's hosted API — `nvidia/nemotron-3.5-lightning-30b-a3b` — and Cosmos 3 Nano Reasoner runs locally as a NIM container. Details in `ARCHITECTURE.md` and `PERCEPTION.md`.)*

Above the physical layer sits a layer that turns a free-form, possibly multi-step instruction into a sequence of pick-and-place calls:

```
Instruction -> [local stack planner  |  Nemotron: decompose into steps] -> [guardrails check each step]
      -> clear any blocking cube -> dispatch to the physical layer -> [verify against real telemetry]
      -> re-plan from the live scene / retry on failure -> repeat until done or genuinely stuck
```

This is what lets *"put the green cube in the tray, then stack the blue cube on top of it"* become a safely sequenced, self-checking plan instead of a single scripted call.

**Why stacking is planned in code.** The right order depends on measured positions (what is reachable, what is covered, what stands where), which a deterministic planner reads exactly and instantly. Nemotron was tried for it first and was unreliable: the 30B model reasons at length, two of three stacking prompts ran out of tokens before writing an answer, and a plan took about 50 s when it worked. The language model handles what it is good at — free-form phrasing — and prompts that name colours or sequence steps still go to it.

**Pure perception (GUI switch).** With the switch on, no simulator coordinates reach the planner or the arm: the overview camera detects each cube (colour masks + depth, corrected against the known table plane), the tray is a fixed constant, and every result is checked by a fresh camera look. Cosmos supplies a second, independent visual check per step and never decides what the robot does.

---

## Why This Approach — vs. What Exists Today

| Today's typical approach | Physyk |
|---|---|
| Fixed, pre-programmed motion paths per task | Understands free-form natural-language instructions |
| Assumes objects are exactly where expected | Perceives live object positions before acting |
| "Motion completed" is treated as "task succeeded" | Real position verification after every action — failures are actually caught |
| A failure needs a human to intervene | Automatic re-plan / retry loop before escalating |
| Plans in the order the words come | Plans from the scene: nearest cube first, blocked cubes cleared, unreachable cubes reported |

## Customer Benefits & Industry Applications

- **Faster changeover**: reconfiguring what a robot does is an instruction, not a re-programming project, which matters most in low-volume / high-mix manufacturing, kitting and warehouse sorting.
- **Fewer silent failures**: real verification and automatic recovery reduce the "robot said it worked but it didn't" problem.
- **Natural-language operation**: floor operators can direct the system without robotics programming expertise.

See `INDUSTRY_APPLICATIONS.md`.

---

## Challenges We Faced

Real problems hit and (mostly) fixed along the way — kept here honestly rather than polished away. The project has two lines of history: the original build (22 August) and a later cube-and-tray rebuild that was dropped in October in favour of restarting from the original (`HISTORY.md`). Challenges from both are listed; the ones that apply to the code as it is now are marked.

- **One-fingered grasp bug** (applies): one of the gripper's two fingers shipped with zero drive strength, so only one finger ever closed. Fixed at the gain level.
- **Blind motion planner, then the opposite problem** (applies in part): the planner did not know about the table, tray or other cubes. Registering obstacles seemed to fix it, but the controller's reset silently clears them, and on the later branch activating them made results worse (12/15 against 14/15 picks). Clearance now rests on carry height and vertical descents.
- **Grasp timing** (applies): the gripper started to lift before it had finished closing, so it grabbed an edge and slipped. Fixed by giving the close and settle phases real time.
- **Camera framing**: cameras originally missed the table edges and cropped the robot out of frame. Fixed by repositioning and widening the lenses.
- **Fake success reporting** (applies): task verification used to be a hardcoded "always pass". Replaced with a real position check.
- **Perception read cubes about 7 mm too low** (applies): the first camera-only stacking attempt was rejected by the height guardrail. The suspected cause (camera calibration) was wrong: the colour mask missed each cube's near-white *top face*. Growing the mask onto the top face took the height error to 0.
- **Picks took about two minutes of wall time** (applies): the controller counts its phases per loop iteration, and the loop ran at only about 6 iterations/s because every camera was rendered twice per frame. Skipping the duplicate render, dropping cameras and encoding the streams every second frame brought the idle loop to about 60 iterations/s. Raising the motion speed instead made picks miss, because the controller is an open-loop timer.
- **"Stack" meant side by side**: one path gave every cube its own tray slot. Stacking is now planned from positions (tray, table or neither).
- **Nemotron could not plan stacking reliably** (applies): a thinking model ran out of its token budget on "stack the cubes", and took 20-50 s when it did answer. Stacking moved to a local planner that needs no model and is tested offline.
- **A closed-loop planner can loop forever** (applies): the offline agent test found that a step the guardrail rejected was simply re-generated by the re-plan, forever. Rejected cubes are now excluded from later plans and plan rewrites are capped.
- **Cosmos rated every tray task "low feasibility"** (applies): it did not recognise the flat yellow slab as a tray. The prompt now says what the objects look like.
- **Cubes sinking into the table** (applies): a cube can settle a couple of millimetres below its nominal height, which the guardrail's floor used to reject. The floor now allows for it.
- **A pick too close to the base** (applies): a cube 0.33 m from the base drove joint 6 into its limit and the grasp failed. The guardrail refuses closer targets; Randomize and free spots stay at 0.42 m or more.
- **Isaac Sim on a too-new host OS** (applies): Ubuntu 26.04's glibc is newer than Isaac Sim supports (pip installs segfault). Solved by running the official Isaac Sim 6.0.1 container.
- **Cosmos 3 does not fit next to the sim by default** (applies): the NIM's defaults crash on a 24 GB laptop GPU shared with Isaac Sim. Fixed with the FP8 build, an 8192-token context, a 65% GPU share and CUDA graphs off.
- **A ceiling broke the camera-only calibration** (applies): a closed room darkened the table, the detector's table fit shrank from about 24,000 to about 1,000 pixels, and cube heights read 11-18 mm off. The fit's brightness floor is now relative to the table's own brightness.
- **A learned-policy (GR00T) experiment** (removed): did not complete the task reliably on this scene; the code was removed.
- **Arm posture and long carries (open)**: the elbow posture is not controlled, and on the later controller a long sideways carry overshot its destination by about 21 cm on 4 of 4 runs. Not yet measured on this controller; see `TECHNICAL_DEEP_DIVE.md` §5.
