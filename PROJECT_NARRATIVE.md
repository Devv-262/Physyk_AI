# Physyk — Agentic Pick-and-Place

> **New to the project?** Start with `HOW_IT_WORKS.md` — a plain-English walkthrough with no code.

## Problem Statement

Most robots in a factory today can only do one thing, one way, forever. You program a robot
arm to pick up a part from exactly position X and put it down at exactly position Y, and
that's all it will ever do — the moment the part moves, the layout changes, or someone gives
it a new instruction in plain English, it breaks. Re-programming a robot for a new task
usually means an engineer sitting down and hand-coding new motion paths, which can take days.

**Physyk asks a simple question:** what if you could just *tell* the robot what to do — in
plain English, with the layout allowed to change — and have it figure out the rest, verify
its own work, and recover when something goes wrong?

## Objectives

1. **Understand natural-language instructions** ("pick up the red cube and place it in the
   tray", "stack the cubes") without hand-coded, per-task programming.
2. **Adapt to a changing scene** — object positions and rotations aren't fixed or hardcoded; the robot
   perceives where things actually are before acting.
3. **Verify real outcomes, not assumptions** — confirm a task actually succeeded using live
   sensor/position data, not just "the motion finished so it must have worked."
4. **Recover from failure automatically** — retry, re-plan, or escalate when a pick or
   placement doesn't go as expected, instead of silently reporting success.

## Core Capabilities

- **Instruction parsing** — extracts the source object and destination from natural-language commands.
- **Stack-aware planning** — "stack the cubes" is a vertical pile whose order comes from the cubes' positions (easiest
  to pick first, never by colour), on the tray or on the table; a cube with another cube on top is cleared first.
- **Physics simulation** — a Franka Panda arm in NVIDIA Isaac Sim, with real gravity, contact and friction, on a lab scene.
- **Deterministic control** — a validated 10-phase grasp-and-place state machine, with closed-loop additions: it waits for
  the hand to arrive, lifts clear of other cubes, releases on contact, and turns the hand to the cube's faces.
- **Motion planning** — RMPFlow generates the joint targets; avoidance of the table, tray and cubes is off by default because enabling it made results worse.
- **Pure perception mode** — a GUI switch that finds the cubes *by looking* through the overview camera (colour
  segmentation + depth, ~20 ms) instead of reading simulator coordinates. The arm, the agent and the outcome check then
  use only camera positions. See `PERCEPTION.md`.
- **Outcome verification** — compares a live position (simulator telemetry, or a fresh camera look in pure-perception
  mode) with the target to determine actual task success.

> **Scene:** a life-science lab (floor, walls, bench, fume hood — visual only), a work table with a yellow tray and three
> cubes, and the arm. `PHYSYK_BACKGROUND=grid` restores the old blue grid.

> **Running locally:** everything runs on one laptop (Isaac Sim in Docker, Cosmos as a local NIM, Nemotron through
> NVIDIA's hosted API). Setup and commands are in `instructions.md`.

---

## 1. Pick-and-Place — the physical layer

At the bottom of everything is one simple, reliable capability: **a Franka Panda robot arm
that can pick up an object from wherever it actually is, and place it wherever it's told to.**
This runs inside NVIDIA Isaac Sim, so every grasp, lift and placement is real simulated physics, not animation.

- The arm figures out *what* to pick up and *where* to put it down from the instruction text.
- Motion is a validated 10-phase sequence (approach → descend → grasp → lift → transport → place →
  release → retreat) with collision-aware path planning around the table and tray.
- After every task, the system checks the object's **real final position** against where it was
  supposed to land — success or failure is measured, not assumed.

## 2. Agentic Orchestration — Nemotron & Cosmos, and Pure Perception

*(Nemotron runs on NVIDIA's hosted API — `nvidia/nemotron-3-super-120b-a12b` — and Cosmos 3 Nano Reasoner
runs locally as a NIM container. Details in `ARCHITECTURE.md` and `PERCEPTION.md`.)*

Above the physical layer sits a reasoning layer that turns a free-form, possibly multi-step instruction into a
sequence of pick-and-place calls:

```
Instruction → [Nemotron: decompose into steps  (or the stack planner)] → [guardrails check each step]
      → clear any blocking cube → dispatch to the physical layer → [verify against real telemetry]
      → retry / re-plan on failure → repeat until done or genuinely stuck
```

This is what lets *"put the green cube in the tray, then stack the blue cube on top of it"* become a
safely-sequenced, self-checking plan instead of a single scripted call.

**Pure perception (GUI switch).** With the switch on, no simulator coordinates reach the planner or the arm: the
overview camera detects each cube (colour masks + depth, corrected against the known table plane), the tray is a fixed
constant, and every result is checked by a fresh camera look. Cosmos supplies a second, independent visual check per
step and never decides what the robot does. The Perception Log page shows the annotated camera frames with bounding
boxes. With the switch off the deterministic pipeline runs.

---

## Why This Approach — vs. What Exists Today

| Today's typical approach | Physyk |
|---|---|
| Fixed, pre-programmed motion paths per task | Understands free-form natural-language instructions |
| Assumes objects are exactly where expected | Perceives live object positions and rotations before acting |
| "Motion completed" is treated as "task succeeded" | Real position verification after every action — failures are actually caught |
| A failure needs a human to intervene | Automatic retry / re-plan loop before escalating |
| Plans in the order the words come | Plans from the scene: easiest cube first, blocked cubes cleared |

## Customer Benefits & Industry Applications

- **Faster changeover** — reconfiguring what a robot does is an instruction, not a re-programming project, which matters
  most in low-volume/high-mix manufacturing, kitting and warehouse sorting.
- **Fewer silent failures** — real verification and automatic retry reduce the "robot said it worked but it didn't" problem.
- **Natural-language operation** — floor operators can direct the system without robotics programming expertise.

See `INDUSTRY_APPLICATIONS.md`.

---

## Challenges We Faced

Real problems hit and (mostly) fixed along the way — kept here honestly rather than polished away:

- **One-fingered grasp bug** — one of the gripper's two fingers shipped with zero drive strength, so only one finger ever
  closed. Fixed at the gain level.
- **Blind motion planner, then the opposite problem** — the planner didn't know about the table, tray or other cubes, so the arm could
  clip through them. Registering obstacles seemed to fix it, but the controller's reset silently cleared them, so they were never active;
  when they were made active, results got worse (12/15 vs 14/15 picks on identical layouts). The arm now runs with no avoidance and relies
  on carry height and vertical descents.
- **Grasp timing** — the gripper started to lift before it had finished closing, so it grabbed an edge and slipped. Fixed
  by giving the close/settle phases real time.
- **Camera framing** — cameras originally missed the table edges and cropped the robot out of frame. Fixed by repositioning
  and widening the lenses.
- **Fake success reporting** — task verification used to be a hardcoded "always pass". Replaced with a real position check.
- **Perception read cubes ~7 mm too low** — the first camera-only stacking attempt was rejected by the height guardrail.
  The suspected cause (camera calibration) was wrong: the detector's colour mask missed each cube's near-white *top face*.
  Growing the mask onto the top face took height error to 0 and 3D error from 8.4 mm to 3.0 mm.
- **Picks took ~2 minutes of wall time** — the grasp controller counts its phases per loop iteration, and the loop ran at
  only ~6 iterations/s because every camera was rendered twice per frame. Skipping the duplicate render and dropping
  cameras took it to ~40/s: a pick is now 17-20 s. Raising the motion speed instead made picks miss — the controller is an
  open-loop timer.
- **The arm swooped and arrived late** — the transport phase lasts ~16 loop iterations, so the arm did the whole move
  during the descent, diagonally, and was still arriving when the gripper released. A per-joint trace showed it; the agent
  now waits for the hand to arrive before each phase ends.
- **The hand never rotated** — every cube was gripped at yaw 0, so a turned cube was pinched at its corners, and cubes
  were always spawned straight. The hand now turns to the cube's faces (and away from neighbours), and "Randomize" turns
  the cubes. The first version read the yaw wrongly (the cubes are scaled prims) and still grasped corners; the trace's
  finger width (55-74 mm instead of 53 mm) exposed it.
- **A release that dropped cubes** — the early-release-on-contact rule fired at the start of the lowering phase for the 3rd
  level of a stack, because the carried cube was already below that height. It now only fires for a genuine descent.
- **"Stack" meant side by side** — one path gave every cube its own tray slot. Stacking is now planned from the positions
  (tray, table or neither), and Nemotron's prompt states the rule.
- **Cosmos rated every tray task "low feasibility"** — it did not recognise the flat yellow slab as a tray. The prompt now
  says what the objects look like.
- **Cubes sinking into the table** — a cube can settle a couple of millimetres below its nominal height, which the
  guardrail's floor used to reject. The floor now allows for it.
- **Isaac Sim on a too-new host OS** — Ubuntu 26.04's glibc is newer than Isaac Sim supports (pip installs segfault).
  Solved by running the official Isaac Sim 6.0.1 container.
- **Cosmos 3 doesn't fit next to the sim by default** — the NIM's defaults crash on a 24 GB laptop GPU shared with Isaac
  Sim. Fixed with the FP8 build, an 8192-token context, a 65% GPU share and CUDA graphs off.
- **A pick too close to the base** — a cube spawned 0.33 m from the base drove joint 6 into its limit and the grasp failed. Cubes now
  stay 0.38-0.66 m from the base and the agent refuses closer targets.
- **Arm posture and rare swings (open)** — the elbow posture is not controlled and one joint swing was seen that only reproduces under
  specific conditions; see `TECHNICAL_DEEP_DIVE.md` §5.
