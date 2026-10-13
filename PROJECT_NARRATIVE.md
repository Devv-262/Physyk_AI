# Physyk — Agentic Pick-and-Place

> **Current scene (updated):** the red/green/blue cubes and the yellow tray have been **removed**. The table now holds
> five pieces of laboratory glassware — three Florence flasks (blue liquid, red liquid, empty) and two beakers (red
> liquid, green liquid) — plus a three-level shelf behind the robot. Pure perception finds them by **depth geometry,
> then names them by liquid colour** (99.5% of 200 live detections correct, 1.7 mm mean error, ~20 ms). **Picking
> glassware is not implemented yet** (it needs a sideways neck/body grasp planner): the agent can see and plan around
> them and refuses picks with a clear message. Sections below that describe cubes, the tray and stacking record how the
> earlier cube pipeline worked (100% of the numbers there were measured on that scene).

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
   tray") without hand-coded, per-task programming.
2. **Adapt to a changing scene** — object positions aren't fixed or hardcoded; the robot
   perceives where things actually are before acting.
3. **Verify real outcomes, not assumptions** — confirm a task actually succeeded using live
   sensor/position data, not just "the motion finished so it must have worked."
4. **Recover from failure automatically** — retry, re-plan, or escalate when a pick or
   placement doesn't go as expected, instead of silently reporting success.
5. **Explore a fully learned, vision-driven alternative** (a Vision-Language-Action model)
   alongside the reliable scripted path, as a research/demo track toward robots that learn
   physical skills from demonstration rather than being explicitly programmed for them.

## Core Capabilities

- **Instruction Parsing** — extracts the source object and destination target from
  natural-language commands.
- **Physics Simulation** — executes robot motion using a Franka Panda arm in NVIDIA Isaac
  Sim, with real gravity, contact, and friction.
- **Deterministic Control** — uses a validated 10-phase grasp-and-place state machine as the
  default execution path.
- **Motion Planning** — RMPFlow generates collision-aware joint targets for safe arm movement.
- **Pure Perception mode** — a GUI switch that finds the cubes *by looking* through the isometric
  overview camera (color segmentation + depth, ~20 ms) instead of reading simulator coordinates.
  The arm, the agent and the outcome check then use only camera positions. See `PERCEPTION.md`.
- **Outcome Verification** — compares a live position (simulator telemetry, or a fresh camera look in
  pure-perception mode) with the target to determine actual task success.
- **VLA Control (parked)** — GR00T-N1.7 integration exists in the repo but is not plugged in; see §2.

> **Scene:** the sim opens in a life-science-lab setting (visual only; `PHYSYK_BACKGROUND=grid` restores the old blue grid).

> **Running locally:** everything now runs on one laptop (Isaac Sim in Docker, Cosmos as a local NIM,
> Nemotron through NVIDIA's hosted API). Setup and commands are in `instructions.md`.

---

## 1. Pick-and-Place — the physical layer

At the bottom of everything is one simple, reliable capability: **a Franka Panda robot arm
that can pick up an object from wherever it actually is, and place it wherever it's told to.**
This runs inside NVIDIA Isaac Sim, a physics simulator, so every grasp, lift, and placement is
real simulated physics (gravity, contact, friction) — not animation.

- The arm figures out *what* to pick up and *where* to put it down from the instruction text.
- Motion is handled by a validated 10-step sequence (approach → descend → grasp → lift →
  transport → place → release → retreat) using collision-aware path planning, so the arm
  routes around other objects instead of clipping through them.
- After every task, the system checks the object's **real final position** against where it
  was supposed to land — success or failure is measured, not assumed.

This scripted path is the **reliable, default way** the robot executes tasks today.

## 2. GR00T & VLA — the experimental, learned alternative (currently parked)

> **Status: parked.** The GR00T service is no longer started by `run.sh`, the GUI's VLA button and status
> pill are gone, and `vla:` prompts are rejected by the backend. The code, recorder and fine-tuning scripts
> remain in the repo for later. The camera-only pipeline in §3 is the current alternative to the
> deterministic path.

Alongside the scripted path, Physyk has an **experimental** second way of controlling the arm:
**NVIDIA's GR00T-N1.7**, a Vision-Language-Action (VLA) model — a type of AI that looks at
camera images and directly predicts robot motion, the same way a language model predicts the
next word, except its "words" are joint movements.

- **How it works:** the model looks at a camera image of the scene plus the arm's current
  wrist camera view, and predicts a short sequence of moves (16 steps at a time). After each
  sequence, it looks again and predicts the next one — a closed loop of *observe → act →
  observe again*, not a single blind guess.
- **Why it's separate from the reliable path:** this GR00T model was originally trained on a
  different simulated benchmark (LIBERO), not on Physyk's own table/cubes/cameras — so out of
  the box, it doesn't reliably pick up Physyk's objects. It's wired in as an opt-in,
  experimental mode (triggered with a `vla:` prefix) specifically so it can be demonstrated
  and improved without risking the reliability of the main system.
- **What we're doing about it:** fine-tuning — showing the model example demonstrations
  recorded from Physyk's *own* scene, so it learns to recognize *these* cubes, *this* camera
  angle, and *this* table, instead of relying on what it learned from a different simulator.
  Early fine-tuning attempts (see Challenges below) show real, scene-relevant motion, but not
  yet reliable grasp success — more training examples are the clear next step.

---

## 3. Agentic Orchestration — Nemotron & Cosmos, and Pure Perception

*(Nemotron runs on NVIDIA's hosted API — `nvidia/nemotron-3-super-120b-a12b` — and Cosmos 3 Nano Reasoner
runs locally as a NIM container. Details in `ARCHITECTURE.md` and `PERCEPTION.md`.)*

Above the physical pick-and-place layer sits a reasoning layer that turns a free-form,
possibly multi-step instruction into a sequence of pick-and-place calls, using an LLM
(Nemotron) to plan and a perception layer (Cosmos) to help interpret the scene:

```
Instruction → [Nemotron: decompose into steps] → [guardrails check each step]
      → dispatch to the physical layer above → [verify against real telemetry]
      → retry / re-plan on failure → repeat until done or genuinely stuck
```

This is what allows an instruction like *"put the green cube in the tray, then stack the blue
cube on the red cube"* to become a safely-sequenced, self-checking plan instead of a single
scripted call.

**Pure perception (GUI switch).** With the switch on, no simulator coordinates reach the planner or the
arm: the overview camera detects each cube (color masks + depth, corrected against the known table
plane), the tray is a fixed constant, and every result is checked by a fresh camera look. It stacks —
verified end to end: red on the tray, blue on red, green on blue, 3/3 steps, ~3 mm mean position error,
layers 52 mm apart. Cosmos supplies a second, independent visual check per step and never decides what the
robot does. The Perception Log page (top of the GUI) shows the annotated camera frames with bounding boxes.
With the switch off the original deterministic pipeline runs unchanged.

---

## Why This Approach — vs. What Exists Today

| Today's typical approach | Physyk |
|---|---|
| Fixed, pre-programmed motion paths per task | Understands free-form natural-language instructions |
| Assumes objects are exactly where expected | Perceives live object positions before acting |
| "Motion completed" is treated as "task succeeded" | Real position verification after every action — failures are actually caught |
| A failure needs a human to intervene | Automatic retry / re-plan loop before escalating |
| Either fully scripted OR a black-box learned model, rarely both | Reliable scripted execution by default, with a learned (VLA) path available and improvable side-by-side, so learned-AI robotics can be evaluated without giving up reliability |

## Customer Benefits & Industry Applications

- **Faster changeover** — reconfiguring what a robot does is an instruction, not a
  re-programming project, which matters most in low-volume/high-mix manufacturing, kitting,
  and warehouse sorting.
- **Fewer silent failures** — real verification and automatic retry reduce the "robot said
  it worked but it didn't" problem that costs downstream time to discover manually.
- **Natural-language operation** — floor operators can direct the system without robotics
  programming expertise, lowering the skill bar for day-to-day operation.
- **A visible path to learned robotics** — the VLA track gives an organization a live,
  working example of where foundation-model robotics is headed, without betting the whole
  operation's reliability on a model that isn't production-ready yet.

---

## Challenges We Faced

Real problems hit and (mostly) fixed along the way — kept here honestly rather than
polished away:

- **One-fingered grasp bug** — one of the gripper's two fingers shipped with zero drive
  strength, meaning only one finger was ever actually closing. Fixed at the asset/gain level.
- **Blind motion planner** — the path planner didn't know about the table, tray, or other
  cubes as obstacles for a long stretch of development, so the arm could clip through them.
  Fixed by registering real collision obstacles.
- **Grasp timing** — the gripper was starting to move (lift) before it had actually finished
  closing on the object, causing it to grab an edge and later slip mid-motion. Fixed by
  giving the close/settle phases more real time before motion resumes.
- **Camera framing** — cameras originally missed the edges of the table and cropped the robot
  itself out of frame; separately, the wrist camera couldn't see cubes near the table's edges
  during approach. Both addressed by widening the relevant camera field-of-view.
- **Fake success reporting** — task verification used to be a hardcoded "always pass," so a
  failed pick could be silently reported as successful. Replaced with a real position check
  against live simulator telemetry.
- **VLA training data volume** — a first fine-tuning pass on the experimental GR00T VLA path
  used a small number of demonstration episodes; the model showed real, scene-relevant motion
  but not yet reliable grasp completion — it tends to repeat a similar motion each step rather
  than visually correcting itself. More demonstration episodes per skill are the identified
  fix, already scoped for a future pass.
- **Out-of-domain VLA checkpoint** — the GR00T model wasn't trained on Physyk's own cubes,
  table, or camera angles to begin with, so out-of-the-box performance on colored cubes is
  poor — expected, and the reason the fine-tuning work above exists at all. (VLA is parked; the
  perception pipeline replaced it as the camera-driven track.)
- **Perception read cubes ~7 mm too low** — the first camera-only stacking attempt was rejected by the
  height guardrail. The suspected cause (camera calibration) was wrong: a table-plane fit showed the camera
  was accurate to ~1 mm. The real cause was the detector's color mask missing each cube's near-white *top
  face*, so the "top" it measured was really the top edge of the side faces. Fixed by growing the mask onto
  the top face; height error went to 0 and 3D error from 8.4 mm to 3.0 mm.
- **Picks took ~2 minutes of wall time** — the grasp controller counts its phases per loop iteration, and the loop
  ran at only ~6 iterations/s because every camera was rendered twice per frame. Skipping the duplicate render and
  dropping two of five cameras took it to ~40/s: a pick is now ~25 s and a 3-cube stack ~90 s, with identical
  accuracy. (Raising the motion speed instead made picks miss — the controller is an open-loop timer — and on battery
  the laptop GPU is power-capped, so keep it plugged in.) Details in `PERCEPTION.md`.
- **Isaac Sim on a too-new host OS** — Ubuntu 26.04's glibc is newer than Isaac Sim supports (pip installs
  segfault). Solved by running the official Isaac Sim 6.0.1 container; the window shows through X11.
- **Cosmos 3 doesn't fit next to the sim by default** — the NIM's defaults (bf16, 262k-token context, CUDA
  graphs) crash on a 24 GB laptop GPU shared with Isaac Sim. Fixed with the FP8 build, an 8192-token
  context, a 65% GPU share and CUDA graphs off (`start_cosmos_docker.sh`).
