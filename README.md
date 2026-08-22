# Physyk — Agentic Pick-and-Place

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
- **VLA Control** — GR00T-N1.7 predicts end-effector motion from scene images, wrist-camera
  frames, and robot pose.
- **Closed-Loop Execution** — executes 16-step action chunks and re-observes the environment
  after each chunk.
- **Outcome Verification** — compares live simulator telemetry with the target position to
  determine actual task success.

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

## 2. GR00T & VLA — the experimental, learned alternative

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

## 3. Agentic Orchestration — Nemotron & Cosmos

*(Architecture and implementation owned separately — summarized here for context only.)*

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
scripted call. *(Full architecture diagram to be provided separately by the orchestration
owner.)*

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
  poor — expected, and the reason the fine-tuning work above exists at all.
