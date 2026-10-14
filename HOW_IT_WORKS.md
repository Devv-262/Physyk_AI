# Physyk in Plain English

*A robot arm you can talk to — and that checks its own work.*

This page explains the project without code. For the technical details, see `README.md`, `ARCHITECTURE.md`,
`TECHNICAL_DEEP_DIVE.md` and `PERCEPTION.md`; to run it, see `instructions.md`.

---

## 1. The idea in one minute

Most factory robots do one thing, one way, forever. Move a part one centimetre and they fail; ask for a new
task and an engineer has to re-program them.

Physyk is a **simulated robot arm that takes plain-English instructions**, figures out the steps, finds the
objects by looking at them, moves, and then **looks again to check it really worked**.

> Say: *"Put the red cube in the tray, then stack the green cube on the blue cube."*
> The arm does it, and tells you whether it actually succeeded.

Everything runs in a physics simulator — a virtual robot, a lab bench, a yellow tray and three coloured cubes — on
one laptop.

---

## 2. The cast: who does what

Think of a small team. Each member has one job.

| Role | Who | Job |
|---|---|---|
| **The planner** | *Nemotron*, a large AI language model (NVIDIA, 120 billion parameters, used over the internet) | Reads your sentence and breaks it into ordered steps. |
| **The eyes** | A camera plus a simple colour-finding program | Finds each cube in the picture, works out where it is in 3D and how it is turned. |
| **The checker** | *Cosmos 3 Nano Reasoner*, an AI model that understands images (NVIDIA, runs on the laptop) | Looks at the scene and says "yes, the blue cube is on the red one." It's a second opinion and never steers the robot. |
| **The body** | *Isaac Sim* (NVIDIA's robot simulator) with a Franka Panda arm | Moves the arm using a proven motion planner, in a realistic physics world. |
| **The guard** | A set of simple rules | Refuses any step that would reach outside the arm's safe working area. |

A useful way to remember it: **Nemotron decides, the camera sees, Isaac Sim moves, Cosmos double-checks.**

---

## 3. What happens when you type a sentence

```
 you type a sentence
        |
        v
 [1] Planner            -> "step 1: red cube to tray, step 2: ..."
        |
        v   (for each step)
 [2] Guard              -> is this step inside the safe area? if not, stop
        |
 [3] Clear the way      -> is another cube sitting on the one we need? move it off first
        |
 [4] Eyes               -> where exactly is the cube right now, and how is it turned?
        |
 [5] Body               -> turn the hand to the cube, grab, lift, carry, place
        |
 [6] Check              -> look again: is the cube where it should be?
        |
        +--> yes: next step
        +--> no : planner decides: try again, change the step, or re-plan the rest
```

In more detail:

1. **The planner** turns language into a short list of steps. If the sentence mentions something that isn't
   on the table, it returns nothing and the request is rejected instead of guessed at. A request like
   "stack the cubes" is planned by ordinary code from the cubes' positions, not by the language model (see §5).
2. **The guard** is plain rules, not AI. Every target must be inside the area the arm can safely reach.
3. **Clearing the way.** A cube with another cube on top of it cannot be picked. The agent notices this from
   the positions and first moves the top cube to a free spot on the table, with room for the hand around it.
4. **The eyes** find the cube (next section).
5. **The body** uses a ready-made grasping routine with ten phases: move above the cube, go down, settle,
   close the gripper, lift, carry, go down at the destination, let go, go back up, return home. A motion
   planner steers the arm the whole way. On top of that routine Physyk adds a few rules of its own (§6).
6. **The check** makes sure the gripper really holds something (finger width), then looks at where the cube
   ended up. The result must land within about 3 cm of the target; in practice it lands within about 5 mm.

---

## 4. Two ways to find the cubes

There is a switch in the app: **"Pure perception — camera only"**.

**Switch off — the "cheat sheet" mode.** The simulator simply tells the robot where every cube is. Very
reliable, but it's not how a real robot would work, because a real robot has no cheat sheet.

**Switch on — the "real eyes" mode.** The robot is told nothing. It finds the cubes the way you would:

1. A camera looks at the table and also records how far away each pixel is (a "depth image").
2. A colour filter picks out the red, green and blue cube pixels. A rectangle is drawn around each — those
   are the **bounding boxes** you see in the video. The yellow tray gets a box too, labelled "fixed",
   because it never moves.
3. Using the depth, each pixel becomes a point in 3D space. The cube's centre is the best-fitting 5 cm square
   through those points, and the square's angle tells how the cube is turned.
4. That position and angle go to the robot as the target.

If you shuffle the cubes, the robot never sees the new positions — it has to find them again by looking.

**How accurate?** About **3 millimetres** on average, found in about **20 milliseconds**.

---

## 5. Stacking

"Stack" always means a vertical pile — cubes on top of each other, never side by side. The agent decides the
**order from where the cubes are right now, not from their colours**:

- A cube with nothing on top of it and close to the robot is easy to pick, so it goes first.
- With no surface named ("stack the cubes"), the easiest cube is the **base** and does not have to move; the
  others are lifted onto it, easiest first.
- "…in the tray": the easiest cube is set in the tray as the base, and the rest are stacked on it.
- "…on the table": the pile stays on the table. If every cube is already stacked in the tray, a new pile is
  started on a free table spot, taking the old pile apart from the top.
- If the cubes are already one pile, the agent says so and does nothing.

To put one cube on another, the robot looks at the cube below, reads its height, and aims for one cube-height
(5 cm) above it. The height always comes from what the camera measured, never from a lookup.

In test runs the robot built three-cube towers in the tray and on the table, in about 20 seconds per cube,
with the layers 50-52 mm apart (ideal: 50 mm) and typically 5 mm off-centre.

---

## 6. How the arm handles the cubes

The grasping routine is a timer-driven script — it does not watch the arm. Physyk adds closed-loop rules where
that matters:

- **Turn to the cube.** If a cube is turned 30 degrees, the hand turns 30 degrees too, so the fingers close on a
  pair of flat faces instead of corners. If a neighbouring cube is very close, the hand picks the turn whose open
  fingers sweep clear of it (for example 90 degrees from the default). On the way to the destination the hand
  turns again so a stacked cube lines up with the one below.
- **Wait until it arrives.** The script's "move" phases are short; the arm needs longer. The agent waits until the
  hand is within 1.5 cm of the target, so the arm carries the cube across at a safe height and then goes straight
  down, instead of swooping diagonally.
- **Lift high enough.** Before carrying, the cube must clear the top of the tallest other cube.
- **Let go on contact.** When the cube reaches the surface, or stops going down, the gripper opens at once rather
  than pushing on a cube it already put down.
- **Fingers wide open.** Whenever the hand is empty — on the approach and between tasks — the gripper is opened
  fully, so it never goes down onto a cube half-closed.

---

## 7. How it checks and fixes itself

- **Did it grab something?** The gripper's own finger width tells it. If it closed on nothing, it stops
  straight away instead of carrying thin air across the table.
- **Did the cube land right?** After placing, the camera looks again. No sighting counts as "not confirmed".
- **Is the picture borderline?** A measurement can sit right on a safety limit, so the system looks again
  (up to three times) before giving up. A cube may also sink a millimetre or two into the table; the guard allows for that.
- **Did something go wrong?** The planner is shown what happened and chooses one of three options: **retry**
  the same step, **change** just that step, or **re-plan** everything that's left. If the arm's own planner refuses a step, that is
  reported with its reason and not retried.
- **A second opinion.** Cosmos looks at the scene before and after each step. It's told what the objects look like (the
  flat yellow slab is the tray); without that it kept answering "the tray isn't visible". If Cosmos is slow or offline,
  the robot carries on without it.

---

## 8. Where everything runs

| Part | Where |
|---|---|
| The robot simulator (Isaac Sim) | On the laptop, inside a container, with no window by default |
| Cosmos (the image-understanding checker) | On the laptop, inside a container |
| The app and web page you click on | On the laptop |
| Nemotron (the planner) | NVIDIA's servers, over the internet |

The laptop has an NVIDIA RTX 5090 GPU with 24 GB of memory. Isaac Sim uses about 3 GB of it and Cosmos about
16 GB. **It needs to be plugged in:** on battery the graphics chip slows down a lot.

**Speed:** a single pick-and-place takes about **17-20 seconds** (camera-only mode **22-30**); a three-cube tower
about a minute.

---

## 9. What you see on screen

- **Live camera views:** the overview and the wrist camera. The overview shows bounding boxes in camera-only mode.
- **Agent Plan panel:** the steps, with their status (running, done, rejected, retrying). Steps the agent adds
  itself (like moving a blocking cube) are marked "inserted".
- **Logs along the top:**
  - *Nemotron log* — the planner's reasoning.
  - *Cosmos log* — each visual check, with the exact image it looked at.
  - *Perception log* — the camera's detections, with bounding boxes and estimated positions.
- **Service lights:** whether the simulator, planner and checker are online. The simulator light is on whenever its
  container is running.

---

## 10. Folder map

```
Physyk_AI/
├── README.md, HOW_IT_WORKS.md, ...   the documents (start here)
├── run.sh, start_*_docker.sh         start the app, the simulator and Cosmos
├── workspace/                        the application itself
│   └── physyk_perception/            the camera-vision code
├── assets/                           the robot's 3D model files
└── logs/                             run logs
```

---

## 11. Honest limits

- It's a **simulation**. A real robot would need a real camera, real calibration and noise handling.
- It works for **three coloured cubes on a known table and tray**, under the default lighting. It is not a
  general object finder.
- The tray and table are assumed fixed.
- Some internal parts still rely on the simulator, for example the planner's obstacle map, the gripper sensor
  and the touch detection used to release a cube.
- Cubes are only reachable within about 0.66 m of the robot; the hand points straight down and cannot grasp further out.
- Picks are verified on single runs and short sequences, not over a large randomised batch. Two cubes touching each
  other has not been tested.

---

## 12. Glossary

| Term | Meaning |
|---|---|
| **Simulator** | A video-game-like physics world that behaves like the real one (gravity, friction, contact). |
| **Isaac Sim** | NVIDIA's robot simulator. |
| **Franka Panda** | The robot arm model used here. |
| **LLM** | A large language model — an AI that reads and writes text. Nemotron is one. |
| **Depth image** | A picture where each pixel stores its distance from the camera. |
| **Bounding box** | A rectangle drawn around a detected object. |
| **Motion planner** | Software that works out how to move the arm without hitting anything. |
| **Guardrail** | A simple rule that blocks unsafe actions. |
| **Container** | A self-contained package that runs a program with everything it needs. |
| **Yaw** | How far something is turned about the vertical axis. |
