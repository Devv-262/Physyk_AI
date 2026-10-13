# Physyk in Plain English

*A robot arm you can talk to — and that checks its own work.*

> **Current scene (updated):** the red/green/blue cubes and the yellow tray have been **removed**. The table now holds
> five pieces of laboratory glassware — three Florence flasks (blue liquid, red liquid, empty) and two beakers (red
> liquid, green liquid) — plus a three-level shelf behind the robot. Pure perception finds them by **depth geometry,
> then names them by liquid colour** (99.5% of 200 live detections correct, 1.7 mm mean error, ~20 ms). **Picking
> glassware is not implemented yet** (it needs a sideways neck/body grasp planner): the agent can see and plan around
> them and refuses picks with a clear message. Sections below that describe cubes, the tray and stacking record how the
> earlier cube pipeline worked (100% of the numbers there were measured on that scene).


This page explains the project without code. For the technical details, see `README.md`,
`ARCHITECTURE.md`, `PERCEPTION.md` and `TECHNICAL_DEEP_DIVE.md`; to run it, see `instructions.md`.

---

## 1. The idea in one minute

Most factory robots do one thing, one way, forever. Move a part one centimetre and they fail; ask for a new
task and an engineer has to re-program them.

Physyk is a **simulated robot arm that takes plain-English instructions**, figures out the steps, finds the
objects by looking at them, moves, and then **looks again to check it really worked**.

> Say: *"Put the red cube in the tray, then stack the green cube on the blue cube."*
> The arm does it, and tells you whether it actually succeeded.

Everything runs in a physics simulator (a virtual robot, table, tray and three coloured cubes), on one laptop.

---

## 2. The cast: who does what

Think of a small team. Each member has one job.

| Role | Who | Job |
|---|---|---|
| **The planner** | *Nemotron*, a large AI language model (NVIDIA, 120 billion parameters, used over the internet) | Reads your sentence and breaks it into ordered steps. |
| **The eyes** | A camera plus a simple colour-finding program | Finds each cube in the picture and works out where it is in 3D. |
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
 [1] Planner (Nemotron)  -> "step 1: red cube to tray, step 2: ..."
        |
        v
 [2] Guard               -> is this step inside the safe area? if not, stop
        |
        v   (for each step)
 [3] Eyes                -> where exactly is the cube right now?
        |
        v
 [4] Body                -> plan a safe path, grab, lift, carry, place
        |
        v
 [5] Check               -> look again: is the cube where it should be?
        |
        +--> yes: next step
        +--> no : planner decides: try again, change the step, or re-plan the rest
```

In more detail:

1. **The planner** turns language into a short list of steps. If the sentence mentions something that isn't
   on the table, it returns nothing and the request is rejected instead of guessed at.
2. **The guard** is plain rules, not AI. Every target must be inside the area the arm can safely reach.
3. **The eyes** find the cube (next section).
4. **The body** uses a ready-made grasping routine with ten phases: move above the cube, go down, settle,
   close the gripper, lift, carry, go down at the destination, let go, go back up, return home. A motion
   planner steers around the table, tray and other cubes the whole way.
5. **The check** makes sure the gripper really holds something (finger width), then looks at where the cube
   ended up. The result must land within about 3 cm of the target.

---

## 4. Two ways to find the cubes

There is a switch in the app: **"Pure perception — camera only"**.

**Switch off — the "cheat sheet" mode.** The simulator simply tells the robot where every cube is. Very
reliable, but it's not how a real robot would work, because a real robot has no cheat sheet.

**Switch on — the "real eyes" mode.** The robot is told nothing. It finds the cubes the way you would:

1. A camera looks at the table and also records how far away each pixel is (a "depth image").
2. A colour filter picks out the red, green and blue cube pixels. A rectangle is drawn around each — those
   are the **bounding boxes** you see in the video.
3. Using the depth, each pixel becomes a point in 3D space. The cube's centre is the best-fitting 5 cm square
   through those points.
4. That position goes to the robot as the target.

If you shuffle the cubes, the robot never sees the new positions — it has to find them again by looking.
The table and tray never move, so those are fixed numbers.

**How accurate?** About **3 millimetres** on average, found in about **20 milliseconds**.

---

## 5. Stacking

Stacking uses the same steps. To put the green cube on the blue cube, the robot looks at the blue cube, reads
its height, and aims for one cube-height (5 cm) above that. The height always comes from what the camera
measured, never from a lookup.

In the test run, the robot built a three-cube tower on the tray (red, then blue, then green) in **87 seconds**,
with the layers 52 mm apart (ideal: 50 mm) and less than 7 mm off-centre.

---

## 6. How it checks and fixes itself

- **Did it grab something?** The gripper's own finger width tells it. If it closed on nothing, it stops
  straight away instead of carrying thin air across the table.
- **Did the cube land right?** After placing, the camera looks again. No sighting counts as "not confirmed".
- **Is the picture borderline?** A measurement can sit right on a safety limit, so the system looks again
  (up to three times) before giving up.
- **Did something go wrong?** The planner is shown what happened and chooses one of three options: **retry**
  the same step, **change** just that step, or **re-plan** everything that's left.
- **A second opinion.** Cosmos looks at the scene before and after each step. It agreed ("verified", 99%
  confidence) in the test runs. If Cosmos is slow or offline, the robot carries on without it.

---

## 7. Where everything runs

| Part | Where |
|---|---|
| The robot simulator (Isaac Sim) | On the laptop, inside a container |
| Cosmos (the image-understanding checker) | On the laptop, inside a container |
| The app and web page you click on | On the laptop |
| Nemotron (the planner) | NVIDIA's servers, over the internet |

The laptop has an NVIDIA RTX 5090 GPU with 24 GB of memory. Isaac Sim uses about 3 GB of it and Cosmos about
16 GB. **It needs to be plugged in:** on battery the graphics chip slows down a lot.

**Speed:** a single pick-and-place takes about **25 seconds**; the three-cube tower about **90 seconds**.

---

## 8. What you see on screen

- **Live camera views:** an overview, a front view and a wrist camera.
- **Agent Plan panel:** the steps, with their status (running, done, rejected, retrying).
- **Logs along the top:**
  - *Nemotron log* — the planner's reasoning.
  - *Cosmos log* — each visual check, with the exact image it looked at.
  - *Perception log* — the camera's detections, with bounding boxes and estimated positions.
- **Service lights:** whether the simulator, planner and checker are online.

---

## 9. Folder map

```
Physyk_AI/
├── README.md, HOW_IT_WORKS.md, ...   the documents (start here)
├── run / start scripts               start the app, the simulator and Cosmos
├── workspace/                        the application itself
│   └── perception/                   the camera-vision code
├── assets/                           the robot's 3D model files
└── logs/                             run logs
```

---

## 10. Honest limits

- It's a **simulation**. A real robot would need a real camera, real calibration and noise handling.
- It works for **three coloured cubes on a known table and tray**, under the default lighting. It is not a
  general object finder.
- The tray and table are assumed fixed.
- Some internal parts still rely on the simulator, for example the planner's obstacle map and the gripper
  sensor.
- We have shown individual picks and a full tower working, but haven't measured a success rate over many
  randomised runs.

**Parked for now:** a more experimental approach (a model called GR00T that predicts robot moves straight from
camera images) was tested and set aside — it wasn't reliable on this scene without more training data.

---

## 11. Glossary

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
| **VLA** | "Vision-Language-Action" model: an AI that goes straight from pictures and words to robot moves. |
