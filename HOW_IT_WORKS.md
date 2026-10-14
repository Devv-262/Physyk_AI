# Physyk in Plain English

*A robot arm you can talk to — and that checks its own work.*

This page explains the project without code. For the technical details, see `README.md`, `ARCHITECTURE.md`, `TECHNICAL_DEEP_DIVE.md` and `PERCEPTION.md`; to run it, see `instructions.md`.

---

## 1. The idea in one minute

Most factory robots do one thing, one way, forever. Move a part one centimetre and they fail; ask for a new task and an engineer has to re-program them.

Physyk is a **simulated robot arm that takes plain-English instructions**, figures out the steps, finds the objects by looking at them, moves, and then **looks again to check it really worked**.

> Say: *"Put the red cube in the tray, then stack the green cube on the blue cube."*
> The arm does it, and tells you whether it actually succeeded.

Everything runs in a physics simulator — a virtual robot, a work table with a yellow tray and three coloured cubes, set in a factory floor — on one laptop.

---

## 2. The cast: who does what

| Role | Who | Job |
|---|---|---|
| **The planner for stacking** | A small program written for this project | Decides, from where the cubes are, which cube is the base, which moves next, and where a blocking cube goes. No AI model involved. |
| **The planner for everything else** | *Nemotron*, a large AI language model (NVIDIA, used over the internet) | Reads your sentence and breaks it into ordered steps. |
| **The eyes** | A camera plus a simple colour-finding program | Finds each cube in the picture and works out where it is in 3D. |
| **The checker** | *Cosmos 3 Nano Reasoner*, an AI model that understands images (runs on the laptop) | Looks at the scene and says "yes, the blue cube is on the red one." A second opinion that never steers the robot. |
| **The body** | *Isaac Sim* (NVIDIA's robot simulator) with a Franka Panda arm | Moves the arm using a proven motion planner, in a realistic physics world. |
| **The guard** | A set of simple rules | Refuses any step that would reach outside the arm's safe working area. |

A useful way to remember it: **the stacking planner and Nemotron decide, the camera sees, Isaac Sim moves, Cosmos double-checks.**

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
 [4] Body               -> grab, lift, carry, place
        |
 [5] Check              -> look again: is the cube where it should be?
        |
        +--> yes: next step
        +--> no : try again, or plan again from what is really on the table
```

1. **The planner** turns language into a short list of steps. "Stack the cubes" is planned by ordinary code from the cubes' positions (section 5); anything else goes to Nemotron. If the sentence mentions something that is not on the table, the request is rejected instead of guessed at.
2. **The guard** is plain rules, not AI. Every target must be inside the area the arm can safely reach, and not too close to the robot's base.
3. **Clearing the way.** A cube with another cube on top of it cannot be picked. The agent notices this from the positions and first moves the top cube to a free spot on the table.
4. **The body** uses a ready-made grasping routine with ten phases: move above the cube, go down, settle, close the gripper, lift, carry, go down at the destination, let go, go back up, return. A motion planner steers the arm the whole way.
5. **The check** makes sure the gripper really holds something (finger width), then looks at where the cube ended up. The result must land within about 3 cm of the target.

---

## 4. Two ways to find the cubes

There is a switch in the app: **"Pure perception — camera only"**.

**Switch off — the "cheat sheet" mode.** The simulator simply tells the robot where every cube is. Very reliable, but it is not how a real robot would work, because a real robot has no cheat sheet.

**Switch on — the "real eyes" mode.** The robot is told nothing. It finds the cubes the way you would:

1. A camera looks at the table and also records how far away each pixel is (a "depth image").
2. A colour filter picks out the red, green and blue cube pixels. A rectangle is drawn around each — those are the **bounding boxes** you see in the video. The yellow tray gets a box too, labelled "fixed", because it never moves.
3. Using the depth, each pixel becomes a point in 3D space. The cube's centre is the best-fitting 5 cm square through those points.
4. That position goes to the robot as the target.

If you shuffle the cubes, the robot never sees the new positions — it has to find them again by looking. It finds them to within a few millimetres in a few tens of milliseconds.

---

## 5. Stacking

"Stack" always means a vertical pile — cubes on top of each other, never side by side. The planner decides **from where the cubes are right now, not from their colours**:

- It finds the **piles** that already exist. A cube with something on it is "covered" and can never be picked, so a pile is always taken apart from the top.
- With no surface named ("stack the cubes"), the biggest pile that already stands somewhere reachable is the one to build on; its bottom cube does not have to move.
- "…in the tray": the pile has to stand in the tray. If one already does, the others go on it. If none does, one cube is carried into the tray first, and the rest go on top.
- "…on the table": the pile has to stand on the table, never in the tray. If every cube is already stacked in the tray, a new pile is started on a free table spot, taking the old pile apart from the top.
- If the cubes are already one pile on the right surface, the agent says so and does nothing.
- Cubes standing close together are picked in an order that keeps the open gripper away from their neighbours where it can.
- After **every** step the planner looks at the table again and plans the rest from what it sees. If a cube is dropped, knocked over or moved by hand, the next step simply adapts.
- If a cube is out of the arm's reach (too close to the base, or off the working area), the agent says which one and does nothing, instead of trying something impossible.

When the robot needs to put a cube "out of the way", the free spot is chosen carefully: inside the arm's comfortable reach, away from the robot base and from the tray, with room for the open hand around it, and the shortest trip from where the cube is now. "Randomize" uses the same safe area, so a random layout never puts a cube where the arm cannot reach it.

---

## 6. How the arm handles the cubes

The grasping routine is a timer-driven script — it does not watch the arm. Each phase gets a fixed amount of time, which is why the arm's speed is kept at a setting that is known to work (faster settings make the script outrun the arm). Things to know about what it does and does not do:

- The hand always points straight down and is **not turned** to the cube's faces, so a rotated cube is gripped at an angle.
- Between phases it does **not** wait for the arm to arrive. On long sideways carries the arm can overshoot the destination and come back (seen on an earlier version of the same controller; not yet measured on this one).
- The gripper opens fully on the approach, and a grasp that closes on nothing is detected and aborted.

These are the main things a future version should improve; they are listed in `TECHNICAL_DEEP_DIVE.md` §5.

---

## 7. How it checks and fixes itself

- **Did it grab something?** The gripper's own finger width tells it. If it closed on nothing, it stops straight away instead of carrying thin air across the table.
- **Did the cube land right?** After placing, the camera (or the simulator) checks where the cube ended up. No sighting counts as "not confirmed".
- **Is the picture borderline?** A measurement can sit right on a safety limit, so the system looks again (up to three times) before giving up.
- **Did something go wrong?** For a stacking request the planner plans again from what is really on the table. For other requests Nemotron is shown what happened and chooses: **retry** the same step, **change** just that step, or **re-plan** everything that is left. If the arm's own planner refuses a step, that is reported with its reason and not retried.
- **Is the whole goal still true?** At the end, code checks the real positions again. For a stacking request that means: are all the cubes in one pile on the right surface?
- **A second opinion.** Cosmos looks at the scene before and after each step, while the arm is moving. It is told what the objects look like (the flat yellow slab is the tray). If Cosmos is slow or offline, the robot carries on without it.

---

## 8. Where everything runs

| Part | Where |
|---|---|
| The robot simulator (Isaac Sim) | On the laptop, inside a container, with no window |
| Cosmos (the image-understanding checker) | On the laptop, inside a container |
| The app and web page you click on | On the laptop |
| Nemotron (the planner for non-stacking prompts) | NVIDIA's servers, over the internet |

The laptop has an NVIDIA RTX 5090 GPU with 24 GB of memory. Isaac Sim uses a few GB of it and Cosmos about 16 GB. **It needs to be plugged in:** on battery the graphics chip slows down a lot.

---

## 9. What you see on screen

- **Live camera views:** the overview and the wrist camera. The overview shows bounding boxes in camera-only mode.
- **Header:** the NTT DATA logo and the project name, then small labelled groups: the mode (RMPFlow, Agentic or Direct, Camera only or Sim positions, Sim idle or busy), the three services (Isaac Sim, Nemotron, Cosmos) and links to the three log pages.
- **Agent Plan panel:** the steps, with their status (running, done, rejected, retrying), a plain-language answer when the run ends, and the Cosmos check for each step. Steps the agent adds itself (like moving a blocking cube) are marked "inserted"; steps replaced by a new plan say "re-planned from the live positions".
- **Logs:**
  - *Nemotron log* — the planner's reasoning.
  - *Cosmos log* — each visual check, with the exact image it looked at.
  - *Perception log* — the camera's detections, with bounding boxes and estimated positions.
- **Telemetry:** the render loop rate, joint angles, gripper width and end-effector position.

---

## 10. Folder map

```
Physyk_AI/
├── README.md, HOW_IT_WORKS.md, ...   the documents (start here)
├── run.sh, start_*_docker.sh         start the app, the simulator and Cosmos
├── workspace/                        the application itself
│   ├── stack_planner.py              the stacking brain (no AI model)
│   ├── physyk_perception/            the camera-vision code
│   ├── factory_background.py         the factory scenery (visual only)
│   └── scripts/                      offline tests
├── assets/                           reference files for the robot arm
└── logs/                             run logs
```

---

## 11. Honest limits

- It's a **simulation**. A real robot would need a real camera, real calibration and noise handling.
- It works for **three coloured cubes on a known table and tray**, under the default lighting. It is not a general object finder.
- The tray and table are assumed fixed.
- Some internal parts still rely on the simulator, for example the gripper's finger-width sensor.
- Cubes can only be grasped between about 0.38 m and 0.66 m from the robot, with the hand pointing straight down.
- The hand is not turned to the cube, the arm's elbow posture is not controlled, and the grasp script is a timer: long sideways carries may overshoot. This has been measured on an earlier version of the controller, not yet on this one.
- Planning "stack the cubes" is fully tested offline (thousands of random layouts and a fake simulator), but it has not yet been run as a batch on the real simulated arm.
- A request that goes to Nemotron can take 20-50 seconds to plan, because the model reasons at length first.

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
| **Covered cube** | A cube with another cube resting on it. |
| **Safe zone** | The part of the table where the system places cubes: comfortably within reach, away from the robot base and the tray. |
