# Physyk AI — Agentic Pick-and-Place

Tell a simulated Franka Panda arm what to do in plain English. Physyk plans the steps, finds the objects by looking at camera images, executes the motion in NVIDIA Isaac Sim, and checks the real outcome before reporting success.

> **Status:** cube and tray pick-and-place on a life-science-lab scene: three cubes (red, green, blue), a yellow tray, the lab bench and walls, no shelf. Prompts such as "put the red cube in the tray", "stack the cubes", or "stack the cubes on the table" run end to end. The glassware (flask/beaker) variant is kept on the `feature-lab` branch.

## How it works

```
Instruction ─► Nemotron (decompose into steps) ─► guardrails ─► Isaac Sim (Franka Panda, RMPFlow)
                                                                      │
        retry / re-plan ◄── verify against live position ◄── camera perception (+ Cosmos check)
```

| Component | Role |
|---|---|
| **Isaac Sim 6.0.1** (Docker) | Physics simulation (60 Hz) and cameras (overview, wrist) |
| **Nemotron** (NVIDIA hosted API) | Turns free-form instructions into pick/place steps |
| **Cosmos 3 Reasoner** (local NIM) | Independent visual check of each step (informational; never gates the arm) |
| **Pure perception** (GUI switch) | Colour masks + depth find the cubes by looking, ~20 ms; the arm then uses camera poses only |
| **Backend + GUI** | FastAPI server on port 7860, talking to the sim on port 8100 |

## What it can do

- **Pick and place** a cube into the tray, onto another cube, beside another cube ("right of", "next to", ...), or onto a free spot on the table.
- **Stack**: "stack the cubes" always builds a vertical pile. The order is chosen from the live positions, never by colour: cubes with nothing on top and closest to the robot go first. "...in the tray" builds the pile in the tray, "...on the table" builds it on the table, and an existing pile is kept or taken apart from the top.
- **Blocked picks**: if a cube has another cube on top of it, the agent first moves the blocker to a free table spot (at least 15 cm from every cube and 10 cm from the tray), then picks.
- **Rotated cubes**: the hand turns about Z to grip a cube's faces. Between the face-aligned angles it picks the one whose open fingers sweep no neighbouring cube or the tray rim. During transport it turns to the destination alignment, so stacks stay square. "Randomize" gives every cube a random rotation.
- **Verification**: every step is checked against the cube's real final position (deterministic mode) or a fresh camera look (camera-only mode), plus a Cosmos visual check.

## Motion control (what makes the arm behave)

The pick/place controller is NVIDIA's open-loop 10-phase state machine on RMPFlow. These additions keep it accurate:

- **Arrival gating**: the approach and transport phases wait until the hand centre is within 1.5 cm of the target xy (up to 400 extra frames), so travel happens at carry height and the descent is vertical. Without it the arm swept diagonally while descending.
- **Release on contact**: while lowering a cube, the gripper opens as soon as the cube reaches its placing height or stops descending, and holds still while the fingers open. It only fires for a genuine descent.
- **Lift clearance**: transport starts only once the carried cube clears the top of the highest other cube.
- **Fingers**: commanded fully open (8 cm) during the approach and whenever the gripper is idle.
- **Reach limits**: the downward-pointing hand can grasp cubes between about 0.38 m and 0.66 m from the robot base. Randomize and the free-spot search stay inside that, and the agent's guardrail refuses targets closer than 0.36 m (closer, the arm folds until joint 6 reaches its limit and the grasp fails).
- **No RMPFlow obstacles by default**: the table, tray and cubes are not avoidance obstacles (the arm travels above them and descends vertically); active obstacles made results worse in tests.
- Arm drive is the stock Franka one (kp 22918, kd 4584 N·m/rad); motion phases run at 1.25x.

## Requirements

- Linux with an NVIDIA GPU (developed on a 24 GB laptop GPU) and Docker with the NVIDIA runtime
- Access to the Isaac Sim 6.0.1 container and the Cosmos 3 NIM image on NGC
- An NVIDIA API key for the hosted Nemotron model
- Python 3 for the backend

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install fastapi 'uvicorn[standard]' httpx opencv-python-headless numpy pillow

cp .env.example .env      # then add your NVIDIA API key; .env is git-ignored
```

## Run

```bash
./start_sim_docker.sh      # terminal 1: Isaac Sim, headless by default (serves on :8100); Ctrl+C stops it
./start_cosmos_docker.sh   # optional: Cosmos NIM (serves on :8001)
./run.sh                   # terminal 2: backend + GUI on http://localhost:7860

./run.sh --status          # health of everything
./run.sh --no-agent        # skip the LLM; send instructions straight to the sim
./run.sh --stop            # stop the backend
```

`HEADLESS=0 ./start_sim_docker.sh` opens the Isaac Sim window instead. The GUI's Isaac Sim pill shows ONLINE while the `physyk-isaac-sim` container is running. Full setup notes are in `instructions.md`.

### Sim tuning (environment variables, all optional)

| Variable | Default | Effect |
|---|---|---|
| `PHYSYK_SPEEDUP` | 1.25 | Speed-up of the motion phases (settle, close, release never scale) |
| `PHYSYK_SPEEDUP_TRANSIT` / `PHYSYK_SPEEDUP_VERTICAL` | = `PHYSYK_SPEEDUP` | Per-group override (free-space moves / vertical moves). 1.5 and 1.75 made the open-loop controller outrun the arm |
| `PHYSYK_ARM_KD` | 0 (stock) | Arm joint damping; lower = faster but jerkier |
| `PHYSYK_ENCODE_EVERY` | 2 | JPEG-encode the camera streams every Nth step (frees time for physics) |
| `PHYSYK_ARM_OBSTACLES` | `off` | RMPFlow collision avoidance: `off` = none, `static` = table + tray, `all` = plus the cubes. Obstacles are re-registered after every controller reset. `off` gave the best results (see `TECHNICAL_DEEP_DIVE.md` §1.3); `PHYSYK_CUBE_OBSTACLES=1` is the old spelling of `all` |
| `PHYSYK_MIN_RADIUS` | 0.38 | Closest distance (m) from the robot base at which Randomize and the free-spot search place cubes; closer picks drive joint 6 into its limit |
| `PHYSYK_RANDOM_SEED` | unset | Fixes the random layout sequence so test runs are comparable |
| `PHYSYK_JOINT_TRACE` | 1 | Writes `workspace/joint_trace.csv` (every joint's position, command, velocity, error per step) and `[Joint anomaly]` lines to the sim log; 0 turns it off |
| `PHYSYK_PROFILE` | off | Prints where each loop iteration's time goes |

## Tested behaviour

Latest end-to-end run on the real sim (default settings, 10 randomized layouts, stacking in the tray and on the table, 25 picks): **24 of 25 passed**, 2.5-6.6 mm from target (one pick missed), about 20 s per pick. The fastest joint reached 0.68 of its velocity cap (no frame at the cap) and no joint came closer than 0.28 rad to its limit. The one miss was a misaligned grip (finger width 40 mm instead of 53 mm) on a cube turned 36°. Earlier on identical layouts: obstacles off 14/15, table + tray obstacles on 12/15. Open problems are listed in `TECHNICAL_DEEP_DIVE.md` §5. Not yet covered: cubes touching each other, and the full Nemotron path on unusual phrasings. If the arm looks wrong, send the `[Joint anomaly]` lines from `docker logs physyk-isaac-sim`, `workspace/joint_trace.csv` and the prompt used.

## Repository layout

| Path | Contents |
|---|---|
| `workspace/` | Backend, orchestrator, sim service, perception, motion control |
| `workspace/physyk_perception/` | Camera-only cube detection, Cosmos client, step verifier, overlay layout |
| `assets/franka_panda/` | Franka Panda URDF / mesh reference files (the sim loads the arm from Isaac Sim's own asset library) |
| `workspace/scripts/arm_tests/` | End-to-end batch and arm-physics test tools (see its README) |
| `CLAUDE.md` | Working notes for Claude Code sessions in this repo |
| `run.sh`, `start_*_docker.sh` | Launchers |

## Documentation

| Doc | For |
|---|---|
| `HOW_IT_WORKS.md` | Plain-English walkthrough, no code |
| `instructions.md` | Setup, running, GUI use, troubleshooting |
| `ARCHITECTURE.md` | Service topology, the pick-and-place flow, the agent loop, state models, known gaps |
| `TECHNICAL_DEEP_DIVE.md` | Controller phases, hand yaw, closed-loop rules, obstacles, Nemotron prompts, guardrails, Cosmos |
| `PERCEPTION.md` | The camera-only pipeline: detector, calibration, measurements, limits |
| `cosmos_integration.md` | How Cosmos is deployed and used |
| `PROJECT_NARRATIVE.md` | The project write-up: objectives, capabilities, challenges and how they were fixed |
| `HISTORY.md` | The setups the project has gone through (first cloud-VM setup, the lab environment, the glassware lab) and what was removed from this branch |
| `INDUSTRY_APPLICATIONS.md` | Where this applies, and Q&A prep |

## Notes

- Keep the laptop plugged in; a power-capped GPU slows the sim noticeably.
- The sim container (`physyk-isaac-sim`) and port 8100 are shared by every run; stop one before starting another.
- Never commit `.env`; it holds API keys.
- The cube colour thresholds in `workspace/physyk_perception/color_segment.py` were tuned to the studio lighting; re-check them if the lighting or background changes.
