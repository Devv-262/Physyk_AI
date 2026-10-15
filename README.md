# Physyk AI — Agentic Pick-and-Place

Tell a simulated Franka Panda arm what to do in plain English. Physyk plans the steps, finds the objects by looking at camera images, executes the motion in NVIDIA Isaac Sim, and checks the real outcome before reporting success.

> **Status:** three coloured cubes (red, green, blue) and a yellow tray on a work table inside a factory-floor scene (conveyor, yellow robot cell, racking, tool cabinet). Prompts such as "put the red cube in the tray", "put the green cube on top of the blue cube" or "stack the cubes" run end to end. "Stack the cubes" is planned by a local, deterministic planner and is re-planned from the live scene after every step. A GUI switch runs the whole thing from the camera alone, with no simulator coordinates.

## How it works

```
Instruction ─► planner ─► guardrails ─► Isaac Sim (Franka Panda, RMPFlow, PickPlaceController)
                 │                                │
   "stack ..." ─► local stack planner             ▼
   other text ─► Nemotron              verify against live positions
                                       (simulator pose, or a fresh camera look)
                                                  │
        retry / re-plan  ◄────────────────────────┘      Cosmos adds a visual second opinion
```

| Component | Role |
|---|---|
| **Isaac Sim 6.0.1** (Docker) | Physics (60 Hz) and two cameras (overview, wrist) |
| **Nemotron 3.5 Lightning 30B-A3B** (NVIDIA hosted API) | Turns free-form instructions into pick/place moves, one at a time (thinking off, streamed); diagnoses failures |
| **Local stack planner** (`stack_planner.py`) | Plans "stack the cubes" from positions alone: which cube is the base, which moves next, where blockers go |
| **Camera-only perception** (GUI switch) | Colour masks + depth find the cubes by looking, about 20 ms; the arm then uses camera poses only |
| **Cosmos 3 Nano Reasoner** (local NIM) | Independent visual check of each step (evidence only; never gates the arm) |
| **Backend + GUI** | FastAPI on port 7860, talking to the sim on port 8100 |

## What it can do

- **Pick and place** a cube into the tray, onto another cube, beside another cube ("right of", "next to", ...), or onto a free spot on the table.
- **Stack** ("stack the cubes", "... in the tray", "... on the table", "build a tower", "stack them"). Always a vertical pile, planned from positions and never from colour. It handles piles already standing in the tray or on the table (taken apart from the top), partial piles of one or two cubes, cubes standing close together, covered cubes (never picked until the cube on top has moved) and a missing or out-of-reach cube (refused with a clear reason). Details: `ARCHITECTURE.md` §3 and `TECHNICAL_DEEP_DIVE.md` §3.
- **Free spots**: when a cube has to be put out of the way, the spot is inside a safe zone: away from the robot base, inside reach, clear of the tray, and apart from every other cube.
- **Verification**: every step is checked against the cube's real final position (deterministic mode) or a fresh camera look (camera-only mode), plus a Cosmos visual check that runs beside the arm.
- **Failure handling**: a failed stacking step is planned again from the live positions. Other steps get Nemotron's diagnosis (retry, re-plan the step, re-plan the rest).

## Scene

A work table (0.9 x 1.3 m, top at Z = 0), a yellow tray, three 5 cm cubes and the arm. Behind them is a visual-only factory backdrop (`factory_background.py`): polished floor, walls and ceiling, a conveyor with cartons, a yellow robot cell with a safety rail, racking, a tool cabinet and table legs. Nothing in the backdrop has physics. `PHYSYK_BACKGROUND=grid` restores the original blue-grid ground. The backdrop can be rebuilt live with `POST :8100/background/rebuild`.

## Motion control (this branch)

NVIDIA's open-loop 10-phase `PickPlaceController` on RMPFlow, with the Franka's stock drive gains.

- Motion phases run at 1.25x; settle, close and release run at NVIDIA's timing, lengthened for a firm grasp.
- Grasp confirmation reads the gripper's achieved width; a grasp that closed on nothing aborts early.
- Post-place verification compares the cube's real position with its target (3 cm in XY, 3.5 cm in Z).
- The table, the tray and the cubes are registered with RMPFlow as obstacles at start-up; see `TECHNICAL_DEEP_DIVE.md` §1.3 for why they are probably not active in practice.
- The hand is not turned to the cube's faces, and the controller does not wait for the arm to arrive between phases: phases advance on a timer. The known consequences are in `TECHNICAL_DEEP_DIVE.md` §5.
- The agent's guardrail refuses targets outside x 0.20-0.70 m, y -0.45..0.45 m, z -0.01..0.50 m, or closer than 0.36 m to the base (the wrist reaches its joint limit).

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
./start_sim_docker.sh      # terminal 1: Isaac Sim (serves on :8100); Ctrl+C stops the container
./start_cosmos_docker.sh   # optional: Cosmos NIM (serves on :8001)
./run.sh                   # terminal 2: backend + GUI on http://localhost:7860

./run.sh --status          # health of everything
./run.sh --no-agent        # skip the LLM; send instructions straight to the sim
./run.sh --stop            # stop the backend
```

The sim service always runs headless in its container; the GUI at `localhost:7860` is the viewport. Full setup notes are in `instructions.md`.

### Settings (environment variables, all optional)

| Variable | Default | Effect |
|---|---|---|
| `NVIDIA_API_KEY` / `NEMOTRON_API_KEY` | — | Key for the hosted Nemotron model (`.env`) |
| `NEMOTRON_MODEL` | `nvidia/nemotron-3.5-lightning-30b-a3b` | Planner model id |
| `NEMOTRON_BASE_URL` | `https://integrate.api.nvidia.com/v1` | A local OpenAI-compatible server also works |
| `COSMOS_BASE_URL` / `COSMOS_MODEL` | `http://localhost:8001/v1` / `nvidia/cosmos3-nano-reasoner` | The local Cosmos NIM |
| `PHYSYK_STACK_PLANNER` | `code` | `code` = local stack planner; `llm` = let Nemotron plan stacking (slow, unreliable with a thinking model) |
| `PHYSYK_PLANNER` | `stepwise` | `stepwise` = Nemotron splits a request into goals once, then proposes one move at a time against the live scene; `upfront` = the old single decomposition |
| `NEMOTRON_STEP_THINKING` / `NEMOTRON_THINKING` | `off` | Thinking mode for the step planner / for the one-shot calls (decompose, failure decision). `on` restores the chain-of-thought (slow, can fill the token cap) |
| `NEMOTRON_STEP_MAX_TOKENS` / `NEMOTRON_STEP_TEMPERATURE` | `400` / `0` | Completion cap and temperature of the step planner (an answer is 30-60 tokens) |
| `PHYSYK_AGENTIC` | `1` | `0` sends every instruction straight to the sim (`./run.sh --no-agent`) |
| `PHYSYK_ENCODE_EVERY` | `2` | JPEG-encode the camera streams every Nth loop iteration (frees time for physics) |
| `PHYSYK_BACKGROUND` | `factory` | `grid` = original blue-grid ground |
| `PERCEPTION_MODE` | `shadow` | Start-up perception mode; the GUI switch changes it live (leave unset) |
| `PERCEPTION_DETECTOR` | `color` | `cosmos` uses Cosmos's own grounding (1.5-3 s per call; not for live use) |
| `PERCEPTION_ZCAL` | `1` | Table-plane self-calibration of detected heights |
| `PHYSYK_SIM_CONTAINER` | `physyk-isaac-sim` | Container the GUI's Isaac Sim pill follows |

## Tests (offline; no simulator, arm or network)

```bash
python3 workspace/scripts/test_stack_planner.py [layouts]   # planner scenarios + fuzz + safe-zone rules
.venv/bin/python workspace/scripts/test_stack_agent.py      # the agent loop against a fake simulator
```

The planner suite runs 48 named scenarios and thousands of random layouts on every surface (tray, table, anywhere), with noisy positions, 2-4 cubes, covered and out-of-reach cubes, and checks that every plan ends as one pile with the minimum number of moves. It also checks that Randomize and free spots always stay inside the safe zone. The agent test runs the real `NeMoRoboticsAgent.run()` loop against a fake sim that can drop cubes at random. Last results: all passed (see `TECHNICAL_DEEP_DIVE.md` §3.6). `workspace/scripts/test_stepwise_agent.py` (needs the API key) runs the step-at-a-time Nemotron planner through the real agent loop against the same fake simulator.

## Repository layout

| Path | Contents |
|---|---|
| `workspace/isaac_sim_service.py` | Sim loop, scene, controller phases, perception wiring, HTTP API on :8100 |
| `workspace/nemotron_planner.py` | The step-at-a-time Nemotron planner: system prompt, scene-as-relations, milestones, streaming |
| `workspace/physyk_agent_orchestrator.py` | Planner (Nemotron + stack planner), guardrails, dispatch / verify / retry |
| `workspace/stack_planner.py` | The local stacking planner and the shared safe-zone rules |
| `workspace/physyk_main_server.py` | GUI and gateway on :7860 |
| `workspace/physyk_perception/` | Camera detector, provider, overlay layout, Cosmos client, step verifier |
| `workspace/factory_background.py` | The visual-only factory backdrop |
| `workspace/reasoning_dashboard.py` | The Nemotron / Cosmos log pages |
| `workspace/physyk_config.py` | Endpoints, keys and model names (from `.env`) |
| `workspace/static/` | The NTT DATA logo used in the GUI header |
| `workspace/scripts/` | The offline tests |
| `assets/franka_panda/` | Franka Panda URDF / mesh reference files (the sim loads the arm from Isaac Sim's own asset library) |
| `run.sh`, `start_*_docker.sh` | Launchers |

## Documentation

| Doc | For |
|---|---|
| `HOW_IT_WORKS.md` | Plain-English walkthrough, no code |
| `instructions.md` | Setup, running, GUI use, troubleshooting |
| `ARCHITECTURE.md` | Service topology, the pick-and-place flow, the agent loop, the stack planner, state models, known gaps |
| `TECHNICAL_DEEP_DIVE.md` | Controller phases, orchestration, the planner in depth, perception, scene, performance, open problems |
| `PERCEPTION.md` | The camera-only pipeline: detector, calibration, measurements, limits |
| `cosmos_integration.md` | How Cosmos is deployed and used |
| `PROJECT_NARRATIVE.md` | The project write-up: objectives, capabilities, challenges and how they were handled |
| `HISTORY.md` | The setups the project has gone through, including the restart from the 22 August code |
| `INDUSTRY_APPLICATIONS.md` | Where this applies, and Q&A prep |
| `CLAUDE.md` | Working notes for Claude Code sessions in this repo |

## Notes

- Keep the laptop plugged in; a power-capped GPU slows the sim noticeably.
- The sim container (`physyk-isaac-sim`) and port 8100 are shared by every run; stop one before starting another.
- Never commit `.env`; it holds API keys.
- The cube colour thresholds in `workspace/physyk_perception/color_segment.py` are tuned to the default studio lighting. The table-plane calibration adapts to how bright the table is, but re-check the detection if the lighting or the table colour changes.
