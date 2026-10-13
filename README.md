# Physyk AI — Agentic Pick-and-Place

Tell a simulated Franka Panda arm what to do in plain English. Physyk plans the steps, finds the objects by looking at camera images, executes the motion in NVIDIA Isaac Sim, and checks the real outcome before reporting success.

> **Status:** the scene is a lab table with five pieces of glassware (three Florence flasks, two beakers) and a three-level shelf. Perception and planning work; **grasping glassware is still in progress** (see `TODO_GRASPING.md`). The earlier cube/tray pipeline is documented in `PROJECT_NARRATIVE.md`.

## How it works

```
Instruction ─► Nemotron (decompose into steps) ─► guardrails ─► Isaac Sim (Franka Panda, RMPFlow)
                                                                      │
        retry / re-plan ◄── verify against live position ◄── camera perception (+ Cosmos check)
```

| Component | Role |
|---|---|
| **Isaac Sim 6.0.1** (Docker) | Physics simulation and cameras |
| **Nemotron** (NVIDIA hosted API) | Turns free-form instructions into pick/place steps |
| **Cosmos 3 Reasoner** (local NIM) | Independent visual check of each step |
| **Pure perception** | Depth-geometry detection, named by liquid colour; ~20 ms, 1.7 mm mean error |
| **Backend + GUI** | FastAPI server on port 7860, talking to the sim on port 8100 |

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
./start_sim_docker.sh      # terminal 1: Isaac Sim (serves on :8100)
./start_cosmos_docker.sh   # optional: Cosmos NIM (serves on :8001)
./run.sh                   # terminal 2: backend + GUI on http://localhost:7860

./run.sh --status          # health of everything
./run.sh --no-agent        # skip the LLM; send instructions straight to the sim
./run.sh --stop            # stop the backend
```

Full setup notes are in `instructions.md`.

## Repository layout

| Path | Contents |
|---|---|
| `workspace/` | Backend, orchestrator, sim service, perception, motion control |
| `workspace/physyk_perception/` | Camera-only object detection |
| `assets/` | Franka Panda and scene object assets |
| `scripts/` | Helper and diagnostic shell scripts |
| `run.sh`, `start_*_docker.sh` | Launchers |

## Documentation

- `HOW_IT_WORKS.md` — plain-English walkthrough, no code
- `ARCHITECTURE.md` and `TECHNICAL_DEEP_DIVE.md` — design details
- `cosmos_integration.md` — Cosmos setup and use
- `TODO_GRASPING.md` — glassware grasping status and open problems
- `INDUSTRY_APPLICATIONS.md` — where this applies
- `PROJECT_NARRATIVE.md` — original project write-up, challenges and history

## Notes

- Keep the laptop plugged in; a power-capped GPU slows the sim noticeably.
- Never commit `.env`; it holds API keys.
