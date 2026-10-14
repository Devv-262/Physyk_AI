# Physyk — Run Guide (local laptop)

Everything runs on this laptop: Isaac Sim in Docker, the backend and GUI in a Python venv, Cosmos as a local NIM container, and Nemotron through NVIDIA's hosted API.

## What runs where

| Port | Service | Runs as | Started by |
|---|---|---|---|
| **7860** | Web GUI + backend (open this) | `.venv` python, `workspace/physyk_main_server.py` | `./run.sh` |
| **8100** | Isaac Sim (physics, cameras, perception) | Docker `nvcr.io/nvidia/isaac-sim:6.0.1` | `./start_sim_docker.sh` |
| **8001** | Cosmos 3 Nano Reasoner 8B (vision-language model) | Docker NIM `nvcr.io/nim/nvidia/cosmos3-reasoner` | `./start_cosmos_docker.sh` |
| — | Nemotron 3.5 Lightning 30B-A3B (planner) | NVIDIA hosted API, no local process | key in `.env` |

Isaac Sim runs in a container because the host OS (Ubuntu 26.04) is newer than Isaac Sim supports; the container brings its own userland and only needs the NVIDIA driver. The sim service is headless; there is no native window.

## One-time setup

1. **Docker + NVIDIA runtime** working, and logged in to NGC: `docker login nvcr.io` (username `$oauthtoken`, password = an NGC API key).
2. **Pull the images** (large, one time): `docker pull nvcr.io/nvidia/isaac-sim:6.0.1` and `docker pull nvcr.io/nim/nvidia/cosmos3-reasoner:latest`. Cosmos downloads its weights (about 10 GB, FP8) into `~/.cache/nim` on its first start.
3. **Python venv** for the backend:
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install fastapi 'uvicorn[standard]' httpx opencv-python-headless numpy pillow
   ```
4. **API keys**: copy `.env.example` to `.env` and fill it in (never commit `.env`):
   ```
   NVIDIA_API_KEY=nvapi-...      # from build.nvidia.com; used for Nemotron (and as the NGC key for Cosmos's weight download)
   ```
   Endpoints and model names are read by `workspace/physyk_config.py`; every value can be overridden with an environment variable (see the table in `README.md`).

## Starting everything

```bash
# 1. Cosmos (leave running in the background; only needed for Cosmos evidence in the logs)
./start_cosmos_docker.sh &

# 2. Isaac Sim (terminal 1, leave open) — first boot about a minute; Ctrl+C stops it
./start_sim_docker.sh

# 3. Backend + GUI (terminal 2)
./run.sh                    # then open http://localhost:7860
```

The sim starts in the deterministic mode by default. **Do not** set `PERCEPTION_MODE`: the GUI switch controls it.

### Other commands

```bash
./run.sh --no-agent         # every prompt goes straight to Isaac Sim (no Nemotron)
./run.sh --status           # health of everything
./run.sh --stop             # stop the backend only
docker rm -f physyk-isaac-sim   # stop the sim (Ctrl+C in its terminal also stops the container)
docker stop physyk-cosmos       # stop Cosmos
```

`run.sh` starts and stops **only the backend and GUI**; it never launches or stops the sim or Cosmos. After changing sim code (`isaac_sim_service.py`, `stack_planner.py`, `physyk_perception/`), stop the sim and start it again; after changing backend or orchestrator code, run `./run.sh` again. The factory backdrop can be rebuilt without a restart (see below).

## Using the GUI

- **Send prompt to Agent**: on, the prompt is planned (the local stack planner for "stack ..." requests, Nemotron for everything else), guardrailed, executed and verified. Off, the prompt goes straight to the sim as typed.
- **Pure perception — camera only** (the switch under it): off, the arm uses the simulator's exact positions. On, object positions come only from the overview camera, the tray is a fixed constant, and the check after each place is a fresh camera look. It flips live and takes effect on the next pick. See `PERCEPTION.md`.
- **Header pills**: Mode (RMPFlow, Agentic or Direct, Camera only or Sim positions, Sim idle or busy), Services (Isaac Sim, Nemotron, Cosmos: a dot per service, the Isaac Sim pill is on whenever its container runs) and links to the three log pages.
- **Logs**: the Nemotron log (planning reasoning), the Cosmos log (each visual check with the exact image Cosmos saw) and the Perception log (annotated camera frames with bounding boxes and estimated XYZ).
- **Example prompts** (the table holds a red, a green and a blue cube and a yellow tray):
  - `pick up the red cube and place it in the tray`
  - `put the blue cube on top of the green cube`
  - `put the red cube to the right of the blue cube`
  - `stack the cubes` — a vertical pile built from where the cubes are, nearest first, no colour order needed
  - `stack the cubes in the tray`, `stack the cubes on the table`, `build a tower with the cubes`
- **Quick actions**: Randomize places the cubes at random spots inside the safe zone (not near the robot base or the tray, always reachable), Return to Home moves only the arm, Full Reset puts the cubes back to their fixed spots and homes the arm, and Hard Reset stops everything now.

## The scene

The sim opens in a factory-floor scene: floor, walls, ceiling, a conveyor with cartons, a yellow robot cell, racking and a tool cabinet, plus the work table, tray and cubes. Everything except the work table, the tray, the cubes and the arm is **visual only** (no physics, no effect on picking). It lives in `workspace/factory_background.py`:

- `PHYSYK_BACKGROUND=grid ./start_sim_docker.sh` restores the original blue-grid ground.
- Edit the module, then `curl -X POST localhost:8100/background/rebuild` to rebuild it live without restarting the sim.
- `CEILING = False` in that module removes the ceiling (a brighter room); `SCENERY_SCALE` sets the scale of the wall scenery.
- Keep any new scenery neutral and away from the table region of the overview image: the colour detector looks for saturated red, green and blue, and a darker table needs the detector's table calibration (`fit_table_plane`), which adapts to the table's brightness.

## GPU memory (24 GB laptop GPU)

Isaac Sim takes a few GB and Cosmos about 16 GB, leaving a few GB spare. If they ever collide, lower Cosmos's share: `COSMOS_GPU_UTIL=0.60 ./start_cosmos_docker.sh` (default 0.65). Cosmos runs the FP8 build with an 8192-token context and CUDA graphs off; see the comments in `start_cosmos_docker.sh` for why (the defaults crash on this GPU).

## Health and logs

```bash
curl -s localhost:7860/health; curl -s localhost:8100/health; curl -s localhost:8001/v1/models
tail -f logs/physyk_server.log          # the backend (written by run.sh)
docker logs physyk-isaac-sim | tail     # the sim
docker logs physyk-cosmos | tail        # Cosmos
nvidia-smi
```

## Tests (offline; no sim, arm or network)

```bash
python3 workspace/scripts/test_stack_planner.py 4000    # stacking planner + safe-zone rules
.venv/bin/python workspace/scripts/test_stack_agent.py   # the agent loop against a fake simulator
```

## Troubleshooting

- **Sim is slow / a pick takes minutes**: first check the laptop is **plugged in** (on battery the GPU is power-capped). The idle loop should run at about 60 iterations/s (`curl -s localhost:8100/state` shows `fps`).
- **The arm moves strangely** (overshoot on a long carry, a different elbow posture): see `TECHNICAL_DEEP_DIVE.md` §5; this controller is an open-loop timer and the hand is not turned to the cube.
- **Sim hangs at startup with an empty log / the container stays "unhealthy"**: intermittent (about one start in five); `docker rm -f physyk-isaac-sim` and start it again.
- **Cosmos won't start**: see `docker logs physyk-cosmos`; usually GPU memory (stop the sim first, or lower `COSMOS_GPU_UTIL`).
- **Nemotron errors / "planning failed"**: check `.env`; without a working key each prompt falls back to a single direct command. Stacking prompts still work, because the local planner needs no model.
- **"I cannot stack the cubes: the ... cube cannot be reached"**: a cube is closer than about 0.38 m to the robot base or outside the working area, so it can be neither picked nor stacked on. Use Randomize or Full Reset.
- **"target is too close to the robot base (radius ... m, minimum 0.36 m)"**: the same limit, reported by the guardrail for a step planned by hand or by Nemotron.
- **"there is no free spot on the table ..."**: the agent needed to move a cube out of the way but the safe zone has no spot with room for the hand; Randomize or Full Reset first.
- **"target object ... not found in live scene" in pure-perception mode**: the camera did not see that cube (hidden by the arm or another cube). The guardrail re-observes up to 3 times; if it persists, Randomize or switch the perception switch off.
- **Camera-only heights read a centimetre or more off after a scene or lighting change**: check `curl -s localhost:8100/perception` and look at `table_calibration.n` (the number of table pixels used). It should be in the tens of thousands; a few hundred or a thousand means the table fit is failing.

More: `README.md` (overview), `HOW_IT_WORKS.md` (plain English), `ARCHITECTURE.md` and `TECHNICAL_DEEP_DIVE.md` (how it works), `PERCEPTION.md` (the camera-only pipeline), `cosmos_integration.md` (Cosmos).
