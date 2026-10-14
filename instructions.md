# Physyk — Run Guide (local laptop)

Everything runs on this laptop: Isaac Sim in Docker, the backend/GUI in a Python venv, Cosmos as a
local NIM container, and Nemotron through NVIDIA's hosted API.

## What runs where

| Port | Service | Runs as | Started by |
|---|---|---|---|
| **7860** | Web GUI + backend (open this) | `.venv` python, `workspace/physyk_main_server.py` | `./run.sh` |
| **8100** | Isaac Sim (physics, cameras, perception) | Docker `nvcr.io/nvidia/isaac-sim:6.0.1` | `./start_sim_docker.sh` |
| **8001** | Cosmos 3 Nano Reasoner 8B (vision-language model) | Docker NIM `nvcr.io/nim/nvidia/cosmos3-reasoner` | `./start_cosmos_docker.sh` |
| — | Nemotron-3 Super 120B (planner) | NVIDIA hosted API, no local process | key in `.env` |

Isaac Sim runs in a container because the host OS (Ubuntu 26.04) is newer than Isaac Sim supports;
the container brings its own userland and only needs the NVIDIA driver.

## One-time setup

1. **Docker + NVIDIA runtime** working, and logged in to NGC: `docker login nvcr.io`
   (username `$oauthtoken`, password = an NGC API key).
2. **Pull the images** (large, one time): `docker pull nvcr.io/nvidia/isaac-sim:6.0.1` and
   `docker pull nvcr.io/nim/nvidia/cosmos3-reasoner:latest`. Cosmos downloads its weights (~10 GB, FP8)
   into `~/.cache/nim` on its first start.
3. **Python venv** for the backend:
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install fastapi 'uvicorn[standard]' httpx opencv-python-headless numpy pillow
   ```
4. **API keys** — copy `.env.example` to `.env` and fill it in (never commit `.env`):
   ```
   NEMOTRON_API_KEY=nvapi-...                    # from build.nvidia.com
   NEMOTRON_MODEL=nvidia/nemotron-3-super-120b-a12b
   COSMOS_API_KEY=nvapi-...                      # used only as the NGC key for Cosmos's weight download
   ```
   Endpoints and model names are read by `workspace/physyk_config.py`; every value can be overridden
   with an environment variable.

## Starting everything

```bash
# 1. Cosmos (leave running in the background; only needed for Cosmos evidence in the logs)
./start_cosmos_docker.sh &

# 2. Isaac Sim, headless by default (terminal 1, leave open) — first boot ~1 minute; Ctrl+C stops it
./start_sim_docker.sh

# 3. Backend + GUI (terminal 2)
./run.sh                    # then open http://localhost:7860
```

The sim starts in the deterministic mode by default. **Do not** set `PERCEPTION_MODE` — the GUI
controls it (below).

### Other commands

```bash
./run.sh --no-agent         # every prompt goes straight to Isaac Sim (no Nemotron)
./run.sh --status           # health of everything
./run.sh --stop             # stop the backend only
docker rm -f physyk-isaac-sim   # stop the sim (Ctrl+C in its terminal also stops the container)
docker stop physyk-cosmos       # stop Cosmos
HEADLESS=0 ./start_sim_docker.sh   # sim with a window (default is headless)
```

Notes: `run.sh` starts and stops **only the backend/GUI** — it never launches or stops the sim or Cosmos (it just warns if
the sim isn't up on :8100). After changing sim code, stop the sim (Ctrl+C) and start it again; after changing backend
code, just run `./run.sh` again.

### Sim tuning (all optional environment variables)

See the table in `README.md` (`PHYSYK_SPEEDUP*`, `PHYSYK_ARM_KD`, `PHYSYK_ENCODE_EVERY`, `PHYSYK_ARM_OBSTACLES`,
`PHYSYK_MIN_RADIUS`, `PHYSYK_RANDOM_SEED`, `PHYSYK_JOINT_TRACE`, `PHYSYK_PROFILE`). Set them in front of the command, e.g. `PHYSYK_PROFILE=1 ./start_sim_docker.sh`.

## Using the GUI

- **Send prompt to Agent** — on: Nemotron decomposes the prompt into steps, guardrails check each,
  the sim executes and verifies. Off: the prompt goes straight to the sim as typed.
- **Pure perception — camera only** (switch under it) — off: the arm uses the simulator's exact
  positions (deterministic). On: object positions come only from the overview camera, the tray is a
  fixed constant, and the check after each place is a fresh camera look. Flips live, no restart; takes
  effect on the next pick. See `PERCEPTION.md`.
- **Logs (top strip)** — `Nemotron Log` (planning reasoning), `Cosmos Log` (pre/post-step visual checks
  with the exact image Cosmos saw), `Perception Log` (annotated camera frames with bounding boxes and
  estimated XYZ for every pre-pick detection and post-place check).
- **Example prompts** (the table holds a red, a green and a blue cube and a yellow tray):
  `pick up the red cube and place it in the tray`; `put the blue cube on top of the green cube`;
  `put the red cube to the right of the blue cube`; `stack the cubes` (a vertical pile, built from where the cubes are, easiest
  first — no colour order needed); `stack the cubes in the tray`; `stack the cubes on the table`.
  The control buttons `randomize` (random positions **and** rotations inside the arm's reach), `reset` (back to the staging spots)
  and `home` go straight to the sim.
- The **Isaac Sim** light in the header is ONLINE whenever the `physyk-isaac-sim` container is running, even while the sim is still loading.

## Scene background

The sim opens in a life-science-lab setting (floor, walls, a bench body with drawers under the work table, the bench
extended behind the robot, wall shelving and a fume hood). It is **visual only** — no physics, no effect on picking,
and the work table itself is unchanged. `PHYSYK_BACKGROUND=grid ./start_sim_docker.sh` restores the original blue grid.
Keep any new background items **neutral (white/grey)**: the colour detector looks for saturated red, green and blue.
The cube colour thresholds (`workspace/physyk_perception/color_segment.py`) were rebuilt from measured values; re-check them if the
lighting or background changes.

## GPU memory (RTX 5090 laptop, 24 GB)

Isaac Sim ~3.6 GB + Cosmos ~16.3 GB ≈ 20 GB, leaving ~4 GB. If they ever collide, lower Cosmos's share:
`COSMOS_GPU_UTIL=0.60 ./start_cosmos_docker.sh` (default 0.65). Cosmos runs the FP8 build with an 8192-token
context and CUDA graphs off — see the comments in `start_cosmos_docker.sh` for why (defaults crash on this GPU).

## Health & logs

```bash
curl -s localhost:7860/health; curl -s localhost:8100/health; curl -s localhost:8001/v1/models
tail -f logs/physyk_server.log logs/isaac_sim.log logs/cosmos.log
nvidia-smi
```

## Troubleshooting

- **Sim window doesn't open / renderer errors** — the cache folder under `.isaac-cache/` must be writable
  by the container (the launcher handles this); check `logs/isaac_sim.log`.
- **Sim is slow / a pick takes minutes** — first check the laptop is **plugged in**: on battery the GPU is power-capped
  (~800 of 3090 MHz) and the loop drops to ~3 iterations/s. Normal now: ~40 iterations/s, a pick in 17-20 s (22-30 s in
  camera-only mode), a three-cube stack in about a minute. `PHYSYK_PROFILE=1 ./start_sim_docker.sh` prints where each frame's time goes.
- **The arm moves erratically** — send the `[Joint anomaly]` lines (`docker logs physyk-isaac-sim | grep "Joint anomaly"`) and
  `workspace/joint_trace.csv` (written by default; every joint's position, command, velocity and error per step of a pick), plus the prompt used.
- **A pick misses or the arm outruns itself** — leave the speed-ups at the default 1.25; higher values made the open-loop controller
  outrun the arm.
- **Sim hangs at startup with an empty log / the container stays "unhealthy"** — intermittent (about one start in five); `docker rm -f physyk-isaac-sim` and start it again.
- **Cosmos won't start** — see `logs/cosmos.log`; usually GPU memory (stop the sim first, or lower `COSMOS_GPU_UTIL`).
- **Nemotron errors / "planning failed"** — check `.env`; without a working key each prompt falls back to a
  single direct command.
- **"Y-target ... out of bounds [-0.45, 0.45]" right after a randomize** — an object is outside the arm's validated reach. Randomize keeps
  cubes within 0.66 m of the robot and Y from -0.15 to 0.42; restart the sim to pick this up.
- **"target object ... not found in live scene" in pure-perception mode** — the camera did not see that cube (it can be hidden by the
  arm or another cube). The guardrail re-observes up to 3 times; if it persists, `randomize` or switch the perception switch off.
- **"target is too close to the robot base (radius ... m, minimum 0.36 m)"** — the cube is closer to the robot than the downward hand can reach (the wrist hits its joint limit). Move it with `randomize` (which keeps cubes 0.38-0.66 m from the base) or `reset`.
- **"there is no free spot on the table ..."** — the agent needed to move a blocking cube but the table is too crowded for a spot with
  room for the hand; `reset` or `randomize` first.
- **Perception mode rejects a step ("Z-target … out of bounds")** — the orchestrator re-observes up to 3 times
  before rejecting; a persistent failure means the camera can't see the object.

More: `README.md` (overview), `HOW_IT_WORKS.md` (plain English), `ARCHITECTURE.md` and `TECHNICAL_DEEP_DIVE.md` (how it works),
`PERCEPTION.md` (the camera-only pipeline), `cosmos_integration.md` (Cosmos).
