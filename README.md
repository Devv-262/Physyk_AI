# Physyk AI — a robot arm you can talk to

**Track: Physical AI** · Nebius × NVIDIA hackathon · License: MIT (see `LICENSE`)

Type *"put the green cube on the blue cube, then the blue cube on the red cube"* and a Franka Panda arm does it in NVIDIA Isaac Sim, with real physics (cubes fall, slip and stack). Nothing is pre-recorded and no step is scripted per task.

The pattern it demonstrates for language-driven robotics:

**understand the request → plan one move → check it is safe → look at the scene → move the arm → measure what really happened → plan again if it did not work.**

**Central rule: a part that can be wrong never decides something it cannot check.** The language model proposes; plain code checks, executes and measures.

---

## 1. For judges — at a glance

| Question | Answer |
|---|---|
| What is it? | A natural-language robot-arm agent: Nemotron plans, Isaac Sim executes, the result is measured, Cosmos gives a second visual opinion. |
| Why build it? | Language models are good at open-ended instructions and bad at knowing whether a physical action worked. This project separates the two: the model plans one move at a time against the *live* scene, and code verifies each move against real positions. |
| NVIDIA models | **Nemotron** (30B class; planner, streamed, thinking off) and **Cosmos** Reason / Nano Reasoner (vision-language second opinion). Also NVIDIA Isaac Sim 6.0.1, `PickPlaceController` and RMPFlow. |
| Nebius | Nemotron is called through an **OpenAI-compatible endpoint**, so **Nebius Token Factory** is a drop-in planner backend: set three variables (section 5). Status is stated honestly in section 5. |
| Physical AI requirement | The robot is simulated (Isaac Sim, full physics, camera perception). There is no physical hardware, so the demo video shows the key application modules in action: GUI, Nemotron plan, arm executing, verification. |
| Verified | Offline: 12,000 planner layouts, 47 agent-loop cases. On the real sim (15 Oct 2026, default layout): 10 end-to-end prompts, all passed, including chained prompts needing parking moves and a refusal of a missing cube. |
| Honest limits | Section 8. |

---

## 2. What you will see

A web page (`http://localhost:7860`) with a **live camera view**, a **prompt box**, and an **agent plan card** showing each move the system decides on, why, whether it worked, and what the second model thought.

| You type | What happens |
|---|---|
| `put the red cube in the tray` | One pick-and-place (about 20 s). |
| `put the blue cube on top of the green cube` | A stack of two. |
| `put the red cube to the right of the blue cube` | A placement 8 cm to the side. |
| `stack the cubes` / `... in the tray` / `... on the table` | A pile of all three, planned by ordinary code from the positions. |
| `put the green cube on the blue cube, then the blue cube on the red cube` | Chained request: blue is buried under green, so green is moved out of the way first (3 moves for 2 clauses). |
| `put the yellow cube on the red cube` | Refusal ("the yellow cube is not something I can see right now"). Nothing moves. |

---

## 3. How it works

| Part | What it is | Its one job | It never |
|---|---|---|---|
| **Nemotron** | NVIDIA's 30B-class language model | Split the sentence into goals; propose the next single move | Touches the robot, or decides whether a goal is met |
| **Local stack planner** | Python (`stack_planner.py`) | Plan "stack the cubes" exactly from positions | Use colour or any AI |
| **Camera + colour detector** | Simulated overview camera, colour/depth maths (no neural net) | Find each cube's 3D position | Guess: no camera frame, no cube |
| **Isaac Sim + NVIDIA controller** | Physics, rendering, `PickPlaceController`, RMPFlow | Move the arm through a 10-phase grasp | Look at the arm (it follows a timer) |
| **Guardrails** | Plain `if` rules | Refuse out-of-reach or unsafe targets | Get overruled by the AI |
| **Cosmos** | NVIDIA vision-language reasoning model | Look at the picture and give a second opinion per step | Block or steer the arm |

```
 you type a sentence
        │
        ▼
 Agent (orchestrator) ── "stack ..." ──► local stack planner (no AI, exact)
        │
        └─ anything else ──► Nemotron: goals once, then ONE move per turn
        │
        ▼
 guardrail (reach / safety) ──refuse──► stop with the reason
        │ ok
        ▼
 Isaac Sim: resolve coordinates → 10-phase pick-and-place → measure the cube's real position
        │                                  ▲ camera (colour + depth) or simulator pose
        ▼
 verified? yes → next move      no → plan again from the live scene
        └─ in parallel: Cosmos looks at a picture before and after the move (evidence only)
```

Design choices (each closed a failure seen on the real system; full table in `ARCHITECTURE.md` §10):

- **Measure, don't assume.** After every move the cube's final position is compared with the target (3 cm sideways, 3.5 cm vertically) and the gripper width is read to see if it closed on the cube.
- **AI for open-ended parts, code for exact parts.** Stacking has a precise answer, so code solves it (fast, repeatable, testable offline). Free-form sentences go to Nemotron.
- **One move at a time.** The model sees the real scene after every move, and *code* decides when a goal is reached. Letting the model track its own progress looped (14 moves, no result); with code tracking it took 7 calls at about 1 s each.
- **The AI can say no but cannot force a move.** A refusal ends the run with the reason quoted verbatim; nothing is substituted.
- **Thinking off.** With thinking on, "stack the cubes" often hit the 4,096-token cap (22-50 s per plan). With it off, a move is about 1 s.

---

## 4. Our stack

| Layer | Technology |
|---|---|
| Simulation and rendering | **NVIDIA Isaac Sim 6.0.1** (PhysX physics at 60 Hz, RTX rendering), run headless in the official Docker container |
| Robot and motion | Franka Panda from Isaac Sim's asset library; NVIDIA **`PickPlaceController`** (10-phase grasp) driving **RMPFlow** motion policies |
| Perception | Simulated overview camera (960 x 540, RGB + depth); OpenCV + NumPy colour segmentation and depth deprojection; no neural network, about 3 mm mean error |
| Language model (planner) | **NVIDIA Nemotron** 30B class, called over an OpenAI-compatible chat API: NVIDIA hosted API or **Nebius Token Factory**; streaming, thinking off |
| Vision-language model | **NVIDIA Cosmos** Nano Reasoner 8B (NIM container), checks each step from a camera frame |
| Agent and planning | Python: orchestrator with guardrails, dispatch / verify / retry; deterministic stack planner; step-at-a-time Nemotron planner |
| Backend and GUI | Python 3, **FastAPI** + uvicorn, `httpx`; the GUI is one inline HTML page that polls `/state` (about 150 ms) and shows MJPEG camera streams |
| Process layout | Two processes: the sim (Isaac Sim's Python, :8100) and the backend + GUI (venv, :7860), talking plain HTTP/JSON; models are remote or container services |
| Tests | Plain Python scripts against a fake simulator (no GPU needed) |

### Models and services in detail

| Component | Model / service | Where it runs | Role |
|---|---|---|---|
| Planner | **NVIDIA Nemotron 30B class** (default `nvidia/nemotron-3.5-lightning-30b-a3b`) over an OpenAI-compatible API | NVIDIA hosted API (default), **Nebius Token Factory** (configurable, section 5), or any local OpenAI-compatible server | Goals split once, then one move per turn |
| Visual second opinion | **NVIDIA Cosmos 3 Nano Reasoner 8B** (NIM container) | Local NIM container, port 8001 | Pre-check and post-check of each step; never gates the arm |
| Simulator | **NVIDIA Isaac Sim 6.0.1**, Franka Panda asset, PhysX | Docker container (Linux) | Physics, rendering, cameras |
| Motion | NVIDIA `PickPlaceController` + RMPFlow | Inside the sim | 10-phase grasp |

Not used: Nebius Serverless Jobs / Serverless Endpoints, GR00T, Sonic. (A GR00T path was tried in the first setup, did not complete the task reliably on this scene, and was removed.)

---

## 5. Nebius Token Factory

The planner client speaks the OpenAI chat-completions protocol (streaming, `chat_template_kwargs` for thinking off, retry on 429 / 5xx with back-off). Pointing it at Token Factory is configuration only, in `.env`:

```
NEMOTRON_BASE_URL=https://api.tokenfactory.nebius.com/v1
NEMOTRON_API_KEY=<your Token Factory key>
NEMOTRON_MODEL=<a Nemotron model id exactly as listed in the Token Factory catalogue>
```

Check the endpoint with no simulator running: `python tools/check_services.py` lists the served models, sends a tiny planning request through the project's own parser, and prints the latency.

**Status, stated plainly:** all measurements in this repository (about 1 s per move, the 10-prompt sweep) were taken against NVIDIA's hosted API. The Token Factory path is supported by the code but its latency and its handling of `chat_template_kwargs` (thinking off) have **not** been measured here. See `ARCHITECTURE.md` §13 for what to check.

---

## 6. Run it

**Requirements:** Linux, an NVIDIA GPU with driver (developed on a 24 GB laptop GPU), Docker with the NVIDIA runtime, access to the Isaac Sim 6.0.1 container (and optionally the Cosmos NIM image) on NGC, an API key for Nemotron (NVIDIA or Nebius), Python 3.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env          # add your key(s); .env is git-ignored

docker login nvcr.io          # user $oauthtoken, password = an NGC API key
docker pull nvcr.io/nvidia/isaac-sim:6.0.1
docker pull nvcr.io/nim/nvidia/cosmos3-reasoner:latest    # optional; about 10 GB on first start
```

```bash
./run.sh --cosmos &           # optional: the Cosmos second opinion (NIM on :8001)
./run.sh --sim                # terminal 1: Isaac Sim, headless, API on :8100 (first boot about a minute); Ctrl+C stops it
./run.sh                      # terminal 2: backend + GUI -> http://localhost:7860

./run.sh --status             # health of everything
./run.sh --no-agent           # skip the language model; instructions go straight to the sim
./run.sh --stop               # stop the backend only
```

Without Cosmos everything works; the step cards show "NOT CHECKED". Without a working Nemotron endpoint, "stack ..." prompts still work (they need no model) and other prompts fall back to one direct command.

After changing sim code restart the sim; after changing backend or orchestrator code run `./run.sh` again.

### Using the GUI

- **Send prompt to Agent** on: planned, guardrailed, executed, verified. Off: sent straight to the simulator as typed.
- **Pure perception — camera only**: off = exact simulator positions; on = positions only from the overview camera (tray fixed), result check is a fresh camera look. Flips live.
- **Header pills**: mode, service health (Isaac Sim, Nemotron, Cosmos), links to the Nemotron, Cosmos and Perception log pages (every model answer, every image Cosmos saw).
- **Quick actions**: Randomize, Return to Home, Full Reset, and **Hard Reset** (the only interrupt for a running plan).
- **Plan card**: status (PLANNING, RUNNING, VERIFYING, SUCCESS, GOAL_NOT_HELD, PARTIAL, FAILED, REJECTED, ABORTED), one card per move with the model's reason, the Cosmos pre/post check, and a plain-language answer.

---

## 7. Settings (`.env`, all optional)

| Variable | Default | Effect |
|---|---|---|
| `NVIDIA_API_KEY` / `NEMOTRON_API_KEY` | — | Key for the Nemotron endpoint (`NEMOTRON_API_KEY` wins) |
| `NEMOTRON_BASE_URL` / `NEMOTRON_MODEL` | `https://integrate.api.nvidia.com/v1` / `nvidia/nemotron-3.5-lightning-30b-a3b` | Planner endpoint and model id; any OpenAI-compatible server (Nebius Token Factory, a local vLLM) |
| `COSMOS_BASE_URL` / `COSMOS_MODEL` | `http://localhost:8001/v1` / `nvidia/cosmos3-nano-reasoner` | Cosmos endpoint and model id |
| `PHYSYK_PLANNER` | `stepwise` | `stepwise` = goals once, then one move per turn; `upfront` = older whole-plan call |
| `PHYSYK_STACK_PLANNER` | `code` | `code` = local planner; `llm` = Nemotron plans stacking too (slow) |
| `NEMOTRON_STEP_THINKING` / `NEMOTRON_THINKING` | `off` | Thinking mode for step planner / one-shot calls |
| `NEMOTRON_STEP_MAX_TOKENS` / `NEMOTRON_STEP_TEMPERATURE` | `400` / `0` | Completion cap and temperature of the step planner |
| `NEMOTRON_TIMEOUT_S` | `90` | Seconds to wait for one Nemotron call |
| `COSMOS_TIMEOUT_S` / `COSMOS_VERIFY_TIMEOUT_S` / `COSMOS_MAX_TOKENS` | `6` / `6` / `512` | Cosmos limits; raise the first two (30 / 20) if Cosmos runs on another machine |
| `PHYSYK_AGENTIC` | `1` | `0` sends every instruction straight to the sim |
| `PHYSYK_GUI_PORT` / `PHYSYK_SIM_PORT` | `7860` / `8100` | Ports |
| `PHYSYK_BACKGROUND` | `factory` | `grid` = the original blue-grid ground |
| `PERCEPTION_DETECTOR` | `color` | `cosmos` uses Cosmos's own grounding (1.5-3 s per call; not for live use) |

---

## 8. What it can and cannot do

**Can:** pick and place a cube into the tray, onto another cube, beside another cube, or onto a free spot; build a pile from any starting arrangement (piles are taken apart from the top, covered cubes are never picked, an unreachable cube is refused with a reason); chain placements and clear blockers automatically; run entirely from the camera; verify every move; re-plan after a failure.

**Cannot (yet):**

- **The controller is open-loop.** Its ten phases advance on a timer and never check where the arm is. A long sideways carry may overshoot (21 cm was measured on an earlier variant; **not measured on this one**).
- **The hand never turns.** It grips straight down, so a rotated cube is gripped at an angle and touching cubes may not be pickable.
- **No dependable obstacle avoidance.** Safety comes from carrying high and descending vertically.
- **Reach:** cubes closer than about 0.38 m to the base or farther than 0.66 m are refused.
- **Perception** is built for three coloured cubes on the known table under the default lighting; it is not a general object detector.
- **Speed:** a pick takes about 20 s (21-23 s measured); the sim loop runs at 34-46 iterations/s, below the 60 Hz physics rate.
- **Not yet verified:** the planners as a large batch on the real simulator; touching cubes; random layouts in bulk; Nemotron on Nebius Token Factory.

---

## 9. Tests (offline: no simulator, arm or network)

```bash
python3 workspace/scripts/test_stack_planner.py [layouts]   # 48 named scenarios + random layouts + safe-zone rules
.venv/bin/python workspace/scripts/test_stack_agent.py      # the real agent loop against a fake simulator
.venv/bin/python workspace/scripts/test_stepwise_agent.py   # needs a Nemotron endpoint: step-at-a-time planner through the agent loop
```

Last results: 12,000 layout-surface combinations with 0 failures; agent test 47/47; stepwise test 4 cases. Blinding the planner to covered cubes makes 12 of 48 named scenarios fail, so the tests do detect planner bugs. They show the planning logic is right; they cannot show how the physical arm behaves.

---

## 10. Troubleshooting

- **Sim slow / a pick takes minutes:** keep the laptop plugged in. `curl -s localhost:8100/state` shows `fps` (expect 34-46).
- **Sim hangs at start-up with an empty log** (about one start in five): `docker rm -f physyk-isaac-sim` and run `./run.sh --sim` again.
- **Docker: "nvidia-container-cli: driver not loaded":** the running kernel has no NVIDIA module; install the matching `linux-modules-nvidia-...` or boot the previous kernel.
- **Cosmos won't start / pill red:** `docker logs physyk-cosmos`; usually GPU memory (stop the sim first or lower `COSMOS_GPU_UTIL`).
- **Nemotron errors / "planning failed":** check `.env` and run `python tools/check_services.py`. The hosted API rate-limits bursts (HTTP 429, retried with back-off).
- **"... cannot be reached" / "too close to the robot base":** use Randomize or Full Reset.
- **"... is not something I can see right now":** the cube is not on the table (or not visible in camera mode). Nothing moves.
- **Camera-only heights a centimetre off:** `curl -s localhost:8100/perception` and read `table_calibration.n` (tens of thousands healthy; about a thousand = table fit failing).
- **"Plan stopped by hard reset":** somebody clicked Hard Reset in a browser with the GUI open.

---

## 11. What is new for this hackathon

The project existed before the submission period. The history, so judges can separate old from new:

**Existing before the submission period** (the full cube/tray history is on this repository's `master` branch; this branch's own history is the earlier glassware-lab work from `main`, which this branch replaces with the cube/tray project):
- 22 August 2026: first version — Isaac Sim scene, Franka Panda with NVIDIA's `PickPlaceController`, Nemotron decomposing a sentence into steps, guardrails, verification, Cosmos shadow checks.
- September – early October: camera-only perception (colour + depth, about 3 mm mean error), faster picks (about 127 s → 25 s), stack planning.
- 14 October: restart from the 22 August code, adding the local stack planner, factory backdrop and restyled GUI (commits `1e86416`, `4e7e036`).
- 15 October: step-at-a-time Nemotron planning with code-tracked milestones, stacked-cube perception fix, cleaner overlay, UI colours and About dialog (commit `d149bab`).

**Added on the `nebius-nvidia-hackathon` branch:**
- Planner endpoint configuration for Nebius Token Factory (`.env.example`, section 5) and the `tools/check_services.py` connectivity check.
- Configurable ports, endpoints, timeouts and token limits through `.env` (`physyk_config.py`), and Cosmos client / step-verifier hardening for a Cosmos server on another machine.
- `./run.sh` now also starts the sim (`--sim`) and Cosmos (`--cosmos`) containers, replacing the separate launcher scripts.
- MIT `LICENSE`; the documentation consolidated into this README and `ARCHITECTURE.md`.

---

## 12. Repository layout

| Path | Contents |
|---|---|
| `workspace/isaac_sim_service.py` | The simulator program: scene, robot, cameras, controller, perception wiring, HTTP API :8100 |
| `workspace/physyk_agent_orchestrator.py` | The agent: planning dispatch, guardrails, dispatch / verify / retry, Cosmos checks, final answer |
| `workspace/nemotron_planner.py` | Step-at-a-time Nemotron planner: prompt, scene-as-relations, milestones, streaming |
| `workspace/stack_planner.py` | Local stack planner and shared safe-zone rules |
| `workspace/physyk_perception/` | Colour/depth detector, perception provider, overlay, Cosmos client, step verifier |
| `workspace/physyk_main_server.py` | GUI (one HTML page) and gateway :7860 |
| `workspace/factory_background.py` | Visual-only factory backdrop |
| `workspace/reasoning_dashboard.py` | Nemotron / Cosmos / Perception log pages |
| `workspace/physyk_config.py` | Endpoints, keys, model names, ports, timeouts (from `.env`) |
| `workspace/scripts/` | The offline tests |
| `run.sh` | The one launcher: backend + GUI, `--sim`, `--cosmos`, `--status`, `--stop`, `--no-agent` |
| `tools/check_services.py` | Connectivity check for the Nemotron and Cosmos endpoints (no sim needed) |
| `requirements.txt`, `.env.example`, `LICENSE` | Dependencies, settings template, MIT license |

`ARCHITECTURE.md` explains how the parts fit, with the measurements and open problems. Never commit `.env` (it holds API keys).
