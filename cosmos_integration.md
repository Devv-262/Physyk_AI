# Cosmos in Physyk

Cosmos 3 Nano Reasoner is the project's **vision-language checker**: it looks at the overview camera and answers questions
about the scene. It is **evidence only** — it never decides what the robot does, and a slow, wrong or offline Cosmos cannot
block or change a plan.

## Deployment

| Item | Value |
|---|---|
| Model | `nvidia/cosmos3-nano-reasoner` (8B), served as an NVIDIA NIM container |
| Image | `nvcr.io/nim/nvidia/cosmos3-reasoner:latest` |
| Start | `./start_cosmos_docker.sh` (container `physyk-cosmos`, port 8001; `run.sh` never starts or stops it) |
| Endpoint | `COSMOS_BASE_URL` = `http://localhost:8001/v1` (`workspace/physyk_config.py`) |
| Weights | downloaded on first start into `~/.cache/nim` (~10 GB for the FP8 build); the key in `.env` (`COSMOS_API_KEY`) is only the NGC key for that download |
| GPU share | ~16 GB total next to Isaac Sim's ~4 GB on a 24 GB GPU |

The launcher runs it lean because the defaults crash on this GPU: the FP8 profile, an 8192-token context (the default needs
36 GB of KV cache), `NIM_GPU_MEMORY_UTILIZATION` 0.65 (`COSMOS_GPU_UTIL` overrides it) and CUDA graphs off.

## Where it is used

`physyk_perception/cosmos_client.py` (`ask_visual_question`: one image plus a short prompt, temperature 0, 30 s timeout) and
`physyk_perception/step_verifier.py`, called from `physyk_agent_orchestrator.py`:

| Call | When | Asks | Used for |
|---|---|---|---|
| `assess_pick_place` | before a step is dispatched | target / destination visible, obstacle, feasibility (high/medium/low), approach | the "Cosmos Reasoner" panel on the plan card; informational |
| `verify_step` | after the motion | "is the X now at/on Y?" with a confidence | an independent second check next to the position check; a failed check is shown as evidence |

Both receive the clean overview frame (`/camera/scene_raw.jpg`), never the frame with the perception overlay, so drawn labels
cannot bias the model. Every call is logged with the exact image Cosmos saw at `http://localhost:7860/reasoning/cosmos`. An
error or timeout produces "NOT CHECKED" and the plan carries on.

## The prompt needs to describe the scene

The overview is a wide shot — a cube is ~20 px across — and the "tray" is a flat yellow square slab. Asked about "the tray",
Cosmos used to answer that the tray was not visible and rate the step low feasibility. Both prompts now start with a scene key:
the yellow slab IS the tray; there are up to three small 5 cm cubes (red, green, blue) that may be stacked; "on top of a cube"
means resting on its top face. The assessment also defines the rating: *high* when the target is visible with a free top face
and the destination is reachable, *medium* when it is tight, *low* only when the target is covered, blocked or not in the image.

Result on 6 live questions about a flat layout: all *high*, where the tray questions had been *low*. Not yet checked: that it
still answers *low* for a cube that really is covered.

## Troubleshooting

- **Cosmos won't start** — see `logs/cosmos.log`; usually GPU memory (stop the sim first, or lower `COSMOS_GPU_UTIL`).
- **The Cosmos pill is OFFLINE** — `curl -s localhost:8001/v1/models`; the NIM takes a minute to load.
- **Answers are empty** — the call timed out (the model reasons before answering); the step shows "NOT CHECKED".
- Camera-only perception does **not** use Cosmos: it uses colour masks and depth (`PERCEPTION.md`). Cosmos's own grounding
  (`ground_objects`) takes 1.5-3 s per call and returns a point rather than a mask, so it is only a semantic layer.
