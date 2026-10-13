---
title: Physyk Technical Deep Dive
---

# Physyk — Technical Deep Dive

Reference doc for demo/Q&A prep. Goes one level deeper than `ARCHITECTURE.md` on how each
piece actually works: the pick-and-place execution primitive, inverse kinematics and collision
avoidance, GR00T-N1.7 VLA (**now parked**), Nemotron orchestration, Cosmos perception, and the
camera-only pure-perception pipeline. Every claim below traces to the codebase — nothing here is
aspirational. (Line numbers drift as the code changes; the file and function names are the stable pointers.)

For the service topology (local Docker sim, local Cosmos NIM, hosted Nemotron; ports; request lifecycle) see
`ARCHITECTURE.md` §1 and §4 — not repeated here. For the camera-only pipeline in depth see `PERCEPTION.md`.

---

## 1. Pick-and-place execution — the physical primitive

### 1.1 The 10-phase grasp state machine

Every pick-and-place task, whether dispatched directly or via the Nemotron orchestrator, ends
up calling NVIDIA's own validated `PickPlaceController` (`isaacsim.robot.manipulators`), driven
once per physics frame from `isaac_sim_service.py`'s sim loop:

```mermaid
flowchart LR
    P0["0 Moving Above Target"] --> P1["1 Descending to Grasp"]
    P1 --> P2["2 Settling Before Grasp"]
    P2 --> P3["3 Closing Gripper"]
    P3 --> P4["4 Lifting\n(grasp-confirmed here)"]
    P4 --> P5["5 Transporting to Place XY"]
    P5 --> P6["6 Descending to Place Height"]
    P6 --> P7["7 Releasing Gripper"]
    P7 --> P8["8 Ascending"]
    P8 --> P9["9 Returning to Staging"]
    P9 --> V["Real position verification\n(live telemetry vs target)"]
```

Two engineering details worth knowing cold for Q&A:

- **Grasp confirmation is physical, not assumed.** At the phase-3→4 transition, the code reads
  the gripper's actual achieved joint width and checks it against `GRASP_MIN_WIDTH_M` (a
  gripper that closed on nothing settles near its mechanical minimum, not near a cube's real
  width). A failed grasp **early-aborts** the task instead of running phases 5-9 with nothing
  held — this closes a real bug where a failed grasp used to silently run the whole ~30-40s
  motion before anyone found out.
- **A second, related bug found and fixed this session**: `PickPlaceController.reset()` only
  resets the state machine's phase/timer — it never physically re-opens the gripper. A failed
  grasp (which early-aborts before reaching the release phase) left the jaws physically closed,
  so the *next* task began with an already-shut gripper — guaranteed to fail the same way,
  producing visible failure streaks. Fixed by explicitly calling `robot.gripper.open()` at
  every task-start and every abort/reset path.

### 1.2 Inverse kinematics — what's actually used (and what isn't)

The file initializes **two separate IK-capable systems**, but only one is actually load-bearing:

- **`LulaKinematicsSolver`** (`isaac_sim_service.py:715-717`) — NVIDIA's Lula IK engine,
  loaded via `load_supported_lula_kinematics_solver_config("Franka")` (NVIDIA's own bundled
  Franka config, not a custom URDF). **This solver is instantiated and then never referenced
  again anywhere in the file.** It is dead code today — evaluated, wired up, not used for any
  actual motion.
- **RMPFlow**, via `PickPlaceController._cspace_controller` — this is what actually computes
  every joint target in the system, for both the scripted path and the VLA path (the VLA loop
  calls `pick_place_controller._cspace_controller.forward(...)` directly, bypassing the phase
  state machine but still using the same RMPFlow instance).

**Good, honest Q&A answer**: *"We evaluated NVIDIA's Lula kinematics solver and RMPFlow side by
side; RMPFlow's built-in collision-aware planning made it the right default for everything, so
Lula's solver is initialized but not currently wired into the motion path."*

### 1.3 Collision avoidance — RMPFlow obstacle registration

RMPFlow does not automatically know about scene geometry — every obstacle has to be explicitly
registered (`isaac_sim_service.py:769-775`):

```python
rmpflow_motion_policy.add_obstacle(work_table_obj, static=True)
rmpflow_motion_policy.add_obstacle(target_tray_obj, static=True)
for cube_obj in CUBE_OBJECTS.values():
    rmpflow_motion_policy.add_obstacle(cube_obj, static=False)
```

Table and tray are **permanent static obstacles**. All three cubes are **dynamic obstacles** —
critically, this means the arm routes *around* whichever cubes it isn't currently interacting
with. The cube currently being picked (and, for a "stack on X" task, the reference cube being
stacked onto) is **temporarily disabled** as an obstacle for the duration of that one task via
`disable_obstacle`/`enable_obstacle` calls, then re-enabled once the task completes — otherwise
RMPFlow would refuse to let the gripper approach and enclose its own pick target.

This obstacle-registration step didn't always exist — an earlier version of the system had zero
registered obstacles, meaning the arm could clip straight through the table or other cubes; this
was one of the first real physics bugs found and fixed in the project's history.

---

## 2. GR00T-N1.7 — Vision-Language-Action model

> **Status: parked.** Not started by `run.sh`, no GUI control, `vla:` prompts are rejected by the backend.
> Everything below documents the integration as built; the code and fine-tuning tooling stay in the repo.

### 2.1 Architecture (from the loaded checkpoint's own `config.json`)

| Component | Value |
|---|---|
| Model type | `Gr00tN1d7` |
| Vision-language backbone | `nvidia/Cosmos-Reason2-2B` (Qwen3-VL architecture) |
| Backbone layers used | 16 (`select_layer: 16`) |
| Backbone embedding dim | 2048 |
| Action head | Flow-matching diffusion transformer (DiT) |
| Action head layers / heads | 32 layers, 32 attention heads, head dim 48 |
| Diffusion inference steps | 4 (`num_inference_timesteps: 4`) |
| Max state / action dim | 132 |
| Model action horizon (architecture cap) | 40 |
| Image processing | 256×256 target, 230×230 crop, 0.95 crop fraction |
| Precision | bfloat16 |

**A note on precision worth having ready**: NVIDIA's own Isaac-GR00T repo README states N1.7's
default diffusion head as 16 layers (down from 32 in N1.6) and `select_layer: 12` — but this
project's actual loaded checkpoint (`GR00T-N1.7-LIBERO`) shows `num_layers: 32` and
`select_layer: 16` in its own `config.json`. Rather than guess which is "right," both numbers
are cited here — the checkpoint's own file is authoritative for what's actually running, and
likely reflects an earlier fine-tune snapshot from before those defaults changed upstream.

GR00T N1.7's real architectural headline (per the repo README) is a **relative end-effector
action space** shared across robot and human demonstration data — deltas from the current pose
rather than absolute targets, which is the mechanism that lets the same model architecture
pretrain on ~20K hours of human egocentric video and transfer to robot control.

### 2.2 What this project actually sends the model

Only two RGB camera frames, proprioception, and language — **no depth**, confirmed at the exact
call site (`isaac_sim_service.py`'s `query_groot_server`):

```python
payload = {
    "instruction": instruction,
    "scene_image_b64": ...,   # scene/front camera, JPEG
    "wrist_image_b64": ...,   # wrist camera, JPEG
    "ee_pose": [x, y, z, roll, pitch, yaw],
    "gripper_width": ...,
}
```

This isn't a plumbing gap — the checkpoint's own `LIBERO_PANDA` modality config has no depth
channel defined at all. GR00T-N1.7-LIBERO is, by design, a pure RGB+proprioception+language
model here. (The project does have a real depth channel — the overview camera's depth image, used by the
camera-only perception pipeline in §4 and `PERCEPTION.md` — but it isn't fed into GR00T.)

### 2.3 From prediction to motion — the exact chain

```mermaid
flowchart LR
    A["Camera frames + EE pose"] --> B["groot_policy_service.py\nformat_observation()"]
    B --> C["Gr00tPolicy.predict_action_chunk()\n(16, 7) delta array"]
    C --> D["isaac_sim_service.py sim loop:\ntarget_xyz += delta[:3]\n(accumulated per step)"]
    D --> E["pick_place_controller._cspace_controller.forward(\ntarget_end_effector_position=target_xyz)"]
    E --> F["RMPFlow → joint targets → PhysX"]
    F --> G{"16 steps done?"}
    G -- no --> D
    G -- yes --> H["Re-observe: new camera frames\n(closed loop)"]
    H --> A
```

Up to **4 chunks per task** (64 total delta-steps), each chunk followed by a fresh
observation — genuinely closed-loop, not a single blind prediction. This is opt-in via a
`vla:` instruction prefix; the scripted `PickPlaceController` path (§1) remains the default,
reliable executor.

### 2.4 This project's fine-tuning pipeline (built this session)

```mermaid
flowchart LR
    A["groot_episode_recorder.py\n(hooked into sim loop,\nRECORD_EPISODES=1 flag)"] --> B["raw episodes:\nscene.mp4 + wrist.mp4 +\nframes.jsonl per episode"]
    B --> C["convert_raw_episodes_to_lerobot.py\n(LeRobot v2 format,\nGR00T-compatible schema)"]
    C --> D["gr00t/data/stats.py\n(normalization stats)"]
    D --> E["launch_finetune.py\n(default mode: projector +\ndiffusion head only, backbone frozen)"]
    E --> F["checkpoint-N/\n→ GROOT_MODEL_PATH →\ngroot_policy_service.py"]
```

The recorder is a genuinely non-invasive hook — 4 small additions to `isaac_sim_service.py`,
gated behind an env flag, using data (raw camera frames, `sim_telemetry["ee_pos"]`/`gripper`)
that already existed in the sim loop. Only **successful** episodes (per the same real
verification described in §1.1) are written to disk — failed attempts are silently dropped, so
the dataset never needs a separate filtering pass.

**Real runs this session, honestly reported:**

| Dataset | Episodes | Steps | Train loss | Closed-loop result |
|---|---|---|---|---|
| `physyk_cubes_17` | 17 (mixed 3-color) | 1000 | ~0.04 (exact log not retained) | Moves, doesn't converge — near-identical deltas every chunk |
| `physyk_red_21` | 21 (red-only) | 1000 | 0.0425 | Same symptom, marginally different deltas |
| `physyk_red_54` | 54 (red-only, current default) | 1500 | 0.0402 | Deltas shrink chunk-over-chunk (real signal) but still doesn't reliably grasp |

**The honest finding worth stating directly in Q&A**: training loss dropping to ~0.04 does
**not** mean closed-loop task success. That number only measures "did the model predict
something close to what the demonstration did at that exact frame" — it says nothing about
whether the robot completes the task when run in a real feedback loop, where small per-step
errors compound (a well-known imitation-learning failure mode called distribution shift). With
54 episodes that are all structurally near-identical (same cube, same tray, same camera, only
start position varying), the model shows real signs of learning trajectory *shape* without yet
learning to condition tightly enough on *where the cube actually is*. The next lever, based on
this evidence, is more positional/visual diversity in the data — not just more of the same.

---

## 3. Nemotron — agentic orchestration

*(Full architecture and diagrams: `ARCHITECTURE.md` §3-4. This section goes one level deeper
on the actual prompts/schemas.)*

**Serving**: NVIDIA's hosted API (`https://integrate.api.nvidia.com/v1`), model
`nvidia/nemotron-3-super-120b-a12b`, configured in `physyk_config.py` / `.env`. (The earlier setup served
Nemotron-30B-FP8 locally through vLLM on port 8000; that is retired. A local OpenAI-compatible server still works by
setting `NEMOTRON_BASE_URL`.) The hosted model returns its chain-of-thought in a separate `reasoning_content`
field rather than inline before `</think>`; `_message_text()` in the orchestrator folds it back into the
`reasoning</think>answer` shape the parsers and the reasoning log expect. Calls retry on 429/5xx; typical
latency is ~1 s for a one-step plan and ~6-7 s for a multi-step plan.

### 3.1 Decomposition — exact schema

The orchestrator's system prompt (`physyk_agent_orchestrator.py:342-386`) instructs Nemotron to
output an ordered JSON array where every element is exactly one of:

```json
{"action": "pick_place", "target": "<object name>", "destination": "<free text or empty>"}
{"action": "home"}
{"action": "reset"}
```

Key rules baked into the prompt: never invent objects not in the live scene description; one
subgoal per "and/then/after that" clause; but a single-object pick-and-place-in-the-same-clause
stays **one** subgoal, not two; never invent a destination the user didn't state
(`destination: ""`).

### 3.2 Guardrails — deterministic, not LLM-judged

`PhysicalWorkspaceRails` (`physyk_agent_orchestrator.py:85-102`) rejects any Cartesian target
outside a fixed box **before dispatch**, in code, with no LLM in the loop:

```
x: [0.20, 0.70]   y: [-0.45, 0.45]   z: [0.02, 0.50]   (meters)
```

Beyond the workspace box, several other checks are pure code, deliberately not LLM calls, for
speed and determinism:

- `_subgoal_already_satisfied` — is the goal already true (avoids re-picking an already-placed
  cube on a re-run of a compound instruction).
- `_reference_out_of_tray` / `_subgoal_drift_detected` — did a stacking reference object
  physically move or leave the tray since an earlier step verified it, before dispatching the
  next step that depends on it.

**Guardrail retry (camera-derived scenes only).** In pure-perception mode the scene's positions are
measured, so a value can sit right on a limit and fail once (a resting cube read Z = 0.0196 against a 0.02
floor) and pass on the next look. `run()` re-observes and re-validates up to `GUARDRAIL_RETRIES` = 3 times,
`GUARDRAIL_RETRY_DELAY_S` = 0.8 s apart, before rejecting; it is skipped when the scene is the deterministic one.
(The height bias that caused this in the first place was fixed in the detector — see `PERCEPTION.md`.)

### 3.3 Failure handling — the 3-way decision

On a verification failure, `request_failure_decision` posts the real error (`error_mm`),
completed/remaining subgoals, and the live scene to Nemotron, which must return exactly one of:

| Decision | Effect |
|---|---|
| `retry` | One-off physical miss — redispatch the same subgoal |
| `replan_step` | Replace just the failed subgoal's destination, keep the rest |
| `replan_remaining` | Regenerate the entire remaining plan |

Capped at once per run to prevent a runaway replan loop. LLM JSON output is parsed with a
custom balanced-brace extractor (`_extract_json_objects`) — built specifically because a naive
regex broke on nested JSON objects in real Nemotron output.

```mermaid
sequenceDiagram
    participant U as Instruction
    participant N as Nemotron (decompose)
    participant G as Guardrails (code)
    participant S as Isaac Sim
    participant V as Verify (real telemetry)
    U->>N: instruction + live scene
    N-->>U: ordered subgoal array
    loop each subgoal
        U->>G: validate_cartesian_target()
        G-->>U: pass / reject
        U->>S: dispatch
        S-->>V: task complete
        V-->>U: success / fail (real position check)
        alt fail, attempts remain
            U->>N: request_failure_decision(error_mm, scene)
            N-->>U: retry / replan_step / replan_remaining
        end
    end
    U->>U: deterministic final-goal recheck (no LLM call)
```

---

## 4. Cosmos and perception

**Cosmos 3 Nano Reasoner (8B)** runs locally as a NIM container (`nvcr.io/nim/nvidia/cosmos3-reasoner`) on port
8001 — it is not available on the hosted API. It shares the 24 GB laptop GPU with Isaac Sim, so
`start_cosmos_docker.sh` runs it lean: the FP8 build (~10 GB weights, ~16 GB total VRAM), an 8192-token
context, a 65% GPU share, and CUDA graphs off (the defaults — bf16, 262k context, CUDA graphs — crash on this GPU).
Cosmos is **evidence only**: it never decides what the robot does. (The original build used Cosmos-Reason2-2B
through vLLM; the client code is model-agnostic and parses `<think>/<answer>` and `reasoning_content` outputs.)

### 4.1 Perception modes

- **`stage` / `shadow`** (default; GUI switch off) — the simulator's ground-truth poses drive the robot.
  `shadow` additionally runs the camera detector and logs the delta.
- **`vision`** — legacy hybrid: the camera pose drives IK, gated by an 8 cm sanity check against the stage pose with
  fallback to it. Not used by the GUI.
- **`pure`** (GUI switch on: "Pure perception — camera only") — camera only, no simulator coordinates, no fallback;
  the tray is a fixed constant; verification is a fresh camera look. The detector is **color segmentation + depth**
  (`color_segment.py`), not Cosmos: ~20 ms, no GPU, ~3 mm mean error. Cosmos's own grounding (a VLM returning
  pixel points) measured within 2-3 px on a test frame but takes 1.5-3 s per call and returns a point rather than a
  mask, so it stays a semantic layer. Full write-up, measurements and the debugging story: `PERCEPTION.md`.

### 4.2 Real Cosmos call sites

- **Pre-dispatch visual assessment** (`_attach_cosmos_assessment`, `physyk_agent_orchestrator.py`) — before a subgoal
  is dispatched, Cosmos assesses feasibility (target/destination visible, obstacle, approach) from the clean overview
  frame. Informational only.
- **Post-dispatch visual verification** (`_attach_visual_evidence`) — after the motion, Cosmos is asked whether the
  outcome is visually true ("is the blue cube on top of the red cube?") as an independent check next to the real
  position-based verification. Observed: `verified=True`, confidence 0.99 on the stacking run.

Both write to the Cosmos log (`/reasoning/cosmos`) **with the exact image Cosmos was shown**. Cosmos receives the clean
frame (`/camera/scene_raw.jpg`), never the perception overlay, so drawn labels can't bias it. A Cosmos failure or
timeout yields "NOT CHECKED" and never blocks a plan.

---

## 5. Seeing it live

| What | Where |
|---|---|
| Main GUI + live cameras | `http://localhost:7860` |
| Service status panel | Header of the main GUI (Isaac Sim / Nemotron / Cosmos pills) |
| Nemotron full reasoning log | `http://localhost:7860/reasoning/nemotron` |
| Cosmos log (with the images Cosmos saw) | `http://localhost:7860/reasoning/cosmos` |
| Perception log (annotated frames, bounding boxes, XYZ) | `http://localhost:7860/reasoning/perception` |
| Perception snapshot / mode | `http://localhost:8100/perception` (mode, per-object camera XYZ, scoring delta) |
| Start the sim / everything else | `./start_sim_docker.sh`, then `./run.sh` (Cosmos: `./start_cosmos_docker.sh`) |
| Scripted-path-only (no Nemotron) | `./run.sh --no-agent` |
| Live status check any time | `./run.sh --status` |
