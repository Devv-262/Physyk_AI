# Physyk — Full Demo-Day Context Briefing

Written by reading the actual running code (`physyk_agent_orchestrator.py`, `isaac_sim_service.py`,
`groot_policy_service.py`, `nemotron_fastapi_server.py`) and live process/GPU state
(`ps aux`, `nvidia-smi`) — not just the project's own docs. No code was changed to produce this.
Every jargon term (RMPFlow, MoE, KV cache, WebRTC, LIBERO, VLA, DiT...) is explained plainly the
first time it's used, because the previous pass of this briefing was flagged as too vague.

---

## Part 1 — The five moving pieces, and what each one literally is

Five separate OS processes, all on one GPU, talking over plain `http://localhost:<port>` —
confirmed live via `ps aux`:

| # | Process | Port | What it is, concretely |
|---|---|---|---|
| 1 | `physyk_main_server.py` | 7860 | A plain Python (no GPU) FastAPI server. Serves the browser GUI, relays WebSocket log/plan updates, proxies camera JPEGs. Does no AI itself — it's the switchboard. |
| 2 | `isaac_sim_service.py` | 8100 | Runs *inside* Isaac Sim's own bundled Python (`/opt/dlami/nvme/isaac-sim/kit/python/bin/python3`). This is the actual physics simulation: PhysX (rigid body physics engine), RTX rendering, the Franka Panda robot, RMPFlow motion planning, the `PickPlaceController`. One single-threaded loop processes one command at a time from a queue. |
| 3 | `nemotron_fastapi_server.py` | 8000 | A real `vllm.entrypoints.openai.api_server` process. vLLM is NVIDIA/UC-Berkeley's high-throughput LLM serving engine. It has the actual Nemotron model weights loaded in GPU memory and exposes them as an OpenAI-style chat API. |
| 4 | Cosmos Reason 2 server | 8001 | Also a real vLLM process, loaded with `nvidia/Cosmos-Reason2-2B` — a separate, smaller vision-language model (it can read images, not just text). |
| 5 | `groot_policy_service.py` | 8300 | A FastAPI server in its **own separate Python virtual environment** (`Isaac-GR00T/.venv`), because GR00T needs a different, incompatible set of library versions than Isaac Sim's bundled Python has. Loads NVIDIA's GR00T-N1.7 robot-action model. |

**GPU right now** (`nvidia-smi`, live): NVIDIA RTX PRO 6000 Blackwell, 96 GB total VRAM, 94% GPU
utilization, all five processes' GPU-resident pieces running concurrently.

**Why so many separate processes instead of one program?** Each of these needs a different,
often conflicting, software environment (different PyTorch/CUDA/transformers versions) and each
one, if it crashed, shouldn't take the whole demo down with it — Isaac Sim continues running
even if, say, the Cosmos server needs a restart. That's the actual reason for the HTTP-between-
processes architecture, not a stylistic choice.

---

## Part 2 — What each of the three "brains" is actually doing, step by step

### 2a. Nemotron — the planner (turns English into a step list)

This lives in `physyk_agent_orchestrator.py`. Concretely, when you type an instruction:

1. The orchestrator fetches the **live scene** from Isaac Sim (`GET /state`) — real object
   names and X/Y/Z positions, queried fresh, not cached.
2. It builds a prompt (`NEMOTRON_SYSTEM_PROMPT`, verbatim in the code) that says, roughly:
   *"You are the cognitive planner for a real Franka Panda arm. Decompose the user's
   instruction into a JSON array of subgoals, each being `pick_place` (with a target and a
   destination), `home`, or `reset`. Only reference objects that are actually in the scene
   below. If one clause says pick-and-place the same object, that's ONE subgoal, not two."*
   The scene listing (from step 1) is embedded directly into this prompt.
3. It POSTs that to `http://localhost:8000/v1/chat/completions` with `temperature=0.0`
   (deterministic — same input, same output, no creative randomness — important for a
   demoable, repeatable robot).
4. **Nemotron is a "reasoning model"** — meaning before it gives its real answer, it writes out
   a chain-of-thought (its own step-by-step reasoning), which ends with a literal `</think>`
   marker, followed by the actual answer. The code explicitly strips everything before
   `</think>` and only parses JSON from what comes after — this was a real, previously-hit bug:
   at a smaller token budget (1536), the model would spend its *entire* budget on the
   chain-of-thought and get cut off before ever writing the real JSON answer, silently
   degrading every complex prompt into a single blind fallback action. Fixed by raising the
   completion budget to 4096 tokens (and the server's own context window to 12288).
5. The parsed JSON array is the plan — e.g. for *"put the red cube in the tray, then stack the
   blue cube on top of it"*, Nemotron returns:
   ```json
   [
     {"action": "pick_place", "target": "red cube", "destination": "tray"},
     {"action": "pick_place", "target": "blue cube", "destination": "red cube"}
   ]
   ```
6. If Nemotron is unreachable or returns something unparsable, the system does **not** crash or
   hang — it falls back to dispatching your instruction as one single literal command (the
   pre-agentic baseline behavior), so a dead LLM never bricks the whole robot.

Nemotron is called **again**, separately, only when a step actually fails verification — see
Part 3's failure-decision prompt. This is a deliberate "minimize LLM calls" design: trivial
state checks (is this already done? has something drifted?) are done with plain position math
in Python, not by asking the LLM — only real judgment calls go to Nemotron.

### 2b. Isaac Sim / PickPlaceController — the body (turns one subgoal into real robot motion)

Each subgoal becomes a plain-English sentence again (e.g. `"pick up the blue cube and place it
red cube"` → normalized to `"pick up the blue cube and place it on top of the red cube"`) and is
POSTed to Isaac Sim's `/execute`. Inside `isaac_sim_service.py`:

1. **Target resolution** (`resolve_pick_place_targets`) — keyword-matches which object is being
   picked and which is the destination reference, purely from where the words sit in the
   sentence (no LLM here — plain string logic). If a spatial phrase like "right of" or "on top
   of" is present, the object named *after* it is the reference; the other one is the pick
   target.
2. **Where things actually are** — the picking position is the target's **live, current**
   position queried straight from the simulator's own internal scene graph (called "the USD
   stage" — USD = Universal Scene Description, Pixar's/NVIDIA's format for representing a 3D
   scene) — not a cached or assumed position. This matters e.g. when picking a cube that's
   already been stacked on another one; its Z-height only exists in the live state.
3. **Motion planning** — **RMPFlow** (Riemannian Motion Policy Flow) is NVIDIA's real-time,
   collision-aware motion generator: given a target pose and a list of registered obstacles
   (table, tray, other cubes), it continuously computes joint velocities that move the
   end-effector toward the goal while steering around the obstacles — this is what makes the
   arm route *around* a cube instead of clipping through it. The specific object being grasped
   is temporarily un-registered as an obstacle for that task (so RMPFlow doesn't treat the very
   thing it's reaching for as something to avoid), then re-registered afterward.
4. **The `PickPlaceController`** — this is NVIDIA's own pre-built, validated state machine (not
   something built from scratch here) that runs 10 phases per pick-and-place: move above
   target → descend → settle → close gripper → lift → transport → descend to place → release →
   ascend → return to staging. Every frame, it commands real joint torques through PhysX's PD
   controllers (a PD controller drives a joint toward a target position/velocity the way a real
   motor controller would, respecting mass/inertia/friction) — **not** `set_joint_positions()`,
   which would just teleport the joints to a value regardless of physics (that was a real bug,
   fixed).
5. **Real post-place verification** — after the state machine reports done, the target's final
   live position is re-queried and compared to the intended target: XY error checked against
   `POSITION_XY_TOLERANCE_M`, Z error against `POSITION_Z_TOLERANCE_M`. This produces
   `last_task_success` / `last_task_error_mm` — the actual pass/fail signal the whole retry
   loop above depends on. This used to be a hardcoded `0.0` / always-pass — now it's a real
   number from a real comparison.

### 2c. Cosmos Reason 2 — the eyes (perception + visual verification, separate job from Nemotron)

Cosmos is **not** in the planning decision loop — it doesn't decide what to do next. It's a
separate module (`physyk_perception/`) with two distinct jobs, both visible directly in
`physyk_agent_orchestrator.py`:

- **`_attach_cosmos_assessment` (pre-dispatch)** — before a step is dispatched, sends the
  current camera frame to Cosmos with a prompt asking: is the target visible, is the
  destination visible, is there an obstacle, is this feasible, what approach direction looks
  right? This is purely informational (shown in the UI), it never blocks or changes dispatch —
  a deliberate safety choice so a slow/wrong Cosmos read can never stall the robot.
- **`_attach_visual_evidence` (post-motion)** — after a step's motion finishes, sends a new
  camera frame with a question like *"is the green cube resting on the red cube?"* and gets
  back a verified/not-verified verdict plus a plain-English observed sentence and a confidence
  score. This turns a step marked DONE from "the position math says so" into "the position
  math says so, **and** a vision model looked at a picture and agrees" — a second, independent
  channel of evidence, not a replacement for the real position check.
- **Failure handling is asymmetric on purpose**: if the position check already failed, a
  disagreeing Cosmos read is just extra logged evidence — it never overrides. If Cosmos itself
  errors out or times out, the verdict defaults to `verified=True, enforced=False` (shown as
  "NOT CHECKED") — so a dead Cosmos server can never deadlock the demo, only silently lose the
  bonus evidence.

**A detail worth knowing cold for judge Q&A**: `groot_policy_service.py`'s own docstring
describes GR00T-N1.7 as *"Cosmos-Reason2-2B VLM backbone + DiT action head."* That means Cosmos
Reason 2 isn't only a bolt-on perception service in this project — it is architecturally **the
vision component GR00T itself is built from**. GR00T looks at the scene using a Cosmos-family
vision-language backbone, then a separate "DiT action head" (DiT = Diffusion Transformer — a
neural network architecture, originally from image generation, adapted here to generate a
short sequence of robot actions instead of an image) turns that visual understanding into
motion. So Nemotron, Cosmos, and GR00T aren't three unrelated NVIDIA products glued together —
Cosmos genuinely underlies two of the three layers.

---

## Part 3 — The full closed-loop lifecycle for one compound instruction

Traced directly from `NeMoRoboticsAgent.run()`:

```
1. instruction arrives → fetch_live_scene() (real object positions)
2. decompose_task_with_reasoning() → Nemotron returns ordered subgoal list + its own
   chain-of-thought (shown in the UI so you can see WHY it split the instruction that way)
3. for each subgoal, in order:
   a. GUARDRAIL: validate_subgoal() — PhysicalWorkspaceRails checks the resolved target/
      destination XYZ against the arm's real reachable envelope:
        x: [0.20, 0.70] m   y: [-0.45, 0.45] m   z: [0.02, 0.50] m
      (these bounds are shared with GR00T's own safety clamp, so both paths agree on
      "reachable" — not two different definitions of safe)
      → fails → step marked REJECTED, skipped, plan continues with the rest
   b. PRECONDITION CHECK (pure code, no LLM): is the target already at its destination?
      (e.g. re-running "stack red, green, blue" after red is already placed) → skip
      re-dispatch, mark DONE, still get a Cosmos visual check as evidence
   c. DRIFT CHECK (pure code, no LLM) — two independent sub-checks, run BEFORE dispatch:
      - absolute: if this step stacks on a reference that's supposed to be "in the tray,"
        is that reference actually, currently, in the tray region right now (not just
        "unchanged since last look")?
      - relative: has the reference object moved more than 3cm since an EARLIER step in
        THIS plan verified its placement? (something knocked it, e.g. a later grasp bumped it)
      → either one triggers a real Nemotron call BEFORE dispatch to decide what to do,
        rather than dispatching against a premise that's already known to be false
   d. DISPATCH: POST /execute with a unique request_id, then poll /state until Isaac Sim's
      own current_request_id echoes back that exact ID AND busy=True — this is what
      guarantees "the sim is now actually working on THIS command," not stale state from
      a previous or concurrent command (see Part 4 for why this needed three iterations)
   e. WAIT: poll /state until busy=False, 150s timeout. A timeout is ALWAYS counted as a
      failed attempt — never silently treated as "must have succeeded"
   f. VERIFY: read back last_task_success / last_task_error_mm — the real number from
      Part 2b step 5
   g. on FAILURE with attempts remaining (max 3 total): one Nemotron call
      (request_failure_decision) — given the real error in mm, the live scene, what's
      already completed, and what's still planned — returns exactly one of:
        retry            — try the identical subgoal again (one-off physical miss)
        replan_step      — propose a different destination for just this subgoal
        replan_remaining — propose a whole new remaining plan (something changed)
      replan_remaining can fire at most ONCE per run (a hard safety valve against an
      infinite replan loop) — a second request for it is downgraded to a plain retry
4. after all subgoals: a DETERMINISTIC final check (pure position math, explicitly NO extra
   LLM call) — re-compares every step that was marked DONE against ITS CURRENT live
   position, catching the case where step 3's stack accidentally knocked over step 1's
   already-verified placement
5. overall verdict: SUCCESS / GOAL_NOT_HELD (later step disturbed an earlier one) /
   PARTIAL / FAILED / REJECTED
```

Every state transition calls `on_plan_update()`, which pushes the updated plan over WebSocket
to the browser — this is what makes the "agent thinking" panel in the GUI live rather than a
replayed log.

---

## Part 4 — Design decisions that look arbitrary but each closed a real, reproduced bug

Straight from the code comments — these read as over-engineered until you see the specific bug
each one fixes:

- **`request_id`-based dispatch confirmation** is the *third* attempt. (1) No confirmation at
  all → could read leftover state from the *previous* command in the small gap before the sim's
  ~7-8 fps loop even dequeued the new one. (2) `busy=True` alone → also true while the sim was
  mid-task on someone *else's* concurrently-dispatched command. (3) `busy=True` + matching
  instruction text → broke on a byte-identical retry, which is structurally indistinguishable
  from a genuinely new dispatch of the same text. A unique ID per HTTP call has none of these
  three ambiguities.
- **Timeout is never treated as success.** Previously, a timed-out wait would fall through to
  reading `last_task_success`, which could be leftover state from whatever task last genuinely
  finished — possibly unrelated to the one that just timed out.
- **Two separate drift checks, not one** — an *absolute* check (is the reference actually where
  it should be, right now) and a *relative* check (has it moved since it was last verified) each
  independently catch a real failure mode the other one misses: a cube that fell out of the tray
  and just sat there wrong (nothing "moved" after that, so relative-only checking stays silent)
  versus a cube that was correctly placed but got bumped later (nothing was "wrong" about its
  destination text, so absolute-only checking stays silent).
- **Destination normalization bug** — a bare object name as a destination (e.g. "Red Cube"
  instead of "on top of the Red Cube") used to fall through to a generic tray-routing fallback
  in Isaac Sim's own keyword matcher, which actually knocked over an already-correct stack while
  trying to reposition a cube that was never meant to move. Fixed by canonicalizing an ambiguous
  bare-object destination to an explicit "on top of" phrase before it's ever dispatched.
- **Never invent a destination** — if the user only said "pick up the red cube" with no
  placement stated, the system used to default the empty destination to "tray" — silently
  fabricating an instruction the user never gave. Fixed to leave it truly empty (pick-only) when
  that's genuinely all that was asked.

---

## Part 5 — Your questions, answered precisely

### Q1: Why run Nemotron locally instead of just making API calls?
Three concrete, stacked reasons:
1. **Latency and reliability under a live demo.** A single compound instruction can trigger
   *multiple* Nemotron calls (one decomposition + up to one failure-diagnosis call per failed
   step, capped at 3 attempts/step). Each call is a real round trip; over conference wifi to a
   remote API, that's multiple points where the whole demo could stall waiting on a network hop
   it doesn't control. Local inference on the same box removes network variance entirely — the
   only latency is real GPU compute time.
2. **This is (per `PLAN.md`'s own framing) an NVIDIA-stack hackathon** — "NVIDIA NeMo Agent
   Toolkit / NeMo Guardrails + Nemotron-30B" is stated as the actual point of the submission.
   Standing up a real local vLLM server with real Nemotron weights loaded is itself part of what's
   being demonstrated — a hosted-API call wouldn't show that NVIDIA's own inference stack (vLLM,
   FP8 quantization, the model itself) is actually running.
3. **No usage-billing or availability risk mid-demo** — no per-token cost spike from heavy
   iteration, and no dependency on an external service being reachable/up during Q&A.
   Trade-off, stated plainly: local *isn't* strictly "better engineering" — it's the deliberate
   trade against how much simpler a hosted API call would have been to set up.

### Q2: Why so much VRAM constraint even though the Brev dashboard offered bigger GPU options?
The GPU is **not** small — confirmed live: RTX PRO 6000 Blackwell, 96 GB VRAM. The constraint
isn't capacity, it's how vLLM (the serving engine both Nemotron and Cosmos run on) claims memory:
- The flag `--gpu-memory-utilization` tells vLLM what **fraction of total GPU memory** to
  pre-reserve for its KV cache (KV cache = the running memory of "what the model has already
  read/generated in this conversation," reused across every request so the model doesn't
  recompute from scratch each token — this is *the* mechanism that makes an LLM server fast, and
  it needs a large pre-allocated memory pool to do it well).
- **The critical gotcha**: vLLM reserves that fraction **the instant it starts**, whether or not
  it's actually using it yet. `nvidia-smi`'s "free" memory number looks fine right up until a
  *second* vLLM process tries to start on the same card and OOMs (out-of-memory crashes) — because
  the first process's memory was never "free," just pre-reserved and idle.
- At vLLM's own **default** of `--gpu-memory-utilization 0.9`, Nemotron alone would claim ~86 GB
  of the 96 GB card — leaving no room for Cosmos-Reason2 (needs ~13 GB) or GR00T at all.
- **Confirmed live** (`ps aux`): Nemotron is actually launched with `--gpu-memory-utilization 0.55`
  (~53 GB budget), and Cosmos with `--gpu-memory-utilization 0.14` (~13 GB). This was a
  deliberate, explicit shrink from vLLM's own default, specifically so the two coexist.
So: **a bigger Brev GPU instance would not have solved this.** The problem isn't "not enough
total VRAM," it's "multiple independent vLLM processes, each defaulting to assume it owns the
whole card, sharing one card." You'd hit the exact same conversation on a 141 GB card once you
tried to add a fourth model, just later. The fix is (and was) capping each server's utilization
fraction explicitly, not buying more capacity.

### Q3: Why was Cosmos even needed?
Nemotron plans from **text**, not pixels — it reads a scene description built from the
simulator's own internal object-position database (the USD stage), which is privileged,
ground-truth information a real factory robot's camera would never have. That's a real gap in
the "this generalizes to a real robot" story. Cosmos closes two different halves of that gap:
1. **Perception** — instead of asking the simulator "where secretly is the red cube," look at
   a camera image, find the cube in the pixels, and (using the camera's depth data and known
   position) compute a real-world 3D coordinate from that. This is what "seeing" actually means
   here, versus privileged access.
2. **Verification** — instead of only trusting "the motion planner reported success," look at
   a *new* camera frame after the motion and ask a vision model whether the outcome you wanted
   is actually visible (e.g. "is the cube now resting on top of the other cube?").
It's integrated with a hard safety seam so it can never make the demo *worse*: a
`PERCEPTION_MODE` setting (`stage` / `shadow` / `vision`) lets Cosmos run *alongside* the
existing ground-truth path first — in `shadow` mode, the robot is still driven entirely by the
simulator's own known positions, while Cosmos's estimate is computed and logged next to it
purely to show how close it is (e.g. a documented ~6mm delta). Only in `vision` mode does
Cosmos's estimate actually drive the arm, and even then with an automatic fallback back to the
ground-truth position if Cosmos's read looks implausible.

### Q4: Why can't GR00T be perfect even for such small, simple-looking tasks?
The specific checkpoint deployed is **`GR00T-N1.7-LIBERO`** (the `libero_10` variant) — this was
fine-tuned by NVIDIA on **LIBERO**, a *different* published robot-manipulation benchmark, with
its own simulated tables, cubes, textures, lighting, and camera placement. It was never trained
or fine-tuned on **Physyk's own scene** at all. This is a textbook case of **domain shift**: a
vision-conditioned policy's sense of "what a graspable cube looks like from this angle" and
"what a successful grasp trajectory looks like" is learned entirely from what it was shown during
training — and it was shown a different simulator's rendering, not this one.
This was independently verified (not assumed): the team fed the model three different synthetic
images and got three meaningfully different predicted actions — proving the model genuinely
*looks* at its input rather than ignoring it (ruling out "it's secretly a stub"). But "does real
inference" and "is competent at this specific task" are two separate claims, and only the first
one holds. The project's own diagnosis: it "tends to repeat a similar motion each step rather
than visually correcting itself" — consistent with a model pattern-matching toward
out-of-domain visual features it half-recognizes, rather than genuinely understanding this
particular scene.

### Q5: What would have worked better than GR00T here? How would it have worked? Why not use that instead?
Given the real constraints (one hackathon build window, one shared GPU, one specific tabletop
scene):
- **Fine-tuning GR00T on Physyk's own demonstrations (what's already the plan)** — this is the
  correct fix, not a different model choice. `groot_episode_recorder.py` and
  `scripts/generate_finetune_episodes.py`/`scripts/convert_raw_episodes_to_lerobot.py` already
  exist in the codebase for exactly this. Early attempts already show real, scene-relevant
  motion (not random flailing) — the identified gap is simply **not enough demonstration
  episodes yet**, which is a data-volume problem, not an architecture problem.
- **A small, from-scratch behavior-cloning policy trained only on this scene** (e.g. a
  lightweight diffusion policy or an ACT-style transformer, both established, smaller
  alternatives in robot learning) would very plausibly converge to reliable grasping on *this
  specific task* faster, with fewer demonstrations, precisely because it never has to unlearn
  a different domain first. **Why not do that instead**: it would produce a narrow, single-task
  skill with no language-conditioning, no generalization story — it wouldn't demonstrate
  "vision-language-action foundation model," which is the actual narrative this track of the
  project is telling. It would solve the demo but abandon the thing being showcased.
- **A different pretrained foundation VLA** (OpenVLA, Octo, π0, or an RT-2-style model) — all of
  these have the *identical* fundamental issue: any pretrained generalist policy needs either a
  benchmark whose visual domain already closely matches your scene, or genuine fine-tuning on
  your own data, neither of which is free. None ship "grasp this exact cube on this exact table"
  out of the box. Given that unavoidable requirement either way, GR00T-N1.7 is a reasonable pick
  specifically *because* it's NVIDIA's own current-generation VLA — same logic as choosing
  Nemotron: demonstrating NVIDIA's own model stack is part of the point of the submission, not
  incidental to it.
Bottom line: nothing strictly better was available and skipped. The honest, accurate framing for
judges is: *"correct model, correctly identified fix already in progress (fine-tuning), currently
gated on demonstration-data volume, not on a wrong architectural choice."*

### Q6: Why not direct, live Isaac Sim simulation instead of just an HTTPS feed?
It already **is** direct live simulation for the interactive view — this isn't an either/or.
There are genuinely two different video channels serving two different jobs, both real:
- **Isaac Sim's native WebRTC stream, port 8211** (`isaac-sim.streaming.sh`, listed as a
  required port in `BREV_PORT_CONFIG.md`). WebRTC (Web Real-Time Communication) is a real-time,
  low-latency interactive video protocol built for exactly this — it *is* the live, full,
  interactive 3D Isaac Sim viewport, streamed to a browser, because Isaac Sim runs headless
  (no monitor attached) on the remote Brev GPU box and this is how you get its rendered pixels
  to a laptop browser at all.
- **The MJPEG camera feed, port 8100** (`isaac_sim_service.py`'s `/camera/*.stream` routes).
  MJPEG (Motion JPEG) is a much simpler format — literally a continuous sequence of individual
  JPEG images sent one after another over one HTTP connection. This exists as a *second,
  separate, lighter-weight* channel, specifically because:
  - GR00T and Cosmos both need a literal JPEG image as their model input (not a video stream
    protocol) — the MJPEG endpoint is trivial to grab one frame from for that purpose.
  - Showing 5 separate named camera angles (overview/front/side/top/wrist) individually in the
    plain web GUI is far cheaper to implement with 5 independent `<img>`-style MJPEG tags than
    embedding 5 separate WebRTC viewports.
So the honest answer is: the interactive 3D "real simulation" view already exists and is live
(port 8211); MJPEG is the separate machine-consumption and lightweight-preview channel, not a
downgrade substituting for the real thing.

### Q7: Why is the (MJPEG) feed only about 10 fps?
Real, physical GPU contention — not an artificial rate limit chosen in code. Breaking down what's
actually competing for the same GPU at once, confirmed live via `nvidia-smi` showing 94%
utilization with all five processes resident:
- Isaac Sim is simultaneously running **real PhysX physics simulation** and **RTX rendering
  for five separate 720p camera views** (overview, front, side, top, wrist) every single frame.
- On the *same physical GPU*, at the *same time*, three separate model-inference servers
  (Nemotron-30B, Cosmos-Reason2-2B, GR00T) are also doing real compute whenever they're called.
- The MJPEG server code itself checks for a new frame roughly every 16ms (a ~60Hz *check*
  interval — that's how often it *looks* for a new frame to send), but it can only actually
  send a **new** frame as fast as Isaac Sim's own combined physics+render loop produces one —
  and that loop, under this concurrent GPU load, runs at roughly 7-10 fps (referenced directly
  in `ARCHITECTURE.md` as "the ~7-8fps sim loop," and matched by the orchestrator's own 10Hz
  telemetry poll rate, tuned to that reality). The 60Hz check is a polling rate, not a
  production rate — it doesn't create frames faster than Isaac Sim renders them.
This is a real, load-driven number, not a deliberately chosen cap — and it's a tolerable trade
here because the orchestration/verification logic only needs a frame per *step* (not per video
frame), and the smooth, high-fps visual experience for judges is the separate WebRTC viewport
(Q6), which isn't sharing the same bottleneck the same way.

### Q8: How is Nemotron actually implemented — is it "proper" Nemotron itself, or something dressed up as it?
It's the real thing, confirmed by reading the live process command line directly (`ps aux`), not
just trusting a doc:
```
/opt/dlami/nvme/physyk/workspace/.venv-nemotron/bin/python3 -m vllm.entrypoints.openai.api_server \
  --model /opt/dlami/nvme/physyk/models/Nemotron-30B-FP8 \
  --served-model-name nemotron --host 0.0.0.0 --port 8000 --dtype auto \
  --max-model-len 12288 --gpu-memory-utilization 0.55 --trust-remote-code
```
That's a real, unmodified `vllm` server, loading real downloaded weights for
**`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8`** (confirmed in `nemotron_fastapi_server.py`'s
own `/v1/models` route) — NVIDIA's actual published Nemotron checkpoint. A few specifics worth
knowing to explain confidently:
- **30B / A3B** — this is a Mixture-of-Experts (MoE) model: 30 billion total parameters exist,
  but only about 3 billion ("A3B" = "Active 3B") are actually computed per token generated. This
  is *why* it's viable to co-host a "30B" model alongside two other GPU-resident models on one
  card — the compute cost per call is much closer to a 3B model's than a dense 30B model's would
  be, even though the memory footprint for the full weight set is real.
- **FP8** — the weights are stored in 8-bit floating point (versus the more common 16-bit),
  roughly halving memory footprint versus a 16-bit version of the same model, at some (generally
  small, well-studied) precision cost — a standard, deliberate quantization trade to make a 30B
  model fit comfortably alongside the other services.
- **"Reasoning"** — this specific checkpoint is a reasoning-tuned variant, meaning it's trained
  to emit visible step-by-step chain-of-thought before its final answer (the `</think>` marker
  behavior described in Part 2a) — this is a real, load-bearing property the prompt/parsing code
  is built around, not an incidental detail.
One honest historical caveat worth having ready: `dev-progress.md` records that earlier in
development, Nemotron's output was genuinely "computed and discarded" — the orchestrator called
it, logged it, but a separate rule-based path actually drove the robot. That's fixed now (the
code above is real, current, and confirmed to drive dispatch) — worth knowing only so you're not
caught off guard if a judge has read an older internal doc.

### Q9: How is the current pick-and-place math actually set up inside Isaac Sim?
Covered in full detail in Part 2b above — the short version: keyword-match which object/relation
the sentence describes → query that object's real live 3D position from the simulator's internal
scene graph → compute a placement position by destination type (stack = reference position + 2×
cube-half-height; tray-relative phrase = reference + directional offset clamped to tray bounds;
generic tray/bin language = first unoccupied of 3 fixed tray slots, checked against other cubes'
live positions; no destination given = place back where it was picked up) → hand that pick
position + place position to NVIDIA's own `PickPlaceController`, which runs its real 10-phase
grasp state machine through RMPFlow motion planning and real PD-driven PhysX joint control → after
completion, re-query the object's real final position and compare it to the intended target
within a real tolerance to produce the actual success/fail signal.

### Q10: How can the current agentic pipeline be made better?
Concrete, specific next steps — most already explicitly identified in the project's own docs,
not invented for this briefing:
- **Flip Cosmos from `shadow` to `vision` mode for the live demo itself** — right now, per
  `cosmos_integration.md`, Cosmos can run alongside ground truth purely to display the delta;
  the stronger, more honest claim ("the robot's motion was actually driven by what a camera saw")
  requires actually flipping the mode, with its documented automatic fallback still in place as
  a safety net.
- **Replace the hand-rolled `PhysicalWorkspaceRails` bounds check with real NeMo Guardrails**
  (NVIDIA's actual guardrails framework) — the current check is real and functional, but a
  proper NeMo Guardrails config would be both a stronger safety story and more directly aligned
  with what an NVIDIA-stack hackathon likely wants to see used.
- **Run a real N-trial randomized success-rate measurement** (e.g. 20 runs with randomized cube
  placement, tracking actual pass/fail) — explicitly flagged in `ARCHITECTURE.md` as not yet
  done, and now possible since real verification only recently landed. A real success-percentage
  number is a much stronger demo artifact than a single successful run.
- **More GR00T fine-tuning demonstration episodes** — the tooling already exists
  (`groot_episode_recorder.py`, the two conversion/generation scripts in `workspace/scripts/`);
  this is a data-collection task, not new engineering.
- **Delete the confirmed-dead `physyk_lula_controller.py`** — small, but a judge reading the
  live codebase will notice unreachable legacy motion code sitting next to the real one.
- **Surface Cosmos's post-motion visual evidence more prominently in the UI** — the data
  (`visual_verified`, `visual_confidence`, `visual_observed`) is already computed and attached
  to each step; making it visible per-step in the plan panel turns "step marked DONE" into a
  visible, inspectable receipt rather than an opaque checkmark.

### Q11: How is the current agentic pipeline actually implemented by Nemotron + Cosmos together?
The precise division of labor, since this is the most commonly blurred point:
- **Nemotron is the only model inside the actual decision loop** — it's the one called by
  `decompose_task_with_reasoning()` (initial planning) and `request_failure_decision()`
  (recovery decisions). It is the "what should happen next" authority.
- **Cosmos is a separate, additive perception/verification module**, called from two specific
  hook points (`_attach_cosmos_assessment` before dispatch, `_attach_visual_evidence` after
  motion) — it never decides what to dispatch next, and by design it can never block, delay, or
  override a decision; any failure inside it degrades to "NOT CHECKED" rather than stopping
  anything.
- **The dependency runs one direction**: Cosmos's outputs (visual confirmation, visual
  disagreement) are logged as evidence that *feeds into what a human or judge reads*, and in the
  `vision` perception mode, Cosmos's pose estimate can literally become the coordinate IK uses —
  but Nemotron's planning logic doesn't consult Cosmos's assessments to make its retry/replan
  decisions; those decisions are driven by the deterministic position-verification telemetry from
  Part 2b.
So: **Nemotron plans and decides. Cosmos perceives and provides a second, independent line of
evidence. Isaac Sim executes and produces the ground-truth verification signal that actually
drives the loop.** Deliberately separated jobs, not two models competing to do the same thing —
this exact framing ("don't let Nemotron and Cosmos compete, give them explicit jobs") is a
direct design note in `agentic-plan.md`.

---

## Part 6 — Future use cases (for roadmap/judge Q&A)

- **Low-volume/high-mix manufacturing & kitting** — reconfiguring what the robot does becomes a
  typed sentence instead of a re-programming project; most valuable exactly where product runs
  are too short to justify hand-coding new motion paths for each one.
- **Warehouse sorting/picking** — the multi-step decomposition already demonstrated ("sort all
  three cubes by color into bins") generalizes directly to SKU sorting once object recognition
  generalizes past colored cubes to real product classes.
- **Human-operable robotics without robotics programming expertise** — floor operators direct
  the system in plain language; the guardrail layer is precisely what makes it safe to allow
  that level of access at all, rather than requiring a trained integrator for every change.
- **A live, honest testbed for evaluating foundation-model robotics** — the side-by-side
  architecture (reliable scripted `PickPlaceController` as default, GR00T VLA as an explicitly
  opt-in alternative) is itself reusable as a harness for evaluating *any future* VLA checkpoint
  against a known-reliable baseline, without ever betting the whole system's reliability on a
  model that isn't production-ready.
- **Sim-to-real transfer** — once Cosmos-driven visual pose estimation is trustworthy end-to-end
  (real camera + depth → world coordinate, not privileged simulator access), this pipeline is
  structurally much closer to what a *real* camera-equipped robot arm in a real factory would
  need, versus a system that only ever worked because it could silently read ground truth out of
  a simulator.
- **Cosmos 3 migration** — `cosmos_integration.md` explicitly notes NVIDIA has moved active
  Cosmos development to "Cosmos 3," and the Cosmos Reason 2 repo carries a deprecation pointer.
  Deliberately deferred here ("do not attempt a migration the night before a demo") — a
  legitimate, low-risk item to name on a roadmap slide as forward awareness, not as unaddressed
  technical debt.

---

*This file lives at `/opt/dlami/nvme/physyk/CONTEXT_BRIEFING.md` (this project's own directory,
as requested). Nothing in the running system was modified to produce it.*

---

## Part 7 — Demo-Day Script (your part: explaining the agentic orchestration + live demo)

Written to be spoken, not read verbatim — paragraphs are ~15-25 second beats. Bracketed `[...]`
lines are actions/cues, not speech. Timings assume a ~6-8 minute slot; trim the bracketed
"if time" asides first if you're running short.

### 7.1 — Open: what problem this solves (30s)

> "Most robots in a factory today do one thing, one way, forever. You program an arm to pick up
> a part from exactly position X and place it at exactly position Y — the moment the part moves,
> the layout changes, or someone gives it a new instruction, it breaks. Re-programming it is an
> engineer sitting down and hand-coding new motion paths, which takes days.
>
> Physyk asks: what if you could just *tell* the robot what to do, in plain English, with the
> layout allowed to change — and have it figure out the rest, check its own work, and recover
> when something goes wrong?"

[Cue: have the GUI open, empty instruction box, cubes already randomized so it's visibly not a
fixed layout.]

### 7.2 — The architecture, in one breath (30s)

> "There are three layers. At the bottom, a Franka Panda arm in NVIDIA Isaac Sim — real physics,
> real gravity, real friction, not animation. On top of that sits the actual agentic layer: an
> LLM, Nemotron, that takes a free-form instruction and decomposes it into a sequence of
> pick-and-place steps, with a vision model, Cosmos Reason 2, verifying what's actually happening
> in the scene. And alongside the reliable path, an experimental one: GR00T, a vision-language-
> action model that predicts robot motion directly from camera images, as a look at where
> foundation-model robotics is headed."

### 7.3 — The core explanation: agentic orchestration (this is your centerpiece — 90-120s)

> "Here's the part that makes this 'agentic' rather than just 'a robot that understands
> sentences.' The old version of this system was: prompt goes in, a plan comes out, the arm runs
> every step, and it just reports success — whether or not anything actually happened correctly.
>
> The current version is a closed loop. Watch what happens for one instruction:
>
> First, the system looks at the *actual* current scene — real object positions read live off
> the simulator, not a guess — and hands that, plus the instruction, to Nemotron. Nemotron
> breaks it into an ordered list of steps. [If asked how: it's a real 30-billion-parameter
> reasoning model, running locally on this GPU via vLLM — you can literally watch its
> chain-of-thought in the UI as it decides how to split the instruction.]
>
> Before any step is ever sent to the robot, it passes through a safety guardrail — a hard check
> that the target position is inside the arm's actual reachable workspace. If Nemotron ever
> proposed something physically impossible, this rejects it before the robot moves at all.
>
> Then — and this is the important part — before dispatching each step, the system asks two
> questions purely from live data, no LLM needed: is this already done? And has anything changed
> since the plan was made — did an earlier step's placement get knocked out of position by
> something that happened after it? If either is true, it doesn't blindly plow ahead.
>
> The step executes on the real physics engine. Afterward, the system doesn't just trust that the
> motion finished — it re-reads the object's real final position and compares it, in millimeters,
> to where it was supposed to end up. That comparison is the actual truth signal.
>
> If it failed, the system doesn't just retry blindly, and it doesn't just give up. It goes back
> to Nemotron with the real error, the real scene, and asks it to *diagnose*: was this a one-off
> physical miss — try again — or does the destination itself need to change — replan just this
> step — or has something bigger shifted, meaning the rest of the plan needs to be reconsidered.
>
> And only once every step is actually done does it run one last check: did completing a *later*
> step accidentally undo an *earlier*, already-verified one? Only if the whole original goal
> still holds, end to end, does it report success."

[Cue: this is where a judge's eyes glaze over unless you slow down on the phrase "the truth
signal" — it's the single sentence that makes the whole loop legitimate. Consider saying it
twice, once early once at the end.]

### 7.4 — The one-sentence version, if you need to compress (10s)

> "Plan, act, actually look at what happened, and only go around again — retry, replan, or
> report failure — based on what really happened, not what was supposed to happen."

### 7.5 — Live demo sequence

**Demo A — the reliable path, single step (30s)**
1. [Type]: `"pick up the red cube and put it in the tray"`
2. Narrate while it runs: point at the Nemotron reasoning panel (it decomposes to one step since
   it's one clause), the plan panel step going RUNNING → VERIFYING → DONE, and the live camera
   feed showing the actual grasp.
3. Land the point: "That DONE isn't the planner assuming success — it's a real position check,
   in millimeters, against where the cube actually ended up."

**Demo B — multi-step decomposition (45s)**
1. [Type]: `"stack the red, green and blue cubes on the tray, with red at the bottom, green in
   the middle, and blue at the top"`
2. Narrate: "One sentence, three separate physical actions. Watch the plan panel — Nemotron just
   split this into three ordered steps on its own." Point at each step going RUNNING → DONE.
3. If it's already partially done from an earlier run: "See this step didn't even re-execute —
   the system checked the live scene, saw the red cube's already at the bottom of the stack,
   and skipped it instead of blindly re-picking something already correct."

**Demo C — the "it's not scripted" proof: perturbation / failure recovery (60s — the strongest
beat if you have time for only one extra demo)**
1. Start a multi-step instruction.
2. Mid-plan, physically disturb the scene — nudge/reposition a cube in the viewport, or use the
   "randomize" control if a manual nudge isn't available — so a later step's assumption
   ("stack on the red cube") is now false.
3. Narrate as it happens: "I just changed the scene mid-plan. Watch — the system checks, before
   dispatching this next step, whether its assumption still holds. It doesn't. It's now going
   back to Nemotron with the real problem — and rather than a scripted retry, it's genuinely
   deciding whether to try again or replan."
4. Land it: "This is the difference between a script that runs steps in order, and something
   that's actually watching what's true and reacting to it."

**Demo D — if there's time: the perception panel (Cosmos)**
1. Point at the "👁 Perception" card if visible / toggle `shadow` mode.
2. "Cosmos Reason 2 is a separate vision model running alongside — it looks at the same camera
   frame and estimates where objects are on its own, purely from pixels and depth. Right now it's
   in shadow mode: it's not driving the arm, just being checked against the simulator's own
   ground truth live, and it's within about 6 millimeters, consistently."
3. [If you flip to `vision` mode live]: "And now, if I flip this switch, that visual estimate is
   what's actually driving the arm — not privileged access to the simulator's internal state."

**Demo E — the honest one, if asked "does the learned model work?" (don't volunteer, but be
ready)**
1. [Type]: `"vla: pick up the blue cube"`
2. "This is GR00T — NVIDIA's vision-language-action model, predicting motion directly from
   camera pixels, no scripted controller involved. It's genuinely doing real inference — we
   verified that by feeding it different images and confirming it predicts different actions,
   ruling out a stub. It's not yet reliable on this exact scene, because this checkpoint was
   fine-tuned on a different benchmark, not on our own cubes and table — and that's an honest,
   known gap, not something we're hiding. Fine-tuning it on our own demonstrations is the
   scoped next step."

### 7.6 — Anticipated judge questions + short, direct answers

- **"Is any of this hardcoded / scripted to look impressive?"** — "The decomposition, the
  guardrail check, the failure diagnosis, and the perception estimate are all real model calls
  you can watch happen live in the reasoning panel. The only things that are deterministic code,
  not an LLM call, are the trivial state checks — is this already done, has this moved — by
  design, to minimize unnecessary LLM calls. That split is deliberate, not a shortcut."
- **"What happens if Nemotron is down / times out?"** — "It falls back to dispatching the
  instruction directly, single-step, rather than hanging or crashing — the reliable physical
  layer never depends on the cognitive layer being up."
- **"What happens if a step just keeps failing?"** — "Three attempts max per step — one retry,
  one Nemotron-diagnosed correction — and then it reports FAILED and aborts the remaining plan
  honestly, rather than looping forever or silently claiming success."
- **"Why isn't the learned model (GR00T) the main path?"** — "Because it's not reliable on this
  scene yet, and we'd rather show you an honest, working, verified system than a black box we're
  hoping works on demo day. The scripted path is the reliable default specifically so the learned
  path can be shown, improved, and evaluated without risking the whole demo."
- **"How do you know 'success' is real and not just the planner returning?"** — this is the
  one to really land: "Because success is a live position comparison, in millimeters, against
  the object's actual final location in the physics simulation — not 'the motion finished' and
  not 'the LLM said so.'"

### 7.7 — Close (15s)

> "So: natural language in, a real plan out, real physics execution, real verification of what
> actually happened — and when something doesn't go as expected, it notices, diagnoses, and
> recovers, instead of silently reporting success. That loop — plan, act, verify, adapt — is the
> actual agentic part, and it's what's running right now, live, on this GPU."

[Cue: end on the live GUI, not a slide, if possible — the reasoning panel mid-thought is a
strong visual to leave on screen during Q&A.]
