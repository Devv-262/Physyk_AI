# Physyk AI — notes for Claude

A Franka Panda in NVIDIA Isaac Sim 6.0.1 (Docker) that takes plain-English instructions ("stack the cubes in the tray") and does the
pick-and-place: Nemotron (hosted API) plans, the arm executes with NVIDIA's `PickPlaceController` + RMPFlow, the result is verified against
real positions (or a fresh camera look in camera-only mode), and Cosmos (local NIM) gives a second visual opinion.

**Current scene:** three cubes (red / green / blue) + a yellow tray on a lab bench. No shelf. The glassware (flask / beaker + shelf) variant
is **not** here: it lives on the GitHub branches `main` and `feature-lab`. Do not mix them.

## Run

```bash
./start_sim_docker.sh        # Isaac Sim, headless by default (HEADLESS=0 for a window), API :8100, Ctrl+C stops the container
./start_cosmos_docker.sh     # optional, Cosmos NIM :8001
./run.sh                     # backend + GUI on http://localhost:7860   (--no-agent | --status | --stop)
```

Backend Python is `.venv/bin/python` (3.14). Secrets are in `.env` (never commit). Sim/backend code changes need a sim restart / `./run.sh`.
Environment switches (speed, obstacles, trace, seed, ...) are in the `README.md` table.

## Where things are

| What | Where |
|---|---|
| Sim loop, controller phases, perception wiring, `/execute` `/state` | `workspace/isaac_sim_service.py` |
| Planner: Nemotron decomposition, stack planner, guardrails, dispatch / verify / retry | `workspace/physyk_agent_orchestrator.py` |
| GUI + gateway | `workspace/physyk_main_server.py` |
| Camera-only detection, Cosmos client, step verifier | `workspace/physyk_perception/` |
| Test tools (end-to-end batch, arm physics harness) | `workspace/scripts/arm_tests/` (see its README) |
| Docs | `README.md` (overview + env table), `HOW_IT_WORKS.md` (plain English), `ARCHITECTURE.md`, `TECHNICAL_DEEP_DIVE.md` (**§1.8 arm physics, §5 current challenges**), `PERCEPTION.md`, `cosmos_integration.md`, `instructions.md` (run guide), `PROJECT_NARRATIVE.md` |

Read `README.md` and `TECHNICAL_DEEP_DIVE.md` §5 first in a new session. Keep the docs in sync when behaviour changes, and keep them
plain, honest and limited to what the code does now (no VLA / GR00T / glassware material on this branch's docs).

## State (see git log for details)

Working: tray / table / relative placements, "stack the cubes" planned from positions (easiest first, never by colour), blocker clearing,
hand yaw aligned to the cube with neighbour avoidance, arrival gating on approach and transport, release on contact, joint trace + anomaly
log. Latest end-to-end run on the real sim: 24 of 25 picks passed on 10 random layouts, ~20 s per pick.

Open problems (details in `TECHNICAL_DEEP_DIVE.md` §5): the arm's **elbow posture is not controlled** (the user reports the elbow pointing
outwards on the first pick; forward kinematics of the recorded joints says it sits in the base-hand plane, so ask for a screenshot; the
candidate fix is a preferred null-space joint pose for RMPFlow, `set_cspace_target`, not implemented); a rare joint swing that only
reproduces under specific conditions; open-loop controller timing (only phases 0 and 5 wait for the arm); joint 7 kick when the gripper
closes; untested: touching cubes, large batches, colour thresholds on the lab background.

## Rules for working here

- **One sim.** The sim is the container `physyk-isaac-sim` on port 8100, and it is the user's. Run tests on that instance only: the user said
  not to create an isolated second copy of the sim. Stop / restart that same container when a test needs different settings, and say so first
  if they might be using it. The standalone harness scripts in `workspace/scripts/arm_tests/` start their own throw-away container, so ask
  before using them.
- **Git / GitHub.** The user pushes themselves: this machine has no GitHub credentials. Do not look for tokens and do not push or force-push.
  Reading the public repo with `git fetch` works. Never touch `main` / `feature-lab` (glassware). Commit locally when asked.
- **Never `pkill -f` / `kill` by a pattern that appears in your own command line** (it kills the shell, exit 144). Use `pgrep -f 'pat[t]ern'`
  or a PID.
- **Starting the sim:** launch in the background, poll `/health` in a bounded loop (about 5 minutes). Roughly one start in five hangs with an
  empty `docker logs` and an "unhealthy" container: `docker rm -f physyk-isaac-sim` and start again.
- **Long jobs:** run in the background and give a one-line status before waiting. The user interrupts silent waits ("what's lagging", "is anything
  stuck").
- **Answer what was asked.** "Summary of the last test" means that test, not the whole project. Separate what was measured from what is a
  hypothesis, and do not report a fix as working without a run on the sim.

## Gotchas learned the hard way

- `PickPlaceController.reset()` clears every registered RMPFlow obstacle. `_controller_reset()` re-registers them per `PHYSYK_ARM_OBSTACLES`
  (default `off`; active obstacles gave worse picks: 12/15 and 2/5 vs 14/15).
- The controller is an open-loop timer: phase 5 lasts ~16 steps while the arm needs ~160. Speed-ups above 1.25x (and lowering RMPFlow's
  velocity cap) make it outrun the arm. Lower damping did not help; stock kp / kd (22918 / 4584 N·m/rad) are the smoothest.
- Cube prims are scaled 0.05: strip the scale (`RemoveScaleShear()`) before reading rotation. The `panda_rightfinger` frame sits ~2.5 cm off the
  hand centre: judge arrival on `panda_hand`.
- Cubes closer than ~0.36 m to the base drive joint 6 into its limit (3.7525 rad) and the grasp fails: Randomize keeps 0.38-0.66 m, the
  guardrail refuses closer targets.
- Cube colour thresholds in `color_segment.py` were rebuilt from measured values and are not re-checked on the lab background.
- `workspace/joint_trace.csv` (every joint per step of a pick) and `[Joint anomaly]` log lines are on by default; `PHYSYK_RANDOM_SEED` makes
  Randomize repeatable for A/B tests.
