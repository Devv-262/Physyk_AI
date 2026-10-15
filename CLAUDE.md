# Physyk AI — notes for Claude

A Franka Panda in NVIDIA Isaac Sim 6.0.1 (Docker) that takes plain-English instructions ("stack the cubes in the tray") and does the pick-and-place. "Stack ..." is planned by a **local deterministic planner** (`workspace/stack_planner.py`); other prompts are planned by Nemotron one move at a time (`workspace/nemotron_planner.py`, thinking off, streamed; hosted API, `nvidia/nemotron-3.5-lightning-30b-a3b`). The arm executes with NVIDIA's `PickPlaceController` + RMPFlow, the result is verified against real positions (or a fresh camera look in camera-only mode), and Cosmos (local NIM) gives a second visual opinion in a background thread.

**Current scene:** three cubes (red / green / blue) and a yellow tray on a work table, inside a visual-only factory backdrop (`factory_background.py`). No shelf. The glassware (flask / beaker + shelf) variant is **not** here: it lives on the GitHub branches `main` and `feature-lab`. Do not mix them.

**Lineage (see `HISTORY.md` §5):** this code is the 22 August code (commit `80a0e7f`) plus the perception / agent / tooling work of the later cube-lab rebuild, the local stack planner, the factory scene and the restyled GUI. The rebuild's arm-control work (hand yaw, arrival gating, release on contact, obstacle modes, joint trace, arm test tools) was **dropped on purpose** and lives in git history (`git show ef67e16:<path>`, local tag `backup-before-aug22-rewind`). Do not re-add it unless asked.

## Run

```bash
./start_sim_docker.sh        # Isaac Sim, headless, API :8100, Ctrl+C stops the container
./start_cosmos_docker.sh     # optional, Cosmos NIM :8001
./run.sh                     # backend + GUI on http://localhost:7860   (--no-agent | --status | --stop)
```

Backend Python is `.venv/bin/python`. Secrets are in `.env` (git-ignored, never commit). The sim container mounts `workspace/` read-write, so a sim code change needs a sim restart; the backdrop can be rebuilt live (`curl -X POST localhost:8100/background/rebuild`). A backend / orchestrator change needs `./run.sh`. Settings are in the `README.md` table.

## Where things are

| What | Where |
|---|---|
| Sim loop, scene, controller phases, perception wiring, `/execute` `/state` | `workspace/isaac_sim_service.py` |
| Planner (Nemotron + local stack planner), guardrails, dispatch / verify / retry, closed-loop re-planning | `workspace/physyk_agent_orchestrator.py` |
| Stack planner and the shared safe-zone rules (Randomize, free spots) | `workspace/stack_planner.py` |
| GUI (inline HTML) + gateway | `workspace/physyk_main_server.py` |
| Camera-only detection, Cosmos client, step verifier | `workspace/physyk_perception/` |
| Factory backdrop (visual only) | `workspace/factory_background.py` |
| Nemotron / Cosmos log pages | `workspace/reasoning_dashboard.py` |
| Offline tests | `workspace/scripts/test_stack_planner.py`, `test_stack_agent.py`; `test_stepwise_agent.py` calls the real Nemotron API against the fake sim |
| Docs | `README.md` (overview + env table), `HOW_IT_WORKS.md` (plain English), `ARCHITECTURE.md`, `TECHNICAL_DEEP_DIVE.md` (**§5 open problems**), `PERCEPTION.md`, `cosmos_integration.md`, `instructions.md` (run guide), `PROJECT_NARRATIVE.md`, `INDUSTRY_APPLICATIONS.md`, `HISTORY.md` |

Read `README.md` and `TECHNICAL_DEEP_DIVE.md` §5 first in a new session. Keep the docs in sync when behaviour changes, and keep them plain, honest and limited to what the code does now (no VLA / GR00T / glassware material).

## State

Working: tray / table / relative placements; "stack the cubes" planned locally from positions for any starting arrangement and re-planned from the live scene after every step; **other prompts planned by Nemotron one move at a time** (goals split once, then one move per turn against the live scene, code decides when a goal is reached, thinking off, streamed; a cube that is not visible or a robot refusal ends the run, nothing is substituted); blocker clearing and parking; camera-only perception that keeps the cubes of a stack (shadowed lower cubes) and a cube carried by the arm; clean overlay (tight rectangles, labels beside them); guardrail retry on camera scenes; Cosmos checks in background threads; the NTT DATA themed GUI with green / red service and result colours, an About dialog and a smaller stage pill. Offline tests pass (12,000 planner combinations, 47 agent-loop cases, 4 stepwise cases against the fake sim with the real Nemotron API).

Verified on the real sim (2026-10-15, one default layout each): 10 end-to-end prompts (single moves, "right of", stack in the tray / on the table, a 3-goal and a 4-goal chained prompt needing parking moves, a missing cube refused, deterministic and camera-only), all passed; the 4-goal prompt twice more.

**Not verified:** pick wall time on this controller as a batch; touching cubes; random layouts in bulk; the colour thresholds beyond the default layout in the dark room; the failure-diagnosis call with thinking off (one try, malformed answer treated as a retry); prefix caching on the hosted API (reported 0 cached tokens).

Open problems (details in `TECHNICAL_DEEP_DIVE.md` §5): the controller is an open-loop timer with no hand yaw, so a long sideways carry may overshoot (measured at about 21 cm on the dropped rebuild's controller, **not yet measured on this one**), the elbow posture is not controlled, RMPFlow obstacles are probably cleared by the controller's reset, the sim start-up occasionally hangs, the sim loop runs at 28-55 iterations/s (below 60, so it runs slower than real time; rendering every step is the main cost).

## Rules for working here

- **One sim.** The sim is the container `physyk-isaac-sim` on port 8100, and it is the user's. Run tests on that instance only: the user said not to create an isolated second copy of the sim. Stop / restart that same container when a test needs different settings, and say so first if they might be using it.
- **Git / GitHub.** The user pushes themselves unless they ask in the moment: this machine has no GitHub credentials. Do not look for tokens, never force-push, never touch `main` / `feature-lab` (glassware). Commit locally when asked. Plain `git push` of a fast-forward may be tried when the user explicitly asks; if it fails on credentials, report it and stop.
- **Never `pkill -f` / `kill` by a pattern that appears in your own command line** (it kills the shell, exit 144). This includes `pgrep -f 'x'` whose pattern also appears anywhere else in the same command. Look PIDs up in one command, then kill by PID in a separate one (`pgrep -f 'pat[t]ern'` for the lookup).
- **Starting the sim:** launch in the background, poll `/health` in a bounded loop (about 5 minutes). Roughly one start in five hangs with an empty `docker logs` and an "unhealthy" container: `docker rm -f physyk-isaac-sim` and start again.
- **Long jobs:** run in the background and give a one-line status before waiting. The user interrupts silent waits ("what's lagging", "is anything stuck").
- **Answer what was asked.** Separate what was measured from what is a hypothesis, and do not report a fix as working without a run on the sim (or, for the planner / agent loop, the offline tests).
- **Aesthetics requests are aesthetics only.** When asked for scene or UI changes, do not touch the arm, the table, the tray, the cubes, the physics or the controller. Scenery is visual-only (`VisualCuboid`); the physics ground plane stays where it is.

## Gotchas learned the hard way

- **The GUI's Hard Reset button is the only thing that stops a plan from outside.** If a test run shows "Plan stopped by hard reset", someone clicked it (the user's browser has the page open); it is not a code path.
- **Seeing the GUI:** `google-chrome --headless=new --no-sandbox --disable-gpu --virtual-time-budget=7000 --window-size=1700,1250 --screenshot=out.png http://localhost:7860` gives a screenshot of the live page (add `--enable-logging=stderr --v=0` to catch JavaScript errors). A scene frame is `curl localhost:8100/camera/scene_raw.jpg`; a frame fetched right after a hot rebuild can be stale for a few seconds.
- **The overview camera looks steeply down**, so it sees only the bottom ~1.4 m of the far walls; wall scenery is built at half scale (`SCENERY_SCALE`). Do not move the camera: the detector's calibration depends on it.
- **A closed ceiling darkens the table.** The detector's table fit (`fit_table_plane`) uses a brightness floor relative to the table's own median; if camera-only heights drift, check `table_calibration.n` in `/perception` (tens of thousands = healthy, about a thousand = failing).
- **Stack planner contract:** the sim and the planner import the same safe-zone constants from `stack_planner.py`; change them there, not in two places. A cube out of reach makes the planner refuse (`StackPlan.blocked`); the agent excludes any cube the guardrail or sim refused from later plans (otherwise the re-plan loops).
- **Nemotron stacking was tried and dropped:** with the thinking model, "stack the cubes" often hit the 4096-token limit. `PHYSYK_STACK_PLANNER=llm` still exists but is not the default.
- **GUI look:** dark neutral base with muted steel-blue accents, the NTT DATA logo beside the heading, small grouped header pills, no emoji. The user rejected a blue-washed background and bright accent colours; keep it subtle.
- `PickPlaceController.reset()` clears registered RMPFlow obstacles (measured on the earlier branch); the sim registers them once at start-up, so they are probably inactive.
- The cube prims are scaled 0.05; `panda_rightfinger` sits about 2.5 cm off the hand centre. Cubes closer than about 0.36 m to the base drive joint 6 into its limit and the grasp fails.
