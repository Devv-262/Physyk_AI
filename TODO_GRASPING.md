# Glassware grasp-and-place: current state and to-do (consolidated 2026-10-12; everything is STOPPED)

Goal: type "pick the flask with red liquid and place it on the middle shelf" and have the Franka do it in Isaac Sim:
vessels held from the side, carried upright (no spilling), placed on a 3-tier shelf or the table, picked back off the
shelf, refusing clearly when it cannot, all through the Nemotron planner + guardrails + verification, in deterministic mode
and in camera-only ("pure perception") mode.

**Project location: `~/Desktop/Physyk_AI-master`** (moved on 2026-10-13 from `~/Desktop/physyk-ai/Physyk_AI-master`; the venv and
absolute paths were fixed; nothing else on the Desktop is used by it).

## 0. State right now

Nothing is running: no sim container, no Cosmos, no backend (ports 7860 / 8100 / 8001 are free). All code and maps are saved
in the workspace. To start again: `./start_sim_docker.sh` (add `HEADLESS=1` for no window), `./start_cosmos_docker.sh`
(optional, ~16 GB VRAM), `./run.sh` (backend + GUI). Restart BOTH the sim and the backend after pulling these changes.

## 1. What works (measured; headless sim, camera-only mode unless noted)

| Check | Result |
|---|---|
| Sideways grasp holds flasks / beakers | width confirmed: flask ~26 mm, beaker ~54.6 mm |
| 4 tasks back to back | red beaker -> bottom 1.9 mm, blue flask -> middle 9.9 mm, green beaker -> top 1.2 mm, red flask -> middle 6.6 mm; arm home after each |
| 5-step sequence (final code) | red beaker -> middle 0.7 mm; green -> bottom 1.3 mm; green bottom -> top 1.1 mm; green top -> table 17.9 mm (52 s); red middle -> table 1.5 mm |
| Pick FROM a shelf (rack-camera pose) | red beaker bottom shelf -> table 1.6 mm |
| Arm after a job / home / hard reset | stays at home (0.001 rad) - the "jumps back to the retreat pose" bug is fixed |
| Rack-camera detector (`rack_detect.py`) | 4/4 vessels on 3 tiers, right key/tier/colour, 1.2-5.8 mm; boxes drawn on the rack view in pure mode |
| Occupancy / "shelf full" | second vessel REFUSED at an occupied slot, ACCEPTED at a free one (offline); slots are filled in order x = 0.25, 0.05, 0.45 |
| Refusals | the sim stops with a plain-English reason; the orchestrator does not retry, substitute another destination/object, or call the model again; the GUI plan card shows the reason in a red box |
| Orchestrator offline tests | 68/68 (`workspace/test_dest_parsing.py`, run with `../.venv/bin/python`) |
| Offline slot coverage (5 vessels, other vessels as obstacles, 9 slots each, roll logic) | 34/45; green/red beaker 8/9, empty flask 7/9, red flask 6/9, blue flask 5/9 |

## 2. NOT verified yet (please test first when resuming)

1. **Scene given to Nemotron includes shelf vessels** (camera-only mode). Your screenshot: "put the red beaker on the table"
   with the red beaker on the top shelf was REJECTED with 0 steps because the overview camera cannot see the shelf.
   Fix applied (`telemetry_object_view` adds rack-camera detections; empty-plan message lists what is visible; GUI shows the
   plan's answer). My live check of it FAILED to run - it collided with your session on port 8100 (see section 7).
2. The GUI changes in a browser (answer/failure box on the plan card, example buttons removed, rack camera in the dropdown).
   The GUI JavaScript could not be syntax-checked (no Node here). If the plan card looks broken, look at
   `physyk_main_server.py` near `plan.answer` and `escHtml`.
3. Cosmos' post-motion check for shelf destinations now uses the rack camera (`step_verifier.verify_step(shelf=True)`).
4. The full agent path with the real Nemotron: "place all objects in the rack", a full shelf, a blocked pick, "back to its
   spot", shelf -> table, shelf -> other tier (offline tests pass with a stubbed model only).
5. Speed: arm damping kd 4584 -> 1500 (`PHYSYK_ARM_KD`, expected ~3x faster travel) and the slower, gentler set-down
   (10 frames/cm, 19 for the last 4 cm, fingers open gradually, 45-frame pause). Does the red flask still topple?
6. Hard Reset while a plan is being computed (planning runs in a worker thread and is cancelled at its next check).

## 3. Open problems (most blocking first)

1. **Top shelf -> table spots on the shelf side fail** (offline, `scripts/planner_tests/eleven.py`): from the top tier the arm
   is in an extended-elbow configuration (joint 4 = -0.80, joint 6 = 1.86); coming down to the table with the hand locked at 40 deg pitch,
   joint 6 (wrist pitch) hits its limit (0.04 rad margin) at about (0.325, 0.387, 0.22). From the middle tier the elbow is bent
   (joint 4 = -2.19, wrist 2.46) and the same spot works (margin 0.28 rad, so borderline). Workaround in place: for the top tier the table
   spots are tried far side first, and neighbours of a failed spot (< 15 cm) are skipped. **Proposed fix (not done, you were
   asked): at the pick, try several IK configurations (branches) for the same grasp pose and keep the one whose carry completes.**
2. **Planning time of long carries is ~30 s** (A* over the reach map with Python collision checks). Failed slots cost up to
   the 40 s search budget. Ideas: precompute per-slot carry paths per layout, vectorise the collision test, tighter budgets.
3. **Some slots/vessels still fail offline**: slot x = 0.45, and top-tier slots for flasks starting from the far side (search
   hits the 40 s budget; may succeed with more time).
4. **Far table spot accuracy**: green beaker top -> (0.373, -0.532) landed 17.9 mm off (passes, tolerance 30 mm) - edge of reach.
5. **A vessel knocked over cannot be re-detected** (lies on its side; "not visible"). Use Full Reset.
6. **Near vessels (radius < ~0.47 m) cannot be picked** with the side grasp; randomize keeps vessels at 0.47-0.78 m.
7. **Shelf placements are still verified from the simulator pose** (pure mode) - use the rack detections for verification.
8. Glass slipping in the fingers during carries is unmeasured; playback has no stuck/tilt abort yet (an arrival timeout just advances).

## 4. To-do list

Planner:
- [ ] Multi-branch pick (problem 1) and re-run `eleven.py` / the 5-step sequence.
- [ ] Cut carry planning time (problem 2); re-measure top-tier coverage with the 40 s budget (`eight.py`).
- [ ] Add a stuck/tilt abort to the playback loop; log arm error, fingertip height, finger width per waypoint.

Live / sim:
- [ ] Restart sim + backend; run the section-2 checks in order (1 first). Send logs + frames for anything odd.
- [ ] Success-rate run: 10 repetitions per vessel type (table pick-and-put-back, then each tier, shelf -> table, shelf -> shelf).
- [ ] Verify shelf placements with the rack camera in pure mode (replace the sim-pose check).

Perception / agent / GUI:
- [ ] Rack detector hardening: partly hidden vessels, arm in view, two vessels on one tier; show tier in the Perception Log page.
- [ ] Review the GUI in a browser; check the plan card for REJECTED / FAILED / SUCCESS.
- [ ] Test the real Nemotron on the vague prompts ("place all objects in the rack", "tidy up"), full shelf, blocked pick.

Docs (last, with measured numbers only): README, HOW_IT_WORKS, instructions (+ `workspace/instructions.md`),
ARCHITECTURE, TECHNICAL_DEEP_DIVE still describe cubes and point to deleted docs.

## 5. Design decisions (and why)

1. **Hand pitched 40 deg down, constant after the grasp** (`HAND_PITCH_DEG`). A level hand has no IK solutions at heights
   ~0.25-0.40 m, so the arm could not go between table and shelf heights. 35-50 deg removes the gap. Constant pitch keeps the
   vessel upright (you asked: joint 6 must not turn after the pick; liquid would spill).
2. **Everything while holding a vessel is a locked-orientation Cartesian path** planned up front in joint space (Lula IK),
   played back with arrival checks. Joint-space interpolation is only for the empty arm. Tilt checked at every sample (5 deg).
3. **Hand roll fixed per plan.** The gripper is symmetric, so each hand pose has two IK families (roll 0 / roll 180). They are
   different arm configurations with different reach and must not mix while a vessel is held. One reach map per roll
   (`reach_map_p40_r0.npz` = 90% of cells up to z 0.77; `reach_map_p40_r180.npz` = 41%, nothing above z 0.59); the sim tries roll 0 then 180.
   This fixed "almost all middle/top picks fail".
4. **Lula IK quirk:** a tight orientation tolerance makes it ignore the warm start. `_ik_a` solves loosely first, then refines,
   and rejects anything not within 4 deg of the exact target rotation (catches flips and tilt).
5. **Carry path = A* over the reach map** (cells whose stored joint configs are mutually continuous), refined to 1 cm IK steps
   and collision / tilt checked; soft cost keeps carries >= 12 cm from planks.
6. **Shelf on the robot's +Y side, wall-mounted and STEPPED** (tier tops z = 0.08 / 0.36 / 0.64; tier fronts y = 0.44 / 0.53 / 0.62;
   x -0.05..0.65; back panel y 0.86). Behind the robot needed the base joint near its limit; with the pitched hand the forearm
   rises on the way out, so no plank may overhang the slot below.
7. **Ring layout** on the table (radius ~0.6 m, ~19 cm apart: (0.485,-0.353), (0.571,-0.185), (0.600,0), (0.571,0.185), (0.485,0.353)).
8. **Fullness = collision**: other vessels (table and shelf) are obstacles for arm, hand and carried vessel; there is no slot bookkeeping.
9. **Refusal policy (your decision): stop and show the reason.** No alternative destination or object, no extra model call.
   The reason is plain English, without planner coordinates (Nemotron once mistook a failure point for the destination).
10. **Camera-only mode uses the rack camera for shelf vessels** (pick pose, obstacles, "next to X", table-spot search, scene given to
    Nemotron); if the rack camera misses a vessel the simulator says is on the shelf, it falls back to the simulator pose and logs it.
11. **Home / reset / hard stop return along a paced joint path**, never snap to the home angles.

## 6. How to run things

- Rebuild the reach maps (after changing `HAND_PITCH_DEG`, the IK settings or the shelf; ~1-2 min, builds both rolls):
  `docker run --rm --gpus all -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -v $PWD/workspace:/workspace -w /workspace --entrypoint /isaac-sim/python.sh nvcr.io/nvidia/isaac-sim:6.0.1 reach_map.py`
  (`workspace/` must be world-writable: `chmod a+rwx workspace`). Delete `reach_map_p40_r*.npz` to force a rebuild.
- Offline planner tests: `workspace/scripts/planner_tests/` - `scan.py`/`scan2.py` (pickable region), `one.py`..`ten.py`, `eleven.py`
  (why a carry fails: joint margins), `rack_test.py` (rack detector on saved frames), `fk.py`. Run e.g.
  `docker run --rm --cpus=4 --gpus all -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -v $PWD/workspace:/workspace -v $PWD/workspace/scripts/planner_tests:/scratch --entrypoint /isaac-sim/python.sh nvcr.io/nvidia/isaac-sim:6.0.1 /scratch/eight.py`
  (each run starts a headless Isaac Sim, 30-60 s; run in the background).
- Live scripts used for the numbers above are in the session scratchpad (`four_tasks.py`, `repro3.py`): they POST `/execute`,
  poll `/state`, save `/camera/scene_raw.jpg` and `/camera/rack.jpg`.
- Orchestrator tests (no sim, no network): `cd workspace && ../.venv/bin/python test_dest_parsing.py`.
- Env knobs: `PHYSYK_ARM_KD` (arm damping, 0 = stock), `PHYSYK_SPEEDUP`, `PHYSYK_PROFILE`.

## 7. Pitfalls learned

- **Only one thing may use the sim on port 8100, and the container name `physyk-isaac-sim` is shared.** My headless test runs and
  your GUI/backend collided: commands from your session reached my test sim, and my cleanup (`docker rm -f physyk-isaac-sim`)
  can kill YOUR sim. Next time: ask first, and give test sims their own container name and port (add an env override in
  `start_sim_docker.sh`).
- A docker run whose output pipe closes (interrupted command) hangs as an "unhealthy" container - remove it.
- `rep.create.camera` has a 1.0 m default near clip (anything closer is invisible): use a plain USD camera (done for the rack camera).
- `pkill -f` can kill your own shell; `.isaac-cache` and `workspace/` must be writable by the container.
- Don't wait with long sleep loops in tool calls; run in the background, check output files, say what is running.
- Playback bug history: waypoint target after a segment ends must be `seg.q[-1]` (was `seg.q[0]` -> arm left at the retreat pose,
  one-frame jerk at every segment boundary).

## 8. Files touched in this effort

- Sim: `workspace/isaac_sim_service.py` (shelf, rack camera, vessel ring layout, playback loop, threaded planning, candidate slots/table spots,
  rack detection loop + overlay, refusal telemetry, smooth home), `start_sim_docker.sh` (passes `PHYSYK_ARM_KD`).
- Planner: `workspace/side_grasp.py`, `workspace/reach_map.py`, `workspace/reach_map_p40_r0.npz`, `workspace/reach_map_p40_r180.npz`.
- Perception: `workspace/physyk_perception/rack_detect.py` (new), `workspace/physyk_perception/step_verifier.py` (shelf-aware check).
- Agent/GUI: `workspace/physyk_agent_orchestrator.py` (refusal policy, destination parsing from B2, rack frame for Cosmos, scene description),
  `workspace/physyk_main_server.py` (rack camera dropdown, "objects" wording, example buttons removed, plan answer box),
  `workspace/DEST_CONTRACT.md`, `workspace/test_dest_parsing.py`.
- Other: `AGENT_PROMPTS.md` (Task A done by me; B/B2/C done by other chats; D docs not started), `workspace/scripts/planner_tests/*`.
