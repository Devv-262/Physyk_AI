# Project history — the setups Physyk has gone through

A record of how the project got to its current state, so the earlier setups are not lost now that their code has been removed from this
branch. What is current is described in `README.md`, `ARCHITECTURE.md` and `TECHNICAL_DEEP_DIVE.md`; open problems are in
`TECHNICAL_DEEP_DIVE.md` §5.

## 1. First setup — cloud VM, blue grid (initial commit 2026-08-22)

- **Where it ran:** a Brev cloud VM (RTX PRO 6000 Blackwell, 96 GB), Isaac Sim 6.0.1.
- **Scene:** three coloured cubes and a yellow tray on a blue-grid ground plane, a Franka Panda, five cameras (overview, front, side,
  top, wrist).
- **Brains:** Nemotron-30B served locally with vLLM (port 8000); Cosmos-Reason2-2B planned on port 8001 (the 2B model chosen over the 8B to
  fit next to Nemotron's GPU share). Object positions came from the simulator; Cosmos ran in "shadow" mode (logged a delta) or "vision"
  mode (drove the arm, with a fallback to the simulator pose).
- **Motion:** NVIDIA's validated 10-phase `PickPlaceController` on RMPFlow, with the grasp-confirmation and real post-place verification
  that replaced an always-pass stub. A learned-policy (GR00T) path was tried as an experiment, did not complete the task reliably on this
  scene, and has since been removed from the code.
- **Agent:** Nemotron decomposes a sentence into steps; deterministic guardrails; dispatch, verify, retry / replan.
- **Where the code is:** commit `80a0e7f` (the original GitHub master).

## 2. Move to the laptop and the lab environment (September – October 2026)

- **Runs on one laptop** (RTX 5090 laptop GPU, 24 GB, Ubuntu 26.04): Isaac Sim in the official 6.0.1 container because the host's glibc is
  newer than Isaac Sim supports; Cosmos 3 Nano Reasoner 8B as a local NIM (FP8, 8192-token context, 65% GPU share, CUDA graphs off);
  Nemotron-3 Super 120B through NVIDIA's hosted API.
- **Camera-only perception ("pure perception"):** colour masks + depth find the cubes (3.0 mm mean error, ~20 ms); the tray is a fixed
  pose; verification is a fresh camera look. Details and measurements: `PERCEPTION.md`.
- **Speed:** a pick went from ~127 s to ~25 s by skipping the duplicate camera render and dropping cameras (five -> three -> two).
- **Lab environment:** the blue grid was replaced by a life-science lab (floor, walls, a bench body under the work table extended behind the
  robot, wall shelving and a fume hood). It is visual only (no physics); `PHYSYK_BACKGROUND=grid` restores the old grid. Colours in the
  background are kept neutral because the cube detector looks for saturated red, green and blue.
- **Stacking** of three cubes in the tray worked end to end in camera-only mode (layers 52 mm apart, 4-5 mm off-centre).

## 3. Glassware lab (branches `feature-lab` and `main`, consolidated 2026-10-12)

- **Scene:** the cubes and tray were replaced by five pieces of glassware (three Florence flasks: blue liquid, red liquid, empty; two
  beakers: red and green liquid) and a three-tier stepped wall-mounted shelf, with a rack camera looking at the shelf.
- **New machinery:** a sideways neck / body grasp planner (Lula IK, precomputed reach maps, level carries so liquid does not spill),
  glass detection from depth geometry named by liquid colour (99.5% of 200 live detections correct, 1.7 mm mean error), shelf and table
  slot search by collision planning, a refusal policy (stop and show the reason).
- **Result:** placements within 0.7-17.9 mm in headless tests; open problems and a to-do list were written up in `TODO_GRASPING.md`,
  which exists only on those branches.

## 4. Back to the cube / tray lab (branch `master`, 2026-10-13 onwards)

The cube pipeline was rebuilt on the current tree (the git history held only the glassware tree, so the original cube code was recovered from
the August snapshot and the later camera-only work was kept), with the lab background and no shelf. Since then:

- stack planning from the cubes' positions, easiest first, for the tray, the table or neither; blocker clearing;
- the hand turns to the cube's faces and away from neighbours; Randomize rotates the cubes;
- the controller waits for the arm (arrival checks on the approach and transport), lifts clear of other cubes, releases on contact and keeps
  the gripper open when empty;
- an investigation of the arm's physics (`TECHNICAL_DEEP_DIVE.md` §1.8): stiffness and damping are fine; the controller's open-loop timer
  is the source of the lag; RMPFlow obstacles were never active (the controller's reset cleared them) and made results worse when they were
  switched on; cubes closer than 0.36 m to the base drive joint 6 into its limit;
- headless sim by default, per-joint trace and anomaly log, repeatable layouts, test tools in `workspace/scripts/arm_tests/`;
- Cosmos prompts that describe the scene, a status light that follows the sim container, docs rewritten, and the unused code removed
  (below).

Latest end-to-end result: 24 of 25 picks passed on 10 random layouts at about 20 s per pick.

## What was removed from this branch, and where it is

Removed on 2026-10-14 (still in git history: `git show 4ebe30a:<path>`, the last commit that had them):

- the learned-policy experiment: its policy servers, the fine-tuning and episode-recording tools, its launch script, and the hooks for it
  in the sim service and backend;
- earlier setups and one-off tests: the first web server and pipeline, a standalone Franka demo, ROS 2 bridge, motion-policy and
  Lula-controller experiments, interactive and rigorous test scripts, camera tests;
- the retired cloud-VM / local-vLLM scripts (local Nemotron server and watchdog, Cosmos vLLM launcher, the old runners, status and start
  scripts) and the diagnostic shell scripts for the VM;
- the glassware meshes and their generator scripts (the glassware lab itself lives on `feature-lab` / `main`);
- tracked log files and the empty submodule folders.

Docs that were removed earlier and live on other branches / in history: `CONTEXT_BRIEFING.md`, `AGENT_PROMPTS.md`, `TODO_GRASPING.md`.
