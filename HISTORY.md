# Project history — the setups Physyk has gone through

A record of how the project got to its current state, so the earlier setups are not lost. What is current is described in `README.md`, `ARCHITECTURE.md` and `TECHNICAL_DEEP_DIVE.md`; open problems are in `TECHNICAL_DEEP_DIVE.md` §5.

## 1. First setup — cloud VM, blue grid (initial commit 2026-08-22)

- **Where it ran:** a Brev cloud VM (RTX PRO 6000 Blackwell, 96 GB), Isaac Sim 6.0.1.
- **Scene:** three coloured cubes and a yellow tray on a blue-grid ground plane, a Franka Panda, five cameras (overview, front, side, top, wrist).
- **Brains:** Nemotron-30B served locally with vLLM (port 8000); Cosmos-Reason2-2B planned on port 8001. Object positions came from the simulator; Cosmos ran in "shadow" mode (logged a delta) or "vision" mode (drove the arm, with a fallback to the simulator pose).
- **Motion:** NVIDIA's validated 10-phase `PickPlaceController` on RMPFlow, with the grasp-confirmation and real post-place verification that replaced an always-pass stub. A learned-policy (GR00T) path was tried as an experiment, did not complete the task reliably on this scene, and has since been removed from the code.
- **Agent:** Nemotron decomposes a sentence into steps; deterministic guardrails; dispatch, verify, retry / replan.
- **Where the code is:** commit `80a0e7f` (the original GitHub master). The current code is built on top of it.

## 2. Move to the laptop and the lab environment (September – October 2026)

- **Runs on one laptop** (RTX 5090 laptop GPU, 24 GB, Ubuntu 26.04): Isaac Sim in the official 6.0.1 container because the host's glibc is newer than Isaac Sim supports; Cosmos 3 Nano Reasoner 8B as a local NIM (FP8, 8192-token context, 65% GPU share, CUDA graphs off); Nemotron-3 Super 120B through NVIDIA's hosted API.
- **Camera-only perception ("pure perception"):** colour masks + depth find the cubes (3.0 mm mean error, about 20 ms); the tray is a fixed pose; verification is a fresh camera look.
- **Speed:** a pick went from about 127 s to about 25 s by skipping the duplicate camera render and dropping cameras.
- **Lab environment:** the blue grid was replaced by a life-science lab (floor, walls, bench, shelving, fume hood), visual only.
- **Stacking** of three cubes in the tray worked end to end in camera-only mode.

## 3. Glassware lab (branches `feature-lab` and `main`, consolidated 2026-10-12)

- **Scene:** the cubes and tray were replaced by five pieces of glassware (three Florence flasks, two beakers) and a three-tier stepped wall-mounted shelf.
- **New machinery:** a sideways neck / body grasp planner (Lula IK, precomputed reach maps, level carries), glass detection from depth geometry, shelf and table slot search by collision planning, a refusal policy.
- **Result:** placements within 0.7-17.9 mm in headless tests. That work lives on those branches only and is not part of this one.

## 4. The cube / tray lab rebuild (branch `master`, 2026-10-13 to 2026-10-14) — superseded

The cube pipeline was rebuilt on the then-current tree from the August snapshot plus the later camera-only work, with the lab background and no shelf (commits `279f5f2` .. `ef67e16`). It added:

- stack planning from the cubes' positions, easiest first, for the tray, the table or neither; blocker clearing;
- a hand yaw aligned to the cube and away from neighbours; Randomize that rotated the cubes;
- controller changes: arrival checks on the approach and transport phases, a lift-clearance rule, release on contact, the gripper kept open when empty;
- an investigation of the arm's physics: stiffness and damping are fine; the controller's open-loop timer is the source of the lag; RMPFlow obstacles are cleared by the controller's reset and made results worse when re-registered; cubes closer than 0.36 m to the base drive joint 6 into its limit;
- a per-joint trace and anomaly log, repeatable layouts, arm test tools, a headless sim, a Cosmos prompt that describes the scene.

Its latest end-to-end result was 24 of 25 picks passed on 10 random layouts at about 20 s per pick. Live runs also showed the arm overshooting long sideways carries (about 21 cm at the velocity cap, 4 of 4 runs) and approaches, which this line of work never resolved.

## 5. Restart from the 22 August code (2026-10-14) — the current line

Decision: the 22 August code is a better base to develop from than the rebuilt controller, so `master` was reset to that code (as a merge that keeps the whole earlier history as a parent: nothing was force-pushed or lost). The tip of the previous master is `ef67e16` and a local tag `backup-before-aug22-rewind` points at it.

**Kept from the August code:** the arm controller (the original `PickPlaceController` set-up, with its speed-up and lengthened settle / close phases), the simulator service, the scene's table, tray and cubes, and the GUI backend (restyled).

**Brought over from the rebuild (perception, agent and tooling, not the arm changes):**
- the Docker launchers for the sim and Cosmos, `physyk_config.py` and `.env.example`, the laptop `run.sh`;
- camera-only perception: the colour detector, the provider, the overlay layout, the table-plane calibration, the Perception Log page, the GUI switch;
- the rebuild's orchestrator (it replaces the August one, with the new stack planner wired in): stack-aware blocker clearing, guardrail retry on camera scenes, refusals reported verbatim, the plain-language final answer, Cosmos evidence with the exact image shown, the Cosmos scene key;
- the hosted Nemotron client (now `nvidia/nemotron-3.5-lightning-30b-a3b`) and the local Cosmos NIM client.

**Removed:** the GR00T / VLA path and everything for it (policy servers, episode recorder, fine-tuning scripts, the submodule), three of the five cameras (front, side, top), and 73 files of first-setup leftovers: the cloud-VM launch and watchdog scripts, ROS 2 and Lula experiments, older servers and pipelines, one-off tests, logs and diagnostic scripts. None of it was used by the live code.

**New in this line:**
- **The local stack planner** (`stack_planner.py`), because Nemotron was too slow and unreliable for stacking: columns and covered cubes, surface-aware pile choice, an order that avoids picking covered cubes and keeps the gripper clear of neighbours, a refusal for unreachable cubes, closed-loop re-planning after every step, and offline tests (planner and agent loop) with a mutation check. An infinite re-plan loop it found in the agent was fixed.
- **A shared safe zone** for Randomize and free spots (away from the base and the tray, always reachable).
- **Speed:** the duplicate render per frame removed, two cameras instead of five, JPEG streams encoded every second iteration, and Cosmos checks moved into background threads. The idle loop went from 6-7 to about 60 iterations/s.
- **The factory scene** (`factory_background.py`): visual only, hot-rebuildable, built at half scale for the camera's view; a visual extension under the arm base; the detector's table calibration made brightness-relative after a ceiling broke it.
- **A restyled GUI:** the NTT DATA logo and theme (dark neutral base, muted steel-blue accents, no emoji), a compact grouped header, the stack line read from the configured models, tidied copy.
- **Documentation:** all docs rewritten for this line.

**Added after the restart (2026-10-15):**
- Nemotron planning changed to one move at a time (`nemotron_planner.py`): measured on the hosted 30B, thinking off answers in about 1 s and thinking on took 25-60 s and sometimes never answered; the model alone tracking progress looped, so code tracks the goals.
- Perception: the dim room put a stacked cube's sides below the brightness floor and a blue object off the table out-sized the cube, so the lower cube of a stack was dropped; fixed on real frames (blobs with a low floor, the first blob that is a real cube on the table wins) and the height limit raised so a carried cube is tracked.
- GUI: label layout beside each box, status colours, About dialog, smaller stage pill.

**Not brought over (still in the old history if wanted):** the hand-yaw alignment, the arrival and lift-clearance checks, release on contact, the obstacle modes and reach-minimum randomize, the joint trace and anomaly log, and the arm test tools. See `git show ef67e16:<path>`; the arm physics findings are summarised in `TECHNICAL_DEEP_DIVE.md` §5.

## Where to find earlier work

| What | Where |
|---|---|
| Original August code | commit `80a0e7f` |
| The cube / tray rebuild, including its arm changes and arm test tools | commits `279f5f2` .. `ef67e16` on `master` before the restart; local tag `backup-before-aug22-rewind` |
| Glassware lab | branches `feature-lab` and `main` (not part of this line) |
| Removed first-setup files | `git show 80a0e7f:<path>` |

Docs that were removed earlier and live on other branches or in history: `CONTEXT_BRIEFING.md`, `AGENT_PROMPTS.md`, `TODO_GRASPING.md`.
