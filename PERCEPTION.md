---
title: Physyk Pure-Perception Pipeline
---

# Physyk — Pure-Perception Pick-and-Place

A second way to run the same robot: instead of reading object coordinates from the simulator,
the arm finds the cubes **by looking** through the isometric overview camera. It is a GUI switch
(**"Pure perception — camera only"**); with it off, the original deterministic pipeline runs exactly
as before. Only *where the target coordinates come from* changes — motion planning, the grasp state
machine, and the Nemotron agent loop are the same in both modes.

## The idea in one picture

```
overview camera (RGB + depth, fixed)
   -> color mask per cube (red / green / blue)           # no model, ~ms, no GPU
   -> depth of the mask pixels -> 3D points (pinhole model + known camera pose)
   -> fit a 5 cm square to the footprint -> cube center XYZ (+ yaw, used to turn the hand)
   -> table-plane self-calibration corrects the camera's vertical error
   -> picking_position / placing_position
   -> PickPlaceController (10-phase state machine) + RMPFlow (collision-aware planner)
   -> fresh camera look after the place -> verify -> Nemotron retry/replan on failure
```

There is no learned policy in this path. Detection is classical vision; movement is
NVIDIA's `PickPlaceController` driven by RMPFlow, unchanged.

## Modes (`physyk_perception/provider.py`)

| Mode | Who drives the arm | Used by |
|---|---|---|
| `stage` / `shadow` | Simulator ground truth (deterministic). `shadow` also runs perception alongside and logs the delta. | GUI switch **off** (default) |
| `vision` | Camera pose, falling back to the simulator's on a bad detection or an 8 cm sanity-gate miss. Legacy hybrid. | not used by the GUI |
| `pure` | Camera only. No simulator coordinates, no fallback. An object the camera can't see is treated as **absent** and the task is rejected. | GUI switch **on** |

The GUI switch calls `POST /perception_mode` on the backend, which calls `POST /perception/mode` on the sim.

## What is (and isn't) fixed in pure mode

- **Table and tray never move**, so the tray pose is a fixed calibration constant, `(0.42, -0.32, 0.015)`
  (`fixed_poses` in `isaac_sim_service.py`) — not read from the simulator and not detected.
- **Cubes** are always located from the camera, including after a randomize.
- Everything the agent sees in pure mode comes from the camera: `/state.objects` is tagged `source: camera`
  (tray: `fixed calibration`), so Nemotron's scene description, the guardrails, the already-done and drift
  checks all use camera positions.
- Still simulator-side by design: the randomize/reset commands, the motion planner's internal collision
  model (RMPFlow reads the sim's obstacle transforms), and the gripper's own width sensor (grasp
  confirmation). The `stage_pos_m` / `delta_mm` fields in `/perception` are a **scoring readout only** and are
  never consulted for decisions in pure mode.

## Camera and detector

- **Camera:** the isometric overview at (1.85, -1.55, 1.30) m looking at (0.35, 0, 0.28); 960x540; focal 22 mm
  (fx ~ 1008 px). A cube is ~20 px wide. Depth is the `distance_to_image_plane` annotator on the same render
  product. Intrinsics/extrinsics are computed analytically from the camera's creation parameters
  (`SCENE_CAM_INTRINSICS`, `SCENE_CAM_WORLD_TRANSFORM`) and must be kept in sync with the camera setup.
- **Color masks** (`physyk_perception/color_segment.py`): HSV thresholds measured from real frames, restricted
  to the table area in the image (the blue ground grid has nearly the same hue as the blue cube). Two tiers:
  strong color for the cube's side faces, then a growth step into the cube's **top face**, which renders
  almost white under the studio lighting (saturation 40-56) and would otherwise be missed.
- **Position:** every mask pixel is deprojected with its depth. The 3D footprint (top + the two camera-facing
  sides) is fitted with a square of the known 5 cm size — this works even when the top face is hidden by a
  cube stacked on it (the two sides form an "L" that pins the square). Averaging top-face pixels instead is
  biased toward the camera (perspective gives the far part of the face fewer pixels).
- **Height:** the measured top surface minus half a cube.
- **Table-plane self-calibration** (`fit_table_plane`): the bare table is a flat plane at Z = 0; its
  deprojected pixels are plane-fitted (with outlier rejection) and the residual is subtracted from every
  detection. On this scene the fit is nearly perfect (offset ~1 mm), i.e. the camera pose is good — it is a
  safety net, not the main fix.
- Cost: ~20 ms per detection.

## Stacking

Uses the deterministic stacking logic unchanged, with camera poses as inputs: the place target is the
measured top height of whatever is below plus one cube height (5 cm), at that cube's measured X/Y; a cube
already sitting on the reference is found from the camera and stacked on instead. The tray is only used for
"put it in the tray" (a free slot at the fixed tray height, occupancy judged from camera positions).

## Guardrail retry

The orchestrator rejects a step whose target is outside the reachable workspace. On a camera-derived scene a
borderline reading can fail once and pass on the next look, so it now re-observes and re-validates up to 3
times (0.8 s apart) before rejecting. It is skipped for the deterministic scene, where a retry can't help.

## Logs and screenshots

- **Overview feed:** in pure mode it shows bounding boxes with camera-estimated XYZ (labels laid out in a
  column beside the cubes with leader lines) and a `PERCEPTION: CAMERA ONLY` banner. Off: unchanged.
- **Perception Log** (`/reasoning/perception`, link at the top of the GUI): for every pre-pick detection,
  post-place verification and mode switch — the annotated frame, per-object camera XYZ, yaw, and the hidden
  scoring error.
- **Cosmos Log** entries carry the exact image Cosmos was shown. Cosmos receives a clean frame
  (`/camera/scene_raw.jpg`, no overlay) so drawn labels can't bias it.
- Cosmos is evidence only (pre-dispatch assessment, post-motion "is it actually there?"); it never decides
  what the robot does.

## Measured results (headless runs, RTX 5090 laptop; the detection numbers date from before the thresholds were rebuilt)

| Measure | Result |
|---|---|
| Detection latency | ~20-24 ms |
| Cube position error vs hidden truth (4 random layouts x 3 cubes) | 3.0 mm mean, 4.8 mm max; **height error 0 mm** |
| Before the top-face fix | 8.4 mm mean; height read ~7 mm low, which made resting cubes fail the guardrail floor (Z 0.02) |
| Single pick, camera-only | success; landed within ~4-5 mm of target; grasp confirmed |
| **Stack red -> blue -> green on the tray, full agent path** | **SUCCESS 3/3**; layers 52 mm apart (ideal 50); horizontal misalignment 4-5 mm |
| Time (after the frame-rate fix below) | ~24-29 s per pick; the whole 3-cube stack in **87 s** including Nemotron planning (was ~130 s per pick and 6.8 min before) |
| Latest runs (arrival gating, JPEG every 2nd step), camera-only, random rotated layouts | 6 of 6 picks passed, 1.4-10.3 mm, 22-30 s per pick (deterministic: 17-20 s) |

### Why it was slow, and what fixed it

A pick is ~750 controller steps (the `PickPlaceController` counts its phases per loop iteration), so wall time =
750 / (loop iterations per second). The loop was doing ~6-7 iterations/s, i.e. ~2 minutes per pick, although the
robot's own simulated motion is only ~12 s. Profiling found two causes:

1. **The loop rendered every camera twice** — `world.step(render=True)` already renders and fills the annotators, and
   `rep.orchestrator.step()` then rendered all camera products again. Skipping it took the loop from ~6.9 to ~32
   iterations/s with identical picks (fresh frames verified; deterministic and camera-only picks both succeed).
2. **Five cameras, two needed** — the side-profile and top-down cameras were removed, then the front camera too; only the
   overview and wrist cameras remain, which brought the loop to ~43 iterations/s. Perception only uses the overview camera.

Also: **plug the laptop in.** On battery the GPU was capped to ~800 of 3090 MHz (SW power cap) and the loop dropped to
~2.7 iterations/s. Things that did *not* work: raising the motion speed factor to 1.5-1.9x (the controller is an open-loop
timer, so faster phases outrun the arm and the pick misses), and physics-only sub-stepping (`world.step(render=False)`
does not advance the arm in this Isaac Sim build). `PHYSYK_PROFILE=1` prints a per-iteration time breakdown.

Lessons worth remembering: (1) the first suspicion (camera calibration) was wrong — fitting the table plane
showed the camera was fine, and sampling the mask over each face showed it covered 88-96% of the sides but
2-11% of the top; (2) a synthetic test that renders realistic per-face colors reproduced the bug offline
before the fix and now guards against regressions.

## Known limits

- Tuned to the default **studio** lighting; the neon/warm presets shift hues. Use studio for perception.
- The HSV thresholds in `COLOR_SPECS` were rebuilt on 2026-10-13 from the measured pixel values recorded in the file's comments
  (the originals were lost when the scene briefly held glassware) and verified on one saved cube frame. Re-measure them on the
  lab background if detection looks off.
- Colored cubes only, on the known table; not a general object detector. (Grounding DINO / SAM2 / a grasp
  network would be the upgrade path for arbitrary objects; not built.)
- Lower cubes in a stack are less accurate (~8-9 mm in synthetic tests) because their top faces are hidden.
- The tray is fixed by assumption; moving it would need detection re-enabled.
- Perception refreshes at dispatch, at verification, and every ~0.5-1 s in pure mode; the arm can occlude a
  cube momentarily.
- Cube yaw comes from the square fit and is used to turn the hand to the cube's faces (see `TECHNICAL_DEEP_DIVE.md` §1.4). It is
  only defined modulo 90 degrees; the planner treats it that way.
- The perception overlay labels of two objects at nearly the same image spot are separated by the layout, not
  merged.

## Testing without the simulator

`workspace/scripts/perception_synth_test.py` ray-casts the overview camera against the cubes and tray
(with per-face colors measured from a live frame), then checks what the detector recovers for random
scattered layouts and for stacks:

```bash
.venv/bin/python workspace/scripts/perception_synth_test.py
```

Expect roughly 4-5 mm on loose cubes and 4-10 mm for stacks. For a live check, switch perception on in the GUI,
randomize the cubes, and compare `camera_xyz_m` against `score_error_mm` in the Perception Log.

## Files

| File | Role |
|---|---|
| `workspace/physyk_perception/color_segment.py` | color masks, top-face growth, square fit, table-plane fit |
| `workspace/physyk_perception/provider.py` | `PerceptionProvider`: modes, fixed poses, snapshot, calibration |
| `workspace/physyk_perception/overlay_layout.py` | non-overlapping label placement for the overlay |
| `workspace/isaac_sim_service.py` | wiring: decision poses, verification, overlay, log, `/perception*` endpoints |
| `workspace/physyk_agent_orchestrator.py` | guardrail retry, Cosmos evidence with images |
| `workspace/physyk_main_server.py` | GUI switch, `/perception_mode`, Perception Log page |
| `workspace/scripts/perception_synth_test.py` | offline accuracy test |
