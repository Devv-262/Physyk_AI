---
title: Physyk Pure-Perception Pipeline
---

# Physyk — Pure-Perception Pick-and-Place

A second way to run the same robot: instead of reading object coordinates from the simulator, the arm finds the cubes **by looking** through the isometric overview camera. It is a GUI switch (**"Pure perception — camera only"**); with it off, the deterministic pipeline runs. Only *where the target coordinates come from* changes — motion planning, the grasp state machine, the stack planner and the agent loop are the same in both modes.

## The idea in one picture

```
overview camera (RGB + depth, fixed)
   -> colour mask per cube (red / green / blue)             # no model, tens of ms, no GPU
   -> depth of the mask pixels -> 3D points (pinhole model + known camera pose)
   -> fit a 5 cm square to the footprint -> cube centre XYZ (+ yaw)
   -> table-plane self-calibration corrects the camera's vertical error
   -> picking_position / placing_position
   -> PickPlaceController (10-phase state machine) + RMPFlow
   -> fresh camera look after the place -> verify -> retry / re-plan on failure
```

There is no learned model in this path. Detection is classical vision; movement is NVIDIA's `PickPlaceController` driven by RMPFlow, unchanged.

## Modes (`physyk_perception/provider.py`)

| Mode | Who drives the arm | Used by |
|---|---|---|
| `stage` / `shadow` | Simulator ground truth (deterministic). `shadow` also runs the detector alongside and logs the delta. | GUI switch **off** (default) |
| `vision` | Camera pose, falling back to the simulator's on a bad detection or an 8 cm sanity-gate miss. Legacy hybrid. | not used by the GUI |
| `pure` | Camera only. No simulator coordinates, no fallback. An object the camera cannot see is treated as **absent** and the task is refused. | GUI switch **on** |

The GUI switch calls `POST /perception_mode` on the backend, which calls `POST /perception/mode` on the sim. The header pill shows the current source ("Camera only" or "Sim positions").

## What is (and is not) fixed in pure mode

- **The table and the tray never move**, so the tray pose is a fixed calibration constant, `(0.42, -0.32, 0.015)` (`fixed_poses` in `isaac_sim_service.py`) — not read from the simulator and not detected.
- **Cubes** are always located from the camera, including after a randomize.
- Everything the agent sees in pure mode comes from the camera: `/state.objects` is tagged `source: camera` (tray: `fixed calibration`), so Nemotron's scene description, the stack planner, the guardrails and the already-done and drift checks all use camera positions.
- Still simulator-side by design: the randomize / reset commands, the planner's internal collision model, and the gripper's own width sensor (grasp confirmation). The `stage_pos_m` / `delta_mm` fields in `/perception` are a **scoring readout only** and are never consulted for decisions in pure mode.

## Camera and detector

- **Camera:** the isometric overview at (1.85, -1.55, 1.30) m looking at (0.35, 0, 0.28); 960x540; focal length 22 mm (fx about 1008 px). A cube is about 20 px wide. Depth is the `distance_to_image_plane` annotator on the same render product. Intrinsics and extrinsics are computed analytically from the camera's creation parameters (`SCENE_CAM_INTRINSICS`, `SCENE_CAM_WORLD_TRANSFORM`) and must be kept in sync with the camera setup. The camera is the one thing the scene work must not move.
- **Colour masks** (`physyk_perception/color_segment.py`): HSV thresholds measured from real frames, restricted to the table area of the image. Two tiers: strong colour for the cube's side faces, then a growth step into the cube's **top face**, which renders almost white under the studio lighting and would otherwise be missed.
- **Position:** every mask pixel is deprojected with its depth. The 3D footprint (top face plus the two camera-facing sides) is fitted with a square of the known 5 cm size, which works even when the top face is hidden by a cube stacked on it (the two sides form an "L" that pins the square). Averaging top-face pixels instead is biased toward the camera.
- **Height:** the measured top surface minus half a cube.
- **Yaw:** the square fit also gives the cube's yaw (modulo 90 degrees). It is reported in `/perception` and the Perception Log. **This controller does not use it**: the hand is not turned to the cube's faces.
- **Table-plane self-calibration** (`fit_table_plane`): the bare table is a flat plane at Z = 0. Its deprojected pixels are plane-fitted (iteratively, rejecting outliers) and the residual is subtracted from every detection. The pixels used are low-saturation ones inside the table region whose brightness is **at least half the table's own median brightness**. A fixed brightness floor used to be used here; under a darker room (a ceiling) the table fell below it, only about 1,000 pixels survived, the fitted plane tilted, and cube heights read 11-18 mm off. The relative floor restores about 15,000 pixels and 5-7 mm errors in the dark room (about 24,000 and 2.4-4 mm without the ceiling).
- **Cost:** the detector itself takes a few milliseconds; the GUI reports about 45-60 ms per refresh in the current scene (detection plus the periodic table fit).

## Stacking

The stack planner and the resolver work on camera poses as inputs: the place target is the measured top height of whatever is below plus one cube height (5 cm), at that cube's measured X/Y, and a cube already sitting on the reference is found from the camera and stacked on instead. The tray is only used for "in the tray" (a free slot at the fixed tray height, occupancy judged from camera positions). A cube the camera cannot see is absent from the scene, so the planner will not plan with it.

## Guardrail retry

The orchestrator rejects a step whose target is outside the reachable workspace. On a camera-derived scene a borderline reading can fail once and pass on the next look, so it re-observes and re-validates up to 3 times (0.8 s apart) before rejecting. It is skipped for the deterministic scene, where a retry cannot help.

## Logs and screenshots

- **Overview feed:** in pure mode it shows bounding boxes with camera-estimated XYZ (labels laid out in a column beside the cubes with leader lines) and a `PERCEPTION: CAMERA ONLY` banner. These are drawn by the sim into the frame, so they belong to the scene output. Off: unchanged.
- **Perception Log** (`/reasoning/perception`, a link in the header): for every pre-pick detection, post-place verification and mode switch — the annotated frame, per-object camera XYZ, yaw, and the hidden scoring error.
- **Cosmos Log** entries carry the exact image Cosmos was shown. Cosmos receives a clean frame (`/camera/scene_raw.jpg`, no overlay) so drawn labels cannot bias it.
- Cosmos is evidence only; it never decides what the robot does.

## Measured results (live sim, three cubes in the default layout, camera-only mode)

| Scene | Cube error vs the sim's true position | Table-fit pixels |
|---|---|---|
| Original blue-grid ground | 3.4-3.8 mm | — |
| Factory scene, no ceiling | 2.4-4.0 mm | about 24,000 |
| Factory scene, ceiling, fixed brightness floor (before the fix) | 11-18 mm | about 1,000 |
| Factory scene, ceiling, relative brightness floor (current) | 5.0-6.9 mm | about 15,000 |

These are one default layout each, not a batch. Earlier work on the first camera-only setup measured 3.0 mm mean and 4.8 mm worst over 4 random layouts x 3 cubes, with a three-cube stack passing 3/3.

## Known limits

- Tuned to the default **studio** lighting; the neon and warm presets shift hues. Use studio for perception. Anything that darkens the table further needs a re-check of the colour thresholds.
- Coloured cubes only, on the known table; not a general object detector. (A grounding model plus segmentation would be the upgrade path for arbitrary objects; not built.)
- Lower cubes in a stack are less accurate (about 8-9 mm in earlier synthetic tests) because their top faces are hidden.
- The tray is fixed by assumption; moving it would need detection re-enabled.
- Perception refreshes at dispatch, at verification, and every 0.5-1 s in pure mode; the arm can occlude a cube momentarily.
- Scenery in the factory backdrop is kept outside the table region the colour detector looks at; its few saturated accents (blue bins, the signal tower) are far from it.

## Files

| File | Role |
|---|---|
| `workspace/physyk_perception/color_segment.py` | colour masks, top-face growth, square fit, table-plane fit |
| `workspace/physyk_perception/provider.py` | `PerceptionProvider`: modes, fixed poses, snapshot, calibration |
| `workspace/physyk_perception/deproject.py` | pinhole deprojection and projection |
| `workspace/physyk_perception/overlay_layout.py` | non-overlapping label placement for the overlay |
| `workspace/isaac_sim_service.py` | wiring: decision poses, verification, overlay, log, `/perception*` endpoints |
| `workspace/physyk_agent_orchestrator.py` | guardrail retry, Cosmos evidence with images |
| `workspace/physyk_main_server.py` | GUI switch, `/perception_mode`, Perception Log page |
