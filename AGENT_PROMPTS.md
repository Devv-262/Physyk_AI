# Prompts for parallel agents

Shared context (paste at the top of each prompt, or tell the agent to read this file first).

## Shared context

Project: Physyk AI, in `/home/thinkspace/Desktop/Physyk_AI-master`. A Franka Panda in Isaac Sim 6.0.1 (Docker,
API on :8100) is driven by natural language. Nemotron (hosted) plans, guardrails check each step, the sim executes,
and results are verified. Cosmos (local NIM, :8001) gives evidence only. Backend/GUI is `workspace/physyk_main_server.py`
on :7860. The GUI has a "Pure perception - camera only" switch: in that mode positions come from cameras, never from
the simulator's scene graph.

Scene now: five vessels on the table (`flask_blue`, `flask_red`, `flask_empty`, `beaker_red`, `beaker_green`; registry
`SCENE_OBJECTS` in `workspace/isaac_sim_service.py`) and a 3-tier shelf behind the robot. The cubes and tray are gone.
In user-facing text say "objects", never "glassware".

Shelf geometry (world frame, metres; robot base at the origin, table top at z=0):
- X from -0.80 to -0.50, |Y| <= 0.40, open on the +X side (the robot reaches in from +X).
- Rack top surfaces at z = 0.02 (bottom), 0.17 (middle), 0.38 (top, wall-mounted). Planks 0.02 thick.
- Flask: 5x5x9.5 cm, origin at base centre. Beaker: 5x5x7 cm.
Rack camera: at (-0.13, 0, 0.27) looking at (-0.65, 0, 0.24), focal 8 mm, horizontal aperture 20.955 mm, 960x540.
Stream `/camera/rack.stream`, still `/camera/rack.jpg`. In `isaac_sim_service.py` see `RACK_CAM_INTRINSICS`,
`RACK_CAM_WORLD_TRANSFORM`, `rack_annot` (RGB), `rack_depth_annot` (depth). `POST /perception/debug_dump` saves
`workspace/perception_calib/dump_rack_rgb.npy` and `dump_rack_depth.npy`.

Another agent (the main session) is editing `isaac_sim_service.py` (grasp sequencer) and `side_grasp.py`. Do not edit
those two files, and do not start the sim container (port 8100 and the GPU are in use). Work on your own files only
and unit-test offline with saved frames or synthetic data. Do not claim anything works live unless you ran it.

---

## Task A - Rack perception (new files only)

Goal: build `workspace/physyk_perception/rack_detect.py`: detect vessels on the shelf from the rack camera (RGB + depth).
Reuse the ideas in `workspace/physyk_perception/glass_detect.py` (height above a surface from depth, connected
components, classify flask vs beaker by top height and top-slice width, name by liquid colour). Differences: the
reference surface is each tier's own plank (z = 0.02, 0.17, 0.38), so compute height above the tier the pixel
belongs to. Output per vessel: key (same keys as `VESSEL_KEYS` in glass_detect.py), world position (x, y, z of base
centre), tier name (bottom/middle/top), confidence. Also expose `slot_occupancy(detections, slot_xy, tier)` and
`verify_placement(detections, key, expected_xy, tier, tol=0.03)` returning (ok, error_m).

Deliverables: the module; a synthetic test (render fake depth from known vessel poses through
`RACK_CAM_INTRINSICS`/`RACK_CAM_WORLD_TRANSFORM`, run the detector, check mm-level error on all three tiers);
a short doc block at the top of the file. Integration into `provider.py` is a separate step: write
`integration_notes.md` next to the module describing exactly which lines to add, but do not edit `provider.py`.
Report numbers you measured, including failure cases.

## Task B - Destination parsing and orchestrator (orchestrator + prompt only)

Goal: make instructions like "pick the flask with red liquid and place it on the middle shelf" parse into a
structured plan. Files: `workspace/physyk_agent_orchestrator.py` (planner prompt, `validate_subgoal`, destination
handling) only. Define the destination vocabulary: "top/middle/bottom shelf", optionally "left/right/centre of the
shelf", "next to <object>", "on the table", "back to its spot". Each sub-goal must carry `dest_kind`
("shelf"|"table"|"relative"), `shelf_tier`, `shelf_slot` (0..N-1 evenly spaced along Y between -0.30 and 0.30),
and the guardrail must reject an occupied slot or a tier that does not exist, with a clear message. Keep the existing
retry rules (retry only for perception-type failures). The sim side (`resolve_pick_place_targets` in
`isaac_sim_service.py`) will consume these fields; write the exact JSON contract into `workspace/DEST_CONTRACT.md`
and tell the main session about any field you need that does not exist. Test offline with a stubbed Nemotron (no
network) for at least 15 phrasings, including nonsense and impossible ones ("put it on the fourth shelf").

## Task C - GUI cleanup (frontend only)

Goal: `workspace/physyk_main_server.py` HTML/JS only. The quick-task buttons and example prompts are still
cube/tray flavoured; replace them with vessel/shelf examples ("Pick the red flask and place it on the middle shelf",
"Put the green beaker on the top shelf", "Move the empty flask to the bottom shelf"). Say "objects" not
"glassware". The Perception Log page and overlay should show the tier for shelf detections once that data exists
(read it defensively: the field may be missing). Do not change API routes. Verify by loading the page's HTML and
checking the JS parses (e.g. extract the script and run it through `node --check`). Report what you changed.

## Task D - Docs pass (docs only; do this last, after the sim works)

Goal: update `README.md`, `HOW_IT_WORKS.md`, `instructions.md` (and its copy in `workspace/instructions.md`),
`ARCHITECTURE.md`, `TECHNICAL_DEEP_DIVE.md` to describe the current system: vessels, side grasps, shelf, rack
camera, pure perception, no cubes/tray. Remove references to the deleted `PERCEPTION.md`, `cosmos_integration.md`,
`CONTEXT_BRIEFING.md`, `INDUSTRY_APPLICATIONS.md`. Facts must come from the code and from measured numbers the main
session gives you; if a number is not given, say it is unmeasured rather than inventing it.

---

## Task B2 - Orchestrator follow-up (give to the chat that did Task B)

Context update: the shelf moved and the sim now decides placement itself. Read `AGENT_PROMPTS.md` Shared context, then
`workspace/DEST_CONTRACT.md` (your own file) and this update. You may edit only `workspace/physyk_agent_orchestrator.py`,
`workspace/test_dest_parsing.py` and `workspace/DEST_CONTRACT.md`. Do not touch `isaac_sim_service.py`, `side_grasp.py`,
`reach_map.py` or `physyk_main_server.py`, and do not start the sim.

What changed in the sim (all in `isaac_sim_service.py`, already done):
1. The shelf is now on the robot's +Y side, wall-mounted and STEPPED (each higher tier starts further back).
   Tier tops z = 0.08 (bottom), 0.36 (middle), 0.64 (top). Slots are in x (0.25, 0.05, 0.45 in preference order) and
   the slot y is per tier (0.515 / 0.605 / 0.695). "Standing on tier T" means `|z - top_T| < 0.04` and `y > 0.40`.
   The old numbers in DEST_CONTRACT.md section 3 (z = 0.02/0.17/0.38, x = -0.60, slots by y) are WRONG now.
2. The sim, not the orchestrator, decides whether a slot is free: a slot is free only if the vessel can be set
   down there without the arm, hand or vessel hitting anything already there. The orchestrator must NOT reject a
   shelf destination as "occupied" and must not choose slots. Remove `|y - slot_y| < 0.075` occupancy logic and
   slot assignment; keep the tier (bottom/middle/top) only.
3. A tier is optional. "on the shelf" / "in the rack" with no tier means tier = "auto": the sim tries every
   tier and slot, easiest first, and uses the first collision-free one. Emit `dest_kind="shelf"`, `shelf_tier=null`,
   canonical text `on the shelf` for that. The sim already parses `on the top|middle|bottom shelf` and plain `on the shelf`.
4. The sim can now refuse a step, and says why in `/state`: `last_task_refused` (bool) and `last_task_error` (text),
   e.g. "the middle shelf is full: there is no slot where the Red Flask can be set down without hitting something
   already there (last blocker: the hand would hit Blue Flask)" or "cannot pick up the Blue Flask: the arm would hit
   Red Flask". The orchestrator already stops retrying a refusal; extend it so that:
   - the refusal text is quoted verbatim in the step detail and in the final answer Nemotron gives the user
     (in plain language: what was asked, what blocked it, what the user could change);
   - after a refusal Nemotron may propose ONE alternative (another tier, or move the blocker first) via
     `replan_step` / `replan_remaining`, but only if it is a different action; it must never repeat the refused step.
5. Vague multi-object prompts. "place all objects in the rack", "put everything on the shelf", "tidy up" must become
   one `pick_place` sub-goal per object currently on the table (from the live scene), each with
   `dest_kind="shelf"`, `shelf_tier=null`, ordered so the sim's easiest-first filling stays neat: nearest to the
   robot first, flasks before beakers. If an object is refused because the shelf is full, the remaining steps
   still run (they may fit elsewhere) and the final answer lists which objects were placed and which were not,
   each with the reason.
6. Pick refusals: if the sim says the pick itself is blocked, report that clearly; do not retry the same pick.

Deliverables: updated code, updated `DEST_CONTRACT.md` (new geometry, refusal fields, "auto" tier, multi-object
expansion), and extended offline tests (stubbed Nemotron + fake sim returning refusals) covering: explicit tier,
"auto" tier, "place all objects in the rack", a full shelf, a blocked pick, nonsense input. Run them and report
the real results.
