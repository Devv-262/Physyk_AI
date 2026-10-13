# Destination contract (orchestrator <-> sim)

Produced by `workspace/physyk_agent_orchestrator.py` (`parse_destination`, `check_destination`,
`destination_payload`, `compose_final_answer`). Consumed by `resolve_pick_place_targets` in `isaac_sim_service.py`.
Tested offline by `workspace/test_dest_parsing.py` (stubbed Nemotron + scripted fake sim, no network, no container).

Revision B2: the shelf moved and is now stepped on the +Y side; **the sim, not the orchestrator, decides whether a
slot is free**, so the orchestrator no longer knows about slots. Everything about slots in the first version of this
file (z = 0.02/0.17/0.38, x = -0.60, slot y) is gone.

## 1. What the planner emits

Each `pick_place` sub-goal is a JSON object. The orchestrator normalizes every one of them (initial plan,
`replan_step`, `replan_remaining`), so **all fields below are always present**, `null` when not applicable.

| field | values | meaning |
|---|---|---|
| `action` | `"pick_place"` | |
| `target` | free text, e.g. `"red flask"` | object to move (grounded against the live scene) |
| `destination` | free text | **canonical** text: `on the shelf`, `on the top shelf`, `next to the red beaker`, `on the table`, `back to its spot`, or `""` |
| `dest_kind` | `"shelf"` \| `"table"` \| `"relative"` \| `null` | `null` = no destination stated (pick, set back down in place) |
| `shelf_tier` | `"bottom"` \| `"middle"` \| `"top"` \| `null` | only for `dest_kind="shelf"`. **`null` = "auto"**: the user named no tier ("on the shelf", "in the rack"); the sim tries every tier and slot, easiest first |
| `relation` | `right_of` \| `left_of` \| `in_front_of` \| `behind` \| `next_to` \| `null` | only for `relative` |
| `reference` | object key, e.g. `"beaker_red"` \| `null` | only for `relative`; resolved by the guardrail |
| `table_spot` | `"free"` \| `"default"` \| `null` | only for `table`: `free` = "on the table" (any free reachable spot), `default` = "back to its spot" |

Internal extras (not sent to the sim): `reference_text`, `on_shelf_context`, `dest_error`, `notes`.
**Removed in B2:** `shelf_slot`, `shelf_slot_y`. There is no slot in the plan any more.

`notes` carries things the user asked for that the robot cannot honour. Today only a requested slot/side
("left side of the top shelf", "slot 3", "centre of the shelf"): it is ignored, the tier is kept, and the note
("the robot picks the position on the shelf itself, so 'left side' was ignored") is shown as the step's detail.
Stacking phrases ("on top of X", "above X") are mapped to `relation="next_to"`.

## 2. What the sim receives

`POST /execute` (the structured form is sent **next to** the unchanged text instruction):

```json
{
  "instruction": "pick up the blue flask and place it on the shelf",
  "destination": {
    "dest_kind": "shelf", "shelf_tier": null, "shelf_tier_index": null,
    "relation": null, "reference": null, "table_spot": null
  }
}
```

`shelf_tier_index` is 0 = bottom, 1 = middle, 2 = top, `null` for auto. `destination` is omitted when `dest_kind`
is `null`. The canonical text is what the sim's existing parser reads (`on the top|middle|bottom shelf`, plain
`on the shelf`), so the sim works from the text alone; the `destination` object is there for when it wants the
structured form. Other shapes:

```json
{"dest_kind": "shelf", "shelf_tier": "middle", "shelf_tier_index": 1, "relation": null, "reference": null, "table_spot": null}
{"dest_kind": "relative", "relation": "right_of", "reference": "beaker_red", "shelf_tier": null,
 "shelf_tier_index": null, "table_spot": null}
{"dest_kind": "table", "table_spot": "default", "shelf_tier": null, "shelf_tier_index": null,
 "relation": null, "reference": null}
```

`_dispatch` now treats the sim as having started a request as soon as `/state.current_request_id` equals the id
`/execute` returned (it no longer also requires `busy`): a task the sim refuses at planning time can already be idle
again by the next poll, and used to read as "never started".

## 3. Geometry the orchestrator assumes (must equal the sim's constants)

World frame, metres, robot base at the origin, table top z = 0. Object positions in `/state` are **base centres**.

- Tier surface heights, bottom to top: `0.08, 0.36, 0.64` (`SHELF_RACK_TOPS`). "Standing on tier T" means
  `|z - top_T| < 0.04` **and** `y > 0.40` (the shelf is on the +Y side; tiers start at y 0.44 / 0.53 / 0.62).
- Reach limits (`PhysicalWorkspaceRails`) are zone-aware: the table (x 0.20..0.70, |y| <= 0.45, z <= 0.50) **or**
  the shelf box (x -0.05..0.70, y 0.40..0.90, z <= 0.85). A target outside both is "out of bounds".
- `relative` placements use a 0.10 m centre-to-centre gap (`SPATIAL_OFFSET` in the sim): right_of = +Y,
  left_of = -Y, in_front_of = +X, behind = -X, next_to = +Y. The guardrail checks this offset stays in the table
  zone only when the reference stands on the table; for a reference on the shelf the sim clamps.

## 4. Destination vocabulary and parsing rules

The parser works on the destination **text** and is authoritative; the LLM's own `dest_kind` / `shelf_tier` are used
only when it gave no text at all. A model that says `dest_kind="table"` for "top shelf", or `shelf_tier="top"` for
plain "the shelf", is overridden.

| user says | result |
|---|---|
| `top/upper/highest shelf`, `middle/mid/centre shelf`, `bottom/lower/lowest shelf` (also rack/tier/level) | `shelf` + tier |
| `the shelf`, `in the rack`, `on a shelf` | `shelf`, `shelf_tier = null` (**auto**) |
| `first/second/third shelf`, `shelf 2`, `level three` | counted **from the bottom**: 1 = bottom, 2 = middle, 3 = top |
| `fourth shelf`, `shelf 7`, `tier 4` | `shelf` with `shelf_tier` = the user's own word, **rejected** by the guardrail |
| `left side of the top shelf`, `slot 3 of the bottom shelf`, `centre of the shelf` | tier kept (or auto), slot request ignored + a note |
| `next to / beside / near / right of / left of / in front of / behind <object>` | `relative` |
| `on the table`, `on the bench`, `back on the table` | `table`, `table_spot = "free"` |
| `back to its spot`, `put it back`, `where it was` | `table`, `table_spot = "default"` |
| nothing | `dest_kind = null` |
| anything else (`on the moon`, `in the freezer`) | **rejected**: "destination '...' is not understood" |

A relation needs an object word (flask, beaker, a colour, "empty") after it; `right of the shelf` / `on top of the
table` are positions, not relations to an object.

## 5. Vague multi-object prompts

"place all objects in the rack", "put everything on the shelf", "tidy up", "clear the table", "put all the flasks on
the top shelf", "put all beakers in the rack" become **one `pick_place` sub-goal per object standing on the table in
the live scene**, each with `dest_kind="shelf"` and `shelf_tier=null` (or the tier the user named).

- **Which objects:** those not on the shelf with `z < 0.04`, filtered by "flasks" / "beakers" if the user said so.
  Objects already on the shelf are left alone.
- **Order:** flasks before beakers, then nearest to the robot first (`hypot(x, y)`). For the standard scene that is
  `red flask, blue flask, red beaker`.
- **Where it happens:** in code, never by the LLM. A single-clause prompt is expanded **without any LLM call**
  (`_expand_vague_instruction`). A compound prompt ("... then go home") goes to Nemotron, which is told to emit one
  subgoal with `target: "all objects"` (or "all flasks"/"all beakers"); `_expand_vague_subgoals` expands it from the
  live scene.
- **Nothing on the table:** the plan is empty and the user is told "There is nothing to place: no objects are standing
  on the table right now."
- A nonexistent tier word is kept (`put all objects on the fourth shelf` -> three subgoals, each rejected).

## 6. Guardrail rejections (`validate_subgoal` -> `check_destination`)

All destination ones are prefixed `destination: `. Nothing is dispatched for a rejected step.

- `the shelf has no 'fourth' tier — it has 3: top, middle, bottom (top to bottom), or just say 'on the shelf'`
- `destination reference 'purple beaker' not found in live scene`
- `cannot place the Red Flask next to itself`
- `placing it right of the reference: Y-target 0.500m out of bounds [-0.45, 0.45]`
- `destination '...' is not understood — use 'top/middle/bottom shelf' (or just 'on the shelf'), 'next to <object>', 'on the table' or 'back to its spot'`

**The orchestrator never rejects a shelf destination as "occupied" or "full"** — only the sim can tell (it plans the
arm, hand and vessel motion around what is already there).

**Retry rule (unchanged):** only perception-type failures are re-observed, and only on a camera-derived scene:
messages containing `not found` or `out of bounds`. Placement-rule rejections use other wording on purpose and are
never retried.

## 7. Refusals (sim -> orchestrator)

When the sim cannot do a step it does not move and reports, in `/state`:

- `last_task_refused` (bool) and `last_task_error` (text), set atomically with `busy = false`, e.g.
  `the shelf is full: there is no slot where the Blue Flask can be set down without hitting something already there
  (last blocker: the hand would hit Red Flask)` or `cannot pick up the Blue Flask: the arm would hit Red Flask`.

What the orchestrator does with a refusal (all covered by `test_dest_parsing.py`):

1. **Never retried.** The same step fails the same way, so no second dispatch of it.
2. **Quoted verbatim** in the step's `detail` ("refused by the robot: <text>"), in the step's result
   (`refused: true`, `refusals: [...]`, `reason`), in the log, and in the final answer.
3. **One alternative at most.** Nemotron is asked once (the prompt says this is a refusal, "retry" means "give up",
   and names the two allowed moves: another tier / another destination via `replan_step`, or `replan_remaining` whose
   first subgoal moves the named blocker out of the way). The proposal is accepted only if (a) it is a **different
   action** from every refused one (`_step_signature`: object, dest_kind, tier, relation, reference, table_spot),
   (b) its first subgoal passes the guardrail, (c) for `replan_remaining`, that valve has not been used earlier in the
   run. `retry`, an identical step, an invalid proposal or no reply all mean "leave the step refused". If the
   alternative is refused as well, the step ends there (no third attempt, no second proposal).
4. **The plan carries on.** A refused step does not abort the rest (nothing moved, so the world is unchanged); a step
   placed *relative to* a refused object is skipped (`ABORTED`, "it is placed relative to the X, which the robot
   refused to move"). Physical failures (position miss, grasp not confirmed) keep the old rule: retry, then abort.
5. **Pick refusals** (`cannot pick up ...`) are handled exactly the same way; the only useful alternative is moving the
   named blocker first, which the refusal prompt spells out.

A refused step's `status` stays `FAILED` (the GUI already knows it) with `refused: true`; a skipped dependent is
`ABORTED`. The plan's overall status is `SUCCESS`, `PARTIAL` (some done, some not) or `FAILED`.

## 8. The final answer

`run()` returns, and the plan snapshot (`/state.plan`) carries, `answer` (plain-language text) and `summary`:

```json
{"placed": [{"object": "red flask", "where": "on the shelf"}],
 "not_placed": [{"object": "blue flask", "wanted": "on the shelf", "reason": "the robot refused: ...",
                 "refusal": "<verbatim sim text>", "status": "FAILED"}],
 "refusals": [{"object": "blue flask", "wanted": "on the shelf", "text": "<verbatim>"}],
 "worked_around": [ ...refusals whose step was solved by a different action... ]}
```

- An all-clear run gets a deterministic answer (no LLM call).
- Otherwise Nemotron writes it (what was asked, what blocked it, what the user could change), but the result is
  **checked in code**: every verbatim refusal text, every placed and not-placed object name must appear in it; whatever
  is missing is appended from the deterministic account. If Nemotron is down, the deterministic answer is used. A
  paraphrasing or forgetful model therefore cannot drop what the robot actually said.
- The GUI does not render `plan.answer` yet (`physyk_main_server.py` is not part of this task); it is also pushed
  to the log as `[Answer] ...`.

## 9. Not covered / open

- Nothing here ran against the real sim: the tests use a scripted fake that returns refusals in the same
  `/state` fields. Real refusal wording, timing and whether `/execute` ever misses a request are unverified.
- The sim still ignores the `destination` object in `/execute` (it parses the canonical text), and does not yet expose
  `default_position` per object, so "back to its spot" is always dispatched and `table_spot = "free"` has no
  free-spot picker (it falls through to "set back down where it was").
- Slot/side requests are ignored with a note rather than rejected. If you would rather reject them, change the one
  branch in `_normalize_subgoal_destination`.
