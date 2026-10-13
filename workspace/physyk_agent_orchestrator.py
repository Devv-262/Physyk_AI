#!/usr/bin/env python3
"""
Physyk AI — Real Nemotron Agentic Orchestrator (Phase 5)
==========================================================
This used to be a rule-based string-matching task decomposer whose output was computed and
then thrown away by physyk_main_server.py — the real dispatch happened via a separate direct
call right after (see PLAN.md Part 2 / dev-progress.md). That's gone. This module now:

1. Calls the real Nemotron-30B server (nemotron_fastapi_server / vLLM OpenAI server, port
   8000 — see start_nemotron.sh) to decompose a free-form instruction into an ordered list of
   structured pick-and-place subgoals, grounded against the *live* scene state (real object
   positions from isaac_sim_service.py's /state, not a hardcoded object list).
2. Runs a real NeMo-style guardrail check (`PhysicalWorkspaceRails`) against each subgoal's
   resolved target/destination before it is ever dispatched to the robot.
3. Dispatches each subgoal, in order, as a plain-text instruction into the existing
   `/execute` -> `resolve_pick_place_targets()` -> `PickPlaceController` pipeline on
   isaac_sim_service.py (port 8100) — that pipeline is untouched; it's used as a tool/skill,
   exactly as-is. Waits for the sim to go idle and reads back the real post-place verification
   result (`last_task_success` / `last_task_error_mm`).
4. On a verified failure, does NOT follow a fixed retry-then-give-up schedule. It asks
   Nemotron to actually diagnose the failure (real error, live scene, what's already been
   completed) and choose: retry as-is / replan just this step / replan the rest of the plan —
   one LLM call per failure, not a separate call for the decision and the replan. A disturbance
   to an already-completed placement (something knocked over mid-plan) is detected this way,
   not hardcoded.
5. After every subgoal reports DONE, runs one deterministic (code-only, no LLM) final check
   that the original goal still holds — re-comparing each verified placement's position against
   what it was at verification time, catching a later step disturbing an earlier one. Only
   unconditional plan SUCCESS if that holds too.

No local model is loaded in this module or its caller — everything here is a plain HTTP client
of two already-running real servers, so this stays cheap to import from physyk_main_server.py
(which runs under plain system python3, not a GPU-framework venv).
"""

import json
import logging
import math
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from physyk_perception import step_verifier
from reasoning_dashboard import NEMOTRON_REASONING_LOG, COSMOS_REASONING_LOG

# Full-length reasoning text (Nemotron's chain-of-thought, Cosmos's per-step assessment
# paragraphs) used to be rendered inline in the main dashboard's Agent Plan panel — readable
# but made the step list hard to scan at a glance. It now lives on its own page per agent
# instead, at /reasoning/nemotron and /reasoning/cosmos on the SAME port as the main
# dashboard (see reasoning_dashboard.py — dedicated ports 8010/8011 were tried first but
# aren't reachable through Brev/VS Code port forwarding, which only exposes 7860/8211).
# These are just the shared log objects; physyk_main_server.py mounts the actual routes.

logging.basicConfig(level=logging.INFO, format="[NeMo-Agent] %(message)s")
log = logging.getLogger("physyk.orchestrator")

import physyk_config as cfg

ISAAC_SIM_URL = cfg.ISAAC_SIM_URL

HTTP_TIMEOUT_S = 2.0
NEMOTRON_TIMEOUT_S = 90.0   # hosted API + a reasoning model's chain-of-thought: seconds, not ms
PLAN_STEP_TIMEOUT_S = 300.0  # max time to wait for one dispatched subgoal to finish — a real
                              # pick-place has been observed taking 40-45s+ on its own, and
                              # longer under concurrent load; give real margin rather than
                              # risk a live task being cut off mid-motion (a timeout is now
                              # always treated as a failed attempt, never a silent success —
                              # see _wait_for_idle's caller — so this is just patience budget,
                              # not a correctness knob).
POLL_INTERVAL_S = 0.3
MAX_SUBGOAL_ATTEMPTS = 3     # 1 plain retry, then 1 Nemotron-replanned corrective attempt
GUARDRAIL_RETRIES = 3        # re-observations before a guardrail failure on a camera-derived scene stands
GUARDRAIL_RETRY_DELAY_S = 0.8  # a bit over the sim's 0.5 s idle perception refresh


def _scene_is_camera_derived(scene) -> bool:
    """True when the live scene's objects came from the camera (pure-perception mode)."""
    return any(o.get("source") == "camera" for o in (scene.get("objects") or {}).values())


# ─── Real NeMo Safety Guardrails ───────────────────────────────────────────────
# Bounds match VLA_XYZ_MIN/MAX in isaac_sim_service.py — the same reachable-workspace clamp
# already validated there for GR00T VLA safety, reused here per PLAN.md 5.2 rather than
# re-derived, so both safety paths agree on what "reachable" means.

@dataclass
class PhysicalWorkspaceRails:
    """Enforces spatial safety constraints on any Cartesian target before dispatch."""
    x_min: float = 0.20
    x_max: float = 0.70
    y_min: float = -0.45
    y_max: float = 0.45
    z_min: float = -0.01     # a cube centre rests at 0.025 but can sink a few mm into the table; below ~0 is a real error
    z_max: float = 0.50

    def validate_cartesian_target(self, target_xyz, check_z: bool = True) -> Tuple[bool, str]:
        x, y, z = float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2])
        if not (self.x_min <= x <= self.x_max):
            return False, f"X-target {x:.3f}m out of bounds [{self.x_min}, {self.x_max}]"
        if not (self.y_min <= y <= self.y_max):
            return False, f"Y-target {y:.3f}m out of bounds [{self.y_min}, {self.y_max}]"
        if check_z and not (self.z_min <= z <= self.z_max):
            return False, f"Z-target {z:.3f}m out of bounds [{self.z_min}, {self.z_max}]"
        return True, "Safe"


# ─── Live scene / sim HTTP helpers ──────────────────────────────────────────────

def _http_get_json(url: str, timeout: float = HTTP_TIMEOUT_S) -> Optional[Dict[str, Any]]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Physyk-Orchestrator"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        log.warning(f"GET {url} failed: {e}")
        return None


def _get_scene_frame() -> Optional[Tuple[bytes, str]]:
    """Fetches the current overview-camera JPEG straight from Isaac Sim's existing
    /camera/scene.jpg (already served for the live GUI stream) — raw bytes, no decode/
    re-encode needed, for step_verifier.py's visual post-condition checks (Step 7). Returns
    None on any failure; callers must treat that as "no frame available", never raise."""
    try:
        req = urllib.request.Request(f"{ISAAC_SIM_URL}/camera/scene_raw.jpg", headers={"User-Agent": "Physyk-Orchestrator"})
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
            return resp.read(), "image/jpeg"
    except Exception as e:
        log.warning(f"Could not fetch scene frame for visual verification: {e}")
        return None


def _http_post_json(url: str, payload: Dict[str, Any], timeout: float = HTTP_TIMEOUT_S,
                    headers: Optional[Dict[str, str]] = None) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers or {"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode("utf-8"))
        except Exception:
            body = None
        return e.code, body
    except Exception as e:
        log.warning(f"POST {url} failed: {e}")
        return None, None


def fetch_live_scene() -> Dict[str, Any]:
    """Live sim telemetry, including per-object positions (isaac_sim_service.py /state)."""
    return _http_get_json(f"{ISAAC_SIM_URL}/state") or {}


def _resolve_named_key(name: str, scene: Dict[str, Any]) -> Optional[str]:
    """Grounds an LLM-named object/destination phrase to a live scene object's key (e.g.
    "red_cube") rather than just its position — needed to identify *which specific object*
    a destination phrase refers to, so a later check can ask "is this the same object an
    earlier step already placed" rather than just "what's near this phrase right now"."""
    if not name:
        return None
    name_l = name.lower()
    objects = scene.get("objects", {}) or {}
    for key, data in objects.items():
        obj_name = (data.get("name") or "").lower()
        if obj_name and obj_name in name_l:
            return key
    for color in ("red", "blue", "green"):
        if color in name_l and f"{color}_cube" in objects:
            return f"{color}_cube"
    if any(w in name_l for w in ("tray", "bin", "box", "basket")) and "target_tray" in objects:
        return "target_tray"
    return None


def _resolve_named_position(name: str, scene: Dict[str, Any]) -> Optional[List[float]]:
    """Grounds an LLM-named object/destination phrase against live scene object positions.
    Returns None if it can't be resolved to a known object — callers treat that as a
    guardrail-relevant fact (an unresolvable target is either a hallucinated object or a
    relative-offset destination this module doesn't re-derive; the sim's own
    resolve_pick_place_targets does the exact offset math when the subgoal is dispatched)."""
    key = _resolve_named_key(name, scene)
    if key is None:
        return None
    return (scene.get("objects", {}) or {}).get(key, {}).get("position")


_STACK_PHRASES = ("on top of", "stacked on", "stack on", "above")
ALREADY_IN_TRAY_RADIUS_M = 0.13   # generous enough to cover any of the 3 tray slots
ALREADY_STACKED_XY_M = 0.05
ALREADY_STACKED_MIN_DZ_M = 0.015
ALREADY_STACKED_MAX_DZ_M = 0.08
GOAL_DRIFT_TOLERANCE_M = 0.03   # shared by the final-goal check and the pre-dispatch drift
                                  # check below — "has this real-world object moved enough
                                  # to no longer count as where we last verified it"


def _subgoal_already_satisfied(subgoal: Dict[str, Any], scene: Dict[str, Any]) -> bool:
    """Deterministic precondition check — is this pick_place subgoal's target object already
    at its destination? A genuinely agentic planner checks state before acting; this used to
    be missing entirely, confirmed live: re-running the same stacking instruction re-picked a
    cube that was already correctly placed instead of recognizing the goal already held for
    it. Deliberately deterministic (position math), not another LLM call — matches how
    verification/guardrails already work here, and keeps this cheap and reliable rather than
    dependent on the model remembering to reason about it every time."""
    if subgoal.get("action") != "pick_place":
        return False
    target_pos = _resolve_named_position(subgoal.get("target", ""), scene)
    if target_pos is None:
        return False
    dest_text = (subgoal.get("destination") or "").lower()

    if any(p in dest_text for p in _STACK_PHRASES):
        ref_pos = _resolve_named_position(dest_text, scene)
        if ref_pos is None:
            return False
        xy = math.hypot(target_pos[0] - ref_pos[0], target_pos[1] - ref_pos[1])
        dz = target_pos[2] - ref_pos[2]
        return xy < ALREADY_STACKED_XY_M and ALREADY_STACKED_MIN_DZ_M < dz < ALREADY_STACKED_MAX_DZ_M

    if any(w in dest_text for w in ("tray", "bin", "box", "basket")):
        tray_pos = (scene.get("objects", {}) or {}).get("target_tray", {}).get("position")
        if tray_pos is None:
            return False
        xy = math.hypot(target_pos[0] - tray_pos[0], target_pos[1] - tray_pos[1])
        return xy < ALREADY_IN_TRAY_RADIUS_M

    # Relative offsets ("right of X") aren't re-derived here (that's the sim's own exact
    # offset math at dispatch time) — always dispatch rather than guess whether it's already
    # satisfied.
    return False


def _reference_out_of_tray(subgoal: Dict[str, Any], scene: Dict[str, Any]) -> Tuple[bool, str]:
    """Absolute check, distinct from drift: if the destination explicitly says to stack on a
    reference object *in the tray*, verify that reference is actually, currently, inside the
    tray region — not just "hasn't moved since the plan last looked at it". A pure relative
    drift check can't catch this: if a cube fell out of the tray (knocked out during its own
    placement, or bumped by a nearby unrelated grasp) and then simply stayed put in its new,
    wrong spot, nothing "moved since last checked" — the drift check stays silent while the
    plan goes on to stack subsequent cubes onto something that was never really in the tray
    to begin with. Confirmed as a real, distinct gap from a live run: a cube ended up outside
    the tray and later steps kept building on it without anything ever flagging that the
    premise ("in the tray") was already false."""
    if subgoal.get("action") != "pick_place":
        return False, ""
    dest_text = (subgoal.get("destination") or "").lower()
    if not any(k in dest_text for k in _TRAY_DEST_KEYWORDS):
        return False, ""
    if not any(p in dest_text for p in _STACK_PHRASES):
        return False, ""
    ref_key = _resolve_named_key(dest_text, scene)
    if ref_key is None or ref_key == "target_tray":
        return False, ""
    objects = scene.get("objects", {}) or {}
    ref_pos = objects.get(ref_key, {}).get("position")
    tray_pos = objects.get("target_tray", {}).get("position")
    if ref_pos is None or tray_pos is None:
        return False, ""
    xy = math.hypot(ref_pos[0] - tray_pos[0], ref_pos[1] - tray_pos[1])
    if xy > ALREADY_IN_TRAY_RADIUS_M:
        return True, (f"destination says to stack on the reference in the tray, but it's "
                       f"currently {xy * 1000:.0f}mm from the tray — it isn't actually in it")
    return False, ""


def _subgoal_drift_detected(
    subgoal: Dict[str, Any], scene: Dict[str, Any], earlier_steps: List[Dict[str, Any]]
) -> Tuple[bool, str]:
    """Real perceive-before-every-action check, deterministic (no LLM call) — this is what
    was actually missing per live feedback: the orchestrator used to only re-examine state
    *after* a dispatch failed, never *before* dispatching a later step that silently
    depended on an earlier one still being true. Confirmed live: a step 2 dispatch went out
    referencing "the red cube" as a stacking anchor while the red cube's actual placement
    had already been invalidated (a second pick_place inadvertently re-grabbed it) — nothing
    caught that until the resulting placement itself failed verification.

    Two independent checks, either one flags: (1) an *absolute* check — is the reference
    object actually where the destination text says it should be (e.g. really in the tray),
    not just unchanged since we last looked; (2) a *relative* check — has the reference
    object moved beyond real-world tolerance since the earlier step in *this same plan* that
    placed it last verified its position. (1) catches a reference that was already wrong and
    simply never moved again; (2) catches a reference that was right but got disturbed.
    Neither alone is enough — confirmed live, separately, for both failure modes.

    Only checks subgoals with a relative/stacking destination that names another object —
    a plain tray placement has no "reference" to have drifted. Returns (drifted, reason). No
    prior-completed-step reference found for check (2) -> that check stays silent, not a
    claim nothing could be wrong there (the pick target itself is always re-grounded live at
    dispatch time regardless, by the sim's own resolve_pick_place_targets, so an untouched
    target can't silently go stale that specific way)."""
    if subgoal.get("action") != "pick_place":
        return False, ""
    dest_text = subgoal.get("destination") or ""
    if not any(p in dest_text.lower() for p in _RELATION_DEST_KEYWORDS):
        return False, ""

    out_of_tray, reason = _reference_out_of_tray(subgoal, scene)
    if out_of_tray:
        return True, reason

    ref_key = _resolve_named_key(dest_text, scene)
    if ref_key is None:
        return True, f"destination reference '{dest_text}' can no longer be resolved in the live scene"

    for earlier in reversed(earlier_steps):
        earlier_sg = earlier.get("subgoal", {}) or {}
        if earlier_sg.get("action") != "pick_place" or earlier.get("status") != "DONE":
            continue
        if _resolve_named_key(earlier_sg.get("target", ""), scene) != ref_key:
            continue
        expected = earlier.get("final_pos")
        if not expected:
            return False, ""
        ref_pos_now = (scene.get("objects", {}) or {}).get(ref_key, {}).get("position")
        if ref_pos_now is None:
            return False, ""
        drift_m = math.hypot(ref_pos_now[0] - expected[0], ref_pos_now[1] - expected[1])
        if drift_m > GOAL_DRIFT_TOLERANCE_M:
            return True, (f"reference object for '{dest_text}' moved {drift_m * 1000:.0f}mm "
                           f"since step {earlier.get('index')} verified its placement — "
                           f"environment changed since this plan was made")
        return False, ""
    return False, ""


def _describe_scene(scene: Dict[str, Any]) -> str:
    objects = scene.get("objects", {}) or {}
    if not objects:
        return "(scene state unavailable)"
    lines = []
    for data in objects.values():
        pos = data.get("position", [0, 0, 0])
        lines.append(f"- {data.get('name', '?')} at x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")
    return "\n".join(lines)



# ─── Stack awareness (cubes) ─────────────────────────────────────────────────────────────────────────────────
# A cube with another cube on top of it cannot be picked: the blocker has to be moved off first. Positions come from the
# live scene (simulator coordinates, or the camera in pure-perception mode), so this works on whatever stacks exist NOW.
CUBE_SIZE_M = 0.05
STACK_XY_M = 0.035                 # |dxy| below this = the other cube sits on this one
STACK_MIN_DZ_M = 0.035             # ...and is at least this much higher (a full layer is 0.05)
FREE_SPOT_DESTINATION = "on a free spot on the table"
MAX_BLOCKER_MOVES = 4              # safety valve: blocker-clearing steps inserted per run


def _cube_keys(scene: Dict[str, Any]) -> List[str]:
    return [k for k in (scene.get("objects", {}) or {}) if k.endswith("_cube")]


def _cube_pos(scene: Dict[str, Any], key: str):
    return (scene.get("objects", {}) or {}).get(key, {}).get("position")


def _cube_on_top_of(key: str, scene: Dict[str, Any]) -> Optional[str]:
    """The cube resting directly on `key` (the lowest one above it, i.e. the next layer), or None."""
    base = _cube_pos(scene, key)
    if base is None:
        return None
    best, best_z = None, None
    for other in _cube_keys(scene):
        if other == key:
            continue
        p = _cube_pos(scene, other)
        if p is None:
            continue
        if math.hypot(p[0] - base[0], p[1] - base[1]) < STACK_XY_M and p[2] - base[2] > STACK_MIN_DZ_M:
            if best is None or p[2] < best_z:
                best, best_z = other, p[2]
    return best


def _stack_blocker(key: str, scene: Dict[str, Any]) -> Optional[str]:
    """The cube that must be moved before `key` can be picked: the TOP of the pile above `key`, or None if its top is free."""
    blocker, seen = None, {key}
    cur = _cube_on_top_of(key, scene)
    while cur is not None and cur not in seen:
        blocker = cur
        seen.add(cur)
        cur = _cube_on_top_of(cur, scene)
    return blocker


def _cube_name(scene: Dict[str, Any], key: str) -> str:
    return str((scene.get("objects", {}) or {}).get(key, {}).get("name", key)).lower()


_STACK_ALL_RE = re.compile(r"\b(?:stack|pile|tower)\b.*\b(?:all|every|the|these|three|3)?\b.*\bcubes?\b|\bcubes?\b.*\b(?:stack|pile|tower)\b")
_NAMED_CUBE_RE = re.compile(r"\b(?:red|green|blue)\b")


def _ease(scene: Dict[str, Any], k: str) -> float:
    """How easy a cube is to pick: lower = easier. Nothing on top of it first (a covered cube cannot be picked), then the
    closest to the robot, with a penalty for each neighbour within hand-room (the open hand is ~0.11 m wide)."""
    p = _cube_pos(scene, k)
    crowd = sum(1 for o in _cube_keys(scene) if o != k and _cube_pos(scene, o) is not None
                and math.hypot(_cube_pos(scene, o)[0] - p[0], _cube_pos(scene, o)[1] - p[1]) < 0.11)
    return (1.0 if _cube_on_top_of(k, scene) else 0.0) * 10.0 + math.hypot(p[0], p[1]) + 0.08 * crowd


TRAY_XY = (0.42, -0.32)
IN_TRAY_RADIUS_M = 0.13


def _in_tray(scene: Dict[str, Any], k: str) -> bool:
    p = _cube_pos(scene, k)
    return p is not None and math.hypot(p[0] - TRAY_XY[0], p[1] - TRAY_XY[1]) < IN_TRAY_RADIUS_M


def _stack_plan(scene: Dict[str, Any], surface: Optional[str] = None) -> List[Dict[str, Any]]:
    """Subgoals that stack every cube into ONE pile, chosen from the cubes' current positions - never by colour:
      - an existing stack is kept: its bottom cube is the base and what stands on it stays;
      - otherwise the EASIEST cube (nothing on it, closest to the robot, most room) is the base: it is first on the stack and
        does not have to move;
      - the other cubes are stacked on top in order of ease, easiest first."""
    keys = [k for k in _cube_keys(scene) if _cube_pos(scene, k) is not None]
    if len(keys) < 2:
        return []

    def above_count(k):
        n, cur, seen = 0, _cube_on_top_of(k, scene), {k}
        while cur is not None and cur not in seen:
            n, seen = n + 1, seen | {cur}
            cur = _cube_on_top_of(cur, scene)
        return n

    tray = surface == "tray"
    in_tray = [k for k in keys if _in_tray(scene, k)] if tray else []
    base_to_free_spot = False
    # With the tray: the pile stands IN the tray. A pile already there is kept; otherwise the easiest cube is set in the tray first.
    if tray and not in_tray:
        # No pile in the tray yet: build a NEW one there, easiest (uncovered, closest) cube first - an existing tower elsewhere
        # is taken apart from the top, never carried from the bottom.
        base = min(keys, key=lambda k: _ease(scene, k))
        pile = [base]
    elif surface == "table" and all(_in_tray(scene, k) for k in keys):
        # Asked for the table but every cube is in the tray: build a new pile on a free table spot, easiest cube first.
        base = min(keys, key=lambda k: _ease(scene, k))
        pile = [base]
        base_to_free_spot = True
    else:
        # On the table the pile must not use a cube that is in the tray as its base (that would be a pile in the tray).
        cands = [k for k in keys if not _in_tray(scene, k)] if surface == "table" else keys
        base = max(in_tray or cands or keys, key=lambda k: (above_count(k), -_ease(scene, k)))
        pile = [base]                                   # bottom -> top, as it will stand
        cur = _cube_on_top_of(base, scene)
        while cur is not None and cur not in pile:
            pile.append(cur)
            cur = _cube_on_top_of(cur, scene)
    rest = sorted((k for k in keys if k not in pile), key=lambda k: _ease(scene, k))
    goals, top = [], pile[-1]
    if tray and not in_tray:
        # the base cube itself has to be carried to the tray (it is not the one that stays put)
        goals.append({"action": "pick_place", "target": _cube_name(scene, base), "destination": "in the tray"})
    elif base_to_free_spot:
        goals.append({"action": "pick_place", "target": _cube_name(scene, base), "destination": FREE_SPOT_DESTINATION})
    for k in rest:
        goals.append({"action": "pick_place", "target": _cube_name(scene, k),
                      "destination": f"on top of the {_cube_name(scene, top)}"})
        top = k
    return [_validate_subgoal_shape(g) for g in goals]


def _is_single_pile(scene: Dict[str, Any], keys: List[str]) -> bool:
    for k in keys:
        n, cur, seen = 0, _cube_on_top_of(k, scene), {k}
        while cur is not None and cur not in seen:
            n, seen = n + 1, seen | {cur}
            cur = _cube_on_top_of(cur, scene)
        if n == len(keys) - 1:
            return True
    return False


def _expand_stack_instruction(instruction: str, scene: Dict[str, Any]) -> Optional[Tuple[List[Dict[str, Any]], str]]:
    """'stack the cubes' / 'stack all the cubes' (no colours named, no tray) -> an order chosen from the live positions."""
    t = re.sub(r"\s+", " ", instruction.lower().strip())
    if not _STACK_ALL_RE.search(t) or _NAMED_CUBE_RE.search(t) or re.search(r"\b(?:then|and)\b", t):
        return None
    surface = "tray" if re.search(r"\btray\b", t) else ("table" if re.search(r"\b(?:table|tabletop|bench)\b", t) else None)
    goals = _stack_plan(scene, surface)
    if not goals:
        keys = [k for k in _cube_keys(scene) if _cube_pos(scene, k) is not None]
        if len(keys) >= 2 and _is_single_pile(scene, keys):
            return [], "(the cubes are already stacked in one pile - nothing to do)"
        return None
    order = " -> ".join(g["target"] for g in goals)
    if goals and goals[0]["destination"] == "in the tray":
        order = f"{goals[0]['target']} (set in the tray as the base) -> " + " -> ".join(g["target"] for g in goals[1:])
    return goals, (f"(stack order chosen from the live positions, not by colour: base = the easiest cube to pick, then {order} "
                   f"- easiest first)")

# ─── Real Nemotron-30B Task Decomposition ──────────────────────────────────────

NEMOTRON_SYSTEM_PROMPT = """You are the cognitive planner for a real Franka Panda robot arm \
in a tabletop manipulation cell with a red cube, a blue cube, a green cube, and a tray.

Decompose the user's instruction into an ordered JSON array of subgoals. Each subgoal is one \
of exactly these shapes:
  {"action": "pick_place", "target": "<object name>", "destination": "<free-text location, \
e.g. \\"tray\\", \\"right of the red cube in the tray\\", \\"on top of the blue cube\\", or \
"" if none was actually stated>"}
  {"action": "home"}
  {"action": "reset"}

Only reference objects that actually appear in the live scene listed below — never invent an \
object that is not there. If the instruction only makes sense as a single action, return a \
one-element array. If it describes multiple steps ("and", "then", "after that", commas), \
return one subgoal per step, in order.

If the instruction says to pick up an object and immediately states where to put that SAME \
object in the same breath (e.g. "pick up the green cube and put it in the tray", "grab the \
red cube and place it on the blue cube"), that is ONE pick_place subgoal — put the location \
in destination, do not emit a separate pick-only subgoal first. Only split into multiple \
subgoals when the instruction names a DIFFERENT object for a later step, or explicitly \
sequences unrelated actions ("then", "after that", a new imperative verb for a new target).

Example — same object, one clause — exactly one subgoal:
Instruction: "pick up the green cube and put it in the tray"
[{"action": "pick_place", "target": "green cube", "destination": "tray"}]

Example — different objects across steps — two subgoals:
Instruction: "put the red cube in the tray, then stack the blue cube on top of it"
[
  {"action": "pick_place", "target": "red cube", "destination": "tray"},
  {"action": "pick_place", "target": "blue cube", "destination": "red cube"}
]

"Stack" / "pile" / "tower" always means a VERTICAL pile: each cube is placed "on top of" the one before it, never side by side. \
The SURFACE the pile stands on comes from the user's words:
  - "in/on the tray": the tray is the yellow slab; the FIRST (base) cube goes "in the tray", every other cube "on top of the <previous cube>".
  - "on the table" (or no surface named): the pile stands on the TABLE. Never use the tray. The base cube is either left where it is \
(no step for it) or, if it must move, goes "on a free spot on the table"; every other cube goes "on top of the <previous cube>".
Choose the order from the live positions - cubes with nothing on top and closest to the robot first - not by colour. Never give \
several cubes a "tray" destination to stack them.

Example — stack in the tray:
Instruction: "stack the cubes in the tray"
[
  {"action": "pick_place", "target": "red cube", "destination": "in the tray"},
  {"action": "pick_place", "target": "blue cube", "destination": "on top of the red cube"},
  {"action": "pick_place", "target": "green cube", "destination": "on top of the blue cube"}
]

Example — stack on the table (the red cube stays where it is):
Instruction: "stack the cubes on the table"
[
  {"action": "pick_place", "target": "blue cube", "destination": "on top of the red cube"},
  {"action": "pick_place", "target": "green cube", "destination": "on top of the blue cube"}
]

Plan exactly what the user actually asked for — nothing more. If the instruction only says \
to pick up or grasp an object and does not state where to put it, use \
`"destination": ""` — do not invent a placement (e.g. do not assume "the tray") just because \
that's a common destination in this cell. Only fill in "destination" when the user's own \
words actually describe one.

Respond with ONLY the JSON array. No prose, no markdown code fences, no explanation.

Live scene objects:
{scene_desc}
"""

_JSON_ARRAY_RE = re.compile(r"\[.*?\]", re.DOTALL)
_VALID_ACTIONS = {"pick_place", "home", "reset"}


_TRAY_DEST_KEYWORDS = ("tray", "bin", "sort", "transport", "drop", "basket", "box", "container")
_RELATION_DEST_KEYWORDS = _STACK_PHRASES + ("right of", "left of", "in front of", "behind", "next to", "beside")


def _normalize_destination(destination: str) -> str:
    """A bare object name as a destination (e.g. "Red Cube" instead of "on top of the Red
    Cube") is inherently ambiguous to isaac_sim_service.py's own keyword-based resolver —
    confirmed as the real cause of a live incident: with no relation phrase and no tray
    keyword present, the instruction fell through to isaac's own generic tray-routing
    fallback, sending an already-correctly-stacked cube to a free tray slot instead of onto
    the cube actually named — knocking over the real stack while trying to pick it back up.
    If the destination doesn't already contain a recognized tray or relation phrase but does
    name a known cube color, the only sensible reading of "place X <object>" is stacking —
    canonicalize it to say so explicitly, rather than leave it for keyword matching to guess."""
    dest_l = destination.lower()
    if re.fullmatch(r"\s*(?:the\s+)?(?:table|tabletop|table top)\s*", dest_l):
        return "on the table"
    if not destination or any(k in dest_l for k in _TRAY_DEST_KEYWORDS) or any(k in dest_l for k in _RELATION_DEST_KEYWORDS):
        return destination
    if any(color in dest_l for color in ("red cube", "blue cube", "green cube", "red", "blue", "green")):
        return f"on top of the {destination}"
    return destination


def _validate_subgoal_shape(item: Any) -> Dict[str, Any]:
    """Shared shape check for anything Nemotron hands back as a subgoal (initial
    decomposition or a failure-recovery proposal). Raises ValueError on anything malformed."""
    if not isinstance(item, dict) or item.get("action") not in _VALID_ACTIONS:
        raise ValueError(f"Invalid subgoal: {item!r}")
    if item["action"] == "pick_place":
        if not item.get("target"):
            raise ValueError(f"pick_place subgoal missing 'target': {item!r}")
        item["destination"] = _normalize_destination(item.get("destination") or "")
    return item


def _nemotron_chat(payload: Dict[str, Any]) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    """POST to the configured Nemotron endpoint (hosted NVIDIA API by default). Retries on
    429 / 5xx / connection errors with backoff — the hosted free tier is rate limited, and a
    single transient blip should not turn a whole plan into the single-step fallback."""
    url = f"{cfg.NEMOTRON_BASE_URL}/chat/completions"
    headers = cfg.auth_headers(cfg.NEMOTRON_API_KEY)
    status, body = None, None
    for attempt in range(3):
        status, body = _http_post_json(url, payload, timeout=NEMOTRON_TIMEOUT_S, headers=headers)
        if status == 200 or (status is not None and status not in (429, 500, 502, 503, 504)):
            break
        if attempt < 2:
            time.sleep(2.0 * (attempt + 1))
    return status, body


def _message_text(message: Dict[str, Any]) -> str:
    """Normalises a chat message to the `<reasoning></think><answer>` shape the parsers below
    expect. A self-hosted vLLM Nemotron inlines its chain-of-thought before `</think>`; the
    hosted NVIDIA API returns it separately in `reasoning_content` and `content` is answer-only."""
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    if reasoning and "</think>" not in content:
        return f"{reasoning}</think>{content}"
    return content


def _call_nemotron(instruction: str, scene: Dict[str, Any]) -> str:
    # Plain .replace(), not str.format() — the prompt's literal JSON examples are full of
    # `{...}` braces that str.format() would otherwise try to parse as its own placeholders.
    prompt = NEMOTRON_SYSTEM_PROMPT.replace("{scene_desc}", _describe_scene(scene))
    payload = {
        "model": cfg.NEMOTRON_MODEL,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": instruction},
        ],
        "temperature": 0.0,
        # This model is a reasoning model — it emits a chain-of-thought (including draft JSON
        # snippets) before its real answer, confirmed by inspecting a real response (ends with
        # a "</think>" marker, then repeats the actual answer). Bumped from 1536 -> 4096: a
        # real complex multi-step instruction (3+ steps with relative placements) was
        # confirmed live to burn the *entire* 1536-token budget on chain-of-thought alone
        # (finish_reason=="length", cut off mid-reasoning, no closing JSON ever emitted) —
        # this silently degraded every complex prompt to single-step "direct" dispatch, the
        # exact regression reported live. The server's own --max-model-len was raised to
        # 12288 alongside this so the extra completion budget actually fits.
        "max_tokens": 4096,
    }
    status, body = _nemotron_chat(payload)
    if status != 200 or not body:
        raise RuntimeError(f"Nemotron request failed (status={status})")
    try:
        choice = body["choices"][0]
        content = _message_text(choice["message"])
        # Surface truncation explicitly rather than letting it masquerade as "no JSON found"
        # further down — this is the exact failure mode that silently broke complex prompts.
        if choice.get("finish_reason") == "length":
            log.warning(
                f"[Nemotron] Response hit max_tokens before finishing "
                f"({len(content)} chars) — reasoning may be incomplete for: {instruction!r}"
            )
        return content
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Unexpected Nemotron response shape: {e}")


def _split_reasoning(raw_text: str) -> Tuple[str, str]:
    """Splits a reasoning model's raw output into (reasoning, answer). Reasoning is
    everything before </think> (may be empty if the marker is missing or absent — e.g. a
    non-reasoning model, or an answer that came back with no visible chain-of-thought)."""
    if "</think>" in raw_text:
        reasoning, _, answer = raw_text.partition("</think>")
        return reasoning.strip(), answer.strip()
    return "", raw_text.strip()


def _parse_subgoals(raw_text: str) -> List[Dict[str, Any]]:
    # This is a reasoning model: it emits a chain-of-thought (with draft JSON snippets of its
    # own, mid-reasoning) before a "</think>" marker, then repeats the real answer after it —
    # confirmed against a real response. Prefer whatever comes after the last "</think>"; fall
    # back to the raw text if that marker isn't present (non-reasoning models / other formats).
    # Non-greedy search + take the LAST array match, so a stray "[...]" mentioned earlier in
    # the reasoning can't be picked over the actual final answer.
    answer_text = raw_text.rsplit("</think>", 1)[-1]
    matches = _JSON_ARRAY_RE.findall(answer_text)
    if not matches:
        raise ValueError(f"No JSON array found in Nemotron output: {raw_text[:200]!r}")
    parsed = json.loads(matches[-1])
    if not isinstance(parsed, list):
        raise ValueError("Nemotron output parsed but was not a JSON array")
    # An empty array is a legitimate, correct answer — e.g. the model reasoning that an
    # instruction references an object that isn't in the scene (verified against a real
    # response: it explicitly reasoned "cannot reference a non-existent object" and returned
    # []). Don't raise here; let the caller treat that as "no actionable subgoals" rather than
    # falling back to blind direct dispatch of the original (unfulfillable) instruction.
    return [_validate_subgoal_shape(item) for item in parsed]


def decompose_task(instruction: str, scene: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Real Nemotron-backed decomposition. Falls back to a single verbatim-dispatch subgoal
    (today's baseline behavior) if Nemotron is unreachable or returns something unparsable —
    this keeps the reliable single-step path working even when the cognitive layer is down,
    rather than failing the whole instruction."""
    subgoals, _reasoning = decompose_task_with_reasoning(instruction, scene)
    return subgoals


def decompose_task_with_reasoning(
    instruction: str, scene: Optional[Dict[str, Any]] = None
) -> Tuple[List[Dict[str, Any]], str]:
    """Same as decompose_task, but also returns Nemotron's own chain-of-thought text (its
    real reasoning for how it split the instruction into subgoals) so the UI can show *why*
    the plan looks the way it does, not just the resulting steps. Empty string if Nemotron
    was unreachable/failed or returned no visible reasoning."""
    scene = scene if scene is not None else fetch_live_scene()
    stack = _expand_stack_instruction(instruction, scene)
    if stack is not None:
        log.info(f"'{instruction}' expanded from the live cube positions into {len(stack[0])} subgoal(s) (no LLM call).")
        return stack
    try:
        raw = _call_nemotron(instruction, scene)
        reasoning, answer = _split_reasoning(raw)
        subgoals = _parse_subgoals(raw)
        log.info(f"Nemotron decomposed '{instruction}' into {len(subgoals)} subgoal(s): {subgoals}")
        return subgoals, reasoning
    except Exception as e:
        log.warning(f"Nemotron decomposition unavailable/failed ({e}) — dispatching instruction directly.")
        return [{"action": "direct", "instruction": instruction}], f"(planning failed: {e})"


FAILURE_DECISION_SYSTEM_PROMPT = """You are the cognitive planner for a real Franka Panda \
robot arm executing a multi-step plan. One subgoal was actually executed by the robot but \
FAILED real post-placement verification (its final live position was too far from the \
intended target). Diagnose the failure and decide what to do. Respond with ONLY one JSON \
object, no prose, no markdown fences, in exactly one of these three shapes:

  {"decision": "retry", "reason": "<short reason>"}
    Use when this looks like a one-off physical miss (grasp slipped, small error) and the
    exact same subgoal is likely to succeed if just tried again.

  {"decision": "replan_step", "reason": "<short reason>", \
"subgoal": {"action": "pick_place", "target": "<object name>", "destination": "<free-text location>"}}
    Use when the same destination is likely to keep failing (e.g. an occupied/ambiguous tray
    slot) — propose a different, more specific destination for just this one subgoal, keeping
    the rest of the plan as-is.

  {"decision": "replan_remaining", "reason": "<short reason>", \
"subgoals": [{"action": "pick_place", "target": "...", "destination": "..."}]}
    Use when the failure means the REST of the plan (this step onward) needs to be
    reconsidered — e.g. an earlier assumption about the scene no longer holds, or the
    environment changed (an object got knocked over, moved, or isn't where expected). Propose
    the full ordered list of subgoals still needed to achieve the ORIGINAL instruction from
    here, given what's already been completed and the current live scene.

Only reference objects that actually appear in the live scene listed below — never invent one.

Before deciding, explicitly check: for each already-completed subgoal, does that object's \
CURRENT live position below still actually match where that subgoal placed it (e.g. a \
"tray" destination should still show that object near the tray's live position; an "on top \
of X" destination should still show it stacked at X's live position)? If any completed \
subgoal's object has moved away from where it should be, something in the environment \
changed after it was verified (knocked over, picked up again, disturbed) — that is a strong \
signal for "replan_remaining", not just "retry", because the remaining plan's own \
assumptions (destinations relative to that object, tray slot choices, stacking order) may no \
longer be valid either.

Original user instruction: {instruction}
Already-completed subgoals (in order): {completed}
Subgoal that just failed: {failed_subgoal}
Real verification result: {error_mm}
Remaining subgoals that were still planned after this one (in order, not yet attempted): {remaining}
Live scene objects (including where the target object actually ended up):
{scene_desc}
"""

REFUSAL_PROMPT_BLOCK = """
IMPORTANT — THIS WAS A REFUSAL, NOT A MISS. The robot did not move: its own collision planner refused the step, \
with this exact message:
  "{refusal}"
Trying the same thing again cannot work (it fails the same way every time), so "retry" is NOT an option here — \
answering "retry" means "give up on this step". You may propose exactly ONE alternative, and it must be a \
DIFFERENT action from the refused one:
  - "replan_step" with a different destination; or
  - "replan_remaining" whose FIRST subgoal moves the blocker named in the message out of the way, followed by \
the refused step and whatever else is still needed.
Never repeat the refused subgoal unchanged, and never invent an object.
"""

_VALID_DECISIONS = {"retry", "replan_step", "replan_remaining"}


def _extract_json_objects(text: str) -> List[str]:
    """Finds every top-level balanced {...} substring in text, respecting string literals so
    a nested object doesn't truncate the match. A plain non-greedy regex (\\{.*?\\}) breaks
    the instant a real response nests one — confirmed live: a replan_step/replan_remaining
    decision naturally nests a "subgoal"/"subgoals" object, and the regex was matching only
    up to the FIRST inner closing brace, producing a fragment instead of the real decision."""
    spans = []
    depth = 0
    start = None
    in_string = False
    escape = False
    for idx, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    spans.append(text[start:idx + 1])
    return spans


def request_failure_decision(
    instruction: str,
    completed_subgoals: List[Dict[str, Any]],
    failed_subgoal: Dict[str, Any],
    remaining_subgoals: List[Dict[str, Any]],
    scene: Dict[str, Any],
    error_mm: Any,
    refusal: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Asks Nemotron to actually diagnose a real, verified placement failure and choose how to
    respond — retry as-is, replan just this step, or replan the rest of the plan — rather than
    a fixed attempt-number schedule deciding for it. One LLM call per failure (not one for the
    decision and a separate one for the replan itself), per the "minimize LLM calls" goal.
    Returns None on any failure so the caller falls back to a plain retry."""
    prompt = (
        FAILURE_DECISION_SYSTEM_PROMPT
        .replace("{instruction}", instruction)
        .replace("{completed}", json.dumps(completed_subgoals))
        .replace("{failed_subgoal}", json.dumps(failed_subgoal))
        .replace("{remaining}", json.dumps(remaining_subgoals))
        .replace("{error_mm}", str(error_mm))
        .replace("{scene_desc}", _describe_scene(scene))
    )
    if refusal:
        prompt += REFUSAL_PROMPT_BLOCK.replace("{refusal}", refusal)
    payload = {
        "model": cfg.NEMOTRON_MODEL,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Diagnose the failure and decide."},
        ],
        "temperature": 0.0,
        # Bumped 2048 -> 3072 alongside the same server --max-model-len 12288 increase (see
        # _call_nemotron) — a replan_remaining decision has to reason about the whole rest of
        # the plan plus emit a full subgoals array, which is more completion-hungry than a
        # plain retry/replan_step decision.
        "max_tokens": 3072,
    }
    status, body = _nemotron_chat(payload)
    if status != 200 or not body:
        log.warning(f"Failure-decision request failed (status={status})")
        return None
    try:
        choice = body["choices"][0]
        raw = _message_text(choice["message"])
        if choice.get("finish_reason") == "length":
            log.warning("[Nemotron] Failure-decision response hit max_tokens before finishing.")
    except (KeyError, IndexError, TypeError) as e:
        log.warning(f"Unexpected failure-decision response shape: {e}")
        return None
    reasoning, answer_text = _split_reasoning(raw)
    matches = _extract_json_objects(answer_text)
    if not matches:
        log.warning(f"No JSON object found in failure-decision response: {raw[:200]!r}")
        return None
    try:
        item = json.loads(matches[-1])
    except json.JSONDecodeError as e:
        log.warning(f"Failure-decision response wasn't valid JSON: {e}")
        return None
    if not isinstance(item, dict) or item.get("decision") not in _VALID_DECISIONS:
        log.warning(f"Failure-decision response wasn't valid: {item!r}")
        return None
    try:
        if item["decision"] == "replan_step":
            item["subgoal"] = _validate_subgoal_shape(item.get("subgoal"))
        elif item["decision"] == "replan_remaining":
            subgoals = item.get("subgoals")
            if not isinstance(subgoals, list) or not subgoals:
                raise ValueError("replan_remaining without a non-empty 'subgoals' list")
            item["subgoals"] = [_validate_subgoal_shape(sg) for sg in subgoals]
    except ValueError as e:
        log.warning(f"Failure-decision proposal malformed ({e}) — treating as a plain retry.")
        return {"decision": "retry", "reason": f"malformed {item.get('decision')} proposal", "reasoning": reasoning}
    item["reasoning"] = reasoning
    return item


def subgoal_to_instruction(subgoal: Dict[str, Any]) -> str:
    action = subgoal.get("action")
    if action == "home":
        return "return arm to home position"
    if action == "reset":
        return "reset the scene"
    if action == "direct":
        return subgoal.get("instruction", "")
    if action == "pick_place":
        target = subgoal.get("target", "")
        destination = subgoal.get("destination")
        # Confirmed as a real bug: this used to default an empty/missing destination to
        # "tray", which silently turned a plain "pick up the red cube" instruction into
        # "pick up the red cube and place it in the tray" — inventing a placement the user
        # never asked for. Only add a destination clause when Nemotron actually gave one.
        if destination:
            return f"pick up the {target} and place it {destination}"   
        return f"pick up the {target}"
    raise ValueError(f"Unknown subgoal action: {action!r}")


# ─── Orchestrator: Guardrails + Dispatch + Feedback Loop ──────────────────────

# ─── Refusals and the final answer ──────────────────────────────────────────────────────────────────────────
# The sim REFUSES a step it cannot do safely (no collision-free grasp, no free spot to place it) and says why in
# /state: last_task_refused / last_task_error. A refusal is deterministic — the same step fails the same way — so it
# is never retried; Nemotron may propose ONE different alternative, and the plan carries on with the other steps.

def _step_signature(sg: Dict[str, Any]) -> tuple:
    """What makes two pick_place subgoals 'the same action' (same object to the same kind of place)."""
    target = re.sub(r"^(?:the|a|an)\s+", "", (sg.get("target") or "").lower().strip())
    return (target, (sg.get("destination") or "").lower().strip())


def build_outcome_summary(results: List[Dict[str, Any]], refusal_log: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Deterministic placed / not-placed account of a run, built from the per-step results. Refusal texts are kept
    verbatim (never paraphrased) so they can be quoted to the user. `refusals` lists EVERY refusal the robot gave
    during the run, each attributed to the object it was about; `worked_around` is the subset that did not end with
    that step being given up (a different action was taken instead)."""
    placed, not_placed = [], []
    for r in results:
        sg = r.get("subgoal") or {}
        if sg.get("action") != "pick_place":
            continue
        entry = {"object": sg.get("target", "?"), "wanted": sg.get("destination") or "(no destination)"}
        if r.get("status") == "DONE":
            entry["where"] = entry.pop("wanted")
            placed.append(entry)
        else:
            entry["reason"] = r.get("reason") or r.get("status", "not done").lower()
            entry["refusal"] = (r.get("refusals") or [None])[-1]
            entry["status"] = r.get("status")
            not_placed.append(entry)
    refusals = list(refusal_log or [])
    final = {e["refusal"] for e in not_placed if e.get("refusal")}
    return {"placed": placed, "not_placed": not_placed, "refusals": refusals,
            "worked_around": [r for r in refusals if r["text"] not in final]}


def deterministic_answer(instruction: str, summary: Dict[str, Any]) -> str:
    """Plain-language account without an LLM — also the fallback when Nemotron is unreachable."""
    lines = []
    placed, not_placed = summary["placed"], summary["not_placed"]
    if placed:
        lines.append("Done: " + "; ".join(f"the {e['object']} is {e['where']}" for e in placed) + ".")
    for e in not_placed:
        why = f"the robot refused: \"{e['refusal']}\"" if e.get("refusal") else e["reason"]
        lines.append(f"Not done: the {e['object']} ({e['wanted']}) — {why}.")
    if not placed and not not_placed:
        lines.append("Nothing was done.")
    for r in summary.get("worked_around", []):
        lines.append(f"Note: the robot first refused the {r['object']} ({r['wanted']}): \"{r['text']}\" — so a "
                     f"different action was taken.")
    if any(e.get("refusal") for e in not_placed):
        lines.append("You could ask for a different destination, free up space, or move the object named "
                     "as the blocker first.")
    return " ".join(lines)


FINAL_ANSWER_SYSTEM_PROMPT = """You are the voice of a lab robot arm. Write the final answer to the user, in plain \
language, 2-6 sentences, no JSON, no markdown. Use ONLY the facts below; invent nothing.
For every object that was NOT placed say: what the user asked for, what blocked it, and what the user could change \
(for example another destination, freeing space, or moving the named blocker first). When a robot message is given for it, \
QUOTE that message exactly, word for word, inside double quotes. Also say which objects WERE placed and where.

User's instruction: {instruction}
Outcome (JSON): {summary}
"""


def request_final_answer(instruction: str, summary: Dict[str, Any]) -> Optional[str]:
    payload = {
        "model": cfg.NEMOTRON_MODEL,
        "messages": [
            {"role": "system", "content": FINAL_ANSWER_SYSTEM_PROMPT.replace("{instruction}", instruction)
                                                                      .replace("{summary}", json.dumps(summary))},
            {"role": "user", "content": "Write the final answer."},
        ],
        "temperature": 0.0,
        "max_tokens": 2048,
    }
    status, body = _nemotron_chat(payload)
    if status != 200 or not body:
        return None
    try:
        _reasoning, answer = _split_reasoning(_message_text(body["choices"][0]["message"]))
    except (KeyError, IndexError, TypeError):
        return None
    return answer.strip() or None


def compose_final_answer(instruction: str, summary: Dict[str, Any]) -> str:
    """The answer shown to the user. All-clear runs use the deterministic text (no LLM call). When anything was not
    placed or an alternative was needed, Nemotron words it — but the result is CHECKED: every verbatim refusal text
    and every object name must appear in it, otherwise the missing facts are appended from the deterministic account,
    so a paraphrasing or forgetful model can never drop what the robot actually said."""
    needs_llm = bool(summary["not_placed"]) or bool(summary.get("worked_around"))
    answer = None
    if needs_llm:
        try:
            answer = request_final_answer(instruction, summary)
        except Exception as e:                      # the answer must never be what breaks a finished run
            log.warning(f"Final-answer request failed ({e}) — using the deterministic answer.")
    if not answer:
        return deterministic_answer(instruction, summary)
    low = answer.lower()
    missing = []
    for e in summary["not_placed"]:
        if e["object"].lower() not in low or (e.get("refusal") and e["refusal"] not in answer):
            missing.append(e)
    for e in summary["placed"]:
        if e["object"].lower() not in low:
            missing.append({"object": e["object"], "wanted": e["where"], "reason": "", "placed": True})
    for r in summary.get("worked_around", []):
        if r["text"] not in answer:
            missing.append({"object": r["object"], "wanted": r["wanted"], "refusal": r["text"], "reason": "", "worked_around": True})
    if missing:
        extra = []
        for e in missing:
            if e.get("placed"):
                extra.append(f"The {e['object']} is {e['wanted']}.")
            elif e.get("worked_around"):
                extra.append(f"Note: the robot first refused the {e['object']} ({e['wanted']}): \"{e['refusal']}\".")
            else:
                why = f"the robot said: \"{e['refusal']}\"" if e.get("refusal") else e["reason"]
                extra.append(f"The {e['object']} ({e['wanted']}) was not placed — {why}.")
        answer = answer.rstrip() + " " + " ".join(extra)
    return answer


class NeMoRoboticsAgent:
    """Real System-2 orchestrator: Nemotron decomposition -> guardrail gate -> dispatch to the
    existing reliable pick-place primitive -> poll real telemetry for success/failure and
    retry/abort accordingly. No local model, no simulated state — every fact here (scene,
    task completion, verification) comes from the live isaac_sim_service.py telemetry."""

    def __init__(self):
        self.rails = PhysicalWorkspaceRails()
        # Set by request_stop() (wired to the GUI's red "Hard Reset" button) — checked at
        # every poll/loop boundary in run() below so a plan aborts within one poll interval
        # instead of running to its own natural conclusion (retries, replans, and all).
        # threading.Event is safe to set from a different thread than the one running run().
        self._stop_event = threading.Event()
        log.info("Real Nemotron-backed NeMo agentic orchestrator online.")

    def request_stop(self):
        self._stop_event.set()

    def decompose_task(self, instruction: str, scene: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return decompose_task(instruction, scene)

    def decompose_task_with_reasoning(
        self, instruction: str, scene: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], str]:
        return decompose_task_with_reasoning(instruction, scene)

    def validate_subgoal(self, subgoal: Dict[str, Any], scene: Dict[str, Any]) -> Tuple[bool, str]:
        action = subgoal.get("action")
        if action in ("home", "reset", "direct"):
            return True, "OK"
        if action != "pick_place":
            return False, f"unknown action type '{action}'"

        target_pos = _resolve_named_position(subgoal.get("target", ""), scene)
        if target_pos is None:
            return False, f"target object '{subgoal.get('target')}' not found in live scene"
        ok, reason = self.rails.validate_cartesian_target(target_pos)
        if not ok:
            return False, f"target: {reason}"

        # Destination is often a relative phrase ("right of the red cube") the sim resolves
        # itself at dispatch time — only gate on it here when it grounds to a known anchor
        # (a named cube or the tray), as a check against a grossly out-of-workspace request.
        # XY only: an anchor's raw live Z (e.g. the tray's own base, ~0.015m) is not the real
        # placement height — that's computed sim-side by resolve_pick_place_targets (tray
        # surface + cube offset) and was confirmed, by testing this live, to be a completely
        # different number than the anchor's resting Z; checking it here false-rejected every
        # legitimate tray placement.
        dest_pos = _resolve_named_position(subgoal.get("destination", ""), scene)
        if dest_pos is not None:
            ok, reason = self.rails.validate_cartesian_target(dest_pos, check_z=False)
            if not ok:
                return False, f"destination: {reason}"
        return True, "OK"

    def _attach_visual_evidence(self, step: Dict[str, Any], subgoal: Dict[str, Any], enforce: bool) -> None:
        """Cosmos visual post-condition check (Step 7) — writes evidence fields onto
        `step` for the UI (visual_verified/visual_confidence/visual_observed), never raises,
        never blocks: any failure inside verify_step already degrades to a NOT-CHECKED verdict
        (see step_verifier.py's own docstring on why that's asymmetric with a real disagreement)."""
        try:
            target_name = subgoal.get("target", "")
            destination = subgoal.get("destination")
            frame = _get_scene_frame()
            verdict = step_verifier.verify_step(
                lambda: frame, action=subgoal.get("action", ""), obj=target_name,
                destination=destination, enforce=enforce,
            )
            step["visual_verified"] = verdict.verified
            step["visual_enforced"] = verdict.enforced
            step["visual_confidence"] = round(verdict.confidence, 2)
            step["visual_observed"] = verdict.observed
            if verdict.enforced and not verdict.verified:
                log.warning(f"[Visual] Step {step.get('index')} position check passed but Cosmos "
                            f"disagreed: {verdict.observed!r} (confidence {verdict.confidence:.2f}) "
                            f"— shown as evidence, does not change step status.")
            COSMOS_REASONING_LOG.add(
                f"Step {step.get('index')} post-motion visual check ({target_name})",
                f"verified={verdict.verified} (confidence {verdict.confidence:.2f})\n{verdict.observed}",
                image=frame[0] if frame else None,
            )
        except Exception as e:
            # Belt-and-suspenders on top of verify_step's own internal safety — a bug in this
            # brand-new evidence path must never take down a plan that otherwise succeeded.
            log.warning(f"[Visual] Evidence check failed unexpectedly (non-fatal): {e}")
            step["visual_verified"] = True
            step["visual_enforced"] = False
            step["visual_confidence"] = 0.0
            step["visual_observed"] = f"NOT CHECKED (internal error: {e})"

    def _attach_cosmos_assessment(self, step: Dict[str, Any], subgoal: Dict[str, Any]) -> None:
        """Cosmos PRE-dispatch visual reasoning (the "Cosmos 3 Nano Reasoner" panel — target visible / obstacle / feasibility / approach) — writes
        cosmos_* fields onto `step` for the UI. Purely informational, same as
        _attach_visual_evidence: never raises, never blocks or changes what gets dispatched.
        Only meaningful for pick_place (home/reset/direct have no target/destination to look
        at)."""
        if subgoal.get("action") != "pick_place":
            return
        try:
            frame = _get_scene_frame()
            verdict = step_verifier.assess_pick_place(
                lambda: frame, target=subgoal.get("target", ""),
                destination=subgoal.get("destination", ""),
            )
            if not verdict.ok:
                step["cosmos_assessment"] = {"error": f"NOT CHECKED ({verdict.error})"}
                return
            step["cosmos_assessment"] = {
                "target_visible": verdict.target_visible,
                "destination_visible": verdict.destination_visible,
                "obstacle": verdict.obstacle,
                "feasibility": verdict.feasibility,
                "approach": verdict.approach,
                "reasoning": verdict.reasoning,
            }
            log.info(f"[Cosmos] Step {step.get('index')} pre-dispatch assessment: "
                     f"feasibility={verdict.feasibility!r} obstacle={verdict.obstacle!r} "
                     f"— {verdict.reasoning!r}")
            COSMOS_REASONING_LOG.add(
                f"Step {step.get('index')} pre-dispatch — {subgoal.get('target', '')} → {subgoal.get('destination', '')}",
                f"feasibility={verdict.feasibility} · obstacle={verdict.obstacle} · approach={verdict.approach}\n{verdict.reasoning}",
                image=frame[0] if frame else None,
            )
        except Exception as e:
            log.warning(f"[Cosmos] Pre-dispatch assessment failed unexpectedly (non-fatal): {e}")
            step["cosmos_assessment"] = {"error": f"NOT CHECKED (internal error: {e})"}

    @staticmethod
    def _alt_target_allowed(alt_target: str, refused_target: str, refusal: str) -> bool:
        """After a refusal the model may only (a) try the SAME object differently or (b) move an object the refusal
        names as the blocker. Anything else is an unrequested task and is rejected."""
        a, r, txt = (alt_target or "").lower().strip(), (refused_target or "").lower().strip(), (refusal or "").lower()
        if not a:
            return False
        return a in r or r in a or a in txt

    def _refusal_alternative(self, decision: Optional[Dict[str, Any]], refused_signatures: set,
                             scene: Dict[str, Any], remaining_replanned: bool,
                             refused_target: str = "", refusal: str = ""):
        """The one alternative Nemotron may offer after a refusal: ("replan_step", subgoal) or
        ("replan_remaining", [subgoals]). Accepted only if it is a DIFFERENT action from every refused one and its
        first subgoal passes the guardrail. 'retry' (or nothing usable) means: leave the step refused."""
        if not decision:
            return None
        choice = decision.get("decision")
        if choice == "replan_step":
            cand = decision.get("subgoal")
            candidates = [cand] if cand else []
        elif choice == "replan_remaining" and not remaining_replanned:
            candidates = decision.get("subgoals") or []
        else:
            return None
        if not candidates:
            return None
        first = candidates[0]
        if first.get("action") == "pick_place" and not self._alt_target_allowed(first.get("target", ""), refused_target, refusal):
            log.warning(f"Refusal alternative rejected: it is about {first.get('target')!r}, not the refused "
                        f"{refused_target!r} or the object named as the blocker — that would be an unrequested task.")
            return None
        if first.get("action") == "pick_place" and _step_signature(first) in refused_signatures:
            log.warning(f"Refusal alternative rejected: it repeats the refused step ({first}).")
            return None
        ok, why = self.validate_subgoal(first, scene)
        if not ok:
            log.warning(f"Refusal alternative rejected by the guardrail ({why}).")
            return None
        return (choice, cand if choice == "replan_step" else candidates)

    def _dispatch(self, instruction: str) -> bool:
        """Posts the instruction, then blocks until Isaac Sim's own physics loop has actually
        picked up *this specific* dispatch — confirmed via `current_request_id`, a per-request
        ID isaac_sim_service.py now assigns at enqueue time and echoes back in /execute's
        response (see its own command_queue/`_new_request_id` — hardened there specifically
        for this).

        This is the third iteration of this method, each one closing a real bug the previous
        one didn't cover, all found live this session from concurrent manual + orchestrated
        use of the one shared robot command queue:
          1. Originally: no confirmation at all — polling for "idle" immediately after
             dispatch could land in the gap before Isaac Sim's loop (~7-8fps) even dequeues
             the command, reading leftover busy=False/last_task_success from the *previous*
             task and concluding the new one had already finished.
          2. Then: confirmed via busy=True alone — but busy=True is equally true while Isaac
             Sim is mid-task on a completely unrelated command someone else dispatched
             concurrently, so this falsely confirmed "our" dispatch had started while actually
             still watching someone else's in-flight task — misattributing its result to ours.
          3. Then: confirmed via busy=True AND last_instruction text match — closes (2), but a
             byte-identical instruction (e.g. our own plain retry, or two callers happening to
             send the exact same phrase) is structurally indistinguishable from a genuinely
             fresh dequeue of *our* new attempt by text alone.
        request_id is assigned uniquely per HTTP call, so it has none of these ambiguities."""
        status, body = _http_post_json(f"{ISAAC_SIM_URL}/execute", {"instruction": instruction})
        if status != 200 or not body or "request_id" not in body:
            log.warning(f"Dispatch rejected or missing request_id (status={status}, body={body})")
            return False
        my_request_id = body["request_id"]
        deadline = time.time() + 5.0
        while time.time() < deadline:
            state = _http_get_json(f"{ISAAC_SIM_URL}/state")
            # Request id alone confirms the sim dequeued OUR command. busy is not required: a task the sim refuses
            # at planning time can already be idle again by the next poll, and must not read as "never started".
            if state and state.get("current_request_id") == my_request_id:
                return True
            time.sleep(0.05)
        log.warning(f"Sim never confirmed it started processing request #{my_request_id}: {instruction!r}")
        return False

    def _wait_for_idle(self, timeout: float = PLAN_STEP_TIMEOUT_S) -> Optional[Dict[str, Any]]:
        deadline = time.time() + timeout
        state = None
        while time.time() < deadline:
            state = _http_get_json(f"{ISAAC_SIM_URL}/state")
            if state is not None and not state.get("busy", False):
                return state
            time.sleep(POLL_INTERVAL_S)
        log.warning(f"Timed out after {timeout}s waiting for sim to go idle.")
        return state

    def run(self, instruction: str, push_log=None, on_plan_update=None) -> Dict[str, Any]:
        """Closed-loop agentic execution: decompose -> for each subgoal, guardrail-gate ->
        dispatch -> verify against real telemetry. On a verified failure, this does NOT follow
        a fixed retry-then-replan schedule — it asks Nemotron to actually diagnose the failure
        (given the real error, live scene, and what's already been completed) and choose
        retry / replan-this-step / replan-the-remaining-plan, one LLM call per failure. A
        `replan_remaining` decision can only fire once per run (safety valve against a runaway
        loop) — a second request for it is downgraded to a plain retry.

        After every subgoal reports DONE, this does one more thing the previous version
        didn't: a deterministic (code-only, no LLM) final check that the whole original goal
        still holds — re-comparing each already-verified placement's live position against
        what it was at verification time, catching a later step (e.g. a stack) disturbing an
        earlier one. Only unconditional SUCCESS if that holds too.

        Keeps `PickPlaceController`/Isaac Sim's own execution completely untouched — it's
        still just the tool this calls via plain-text `/execute`, per the original design.
        Safe to call from a background thread — every step is a blocking HTTP call/poll, on
        purpose, so it never touches the caller's event loop.

        `on_plan_update(plan_dict)`, if given, is called after every real state transition —
        this is what drives the live "agent thinking" view in the UI, not a log replay."""

        def _log(msg: str, level: str = "INFO"):
            (log.warning if level in ("WARN", "ERROR") else log.info)(msg)
            if push_log:
                push_log(msg, level)

        def _emit():
            if on_plan_update:
                on_plan_update(json.loads(json.dumps(plan)))  # cheap deep copy, JSON-safe

        def _preview_text(sg):
            try:
                return subgoal_to_instruction(sg)
            except ValueError:
                return f"(unrecognized action: {sg.get('action')!r})"

        # Fresh stop-flag for this run — a previous hard-stop must not carry over and abort a
        # plan the user just asked for afterward.
        self._stop_event.clear()

        plan = {"instruction": instruction, "active": True, "overall": "PLANNING", "steps": []}
        _emit()

        scene = fetch_live_scene()
        subgoals, planning_reasoning = self.decompose_task_with_reasoning(instruction, scene)
        # Nemotron's own chain-of-thought for *why* it split the instruction this way — shown
        # in the UI above the step list so "the planner is thinking" is visible, not just its
        # final answer. Truncated defensively; a reasoning model's CoT can run long.
        plan["planning_reasoning"] = planning_reasoning[:2000] if planning_reasoning else None
        NEMOTRON_REASONING_LOG.add(f"Decompose: {instruction[:80]}", planning_reasoning)
        if not subgoals:
            seen = ", ".join(str(o.get("name", "?")).lower() for o in (scene.get("objects") or {}).values())
            if planning_reasoning and planning_reasoning.startswith("(the cubes are already"):
                msg = "The cubes are already stacked in one pile, so there is nothing to do."
            else:
                msg = ("I could not turn that into robot steps: it mentions something I cannot see right now. "
                       + (f"Objects I can currently see: {seen}." if seen else "I cannot see any objects at the moment."))
            _log(f"{msg} — nothing dispatched.", "WARN")
            plan.update(active=False, overall="REJECTED", reason=msg, answer=msg)
            _emit()
            return {"status": "REJECTED", "instruction": instruction, "steps": [], "answer": msg}

        plan["steps"] = [
            {"index": i, "text": _preview_text(sg), "subgoal": sg, "status": "PENDING",
             "detail": None}
            for i, sg in enumerate(subgoals, start=1)
        ]
        plan["overall"] = "RUNNING"
        _log(f"Plan: {len(subgoals)} subgoal(s) for '{instruction}'")
        if planning_reasoning:
            _log(f"[Nemotron reasoning] {planning_reasoning[:300]}", "INFO")
        _emit()

        results = []
        completed_subgoals: List[Dict[str, Any]] = []
        remaining_replanned = False  # replan_remaining allowed once per run — safety valve
        refused_signatures: set = set()   # actions the sim has refused this run — never proposed again
        refused_keys: set = set()         # objects the sim refused to move: later steps that depend on them are skipped
        refusal_log: List[Dict[str, Any]] = []   # every refusal this run, verbatim, attributed to the object it was about
        aborted = False  # set by a hard-stop request — short-circuits straight to ABORTED,
                          # skipping the final-goal check entirely (nothing left to verify)
        blocker_moves = 0
        idx = 0
        while idx < len(subgoals):
            if self._stop_event.is_set():
                aborted = True
                break
            i = idx + 1
            subgoal = subgoals[idx]
            step = plan["steps"][idx]
            n = len(subgoals)
            try:
                text = subgoal_to_instruction(subgoal)
                step["text"] = text
            except ValueError as e:
                _log(f"Step {i}/{n} skipped — {e}", "WARN")
                step.update(status="REJECTED", detail=str(e))
                _emit()
                results.append({"subgoal": subgoal, "status": "REJECTED", "reason": str(e)})
                idx += 1
                continue

            if subgoal.get("action") == "pick_place" and blocker_moves < MAX_BLOCKER_MOVES:
                _tk = _resolve_named_key(subgoal.get("target", ""), scene)
                _blocker = _stack_blocker(_tk, scene) if _tk else None
                if _blocker is not None:
                    blocker_moves += 1
                    _clear = {"action": "pick_place", "target": _cube_name(scene, _blocker),
                              "destination": FREE_SPOT_DESTINATION}
                    _log(f"Step {i}/{n}: the {_cube_name(scene, _tk)} has the {_cube_name(scene, _blocker)} on top of it — "
                         f"moving the {_cube_name(scene, _blocker)} to a free spot first.", "WARN")
                    subgoals.insert(idx, _clear)
                    plan["steps"].insert(idx, {"index": idx + 1, "text": _preview_text(_clear), "subgoal": _clear,
                                               "status": "PENDING",
                                               "detail": f"inserted: clears the {_cube_name(scene, _blocker)} off the "
                                                         f"{_cube_name(scene, _tk)}"})
                    for _j, _st in enumerate(plan["steps"], start=1):
                        _st["index"] = _j
                    _emit()
                    continue

            ok, reason = self.validate_subgoal(subgoal, scene)
            if not ok and _scene_is_camera_derived(scene) and ("not found" in reason or "out of bounds" in reason):
                # A camera-derived scene is measured, so a borderline reading (or an object caught
                # mid-motion / momentarily occluded) can fail the guardrail once and pass on the
                # next look. Re-observe and re-validate a few times before rejecting — but only for
                # perception-type failures (not found / out of reach). Not done for the deterministic
                # scene (exact simulator coordinates — a retry can't change it).
                for attempt in range(1, GUARDRAIL_RETRIES + 1):
                    _log(f"[Guardrail] Step {i}/{n} failed on a camera-derived scene ({reason}) — "
                         f"re-observing (retry {attempt}/{GUARDRAIL_RETRIES})", "WARN")
                    step.update(detail=f"guardrail retry {attempt}/{GUARDRAIL_RETRIES}: {reason}")
                    _emit()
                    time.sleep(GUARDRAIL_RETRY_DELAY_S)
                    scene = fetch_live_scene()
                    ok, reason = self.validate_subgoal(subgoal, scene)
                    if ok:
                        _log(f"[Guardrail] Step {i}/{n} passed on retry {attempt}.", "INFO")
                        break
            if not ok:
                _log(f"[Guardrail] Step {i}/{n} REJECTED: {reason}", "WARN")
                step.update(status="REJECTED", detail=reason)
                _emit()
                results.append({"subgoal": subgoal, "status": "REJECTED", "reason": reason})
                idx += 1
                continue

            # Real precondition check, not blind re-execution: if the target is already at
            # its destination (e.g. re-running "stack red, green, blue" after red is already
            # in the tray), skip dispatching this step entirely rather than needlessly
            # re-picking/re-placing something already correct.
            if _subgoal_already_satisfied(subgoal, scene):
                _log(f"Step {i}/{n} already satisfied — '{text}' is already true in the live "
                     f"scene, skipping.", "INFO")
                step.update(status="DONE", detail="already at destination — skipped")
                if subgoal.get("action") == "pick_place":
                    step["final_pos"] = _resolve_named_position(subgoal.get("target", ""), scene)
                # Cosmos visual check (Step 7) — turns "skipped, assumed already
                # correct" into "skipped, and the model actually looked and confirmed it".
                # enforce=False: nothing was dispatched here, so a disagreeing visual read
                # is evidence worth showing, never a reason to fail an otherwise-correct skip.
                self._attach_visual_evidence(step, subgoal, enforce=False)
                _emit()
                results.append({"subgoal": subgoal, "status": "DONE"})
                completed_subgoals.append(subgoal)
                idx += 1
                continue

            # Real perceive-before-every-action check (deterministic, no LLM call in the
            # common case) — this is the actual gap real ReAct-style feedback pointed at:
            # everything above only re-examines state *after* a dispatch fails; this checks
            # *before* dispatching whether a later step's assumption (e.g. "stack on the red
            # cube") still holds against what's actually happened since. Only escalates to a
            # real Nemotron call when something concrete looks wrong, not on a fixed cadence.
            drifted, drift_reason = _subgoal_drift_detected(subgoal, scene, plan["steps"][:idx])
            if drifted:
                _log(f"Step {i}/{n} pre-dispatch check: {drift_reason}", "WARN")
                step.update(status="REPLANNING", detail=drift_reason)
                _emit()
                decision = request_failure_decision(
                    instruction, completed_subgoals, subgoal, subgoals[idx + 1:], scene, drift_reason
                )
                reason_txt = (decision or {}).get("reason", "")
                choice = (decision or {}).get("decision", "retry")
                planner_reasoning = (decision or {}).get("reasoning", "")
                if planner_reasoning:
                    step["planner_reasoning"] = planner_reasoning[:1500]
                    _log(f"[Nemotron reasoning] {planner_reasoning[:300]}", "INFO")
                    NEMOTRON_REASONING_LOG.add(f"Step {i} failure decision: {choice}", planner_reasoning)
                # Mirrors the post-verified-failure decision-application below (same three
                # outcomes, same guardrail re-check) — this is deliberately a pre-dispatch
                # instance of the exact same real decision call, not a different code path.
                if choice == "replan_step":
                    candidate = decision["subgoal"]
                    ok2, reason2 = self.validate_subgoal(candidate, scene)
                    if ok2:
                        _log(f"Step {i} replanned before dispatch: {candidate} — {reason_txt}", "WARN")
                        subgoal = candidate
                        subgoals[idx] = candidate
                        text = subgoal_to_instruction(candidate)
                        step.update(subgoal=candidate, text=text)
                    else:
                        _log(f"Step {i} pre-dispatch replan_step proposal rejected by guardrail "
                             f"({reason2}) — proceeding with original subgoal.", "WARN")
                elif choice == "replan_remaining" and not remaining_replanned:
                    candidates = decision.get("subgoals") or []
                    if candidates:
                        _log(f"Step {i} pre-dispatch drift triggered a full replan of the "
                             f"remaining {n - idx} step(s) — {reason_txt}", "WARN")
                        subgoals[idx:] = candidates
                        n = len(subgoals)
                        plan["steps"][idx:] = [
                            {"index": idx + 1 + j, "text": _preview_text(sg), "subgoal": sg,
                             "status": "PENDING", "detail": None}
                            for j, sg in enumerate(candidates)
                        ]
                        step = plan["steps"][idx]
                        subgoal = subgoals[idx]
                        text = subgoal_to_instruction(subgoal)
                        remaining_replanned = True
                    else:
                        _log(f"Step {i} pre-dispatch replan_remaining returned nothing usable "
                             f"— proceeding with original subgoal.", "WARN")
                elif choice == "replan_remaining" and remaining_replanned:
                    _log(f"Step {i}: pre-dispatch drift asked for another full replan, but "
                         f"that's already been used once this run — proceeding as originally "
                         f"planned to avoid a runaway replan loop.", "WARN")
                else:
                    _log(f"Step {i}: pre-dispatch decision = proceed as planned — {reason_txt}", "INFO")
                _emit()

            _log(f"Step {i}/{n}: {text}")
            self._attach_cosmos_assessment(step, subgoal)
            step.update(status="RUNNING", detail=None)
            _emit()
            step_done = False
            step_refused = False
            final_state = None
            attempt = 1
            while attempt <= MAX_SUBGOAL_ATTEMPTS:
                if self._stop_event.is_set():
                    aborted = True
                    break
                if not self._dispatch(text):
                    _log(f"Step {i} could not be dispatched (sim busy/unreachable).", "WARN")
                    step["detail"] = "sim busy/unreachable, retrying dispatch..."
                    _emit()
                    time.sleep(1.0)
                    attempt += 1
                    continue
                final_state = self._wait_for_idle()
                if self._stop_event.is_set():
                    # Covers the case where the hard-stop landed while we were inside
                    # _wait_for_idle's poll — isaac_sim_service's own /hard_stop already
                    # dropped busy to False almost immediately, but that's a real abort, not
                    # a real verification result; don't let it fall through and read
                    # (possibly stale) last_task_success as if this attempt actually finished.
                    aborted = True
                    break
                # _wait_for_idle can come back having timed out — sim genuinely still busy,
                # never reached completion. Confirmed as a real bug: without this check, a
                # timeout fell through to reading last_task_success anyway, which is *leftover*
                # from whatever task last actually finished (possibly nothing to do with this
                # attempt) — reported a step DONE while the arm had never actually finished
                # (or ever really executed) this specific attempt at all.
                if final_state is None or final_state.get("busy", True):
                    _log(f"Step {i} attempt {attempt}/{MAX_SUBGOAL_ATTEMPTS} timed out waiting "
                         f"for the sim to finish — not trusting stale telemetry, treating as failed.", "WARN")
                    step.update(status="RETRYING", detail=f"timed out waiting for completion (attempt {attempt}/{MAX_SUBGOAL_ATTEMPTS})")
                    _emit()
                    attempt += 1
                    continue
                # Only pick_place subgoals carry a real verification signal; home/reset/direct
                # are considered done once the sim reports idle again (no verified target).
                if subgoal.get("action") != "pick_place":
                    step_done = True
                    break
                verified = final_state.get("last_task_success")
                if verified is True:
                    step_done = True
                    break
                err_mm = final_state.get("last_task_error_mm")
                # Real grasp-confirmation / displacement signals (isaac_sim_service.py) —
                # folded into the same failure description Nemotron sees, so a confirmed
                # failed grasp gets diagnosed as that, not just a generic position miss.
                grasp_confirmed = final_state.get("last_grasp_confirmed")
                displacement_mm = final_state.get("last_object_displacement_mm")
                refusal = final_state.get("last_task_error") if final_state.get("last_task_refused") else None
                failure_desc = (f"the sim refused the task before moving: {refusal}" if refusal
                                else f"{err_mm}mm position error")
                if grasp_confirmed is False and not refusal:      # (those two are stale after a refusal: nothing moved)
                    failure_desc += "; grasp NOT confirmed (gripper closed on nothing — likely never actually picked up the object)"
                if displacement_mm is not None and not refusal:
                    failure_desc += f"; object moved {displacement_mm}mm from its starting position"
                if refusal:
                    # The sim refused before moving. Quote it verbatim and never retry it (see the policy below).
                    step.setdefault("refusals", []).append(refusal)
                    refusal_log.append({"object": subgoal.get("target", "?"), "wanted": subgoal.get("destination") or "",
                                        "text": refusal})
                    refused_signatures.add(_step_signature(subgoal))
                    _log(f"Step {i} REFUSED by the robot: \"{refusal}\"", "WARN")
                    step.update(status="REPLANNING", detail=f"refused by the robot: {refusal}")
                    # Policy (user decision): after a refusal the step simply FAILS, with the robot's reason shown. No
                    # alternative destination or object is ever substituted - doing something other than what was asked
                    # (another destination, another object) is worse than stopping. No extra model call is made either.
                    step.update(status="FAILED", refused=True, detail=f"refused by the robot: {refusal}")
                    _emit()
                    step_refused = True
                    break
                    attempt = 1
                    continue

                _log(f"Step {i} verification failed (attempt {attempt}/{MAX_SUBGOAL_ATTEMPTS}, "
                     f"{failure_desc})", "WARN")
                step.update(status="RETRYING", detail=f"{failure_desc} (attempt {attempt}/{MAX_SUBGOAL_ATTEMPTS})")
                _emit()
                if attempt >= MAX_SUBGOAL_ATTEMPTS:
                    break

                fresh_scene = fetch_live_scene()
                step.update(status="REPLANNING", detail="Nemotron diagnosing the failure...")
                _emit()
                decision = request_failure_decision(
                    instruction, completed_subgoals, subgoal, subgoals[idx + 1:], fresh_scene, failure_desc
                )
                scene = fresh_scene
                reason_txt = (decision or {}).get("reason", "")
                choice = (decision or {}).get("decision", "retry")
                planner_reasoning = (decision or {}).get("reasoning", "")
                if planner_reasoning:
                    step["planner_reasoning"] = planner_reasoning[:1500]
                    _log(f"[Nemotron reasoning] {planner_reasoning[:300]}", "INFO")
                    NEMOTRON_REASONING_LOG.add(f"Step {i} failure decision: {choice}", planner_reasoning)

                if choice == "replan_step":
                    candidate = decision["subgoal"]
                    ok2, reason2 = self.validate_subgoal(candidate, fresh_scene)
                    if ok2:
                        _log(f"Step {i} replanned (this step only): {candidate} — {reason_txt}", "WARN")
                        subgoal = candidate
                        subgoals[idx] = candidate
                        text = subgoal_to_instruction(candidate)
                        step.update(subgoal=candidate, text=text, status="REPLANNING",
                                    detail=f"Nemotron: {reason_txt or 'replanned this step'}")
                    else:
                        _log(f"Step {i} replan_step proposal rejected by guardrail ({reason2}) — "
                             f"retrying original subgoal instead.", "WARN")
                elif choice == "replan_remaining" and not remaining_replanned:
                    candidates = decision.get("subgoals") or []
                    if candidates:
                        _log(f"Step {i} triggered a full replan of the remaining "
                             f"{n - idx} step(s) — {reason_txt}", "WARN")
                        subgoals[idx:] = candidates
                        n = len(subgoals)
                        plan["steps"][idx:] = [
                            {"index": idx + 1 + j, "text": _preview_text(sg), "subgoal": sg,
                             "status": "PENDING", "detail": None}
                            for j, sg in enumerate(candidates)
                        ]
                        step = plan["steps"][idx]
                        subgoal = subgoals[idx]
                        text = subgoal_to_instruction(subgoal)
                        step.update(status="REPLANNING", detail=f"Nemotron: {reason_txt or 'replanned remaining steps'}")
                        remaining_replanned = True
                        attempt = 0  # fresh attempt budget for the newly-substituted step
                    else:
                        _log(f"Step {i} replan_remaining returned nothing usable — retrying "
                             f"original subgoal instead.", "WARN")
                elif choice == "replan_remaining" and remaining_replanned:
                    _log(f"Step {i}: Nemotron asked to replan the remaining plan again, but "
                         f"that's already been used once this run — retrying as-is instead to "
                         f"avoid a runaway replan loop.", "WARN")
                else:
                    _log(f"Step {i}: Nemotron decision = retry — {reason_txt}", "WARN")
                    step.update(status="RETRYING", detail=f"Nemotron: retry as-is ({reason_txt or 'no reason given'})")
                _emit()
                attempt += 1

            if aborted:
                break

            if step_done and subgoal.get("action") == "pick_place" and final_state:
                # Recorded for the deterministic final-goal check below — this is what "the
                # object was actually here when we verified it" means, in re-checkable form.
                step["final_pos"] = _resolve_named_position(subgoal.get("target", ""), final_state)

            if step_done and subgoal.get("action") == "pick_place":
                # Cosmos visual check (Step 7): a second, independent signal that the
                # step actually looks right, not just measures right. Deliberately additive —
                # this project's real position/grasp/displacement checks above are what
                # actually decided step_done, and stay the sole authority for DONE/FAILED. A
                # disagreeing visual read is logged as real evidence on the step (visible in
                # the UI) rather than silently flipping a passing step to failed; a Cosmos
                # outage or timeout can never block or stall a plan that's positionally correct.
                self._attach_visual_evidence(step, subgoal, enforce=True)

            refusals_hist = list(step.get("refusals") or [])
            if not step_done and step_refused:
                # Nothing moved, so the world is unchanged: record the refusal (verbatim) and carry on with the rest.
                last = refusals_hist[-1]
                step.update(status="FAILED", refused=True, detail=f"refused by the robot: {last}")
                _log(f"Step {i}/{n} not done — refused by the robot: \"{last}\" — continuing with the remaining steps.", "WARN")
                _emit()
                results.append({"subgoal": subgoal, "status": "FAILED", "refused": True, "refusals": refusals_hist,
                                "reason": f"the robot refused: {last}"})
                _rk = _resolve_named_key(subgoal.get("target", ""), scene)
                if _rk:
                    refused_keys.add(_rk)
                scene = fetch_live_scene()
                idx += 1
                continue
            step.update(status="DONE" if step_done else "FAILED")
            _emit()
            _res = {"subgoal": subgoal, "status": "DONE" if step_done else "FAILED"}
            if refusals_hist:
                _res["refusals"] = refusals_hist
            if not step_done:
                _res["reason"] = step.get("detail") or "failed"
            results.append(_res)
            if not step_done:
                _log(f"Aborting remaining plan after step {i}/{n} failed.", "ERROR")
                for later in plan["steps"][idx + 1:]:
                    later["status"] = "ABORTED"
                _emit()
                break
            completed_subgoals.append(subgoal)
            scene = fetch_live_scene()  # refresh grounding for the next subgoal
            idx += 1

        if aborted:
            # Hard-stop requested (GUI's red "Hard Reset" button) — bail out immediately,
            # no final-goal re-check (there's nothing left to verify against; the arm is
            # already homing per isaac_sim_service's own /hard_stop handling), no further
            # Nemotron calls. Whatever subgoal was mid-flight, plus everything after it, is
            # marked ABORTED so the UI reflects exactly what happened, not a fabricated
            # DONE/FAILED for a step that never got to finish.
            _log(f"Plan stopped by hard reset — {idx}/{len(subgoals)} subgoal(s) completed "
                 f"before the stop.", "WARN")
            for later in plan["steps"][idx:]:
                if later["status"] not in ("DONE",):
                    later["status"] = "ABORTED"
            plan.update(active=False, overall="ABORTED")
            _emit()
            return {"status": "ABORTED", "instruction": instruction, "steps": results}

        # Deterministic final-goal verification (code only, no LLM — requirement is to
        # minimize LLM calls). Every individual placement was already verified when its own
        # step finished; this re-checks that a *later* step didn't physically disturb an
        # *earlier*, already-verified one (e.g. knocked over while stacking) — only then is
        # the ORIGINAL goal actually still true, not just "every step individually passed".
        n_done = sum(1 for r in results if r["status"] == "DONE")
        goal_held = True
        if n_done > 0:
            plan["overall"] = "VERIFYING"
            _emit()
            final_scene = fetch_live_scene()
            for step in plan["steps"]:
                if step["status"] != "DONE" or step["subgoal"].get("action") != "pick_place":
                    continue
                recorded = step.get("final_pos")
                if not recorded:
                    continue
                now_pos = _resolve_named_position(step["subgoal"].get("target", ""), final_scene)
                if now_pos is None:
                    continue
                drift_m = math.hypot(recorded[0] - now_pos[0], recorded[1] - now_pos[1])
                if drift_m > GOAL_DRIFT_TOLERANCE_M:
                    goal_held = False
                    step["detail"] = f"disturbed after verification — moved {drift_m * 1000:.0f}mm since"
                    _log(f"Final goal check: '{step['text']}' no longer holds — object moved "
                         f"{drift_m * 1000:.0f}mm after being verified.", "WARN")
                    continue
                # Same absolute check as the pre-dispatch one, run again here as a second
                # layer: a step can individually verify PASS while stacking onto a reference
                # that was never actually in the tray to begin with (resolve_pick_place_targets
                # stacks relative to wherever the reference *currently* is, correct or not) —
                # relative drift alone (above) can't catch that, since nothing "moved" from
                # this step's own perspective.
                out_of_tray, oot_reason = _reference_out_of_tray(step["subgoal"], final_scene)
                if out_of_tray:
                    goal_held = False
                    step["detail"] = oot_reason
                    _log(f"Final goal check: '{step['text']}' no longer holds — {oot_reason}", "WARN")

        if n_done == len(subgoals) and goal_held:
            overall = "SUCCESS"
        elif n_done == len(subgoals) and not goal_held:
            overall = "GOAL_NOT_HELD"
        elif n_done > 0:
            overall = "PARTIAL"
        else:
            overall = "FAILED"
        _log(f"Plan complete: {overall} ({n_done}/{len(subgoals)} subgoals done)",
             "SUCCESS" if overall == "SUCCESS" else "WARN")
        summary = build_outcome_summary(results, refusal_log)
        answer = compose_final_answer(instruction, summary)
        _log(f"[Answer] {answer}", "SUCCESS" if overall == "SUCCESS" else "WARN")
        plan.update(active=False, overall=overall, answer=answer, summary=summary)
        _emit()
        return {"status": overall, "instruction": instruction, "steps": results, "answer": answer, "summary": summary}
