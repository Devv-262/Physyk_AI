"""Visual post-condition checks per plan step (cosmos_integration.md Step 7). Runs
ALONGSIDE the existing real position-based verification already in isaac_sim_service.py /
physyk_agent_orchestrator.py (last_task_success / last_task_error_mm) — never replaces it.
A plan step's DONE/FAILED status is still governed by that real position check; this adds a
second, independent signal (does the scene actually LOOK right, not just measure right) as
visible evidence on each step, matching this project's established rule for combining
verification signals: either signal flagging a problem is worth surfacing, but a passing
position check is not silently overridden by a visual check alone (see
physyk_agent_orchestrator.py's call site for exactly how this evidence is used).

Failure handling is deliberately asymmetric, per the source doc: a visual check that
actually runs and disagrees is a real, reportable "UNVERIFIED" verdict; a check that ERRORS
(Cosmos down, timeout, bad JSON) is NOT a failure — it comes back verified=True,
enforced=False, observed="NOT CHECKED (<reason>)", so a dead Cosmos server can never block
or stall an otherwise-succeeding plan.
"""

import dataclasses
from typing import Callable, Optional, Tuple

from . import cosmos_client

VERIFY_TIMEOUT_S = 6.0


@dataclasses.dataclass
class VerifyVerdict:
    verified: bool       # True unless a real (non-errored) check actually disagreed
    enforced: bool        # False whenever this verdict couldn't be meaningfully checked
    confidence: float     # Cosmos's own reported confidence, 0-1 (0 when not enforced)
    observed: str         # one-sentence description shown in the UI under the step


def verify_step(get_frame: Callable[[], Optional[Tuple[bytes, str]]], action: str, obj: str,
                 destination: Optional[str] = None, enforce: bool = True) -> VerifyVerdict:
    """get_frame() returns (raw_image_bytes, mime_type) for the current live overview camera
    frame, or None if unavailable. `enforce=False` means "ask anyway, for real UI evidence,
    but never let the answer count as a verified failure" — used for the already-satisfied/
    skipped-dispatch case (cosmos_integration.md's own example: no motion happened, so demand
    less from the visual check, but still make the plan card's evidence real, not a blank
    assumption)."""
    try:
        frame = get_frame()
    except Exception as e:
        return VerifyVerdict(True, False, 0.0, f"NOT CHECKED (camera read failed: {e})")
    if not frame:
        return VerifyVerdict(True, False, 0.0, "NOT CHECKED (no camera frame available)")
    image_bytes, mime = frame

    if action == "pick_place" and obj:
        loc = destination or "its intended destination"
        question = (
            f"Look at this image of a tabletop robotic workspace. Is the {obj} now located "
            f"at or on {loc}? Answer with ONLY a JSON object: "
            '{"result": true/false, "confidence": <0.0-1.0>, "observed": '
            '"<one short sentence describing what you actually see>"}'
        )
    else:
        question = (
            f"Look at this image of a tabletop robotic workspace. Describe where the {obj} "
            f"currently is. Answer with ONLY a JSON object: "
            '{"result": true, "confidence": <0.0-1.0>, "observed": "<one short sentence>"}'
        )

    result, latency_ms, err = cosmos_client.ask_visual_question(image_bytes, mime, question, timeout=VERIFY_TIMEOUT_S)
    if err is not None:
        return VerifyVerdict(True, False, 0.0, f"NOT CHECKED ({err})")

    try:
        confidence = float(result.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    observed = str(result.get("observed", "") or "")[:200] or "(no description returned)"
    matched = bool(result.get("result", True))
    verified = matched if enforce else True
    return VerifyVerdict(verified, enforce, confidence, observed)


@dataclasses.dataclass
class AssessVerdict:
    ok: bool               # True unless the check itself errored (never blocks dispatch)
    target_visible: Optional[bool]
    destination_visible: Optional[bool]
    obstacle: str           # short free-text, "" / "none" if Cosmos saw nothing in the way
    feasibility: str        # "low" / "medium" / "high", or "" if not checked
    approach: str            # e.g. "top-right", free text
    reasoning: str            # one-two sentence natural-language explanation, shown in the UI
    error: Optional[str] = None


def assess_pick_place(get_frame: Callable[[], Optional[Tuple[bytes, str]]], target: str,
                       destination: str, timeout: float = VERIFY_TIMEOUT_S) -> AssessVerdict:
    """Pre-dispatch visual reasoning call (cosmos_integration.md's own "COSMOS REASON 2"
    panel: target visible / destination visible / obstacle / feasibility / approach) — purely
    informational, never gates or blocks the dispatch (matches every other Cosmos hook in this
    module: real evidence shown to the operator, zero effect on what the robot actually does).
    Runs BEFORE the pick/place motion starts, unlike verify_step which runs after."""
    try:
        frame = get_frame()
    except Exception as e:
        return AssessVerdict(False, None, None, "", "", "", "", error=f"camera read failed: {e}")
    if not frame:
        return AssessVerdict(False, None, None, "", "", "", "", error="no camera frame available")
    image_bytes, mime = frame

    dest_desc = destination or "no specific destination stated"
    question = (
        f"Look at this image of a tabletop robotic workspace with a Franka Panda arm. A robot "
        f"is about to pick up the {target} and place it at: {dest_desc}. Answer with ONLY a "
        f"JSON object: {{\"target_visible\": true/false, \"destination_visible\": true/false, "
        f"\"obstacle\": \"<short description of anything blocking the path, or 'none'>\", "
        f"\"feasibility\": \"low\"/\"medium\"/\"high\", \"approach\": \"<short recommended "
        f"grasp approach direction, e.g. 'top-right'>\", \"reasoning\": \"<one or two sentence "
        f"explanation of your assessment>\"}}"
    )
    result, latency_ms, err = cosmos_client.ask_visual_question(image_bytes, mime, question, timeout=timeout)
    if err is not None:
        return AssessVerdict(False, None, None, "", "", "", "", error=err)
    return AssessVerdict(
        ok=True,
        target_visible=result.get("target_visible"),
        destination_visible=result.get("destination_visible"),
        obstacle=str(result.get("obstacle", "") or "")[:150],
        feasibility=str(result.get("feasibility", "") or "")[:20],
        approach=str(result.get("approach", "") or "")[:80],
        reasoning=str(result.get("reasoning", "") or "")[:300],
    )
