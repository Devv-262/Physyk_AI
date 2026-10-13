#!/usr/bin/env python3
"""Offline tests for destination parsing, the guardrail, multi-object expansion, refusals and the final answer in
physyk_agent_orchestrator.py (Tasks B and B2).

No network, no sim container: Nemotron is stubbed (decomposition text, failure decisions and the final answer are
all scripted per test, including sloppy or wrong model output) and the sim's /state and /execute are a scripted fake
that can REFUSE a step exactly the way isaac_sim_service.py does (last_task_refused / last_task_error).
Run:  python3 test_dest_parsing.py
"""
import copy
import json
import sys

import physyk_agent_orchestrator as orch

orch.GUARDRAIL_RETRY_DELAY_S = 0.0
orch.POLL_INTERVAL_S = 0.0

TIER_Y = (0.515, 0.605, 0.695)        # the sim's per-tier slot y (informational: the orchestrator no longer uses it)
TOPS = orch.SHELF_RACK_TOPS

# ── live scene: base-centre positions as /state reports them (new stepped shelf on the +Y side) ────────────────
SCENE = {"busy": False, "objects": {
    "flask_blue":   {"name": "Blue Flask",   "keywords": ["blue flask", "flask with blue"],     "position": [0.40, 0.20, 0.002]},
    "flask_red":    {"name": "Red Flask",    "keywords": ["red flask", "flask with red"],       "position": [0.40, 0.00, 0.002]},
    "flask_empty":  {"name": "Empty Flask",  "keywords": ["empty flask", "empty"],              "position": [0.25, 0.605, 0.36]},  # middle tier
    "beaker_red":   {"name": "Red Beaker",   "keywords": ["red beaker", "beaker with red"],     "position": [0.55, 0.10, 0.002]},
    "beaker_green": {"name": "Green Beaker", "keywords": ["green beaker", "beaker with green"], "position": [0.05, 0.695, 0.64]},   # top tier
}}

RESULTS = []


def check(name, problems, note=""):
    RESULTS.append((name, problems, note))


def sg(target, destination, **extra):
    return dict({"action": "pick_place", "target": target, "destination": destination}, **extra)


# ── stubbed Nemotron ─────────────────────────────────────────────────────────────────────────────────────────
def A(*subgoals):
    return "<think>planning...</think>\n" + json.dumps(list(subgoals))


DECOMPOSE = {}                 # instruction -> raw model text
LLM_LOG = []                   # ("decompose"|"decision"|"answer", detail)
DECISION = [None]              # callable(prompt_text) -> dict | None    (the failure-decision reply)
ANSWER_MODE = ["quote"]        # "quote" | "paraphrase" | "down"
ANSWERS = []                   # captured final-answer payload summaries


def stub_chat(payload):
    system, user = payload["messages"][0]["content"], payload["messages"][-1]["content"]
    if user == "Diagnose the failure and decide.":
        LLM_LOG.append(("decision", "REFUSAL" if "THIS WAS A REFUSAL" in system else "miss"))
        reply = DECISION[0](system) if DECISION[0] else None
        if reply is None:
            return 500, None
        return 200, {"choices": [{"message": {"content": "<think>x</think>" + json.dumps(reply)}, "finish_reason": "stop"}]}
    if user == "Write the final answer.":
        LLM_LOG.append(("answer", ANSWER_MODE[0]))
        if ANSWER_MODE[0] == "down":
            return 500, None
        summary = json.loads(system.split("Outcome (JSON): ", 1)[1])
        ANSWERS.append(summary)
        if ANSWER_MODE[0] == "quote":
            bits = [f"I put the {e['object']} {e['where']}." for e in summary["placed"]]
            bits += [f"First the robot said \"{r['text']}\"." for r in summary["worked_around"]]
            for e in summary["not_placed"]:
                bits.append(f"I could not place the {e['object']} ({e['wanted']}): the robot said \"{e.get('refusal') or e['reason']}\". "
                            f"You could try another tier or free up space.")
            text = " ".join(bits)
        else:   # a model that paraphrases and forgets objects
            text = "Some things could not be done because the shelf was crowded."
        return 200, {"choices": [{"message": {"content": "<think>y</think>" + text}, "finish_reason": "stop"}]}
    LLM_LOG.append(("decompose", user))
    if user not in DECOMPOSE:
        return 500, None
    return 200, {"choices": [{"message": {"content": DECOMPOSE[user]}, "finish_reason": "stop"}]}


orch._nemotron_chat = stub_chat

agent = orch.NeMoRoboticsAgent()


def reset_llm():
    LLM_LOG.clear(); ANSWERS.clear()
    DECISION[0] = None
    ANSWER_MODE[0] = "quote"


# ═══ 1. destination parsing (no scene) ═════════════════════════════════════════════════════════════════════════
PARSE = [
    ("on the middle shelf",            dict(dest_kind="shelf", shelf_tier="middle")),
    ("top shelf",                      dict(dest_kind="shelf", shelf_tier="top")),
    ("the lowest rack",                dict(dest_kind="shelf", shelf_tier="bottom")),
    ("the shelf",                      dict(dest_kind="shelf", shelf_tier=None)),          # no tier = auto
    ("in the rack",                    dict(dest_kind="shelf", shelf_tier=None)),
    ("second shelf",                   dict(dest_kind="shelf", shelf_tier="middle")),      # counted from the bottom
    ("third shelf",                    dict(dest_kind="shelf", shelf_tier="top")),
    ("fourth shelf",                   dict(dest_kind="shelf", shelf_tier="fourth")),      # nonexistent: kept for rejection
    ("shelf 7",                        dict(dest_kind="shelf", shelf_tier="7")),
    ("the centre shelf",               dict(dest_kind="shelf", shelf_tier="middle")),
    ("left side of the top shelf",     dict(dest_kind="shelf", shelf_tier="top", slot_request="left side")),
    ("slot 3 of the bottom shelf",     dict(dest_kind="shelf", shelf_tier="bottom", slot_request="slot 3")),
    ("centre of the shelf",            dict(dest_kind="shelf", shelf_tier=None, slot_request="centre of")),
    ("next to the red beaker",         dict(dest_kind="relative", relation="next_to", reference_text="red beaker")),
    ("to the left of the empty flask", dict(dest_kind="relative", relation="left_of", reference_text="empty flask")),
    ("on top of the red beaker",       dict(dest_kind="relative", relation="next_to", reference_text="red beaker")),
    ("on the table",                   dict(dest_kind="table", table_spot="free")),
    ("back on the table",              dict(dest_kind="table", table_spot="free")),
    ("back to its spot",               dict(dest_kind="table", table_spot="default")),
    ("",                               dict(dest_kind=None)),
    ("the moon",                       dict(dest_kind=None, dest_error="is not understood")),
]
for text, want in PARSE:
    got = orch.parse_destination(text)
    probs = []
    for k, v in want.items():
        if k == "dest_error":
            if not got["dest_error"] or v not in got["dest_error"]:
                probs.append(f"dest_error {got['dest_error']!r} lacks {v!r}")
        elif got.get(k) != v:
            probs.append(f"{k}: got {got.get(k)!r}, want {v!r}")
    check(f"parse {text!r}", probs)

# ═══ 2. guardrail ════════════════════════════════════════════════════════════════════════════════════════════
GUARD = [
    # (instruction/destination, target, expectation)
    ("on the middle shelf",  "red flask",   ("ok", None)),
    ("on the shelf",         "red flask",   ("ok", None)),
    ("fourth shelf",         "blue flask",  ("reject", "no 'fourth' tier")),
    ("shelf 7",              "blue flask",  ("reject", "no '7' tier")),
    ("the moon",             "blue flask",  ("reject", "not understood")),
    ("next to the purple beaker", "red flask", ("reject", "'purple beaker' not found")),
    ("left of the red flask",     "red flask", ("reject", "next to itself")),
    ("next to the red beaker",    "blue flask", ("ok", None)),
    ("on the middle shelf",  "yellow flask", ("reject", "'yellow flask' not found")),
]
for dest, target, (verdict, expect) in GUARD:
    g = sg(target, dest)
    orch._validate_subgoal_shape(g)
    ok, reason = agent.validate_subgoal(g, SCENE)
    probs = [] if (ok and verdict == "ok") or (not ok and verdict == "reject" and expect in reason) else [f"ok={ok} reason={reason!r}"]
    check(f"guardrail: {target!r} -> {dest!r}", probs, reason if not ok else "")

# an occupied tier is NOT the orchestrator's call any more: even a tier holding objects passes (the sim decides)
g = orch._validate_subgoal_shape(sg("red flask", "top shelf"))
ok, reason = agent.validate_subgoal(g, SCENE)
check("guardrail: top tier already holds the Green Beaker -> still passes (sim decides slot freeness)", [] if ok else [reason])

# picking FROM the shelf (y 0.605 / z 0.36) and the wall-mounted top tier (z 0.64) are inside the shelf zone
for tgt, key in (("empty flask", "flask_empty"), ("green beaker", "beaker_green")):
    ok, reason = agent.validate_subgoal(orch._validate_subgoal_shape(sg(tgt, "on the table")), SCENE)
    check(f"guardrail: pick {tgt!r} from the shelf is within reach limits", [] if ok else [reason])
far = copy.deepcopy(SCENE); far["objects"]["flask_red"]["position"] = [0.40, 1.50, 0.002]
ok, reason = agent.validate_subgoal(orch._validate_subgoal_shape(sg("red flask", "on the shelf")), far)
check("guardrail: a target at y=1.5 is out of bounds", [] if (not ok and "out of bounds" in reason) else [f"{ok} {reason}"], reason)

# ═══ 3. "place all objects in the rack" — expansion from the live scene ═══════════════════════════════════════
reset_llm()
EXPECT_ORDER = ["red flask", "blue flask", "red beaker"]     # flasks first, nearest first; shelf objects excluded
for phrase in ("place all objects in the rack", "put everything on the shelf", "tidy up", "clear the table"):
    sgs, why = orch.decompose_task_with_reasoning(phrase, SCENE)
    probs = []
    if [g["target"] for g in sgs] != EXPECT_ORDER:
        probs.append(f"order {[g['target'] for g in sgs]}")
    if not all(g["dest_kind"] == "shelf" and g["shelf_tier"] is None and g["destination"] == "on the shelf" for g in sgs):
        probs.append("not all dest_kind=shelf / tier=null / 'on the shelf'")
    check(f"expand {phrase!r}", probs, f"{[g['target'] for g in sgs]}")
check("expansion needs no LLM call", [] if not LLM_LOG else [str(LLM_LOG)])

sgs, _ = orch.decompose_task_with_reasoning("put all the flasks on the top shelf", SCENE)
check("expand 'all the flasks on the top shelf' (tier kept, beakers excluded)",
      [] if [g["target"] for g in sgs] == ["red flask", "blue flask"] and all(g["shelf_tier"] == "top" for g in sgs) else [str(sgs)])
sgs, _ = orch.decompose_task_with_reasoning("put all beakers in the rack", SCENE)
check("expand 'all beakers' -> only the beaker on the table", [] if [g["target"] for g in sgs] == ["red beaker"] else [str(sgs)])

empty_table = copy.deepcopy(SCENE)
for k in ("flask_blue", "flask_red", "beaker_red"):
    empty_table["objects"][k]["position"] = [0.25, 0.605, 0.36]
sgs, why = orch.decompose_task_with_reasoning("place all objects in the rack", empty_table)
check("expand with nothing on the table -> empty plan", [] if sgs == [] and why == orch.NOTHING_TO_PLACE else [f"{sgs} {why}"])

sgs, _ = orch.decompose_task_with_reasoning("put all objects on the fourth shelf", SCENE)
ok, reason = agent.validate_subgoal(sgs[0], SCENE)
check("expansion keeps a nonexistent tier so the guardrail rejects it", [] if (len(sgs) == 3 and not ok and "no 'fourth' tier" in reason) else [f"{len(sgs)} {reason}"])

# compound instruction goes to the LLM; its 'all objects' subgoal is still expanded from the scene
reset_llm()
DECOMPOSE["place all objects in the rack, then go home"] = A(
    sg("all objects", "on the shelf", dest_kind="shelf", shelf_tier=None), {"action": "home"})
sgs, _ = orch.decompose_task_with_reasoning("place all objects in the rack, then go home", SCENE)
check("compound: LLM 'all objects' subgoal expanded + home kept",
      [] if [g.get("target") for g in sgs] == EXPECT_ORDER + [None] and sgs[-1]["action"] == "home" else [str(sgs)])

# ═══ 4. fake sim that can refuse ═════════════════════════════════════════════════════════════════════════════
class FakeSim:
    """Scripted /state + /execute. `script(instruction, destination, n)` -> None (success), ("refuse", text),
    ("miss", err_mm). On success the target is moved to where it was sent (shelf tiers only; other kinds stay put)."""

    def __init__(self, scene, script, camera=False, move_on_success=True):
        self.scene, self.script, self.posts, self.rid, self.polls = copy.deepcopy(scene), script, [], 0, 0
        self.outcome, self.state_calls, self.move_on_success = None, 0, move_on_success
        if camera:
            for o in self.scene["objects"].values():
                o["source"] = "camera"

    def get(self, url, timeout=None):
        self.state_calls += 1
        st = copy.deepcopy(self.scene)
        if self.rid and self.polls < 1:
            self.polls += 1
            st.update(busy=True, current_request_id=self.rid)
            return st
        st.update(busy=False, current_request_id=self.rid, last_task_success=None, last_task_error_mm=None,
                  last_task_refused=False, last_task_error=None)
        kind = self.outcome[0] if self.outcome else None
        if kind == "refuse":
            st.update(last_task_success=False, last_task_refused=True, last_task_error=self.outcome[1])
        elif kind == "miss":
            st.update(last_task_success=False, last_task_error_mm=self.outcome[1])
        elif self.rid:
            st.update(last_task_success=True, last_task_error_mm=3.0)
        return st

    def post(self, url, payload, timeout=None, **kw):
        self.rid += 1
        self.polls = 0
        self.posts.append(payload)
        self.outcome = self.script(payload["instruction"], payload.get("destination"), len(self.posts))
        if self.outcome is None and self.move_on_success:
            self._move(payload)
        return 200, {"status": "accepted", "request_id": self.rid}

    def _move(self, payload):
        text, dest = payload["instruction"], payload.get("destination") or {}
        key = next((k for k, o in self.scene["objects"].items() if any(w in text for w in o["keywords"])), None)
        if key and dest.get("dest_kind") == "shelf":
            tier = dest.get("shelf_tier_index")
            tier = 1 if tier is None else tier
            self.scene["objects"][key]["position"] = [0.25, TIER_Y[tier], TOPS[tier]]


def run_plan(instruction, script, scene=None, camera=False, move_on_success=True):
    sim = FakeSim(scene or SCENE, script, camera, move_on_success)
    orch._http_get_json, orch._http_post_json = sim.get, sim.post
    ag = orch.NeMoRoboticsAgent()
    ag._attach_visual_evidence = lambda *a, **k: None
    ag._attach_cosmos_assessment = lambda *a, **k: None
    out = ag.run(instruction)
    return out, sim


def ok_all(instr, dest, n):
    return None


def step_status(out):
    return [(r["subgoal"].get("target"), r["status"]) for r in out["steps"]]


FULL = ("the shelf is full: there is no slot where the Blue Flask can be set down without hitting something "
        "already there (last blocker: the hand would hit Red Flask)")
TOP_FULL = ("the top shelf is full: there is no slot where the Blue Flask can be set down without hitting something "
            "already there (last blocker: the vessel would hit Green Beaker)")
PICK_BLOCKED = "cannot pick up the Blue Flask: the arm would hit Red Flask"

# ── 4a. explicit tier ────────────────────────────────────────────────────────────────────────────────────────
reset_llm()
DECOMPOSE["put the blue flask on the middle shelf"] = A(sg("blue flask", "middle shelf", dest_kind="shelf", shelf_tier="middle"))
out, sim = run_plan("put the blue flask on the middle shelf", ok_all)
dp = (sim.posts[0].get("destination") if sim.posts else {}) or {}
probs = []
if out["status"] != "SUCCESS": probs.append(out["status"])
if sim.posts[0]["instruction"] != "pick up the blue flask and place it on the middle shelf": probs.append(sim.posts[0]["instruction"])
if (dp.get("dest_kind"), dp.get("shelf_tier"), dp.get("shelf_tier_index")) != ("shelf", "middle", 1): probs.append(str(dp))
if any(k in dp for k in ("shelf_slot", "shelf_slot_y")): probs.append(f"slot fields leaked into the payload: {dp}")
if ("answer", "quote") in LLM_LOG: probs.append("LLM answer used on an all-clear run")
check("run: explicit tier -> payload has tier only, no slot; all-clear answer needs no LLM", probs, out.get("answer", ""))

# ── 4b. auto tier ────────────────────────────────────────────────────────────────────────────────────────────
reset_llm()
DECOMPOSE["put the blue flask in the rack"] = A(sg("blue flask", "on the shelf", dest_kind="shelf", shelf_tier=None))
out, sim = run_plan("put the blue flask in the rack", ok_all)
dp = (sim.posts[0].get("destination") if sim.posts else {}) or {}
probs = []
if out["status"] != "SUCCESS": probs.append(out["status"])
if sim.posts[0]["instruction"] != "pick up the blue flask and place it on the shelf": probs.append(sim.posts[0]["instruction"])
if (dp.get("dest_kind"), dp.get("shelf_tier"), dp.get("shelf_tier_index")) != ("shelf", None, None): probs.append(str(dp))
check("run: auto tier -> shelf_tier null, text 'on the shelf'", probs)

# the LLM naming a tier the user didn't ("the shelf") is overridden by the text
reset_llm()
DECOMPOSE["put the red flask on the shelf please"] = A(sg("red flask", "the shelf", dest_kind="shelf", shelf_tier="top"))
sgs, _ = orch.decompose_task_with_reasoning("put the red flask on the shelf please", SCENE)
check("LLM-chosen tier is overridden when the user named none", [] if sgs[0]["shelf_tier"] is None else [str(sgs[0])])

# ── 4c. "place all objects in the rack" with the shelf filling up: one refusal in the middle ────────────────
def script_one_refused(instr, dest, n):
    return ("refuse", FULL) if "blue flask" in instr else None


for mode in ("quote", "paraphrase", "down"):
    reset_llm()
    ANSWER_MODE[0] = mode
    DECISION[0] = lambda prompt: {"decision": "replan_step", "reason": "same again",
                                  "subgoal": sg("blue flask", "on the shelf")}          # repeats the refused step
    out, sim = run_plan("place all objects in the rack", script_one_refused)
    ans = out["answer"]
    probs = []
    if len(sim.posts) != 3: probs.append(f"{len(sim.posts)} dispatches (expected 3: each object once, the refusal never retried)")
    if step_status(out) != [("red flask", "DONE"), ("blue flask", "FAILED"), ("red beaker", "DONE")]: probs.append(str(step_status(out)))
    if out["status"] != "PARTIAL": probs.append(out["status"])
    if FULL not in ans: probs.append("refusal text not quoted verbatim in the answer")
    for name in ("red flask", "blue flask", "red beaker"):
        if name not in ans.lower(): probs.append(f"answer never mentions the {name}")
    blue_step = next(r for r in out["steps"] if r["subgoal"]["target"] == "blue flask")
    if not blue_step.get("refused") or FULL not in blue_step["reason"]: probs.append("result lacks the refusal")
    if ("decision", "REFUSAL") in LLM_LOG: probs.append("Nemotron was asked for an alternative (policy: a refusal just fails the step)")
    if mode != "down" and ("answer", mode) not in LLM_LOG: probs.append("Nemotron wrote no final answer")
    if mode == "down" and ("answer", "down") not in LLM_LOG: probs.append("answer call not attempted")
    check(f"run: 'place all objects in the rack', one refused, remaining run; final answer [{mode}]", probs, ans)

# step detail carries the verbatim refusal (as shown in the UI plan)
seen = []
reset_llm(); DECISION[0] = lambda p: {"decision": "retry", "reason": "try again"}
sim = FakeSim(SCENE, script_one_refused)
orch._http_get_json, orch._http_post_json = sim.get, sim.post
ag = orch.NeMoRoboticsAgent(); ag._attach_visual_evidence = lambda *a, **k: None; ag._attach_cosmos_assessment = lambda *a, **k: None
out = ag.run("place all objects in the rack", on_plan_update=lambda plan: seen.append(plan))
last = seen[-1]
blue = next(s_ for s_ in last["steps"] if s_["subgoal"]["target"] == "blue flask")
check("run: step detail quotes the refusal verbatim; plan carries the final answer",
      [] if (FULL in (blue["detail"] or "") and blue["status"] == "FAILED" and last.get("answer")) else [f"{blue['status']} {blue['detail']!r}"])

# ── 4d. a full shelf: every object refused ───────────────────────────────────────────────────────────────────
reset_llm()
DECISION[0] = lambda p: {"decision": "retry", "reason": "x"}
out, sim = run_plan("place all objects in the rack", lambda i, d, n: ("refuse", FULL.replace("Blue Flask", "object")))
probs = []
if len(sim.posts) != 3: probs.append(f"{len(sim.posts)} dispatches")
if out["status"] != "FAILED": probs.append(out["status"])
if len(out["summary"]["not_placed"]) != 3 or out["summary"]["placed"]: probs.append(str(out["summary"]))
check("run: full shelf, all three refused -> FAILED, no retries, all listed with the reason", probs, out["answer"])

# ── 4e. policy: a refusal ends the step - no alternative destination is ever substituted ─────────────
reset_llm()
DECOMPOSE["put the blue flask on the top shelf"] = A(sg("blue flask", "top shelf", dest_kind="shelf", shelf_tier="top"))
DECISION[0] = lambda p: {"decision": "replan_step", "reason": "top is full, try the middle",
                         "subgoal": sg("blue flask", "middle shelf", dest_kind="shelf", shelf_tier="middle")}
out, sim = run_plan("put the blue flask on the top shelf", lambda i, d, n: ("refuse", TOP_FULL))
probs = []
if len(sim.posts) != 1: probs.append(f"{len(sim.posts)} dispatches (expected exactly 1: no alternative is tried)")
if out["status"] != "FAILED": probs.append(out["status"])
if TOP_FULL not in out["answer"]: probs.append("the refusal is not quoted in the answer")
if ("decision", "REFUSAL") in LLM_LOG: probs.append("the model was asked for an alternative")
check("run: refusal -> the step FAILS with the robot's reason; no alternative destination is tried", probs, out["answer"])

# the model offers the identical step again, or 'retry': both mean leave it refused
for label, reply in (("identical step", lambda p: {"decision": "replan_step", "reason": "again", "subgoal": sg("blue flask", "top shelf")}),
                     ("'retry'", lambda p: {"decision": "retry", "reason": "again"}),
                     ("no reply", lambda p: None)):
    reset_llm()
    DECISION[0] = reply
    out, sim = run_plan("put the blue flask on the top shelf", lambda i, d, n: ("refuse", TOP_FULL))
    check(f"run: refusal then {label} -> never re-dispatched (1 dispatch), step FAILED",
          [] if (len(sim.posts) == 1 and out["status"] == "FAILED") else [f"{len(sim.posts)} dispatches, {out['status']}"])

# ── 4f. blocked pick: stop and say why (the blocker is NOT moved on the model's initiative) ─────────────────
reset_llm()
DECISION[0] = lambda p: {"decision": "replan_remaining", "reason": "move the blocker first",
                         "subgoals": [sg("red flask", "bottom shelf", dest_kind="shelf", shelf_tier="bottom"),
                                      sg("blue flask", "top shelf", dest_kind="shelf", shelf_tier="top")]}
out, sim = run_plan("put the blue flask on the top shelf", lambda i, d, n: ("refuse", PICK_BLOCKED))
probs = []
if len(sim.posts) != 1: probs.append(f"{len(sim.posts)} dispatches (expected 1: nothing else is moved)")
if out["status"] != "FAILED": probs.append(out["status"])
if PICK_BLOCKED not in out["answer"]: probs.append("pick refusal not quoted in the answer")
check("run: blocked pick -> FAILED with the reason, no other object is moved", probs, out["answer"])

# ── 4g. a step that depends on a refused object is skipped, independent steps still run ──────────────────────
reset_llm()
DECISION[0] = lambda p: {"decision": "retry", "reason": "x"}
DECOMPOSE["put the blue flask on the top shelf then the red beaker next to the blue flask then the red flask on the shelf"] = A(
    sg("blue flask", "top shelf"), sg("red beaker", "next to the blue flask"), sg("red flask", "on the shelf"))
out, sim = run_plan("put the blue flask on the top shelf then the red beaker next to the blue flask then the red flask on the shelf",
                    lambda i, d, n: ("refuse", TOP_FULL) if "blue flask and place it on" in i else None)
probs = []
if [p["instruction"] for p in sim.posts] != ["pick up the blue flask and place it on the top shelf",
                                              "pick up the red flask and place it on the shelf"]: probs.append(str([p["instruction"] for p in sim.posts]))
if step_status(out) != [("blue flask", "FAILED"), ("red beaker", "ABORTED"), ("red flask", "DONE")]: probs.append(str(step_status(out)))
check("run: dependent step skipped after a refusal, the independent one still runs", probs, out["answer"])

# ── 4h. non-refusal failures keep the old behaviour (retry, abort) ───────────────────────────────────────────
reset_llm()
DECISION[0] = lambda p: {"decision": "retry", "reason": "slipped"}
DECOMPOSE["put the blue flask on the top shelf (miss)"] = A(sg("blue flask", "top shelf"))
out, sim = run_plan("put the blue flask on the top shelf (miss)", lambda i, d, n: ("miss", 41.0) if n == 1 else None)
check("run: a position MISS is still retried (2 dispatches) and succeeds",
      [] if (len(sim.posts) == 2 and out["status"] == "SUCCESS" and ("decision", "miss") in LLM_LOG) else [f"{len(sim.posts)} {out['status']} {LLM_LOG}"])
reset_llm()
DECISION[0] = lambda p: {"decision": "retry", "reason": "x"}
out, sim = run_plan("put the blue flask on the top shelf (miss)", lambda i, d, n: ("miss", 41.0))
check("run: three misses abort the plan (existing rule: physical failures still stop everything)",
      [] if (len(sim.posts) == 3 and out["status"] == "FAILED") else [f"{len(sim.posts)} {out['status']}"])

# ── 4i. regression: the pre-dispatch drift branch used an undefined `refusal` (UnboundLocalError) ──────────
reset_llm()
DECISION[0] = lambda p: {"decision": "retry", "reason": "proceed"}
DECOMPOSE["put the red flask on the shelf then the blue flask next to the red flask on the shelf"] = A(
    sg("red flask", "on the shelf"), sg("blue flask", "next to the red flask on the shelf"))
try:
    out, sim = run_plan("put the red flask on the shelf then the blue flask next to the red flask on the shelf", ok_all, move_on_success=False)
    problem = []
    if ("decision", "miss") not in LLM_LOG: problem.append(f"drift branch not exercised: {LLM_LOG}")
except UnboundLocalError as e:
    problem = [f"UnboundLocalError: {e}"]
check("run: pre-dispatch drift branch no longer raises UnboundLocalError", problem)

# ═══ 5. nonsense and impossible input ═════════════════════════════════════════════════════════════════════════
reset_llm()
for phrase, raw in (("make me a sandwich", "<think>not a robot task</think> []"),):
    DECOMPOSE[phrase] = raw
    sgs, _ = orch.decompose_task_with_reasoning(phrase, SCENE)
    check(f"nonsense {phrase!r} -> empty plan", [] if sgs == [] else [str(sgs)])
out, sim = run_plan("make me a sandwich", ok_all)
check("nonsense run -> REJECTED, nothing dispatched", [] if (out["status"] == "REJECTED" and not sim.posts) else [str(out)])

DECOMPOSE["put it on the fourth shelf"] = A(sg("blue flask", "fourth shelf", dest_kind="shelf", shelf_tier="fourth"))
out, sim = run_plan("put it on the fourth shelf", ok_all)
check("impossible tier -> nothing dispatched, reason says which tiers exist",
      [] if (not sim.posts and "no 'fourth' tier" in json.dumps(out)) else [json.dumps(out)[:200]])

DECOMPOSE["put the red flask on the moon"] = A(sg("red flask", "the moon", dest_kind="table"))
out, sim = run_plan("put the red flask on the moon", ok_all)
check("unparseable destination -> nothing dispatched", [] if (not sim.posts and "not understood" in json.dumps(out)) else [json.dumps(out)[:200]])

out, sim = run_plan("tidy up", ok_all, scene=empty_table)
check("'tidy up' with nothing on the table -> REJECTED with a plain-language answer",
      [] if (out["status"] == "REJECTED" and "nothing to place" in out["answer"].lower() and not sim.posts) else [str(out)])

orch._nemotron_chat = lambda payload: (500, None)
sgs, why = orch.decompose_task_with_reasoning("an instruction nobody planned for", SCENE)
check("planner down -> single 'direct' fallback", [] if sgs and sgs[0]["action"] == "direct" else [str(sgs)], why)
orch._nemotron_chat = stub_chat

# ═══ 6. retry rule (perception-type failures only) ═══════════════════════════════════════════════════════════
reset_llm()
DECOMPOSE["put the blue flask on the fifth shelf"] = A(sg("blue flask", "fifth shelf"))
DECOMPOSE["put the yellow flask on the top shelf"] = A(sg("yellow flask", "top shelf"))
_, s_rule = run_plan("put the blue flask on the fifth shelf", ok_all, camera=True)
_, s_perc = run_plan("put the yellow flask on the top shelf", ok_all, camera=True)
check("retry rule: a rule rejection (no such tier) is not re-observed; 'not found' is",
      [] if s_perc.state_calls > s_rule.state_calls + 2 else [f"rule={s_rule.state_calls} perception={s_perc.state_calls}"],
      f"state polls: rule={s_rule.state_calls}, perception={s_perc.state_calls}")

# ═══ report ═════════════════════════════════════════════════════════════════════════════════════════════════
failed = 0
for name, problems, note in RESULTS:
    failed += bool(problems)
    print(f"[{'PASS' if not problems else 'FAIL'}] {name}" + (f"\n        -> {note}" if note else ""))
    for p in problems:
        print(f"        !! {p}")
print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} passed")
sys.exit(1 if failed else 0)
