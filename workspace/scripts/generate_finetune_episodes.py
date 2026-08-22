#!/usr/bin/env python3
"""
Drives isaac_sim_service.py purely over its existing HTTP API (/randomize, /execute, /state)
to generate GR00T fine-tuning demonstration episodes using the already-reliable
PickPlaceController path. Does not import or touch any Physyk module in-process — fully
decoupled from the orchestrator, main server, and sim service internals, same idiom as
physyk_agent_orchestrator.py's own _dispatch/_wait_for_idle polling but reimplemented here
standalone so this script has no dependency on the orchestrator's lock.

Prerequisite: isaac_sim_service.py must be started with RECORD_EPISODES=1 in its environment
(see groot_episode_recorder.py) — this script only triggers tasks, it doesn't write any
episode data itself; the sim service does that via the recorder hooks.

Usage:
    RECORD_EPISODES=1 <start isaac_sim_service.py as usual>
    python3 scripts/generate_finetune_episodes.py --target 40 --max-attempts 60
"""

import argparse
import itertools
import json
import time
import urllib.request

ISAAC_SIM_URL = "http://localhost:8100"
COLORS = ["red", "blue", "green"]
DESTINATIONS = ["in the tray"]  # simple pick-and-place only, per the 30-50 episode scope


def post(path: str, body: dict | None = None, timeout: float = 5.0) -> dict:
    data = json.dumps(body or {}).encode("utf-8")
    req = urllib.request.Request(f"{ISAAC_SIM_URL}{path}", data=data,
                                  headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_state(timeout: float = 5.0) -> dict:
    req = urllib.request.Request(f"{ISAAC_SIM_URL}/state", method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_for_dequeue(request_id, poll_interval: float = 0.2, timeout: float = 10.0) -> bool:
    """/execute enqueues a command but does NOT set busy=True until the sim loop's next
    iteration actually dequeues it (see isaac_sim_service.py's execute_command /
    command_queue.put) — polling for busy=False immediately after /execute can catch a STALE
    busy=False/last_task_success from the previous task and wrongly conclude the new one is
    already done. Mirrors physyk_agent_orchestrator.py's own request_id-confirmation dispatch
    pattern: block until /state.current_request_id echoes back our request_id."""
    start = time.time()
    while time.time() - start < timeout:
        state = get_state()
        if state.get("current_request_id") == request_id:
            return True
        time.sleep(poll_interval)
    return False


def wait_for_idle(poll_interval: float = 0.5, timeout: float = 60.0) -> dict:
    """Poll /state until busy=False, treating a timeout as a failed attempt rather than a
    silent success. Only meaningful to call AFTER wait_for_dequeue has confirmed the sim loop
    actually started processing our specific request."""
    start = time.time()
    last_state = {}
    while time.time() - start < timeout:
        last_state = get_state()
        if not last_state.get("busy", False):
            return last_state
        time.sleep(poll_interval)
    print("[generate] WARNING: wait_for_idle timed out — treating as a failed attempt.")
    return last_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=40, help="number of SAVED (successful) episodes to reach")
    ap.add_argument("--max-attempts", type=int, default=60, help="hard cap on total pick-place attempts")
    ap.add_argument("--checkpoint-every", type=int, default=5)
    ap.add_argument("--colors", type=str, default=None,
                     help="comma-separated color sequence to cycle instead of the default "
                          "red,blue,green — e.g. 'blue,green,green' to fill exact gaps toward "
                          "an even per-color count.")
    args = ap.parse_args()

    saved = 0
    attempts = 0
    color_list = args.colors.split(",") if args.colors else COLORS
    color_cycle = itertools.cycle(color_list)
    color_counts = {c: 0 for c in COLORS}

    print(f"[generate] Target: {args.target} saved episodes, budget: {args.max_attempts} attempts.")
    print("[generate] Make sure isaac_sim_service.py is running with RECORD_EPISODES=1.")

    while saved < args.target and attempts < args.max_attempts:
        attempts += 1
        color = next(color_cycle)
        instruction = f"pick up the {color} cube and place it in the tray"

        try:
            post("/randomize")
            wait_for_idle(timeout=15.0)  # randomize is near-instant, but stay consistent
            resp = post("/execute", {"instruction": instruction})
            if resp.get("status") == "error":
                print(f"[generate] Attempt {attempts}: /execute rejected ({resp.get('message')}) — skipping.")
                continue
            req_id = resp.get("request_id")
            if not wait_for_dequeue(req_id):
                print(f"[generate] Attempt {attempts}: sim never dequeued request {req_id} within timeout — skipping.")
                continue
            final_state = wait_for_idle(timeout=150.0)  # matches physyk_agent_orchestrator.py's
                                                          # own _wait_for_idle timeout — a full
                                                          # pick-place at the sim's actual
                                                          # observed render rate (~7fps) takes
                                                          # ~100-110s wall-clock, not the ~10-20s
                                                          # originally estimated; 45s was far too
                                                          # short and caused spurious early
                                                          # "failed" timeouts + 409 conflicts on
                                                          # the next attempt while the real task
                                                          # was still quietly running.
        except Exception as e:
            print(f"[generate] Attempt {attempts} errored: {e} — skipping.")
            continue

        success = bool(final_state.get("last_task_success"))
        color_counts[color] += success
        if success:
            saved += 1
        print(f"[generate] Attempt {attempts}: '{instruction}' -> "
              f"{'SAVED' if success else 'dropped (failed verification)'} "
              f"(saved so far: {saved}/{args.target})")

        if saved and saved % args.checkpoint_every == 0 and success:
            print(f"[generate] --- checkpoint: {saved} episodes saved, "
                  f"per-color counts so far: {color_counts} ---")

    print(f"[generate] Done. {saved} episodes saved over {attempts} attempts "
          f"({100 * saved / max(attempts, 1):.0f}% success rate). Per-color: {color_counts}")
    if saved < args.target:
        print(f"[generate] NOTE: reached the {args.max_attempts}-attempt budget before hitting "
              f"the {args.target}-episode target — either raise --max-attempts or proceed with "
              f"the {saved} episodes collected so far.")


if __name__ == "__main__":
    main()
