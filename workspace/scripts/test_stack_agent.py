"""End-to-end test of the agent's stacking loop against a FAKE simulator (no Isaac Sim, no arm, no Nemotron).

The fake sim serves /state and /execute on a local port and applies each instruction with the same placement rules the
real sim uses (tray slot / onto the pile top / free spot). It can drop a cube instead of placing it (random step failures)
and refuse unreachable cubes. The real NeMoRoboticsAgent.run() loop drives it, so this exercises planning, closed-loop
re-planning, blocker handling, failure recovery and the final verdict together.

Run from the repo root:   python3 workspace/scripts/test_stack_agent.py
"""
import json
import math
import os
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 8199
os.environ["ISAAC_SIM_URL"] = f"http://127.0.0.1:{PORT}"
os.environ.setdefault("PHYSYK_STACK_PLANNER", "code")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import stack_planner as sp                       # noqa: E402
import test_stack_planner as T                   # noqa: E402
import physyk_agent_orchestrator as orch         # noqa: E402

orch.POLL_INTERVAL_S = 0.02
orch.GUARDRAIL_RETRY_DELAY_S = 0.01
NAME2KEY = {"red": "red_cube", "green": "green_cube", "blue": "blue_cube", "yellow": "yellow_cube"}


class FakeSim:
    def __init__(self, layout, fail_p=0.0, seed=0):
        self.world = {c.key: c for c in T.mk(layout)}
        self.fail_p = fail_p
        self.rng = random.Random(seed)
        self.lock = threading.Lock()
        self.busy = False
        self.req = 0
        self.last = {"last_task_success": None, "last_task_error_mm": None, "last_task_error": None, "last_task_refused": False,
                     "last_object_displacement_mm": None, "last_grasp_confirmed": True}
        self.executed = []

    def state(self):
        with self.lock:
            objs = {k: {"name": c.name.title(), "position": [c.x, c.y, c.z]} for k, c in self.world.items()}
            objs["target_tray"] = {"name": "Target Tray", "position": [0.42, -0.32, 0.015]}
            return {"busy": self.busy, "stage": "READY", "current_request_id": self.req, "objects": objs, **self.last}

    def execute(self, text):
        with self.lock:
            self.req += 1
            rid = self.req
            self.busy = True
            self.last = {**self.last, "last_task_refused": False, "last_task_error": None}
        threading.Thread(target=self._do, args=(rid, text), daemon=True).start()
        return rid

    def _do(self, rid, text):
        time.sleep(0.05)
        t = text.lower()
        with self.lock:
            try:
                tgt = NAME2KEY[t.split("pick up the ")[1].split(" cube")[0]]
                c = self.world[tgt]
                dest = t.split(" place it ")[1] if " place it " in t else ""
                if "free spot" in dest:
                    step = sp.Step(tgt, c.name, "free")
                elif "tray" in dest:
                    step = sp.Step(tgt, c.name, "tray")
                else:
                    rk = NAME2KEY[dest.split("on top of the ")[1].split(" cube")[0]]
                    step = sp.Step(tgt, c.name, "on", rk, self.world[rk].name)
                if math.hypot(c.x, c.y) < 0.36:
                    self.last = {**self.last, "last_task_success": False, "last_task_refused": True,
                                 "last_task_error": f"target is too close to the robot base (radius {math.hypot(c.x, c.y):.2f} m, minimum 0.36 m)",
                                 "last_task_error_mm": None}
                elif sp._covered(c, list(self.world.values())):
                    self.last = {**self.last, "last_task_success": False, "last_task_refused": True,
                                 "last_task_error": f"the {c.name} has another cube on top of it", "last_task_error_mm": None}
                elif self.rng.random() < self.fail_p:        # the cube is dropped somewhere on the table
                    for _ in range(40):
                        x, y = self.rng.uniform(0.30, 0.64), self.rng.uniform(-0.12, 0.40)
                        if sp.reachable(x, y) and not sp.in_tray(x, y) and all(
                                math.hypot(o.x - x, o.y - y) > 0.07 for o in self.world.values() if o is not c):
                            c.x, c.y, c.z = x, y, 0.025
                            break
                    self.last = {**self.last, "last_task_success": False, "last_task_error_mm": 120.0}
                else:
                    err = T.apply_step(self.world, step)
                    if err:
                        self.last = {**self.last, "last_task_success": False, "last_task_refused": True, "last_task_error": err,
                                     "last_task_error_mm": None}
                    else:
                        self.executed.append(text)
                        self.last = {**self.last, "last_task_success": True, "last_task_error_mm": 3.0,
                                     "last_object_displacement_mm": 100.0}
            finally:
                self.busy = False


class Handler(BaseHTTPRequestHandler):
    sim = None

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/state"):
            self._send(200, self.sim.state())
        else:
            self._send(404, b"")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path.startswith("/execute"):
            rid = self.sim.execute(body.get("instruction", ""))
            self._send(200, {"status": "accepted", "request_id": rid})
        else:
            self._send(404, b"")


def run_case(label, layout, instruction, expect_success=True, fail_p=0.0, seed=0, surface=None, expect_status=None, expect_moves=None):
    sim = FakeSim(layout, fail_p, seed)
    Handler.sim = sim
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        # the Cosmos checks are evidence only; keep this test offline and fast
        orch.NeMoRoboticsAgent._attach_visual_evidence = lambda self, *a, **k: None
        orch.NeMoRoboticsAgent._attach_cosmos_assessment = lambda self, *a, **k: None
        agent = orch.NeMoRoboticsAgent()
        res = agent.run(instruction)
        stacked = T.one_pile(sim.world, surface)
        if expect_status:                                   # e.g. REJECTED for "already stacked" / "out of reach"
            ok = res["status"] == expect_status and (expect_moves is None or len(sim.executed) == expect_moves)
        else:
            ok = (res["status"] == "SUCCESS") == expect_success and (stacked == expect_success)
        print(f"  {'ok  ' if ok else 'FAIL'} {label:<58} status={res['status']:<14} one_pile={stacked} moves={len(sim.executed)}")
        if not ok:
            print("       answer:", res.get("answer"))
            print("       world:", [(c.name, round(c.x, 2), round(c.y, 2), round(c.z, 3)) for c in sim.world.values()])
        return ok
    finally:
        srv.shutdown()
        srv.server_close()


def main():
    R, G, B = "red_cube", "green_cube", "blue_cube"
    TX, TY = sp.TRAY_CENTER
    cases = [
        ("separate cubes: stack the cubes", T.SCEN["all separate on the table"], "stack the cubes", True, 0.0, None),
        ("separate cubes: stack the cubes in the tray", T.SCEN["all separate on the table"], "stack the cubes in the tray", True, 0.0, "tray"),
        ("pile in tray -> table", T.SCEN["pile of 3 in the tray, wants the table"], "stack the cubes on the table", True, 0.0, "table"),
        ("pile on table -> tray", T.SCEN["pile of 3 on the table, wants the tray"], "stack the cubes in the tray", True, 0.0, "tray"),
        ("already stacked", T.SCEN["already one pile on the table"], "stack the cubes", True, 0.0, None, "REJECTED", 0),
        ("2 in tray + 1 on table -> table", T.SCEN["pile of 2 in tray + 1 on table"], "stack the cubes on the table", True, 0.0, "table"),
        ("2 stacked + 1 apart, covered cube", T.SCEN["one cube covered by another, third apart"], "stack the cubes", True, 0.0, None),
        ("nearby cubes", T.SCEN["three in a row along Y"], "stack the cubes", True, 0.0, None),
        ("cube out of reach: refused, nothing moves", T.SCEN["pile near the robot base (unreachable)"], "stack the cubes", True, 0.0, None, "REJECTED", 0),
        ("phrase: build a tower", T.SCEN["all separate on the table"], "build a tower with the cubes", True, 0.0, None),
        ("phrase: stack them in the tray", T.SCEN["all separate on the table"], "stack them in the tray", True, 0.0, "tray"),
    ]
    ok = True
    print("== agent loop vs fake sim (no failures) ==")
    for case in cases:
        label, layout, instr, exp, fp, surf = case[:6]
        extra = case[6:]
        ok &= run_case(label, layout, instr, exp, fp, 0, surf, *extra)
    print("\n== random step failures (30% of picks drop the cube) ==")
    bad = 0
    for seed in range(12):
        for instr, surf in (("stack the cubes", None), ("stack the cubes in the tray", "tray"), ("stack the cubes on the table", "table")):
            good = run_case(f"seed {seed}: {instr}", T.SCEN["all separate on the table"], instr, True, 0.3, seed, surf)
            bad += 0 if good else 1
    print(f"\nRESULT: {'ALL PASSED' if ok and not bad else 'FAILURES'} (no-failure cases ok={ok}, failure-injection runs failed={bad}/36)")
    return 0 if ok and not bad else 1


if __name__ == "__main__":
    sys.exit(main())
