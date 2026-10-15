"""Step-at-a-time Nemotron planning against the FAKE simulator (no Isaac Sim, no arm). Unlike test_stack_agent.py this CALLS the
hosted Nemotron API (needs NVIDIA_API_KEY / NEMOTRON_API_KEY in .env), because the thing under test is the model's moves.

Run:   .venv/bin/python workspace/scripts/test_stepwise_agent.py
"""
import os
import sys
import threading
import time
from http.server import ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_stack_agent as A                    # noqa: E402  (sets ISAAC_SIM_URL, imports the orchestrator)
import test_stack_planner as T                  # noqa: E402
import physyk_agent_orchestrator as orch        # noqa: E402


def run(label, instruction, check, status="SUCCESS"):
    sim = A.FakeSim(T.SCEN["all separate on the table"], 0.0, 0)
    A.Handler.sim = sim
    srv = ThreadingHTTPServer(("127.0.0.1", A.PORT), A.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        orch.NeMoRoboticsAgent._attach_visual_evidence = lambda self, *a, **k: None
        orch.NeMoRoboticsAgent._attach_cosmos_assessment = lambda self, *a, **k: None
        agent = orch.NeMoRoboticsAgent()
        t0 = time.time()
        res = agent.run(instruction)
        dt = time.time() - t0
        ok = res["status"] == status and check(sim.world)
        print(f"  {'ok  ' if ok else 'FAIL'} {label:<28} status={res['status']:<9} moves={len(sim.executed):<2} wall={dt:4.1f}s (fake arm 0.05 s/move)")
        for m in sim.executed:
            print("        ", m)
        if not ok:
            print("      answer:", res.get("answer"))
        return ok
    finally:
        srv.shutdown()
        srv.server_close()


def main():
    on = lambda w, a, b: abs(w[a].x - w[b].x) < 0.03 and abs(w[a].y - w[b].y) < 0.03 and w[a].z > w[b].z + 0.03
    tray = lambda w, k: A.sp.in_tray(w[k].x, w[k].y)
    cases = [
        ("one move", "put the red cube in the tray", lambda w: tray(w, "red_cube")),
        ("two clauses", "put the green cube on the blue cube, then the red cube on the green cube",
         lambda w: on(w, "green_cube", "blue_cube") and on(w, "red_cube", "green_cube")),
        ("hard: park and re-use", "pick red over blue, blue on tray, green on blue and then blue on table",
         lambda w: not tray(w, "blue_cube") and w["blue_cube"].z < 0.03),
    ]
    good = sum(run(label, instr, check) for label, instr, check in cases)
    # a cube that is not on the table is refused up front: nothing moves, nothing is substituted
    good += run("missing object", "put the yellow cube in the tray", lambda w: True, "REJECTED")
    print(f"\nRESULT: {good}/4 cases passed")
    return 0 if good == 4 else 1


if __name__ == "__main__":
    sys.exit(main())
