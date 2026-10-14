"""End-to-end batch on the RUNNING sim (port 8100): for each layout it resets, randomizes the cubes, plans a stack with the agent's own
planner (tray on even layouts, table on odd ones) and sends every pick through /execute, recording the result and the rows of
workspace/joint_trace.csv each pick produced. Start the sim with PHYSYK_RANDOM_SEED=<n> to repeat the same layouts across modes.
  usage: .venv/bin/python workspace/scripts/arm_tests/e2e_batch.py <label> <n_layouts> <out.json>
Then:  .venv/bin/python workspace/scripts/arm_tests/analyze_e2e.py <out.json> [more.json ...]"""
import json, time, urllib.request, sys, io, contextlib, os, csv
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workspace"))
with contextlib.redirect_stdout(io.StringIO()):
    import physyk_agent_orchestrator as o
B="http://localhost:8100"; TRACE=str(ROOT / "workspace" / "joint_trace.csv")
mode=sys.argv[1]; N=int(sys.argv[2]); out=sys.argv[3]
def post(i):
    r=urllib.request.Request(B+"/execute",data=json.dumps({"instruction":i}).encode(),headers={"Content-Type":"application/json"},method="POST"); return json.loads(urllib.request.urlopen(r,timeout=5).read())
def state(): return json.loads(urllib.request.urlopen(B+"/state",timeout=5).read())
def nlines(): 
    with open(TRACE) as f: return sum(1 for _ in f)
def run(ins):
    n0=nlines(); t0=time.time(); rid=post(ins)["request_id"]; time.sleep(1.5)
    while True:
        s=state()
        if not s.get("busy") and s.get("current_request_id")==rid: break
        if time.time()-t0>240: print("TIMEOUT",flush=True); break
        time.sleep(0.5)
    time.sleep(0.6); s=state(); return n0, nlines(), dict(wall=round(time.time()-t0,1), ok=s.get("last_task_success"), err=s.get("last_task_error_mm"), grasp=s.get("last_grasp_confirmed"), refused=s.get("last_task_refused"), msg=s.get("last_task_error"))
res=[]
for L in range(N):
    run("reset the scene"); run("randomize the cubes"); time.sleep(1)
    sc=state(); cubes={k:[round(x,3) for x in v["position"]] for k,v in sc["objects"].items() if "cube" in k}
    surface="tray" if L%2==0 else "table"
    goals=o._stack_plan(sc, surface)
    print(f"[{mode}] layout {L+1}/{N} surface={surface} cubes={cubes} goals={[(g['target'],g['destination']) for g in goals]}",flush=True)
    for g in goals:
        ins=f"pick up the {g['target']} and place it {g['destination']}"
        a,b,r=run(ins); r.update(layout=L+1, surface=surface, instruction=ins, rows=[a,b], mode=mode); res.append(r)
        print("   ",ins,"->",r,flush=True)
json.dump(res,open(out,"w"))
print("DONE",len(res),"picks",flush=True)
