import sys, time, numpy as np
sys.path.insert(0, "/workspace")
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from isaacsim.robot_motion.motion_generation import interface_config_loader as L
from isaacsim.robot_motion.motion_generation.lula import LulaKinematicsSolver
import side_grasp as sg
cfg = L.load_supported_motion_policy_config("Franka", "RMPflow")
ks = LulaKinematicsSolver(robot_description_path=cfg["robot_description_path"], urdf_path=cfg["urdf_path"])
P = sg.SideGraspPlanner(ks)
V = {"flask_blue": ("flask", (0.485,-0.353)), "flask_red": ("flask", (0.571,-0.185)), "flask_empty": ("flask", (0.600,0.0)),
     "beaker_red": ("beaker", (0.571,0.185)), "beaker_green": ("beaker", (0.485,0.353))}
tops=(0.08,0.36,0.64); xs=(0.25,0.05,0.45)
def obstacles(excl, extra=()):
    return [(k,np.array([v[1][0],v[1][1],0.0])) for k,v in V.items() if k!=excl]+list(extra)
summary={}
for name,(kind,xy) in list(V.items())[1:4]:
    P.set_vessels(obstacles(name)); others=[np.array(v[1]) for k,v in V.items() if k!=name]
    t=time.time()
    try: segs,q,_=P.plan_pick(kind,xy,0.0,sg.HOME_Q,others)
    except sg.PlanError as e: print("RES",name,"PICK REFUSED:",e,flush=True); continue
    row=[]; why=[]
    for ti,top in enumerate(tops):
        for x in xs:
            try: P.plan_place(kind,(x,P.slot_y(ti)),top,q,others); row.append("ok")
            except sg.PlanError as e: row.append("X"); why.append(f"{('bot','mid','top')[ti]} x{x}: {str(e)[:70]}")
    for w in why[:6]: print("RES   fail",name,w,flush=True)
    print("RES",name,"pick ok; slots [bot x.25,.05,.45 | mid ... | top ...]:"," ".join(row),f"({time.time()-t:.0f}s)",flush=True)
# occupancy: put flask_empty on middle slot x=.25 then try beaker_red there (must be refused) and at x=.05 (free)
kind,xy=V["flask_empty"]; P.set_vessels(obstacles("flask_empty"))
segs,q,_=P.plan_pick(kind,xy,0.0,sg.HOME_Q,[np.array(v[1]) for k,v in V.items() if k!="flask_empty"])
pl,qo=P.plan_place(kind,(0.25,P.slot_y(1)),0.36,q,[])
print("RES placed flask_empty at middle x=.25",flush=True)
occ=("Empty Flask",np.array([0.25,P.slot_y(1),0.36]))
kind,xy=V["beaker_red"]; P.set_vessels([(k,np.array([v[1][0],v[1][1],0.0])) for k,v in V.items() if k not in("beaker_red","flask_empty")]+[occ])
segs,q,_=P.plan_pick(kind,xy,0.0,sg.HOME_Q,[np.array(v[1]) for k,v in V.items() if k not in("beaker_red","flask_empty")])
for x in (0.25,0.05,0.45):
    try: P.plan_place(kind,(x,P.slot_y(1)),0.36,q,[]); print("RES beaker_red -> middle x=%.2f: ACCEPTED"%x,flush=True)
    except sg.PlanError as e: print("RES beaker_red -> middle x=%.2f: REFUSED (%s)"%(x,str(e)[:80]),flush=True)
app.close()
