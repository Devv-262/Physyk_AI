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
def cols(excl): return [(k,np.array([v[1][0],v[1][1],0.0])) for k,v in V.items() if k not in excl]
def others(excl): return [np.array(v[1]) for k,v in V.items() if k not in excl]
tops=(0.08,0.36,0.64)
# timing + coverage for flask_red (the hardest earlier) and flask_empty
for name in ("flask_red","flask_empty"):
    kind,xy=V[name]; P.set_vessels(cols([name]))
    segs,q,_=P.plan_pick(kind,xy,0.0,sg.HOME_Q,others([name]))
    res=[]
    for ti,top in enumerate(tops):
        for x in (0.25,0.05,0.45):
            t=time.time()
            try: P.plan_place(kind,(x,P.slot_y(ti)),top,q,others([name])); res.append("ok%.0fs"%(time.time()-t))
            except sg.PlanError as e: res.append("X%.0fs"%(time.time()-t))
    print("RES",name,"[bot|mid|top x .25 .05 .45]:"," ".join(res),flush=True)
# occupancy: flask_empty on middle .25, then beaker_red must be refused at .25 and accepted at .05
kind,xy=V["flask_empty"]; P.set_vessels(cols(["flask_empty"]))
segs,q,_=P.plan_pick(kind,xy,0.0,sg.HOME_Q,others(["flask_empty"]))
P.plan_place(kind,(0.25,P.slot_y(1)),0.36,q,others(["flask_empty"]))
occ=("Empty Flask",np.array([0.25,P.slot_y(1),0.36]))
kind,xy=V["beaker_red"]; P.set_vessels(cols(["beaker_red","flask_empty"])+[occ])
segs,q,_=P.plan_pick(kind,xy,0.0,sg.HOME_Q,others(["beaker_red","flask_empty"]))
for x in (0.25,0.05):
    try: P.plan_place(kind,(x,P.slot_y(1)),0.36,q,[]); print("RES beaker_red -> middle x=%.2f: ACCEPTED"%x,flush=True)
    except sg.PlanError as e: print("RES beaker_red -> middle x=%.2f: REFUSED (%s)"%(x,str(e)[:90]),flush=True)
app.close()
