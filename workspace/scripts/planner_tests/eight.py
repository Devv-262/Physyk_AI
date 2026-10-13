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
def plan_one(kind, xy, oth, slot_xy, top):
    """Same logic as the sim: try each hand roll; the roll is fixed for pick + carry + place."""
    for roll in sg.ROLLS:
        P.set_roll(roll)
        try:
            segs,q,_=P.plan_pick(kind,xy,0.0,sg.HOME_Q,oth)
            P.plan_place(kind,slot_xy,top,q,oth)
            return "roll%d"%round(np.degrees(roll))
        except sg.PlanError as e:
            last=str(e)[:16]
    return "X:"+last

for name,(kind,xy) in V.items():
    cols=[(k,np.array([v[1][0],v[1][1],0.0])) for k,v in V.items() if k!=name]
    oth=[np.array(v[1]) for k,v in V.items() if k!=name]
    P.set_vessels(cols)
    res=[]
    for ti,top in enumerate((0.08,0.36,0.64)):
        for x in (0.25,0.05,0.45):
            t=time.time()
            r=plan_one(kind,xy,oth,(x,P.slot_y(ti)),top)
            res.append("%s %.0fs"%(r,time.time()-t))
    print("RES",name,"[bot|mid|top x .25 .05 .45]:"," | ".join(res),flush=True)
app.close()
