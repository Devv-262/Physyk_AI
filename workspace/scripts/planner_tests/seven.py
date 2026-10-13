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
table = [("blue",(0.485,-0.353)),("red",(0.571,-0.185))]
P.set_vessels([(n,np.array([x,y,0.0])) for n,(x,y) in table]); oth=[np.array(xy) for _,xy in table]
for name,kind,xy in (("flask_empty","flask",(0.600,0.0)),("beaker_red","beaker",(0.571,0.185))):
    segs,q,_=P.plan_pick(kind,xy,0.0,sg.HOME_Q,oth)
    res=[]
    for ti,top in enumerate((0.08,0.36,0.64)):
        for x in (0.25,0.05,0.45):
            t=time.time()
            try: P.plan_place(kind,(x,P.slot_y(ti)),top,q,oth); res.append("ok%.0fs"%(time.time()-t))
            except sg.PlanError as e: res.append("X%.0fs"%(time.time()-t))
    print("RES",name,"[bot|mid|top x .25 .05 .45]:"," ".join(res),flush=True)
app.close()
