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
P.set_vessels([(n,np.array([x,y,0.0])) for n,(x,y) in table])
oth=[np.array(xy) for _,xy in table]
segs,q,_=P.plan_pick("flask",(0.600,0.0),0.0,sg.HOME_Q,oth)
for label,xy,top in (("bottom .25",(0.25,P.slot_y(0)),0.08),("bottom .05",(0.05,P.slot_y(0)),0.08),("middle .25",(0.25,P.slot_y(1)),0.36),
                     ("top .05",(0.05,P.slot_y(2)),0.64),("table (0.70,0.20)",(0.70,0.20),0.0)):
    t=time.time()
    try:
        pl,qo=P.plan_place("flask",xy,top,q,oth)
        h=P.plan_home(qo)
        print("RES",label,"-> home via:",[s.name for s in h],"(%d waypoints)"%sum(len(s.q) for s in h),"%.0fs"%(time.time()-t),flush=True)
    except sg.PlanError as e:
        print("RES",label,"place X",str(e)[:60],flush=True)
app.close()
