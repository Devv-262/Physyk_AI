import sys, time, numpy as np
sys.path.insert(0, "/workspace")
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from isaacsim.robot_motion.motion_generation import interface_config_loader as L
from isaacsim.robot_motion.motion_generation.lula import LulaKinematicsSolver
import side_grasp as sg, reach_map as rm
cfg = L.load_supported_motion_policy_config("Franka", "RMPflow")
ks = LulaKinematicsSolver(robot_description_path=cfg["robot_description_path"], urdf_path=cfg["urdf_path"])
P = sg.SideGraspPlanner(ks)
table = [("blue",(0.485,-0.353)),("red",(0.571,-0.185))]
P.set_vessels([(n,np.array([x,y,0.0])) for n,(x,y) in table]); oth=[np.array(xy) for _,xy in table]
segs,q,_=P.plan_pick("flask",(0.600,0.0),0.0,sg.HOME_Q,oth)
orig_path = rm.ReachMap.path
def variant(label, budget, use_cost):
    def path(self, s, g, blocked=lambda c,q: False, budget_s=20.0, extra_cost=lambda c: 0.0):
        return orig_path(self, s, g, blocked, budget_s=budget, extra_cost=(extra_cost if use_cost else (lambda c: 0.0)))
    rm.ReachMap.path = path
    for name,xy,top in (("middle .25",(0.25,P.slot_y(1)),0.36),("top .05",(0.05,P.slot_y(2)),0.64)):
        t=time.time()
        try: P.plan_place("flask",xy,top,q,oth); r="ok"
        except sg.PlanError as e: r="X "+str(e)[:50]
        print("RES",label,name,r,"%.0fs"%(time.time()-t),flush=True)
variant("budget20 cost",20.0,True)
variant("budget60 cost",60.0,True)
variant("budget60 NO cost",60.0,False)
app.close()
