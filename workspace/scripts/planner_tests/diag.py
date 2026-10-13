import sys, traceback, numpy as np
sys.path.insert(0, "/workspace")
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from isaacsim.robot_motion.motion_generation import interface_config_loader as L
from isaacsim.robot_motion.motion_generation.lula import LulaKinematicsSolver
import side_grasp as sg, reach_map as rm
cfg = L.load_supported_motion_policy_config("Franka", "RMPflow")
ks = LulaKinematicsSolver(robot_description_path=cfg["robot_description_path"], urdf_path=cfg["urdf_path"])
P = sg.SideGraspPlanner(ks)
print("RES map loaded:", P.reach is not None, flush=True)
for kind in ("flask","beaker"):
    P._hold_h=sg.GRASP_HEIGHT[kind]
    for tier,top in (("bottom",0.10),("middle",0.34),("top",0.58)):
        for x in (0.05,0.25,0.45):
            a=sg._unit2((x,0.62)); v3=P.approach_vec(a)
            grasp=np.array([x,0.62,top+sg.GRASP_HEIGHT[kind]+sg.SET_DOWN_CLEARANCE]); pre=grasp-v3*0.12
            for nm,pt in (("pre",pre),("grasp",grasp)):
                c=P.reach.nearest_reachable(pt)
                if c is None: print("RES",kind,tier,x,nm,"NO reachable cell near",np.round(pt,2),flush=True); continue
                q=P.reach.Q[c]; why=P.collides(q,0.09,P._hold_h)
                print("RES",kind,tier,x,nm,"cell ok; collides:",why,flush=True)
# blocked fraction around the shelf in the map
ok=P.reach.ok
print("RES reachable cells",int(ok.sum()),flush=True)
# pick failure trace
sg.HAND_PITCH_DEG=40.0
try:
    P.plan_pick("flask",(0.368,-0.095),0.0,sg.HOME_Q,[])
except Exception:
    traceback.print_exc()
app.close()
