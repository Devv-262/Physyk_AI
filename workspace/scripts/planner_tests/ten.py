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
# scene of step 4: green beaker on the TOP shelf slot, red beaker on the middle shelf slot, three flasks on the table
tab={"blue":(0.485,-0.354),"red":(0.571,-0.186),"empty":(0.6,-0.002)}
cols=[(n,np.array([x,y,0.0])) for n,(x,y) in tab.items()]+[("Red Beaker",np.array([0.25,0.605,0.36]))]
oth=[np.array(v) for v in tab.values()]
P.set_vessels(cols)
cands=[(0.571,0.185),(0.485,0.353),(0.60,0.0),(0.57,0.21),(0.52,-0.19),(0.346,-0.494),(0.55,0.35)]
for roll in sg.ROLLS:
    P.set_roll(roll)
    try: segs,q,_=P.plan_pick("beaker",(0.25,0.695),0.64,sg.HOME_Q,oth)
    except sg.PlanError as e: print("RES roll",round(np.degrees(roll)),"pick X",str(e)[:70],flush=True); continue
    print("RES roll",round(np.degrees(roll)),"pick from top shelf ok",flush=True)
    for xy in cands:
        t=time.time()
        try: P.plan_place("beaker",xy,0.0,q,oth); print("RES   table",xy,"ok %.0fs"%(time.time()-t),flush=True)
        except sg.PlanError as e: print("RES   table",xy,"X %.0fs:"%(time.time()-t),str(e)[:90],flush=True)
    break
app.close()
