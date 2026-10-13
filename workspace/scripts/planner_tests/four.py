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
table = [("blue",(0.485,-0.353)),("red",(0.571,-0.185)),("empty",(0.600,0.0))]
tcols=[(n,np.array([x,y,0.0])) for n,(x,y) in table]
tother=[np.array(xy) for _,xy in table]
tops=(0.08,0.36,0.64)
def run(label, kind, pick_xy, pick_z, dests):
    t=time.time()
    try: segs,q,_=P.plan_pick(kind,pick_xy,pick_z,sg.HOME_Q,tother)
    except sg.PlanError as e: print("RES",label,"PICK FAIL:",str(e)[:90],flush=True); return
    print("RES",label,"pick ok %.0fs"%(time.time()-t),flush=True)
    for name,xy,top in dests:
        t=time.time()
        try: P.plan_place(kind,xy,top,q,tother); print("RES   ->",name,"ok %.0fs"%(time.time()-t),flush=True)
        except sg.PlanError as e: print("RES   ->",name,"X %.0fs"%(time.time()-t),str(e)[:70],flush=True)
# 1) beaker standing on bottom tier slot x=.25 -> table spots and other tiers
P.set_vessels(tcols)
run("beaker from bottom shelf", "beaker", (0.25,P.slot_y(0)), 0.08,
    [("table (0.70,0.20)",(0.70,0.20),0.0),("table (0.55,0.35)",(0.55,0.35),0.0),("middle slot .05",(0.05,P.slot_y(1)),0.36),("top slot .05",(0.05,P.slot_y(2)),0.64)])
# 2) flask on the middle tier slot x=.25 -> table, bottom, top
P.set_vessels(tcols)
run("flask from middle shelf", "flask", (0.25,P.slot_y(1)), 0.36,
    [("table (0.70,0.20)",(0.70,0.20),0.0),("bottom slot .05",(0.05,P.slot_y(0)),0.08),("top slot .05",(0.05,P.slot_y(2)),0.64)])
app.close()
