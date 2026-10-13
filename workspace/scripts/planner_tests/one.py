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
picks = {"flask_blue": ("flask", (0.266,-0.19)), "flask_red": ("flask", (0.368,-0.095)), "flask_empty": ("flask", (0.470,0.0)), "beaker_red": ("beaker", (0.572,0.095)), "beaker_green": ("beaker", (0.674,0.19))}
for name,(kind,xy) in picks.items():
    t=time.time()
    try:
        segs,q,_=P.plan_pick(kind,xy,0.0,sg.HOME_Q,[]); print("RES",name,"pick ok",round(time.time()-t,1),"s",[ (s.name,len(s.q)) for s in segs],flush=True)
    except sg.PlanError as e:
        print("RES",name,"pick FAIL",e,flush=True); continue
    for ti,(tier,top) in enumerate((("bottom",0.08),("middle",0.36),("top",0.64))):
        for x in (0.05,0.25,0.45):
            t=time.time()
            try:
                pl,qo=P.plan_place(kind,(x,P.slot_y(ti)),top,q,[]); h=P.plan_home(qo); print("RES",name,tier,x,"place+home ok",round(time.time()-t,1),"s",flush=True)
            except sg.PlanError as e:
                print("RES",name,tier,x,"FAIL",str(e)[:90],round(time.time()-t,1),"s",flush=True)
app.close()
