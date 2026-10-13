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
print("RES pick scan: rows y, cols x (flask)  . = ok  X = fail", flush=True)
xs=np.arange(0.30,0.76,0.06)
print("RES      "+" ".join(f"{x:4.2f}" for x in xs))
for y in np.arange(-0.36,0.37,0.06):
    row=[]
    for x in xs:
        try: P.plan_pick("flask",(x,y),0.0,sg.HOME_Q,[]); row.append("  . ")
        except sg.PlanError: row.append("  X ")
    print(f"RES y={y:+.2f} "+" ".join(row),flush=True)
app.close()
