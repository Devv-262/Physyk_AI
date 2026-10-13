import sys, numpy as np
sys.path.insert(0, "/workspace")
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from isaacsim.robot_motion.motion_generation import interface_config_loader as L
from isaacsim.robot_motion.motion_generation.lula import LulaKinematicsSolver
cfg = L.load_supported_motion_policy_config("Franka", "RMPflow")
ks = LulaKinematicsSolver(robot_description_path=cfg["robot_description_path"], urdf_path=cfg["urdf_path"])
for name,q in (("odd pose after task",[2.34,-0.79,-1.43,-2.64,-2.22,2.61,2.38]),("home",[0,-0.785,0,-2.356,0,1.571,0.785])):
    p,R=ks.compute_forward_kinematics("right_gripper",np.array(q))
    print("RES",name,"EE pos",np.round(p,3),"approach (tool z)",np.round(R[:,2],2),flush=True)
app.close()
