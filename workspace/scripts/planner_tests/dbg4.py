import sys, numpy as np
sys.path.insert(0, "/workspace")
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from isaacsim.robot_motion.motion_generation import interface_config_loader as L
from isaacsim.robot_motion.motion_generation.lula import LulaKinematicsSolver
import side_grasp as sg
cfg = L.load_supported_motion_policy_config("Franka", "RMPflow")
ks = LulaKinematicsSolver(robot_description_path=cfg["robot_description_path"], urdf_path=cfg["urdf_path"])
P = sg.SideGraspPlanner(ks)
lo,hi=ks.get_cspace_position_limits(); lo=lo[:7];hi=hi[:7]
def margin(q): return float(np.min(np.minimum(q-lo,hi-q)))
for pitch in (0,20,35,50):
    sg.HAND_PITCH_DEG=float(pitch)
    for r in (0.40,0.55,0.70):
        row=[]
        for z in (0.08,0.12,0.18,0.24,0.30,0.36,0.42,0.50,0.60,0.70):
            pos=np.array([r,0.0,z]); c=P.candidates(pos,np.array([1.0,0.0]),sg.HOME_Q,tries=14)
            row.append("%d/%.2f"%(len(c),max([margin(q) for q in c],default=-1)))
        print("RES pitch",pitch,"r",r,"z .08 .12 .18 .24 .30 .36 .42 .50 .60 .70 ->"," ".join(row),flush=True)
app.close()
