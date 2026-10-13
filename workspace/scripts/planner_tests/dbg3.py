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
print("RES r z  ncand best_margin best_jump_from_home")
for z in (0.10,0.135,0.20,0.30,0.45):
    for r in (0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.70,0.75):
        for ang in (0.0,):
            pos=np.array([r*np.cos(ang),r*np.sin(ang),z]); a=np.array([1.0,0.0])
            c=P.candidates(pos,a,sg.HOME_Q,tries=25)
            if not c: print("RES",r,z,0); continue
            m=max(margin(q) for q in c); j=min(float(np.max(np.abs(q-sg.HOME_Q))) for q in c)
            print("RES",r,z,len(c),round(m,2),round(j,2),flush=True)
app.close()
