import sys, time, numpy as np
sys.path.insert(0, "/workspace")
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from isaacsim.robot_motion.motion_generation import interface_config_loader as L
from isaacsim.robot_motion.motion_generation.lula import LulaKinematicsSolver
import side_grasp as sg, reach_map as rm
cfg = L.load_supported_motion_policy_config("Franka", "RMPflow")
ks = LulaKinematicsSolver(robot_description_path=cfg["robot_description_path"], urdf_path=cfg["urdf_path"])
lo,hi=ks.get_cspace_position_limits(); lo=lo[:7]; hi=hi[:7]
P = sg.SideGraspPlanner(ks)
tab={"blue":(0.485,-0.354),"red":(0.571,-0.186),"empty":(0.6,-0.002)}
oth=[np.array(v) for v in tab.values()]
dest=(0.485,0.353)
def margin(q): return np.minimum(q-lo,hi-q)
def trace(label,pick_xy,pick_z):
    P.set_roll(0.0)
    P.set_vessels([(n,np.array([x,y,0.0])) for n,(x,y) in tab.items()])
    segs,q,_=P.plan_pick("beaker",pick_xy,pick_z,sg.HOME_Q,oth)
    p0,_=P.hand_pose(q)
    a=sg._unit2(dest); v3=P.approach_vec(a)
    grasp=np.array([dest[0],dest[1],0.0+sg.GRASP_HEIGHT["beaker"]+sg.SET_DOWN_CLEARANCE]); pre=grasp-v3*P.standoff_for(dest)
    cells=P.reach.path(p0,pre,lambda c,qq: False,budget_s=30.0)
    print("RES",label,"start hand",np.round(p0,3),"start q",np.round(q,2),"min joint margin to limit %.2f (joint %d)"%(margin(q).min(),int(margin(q).argmin())+1),flush=True)
    print("RES",label,"map path:",None if cells is None else len(cells),"cells",flush=True)
    if not cells: return
    pts=[p0]+cells+[pre]; samples=[]
    for x_,y_ in zip(pts[:-1],pts[1:]):
        n=max(1,int(np.ceil(np.linalg.norm(y_-x_)/0.01)))
        for i in range(1,n+1):
            pt=x_+(y_-x_)*i/n; samples.append((pt,sg._unit2(pt[:2])))
    prev=q; worst=9
    for k,(p,a_) in enumerate(samples):
        try: qq=P.chain([(p,a_)],prev); prev=qq[-1]; m=margin(prev); worst=min(worst,m.min())
        except sg.PlanError:
            m=margin(prev)
            print("RES",label,"CHAIN FAILS at sample",k,"of",len(samples),"pos",np.round(p,3),"| joint margins to limit:",np.round(m,2).tolist(),"| tightest joint",int(m.argmin())+1,flush=True)
            return
    print("RES",label,"chain OK; tightest margin along the path %.2f rad"%worst,flush=True)
trace("from MIDDLE shelf",(0.25,0.605),0.36)
trace("from TOP shelf",(0.25,0.695),0.64)
app.close()
