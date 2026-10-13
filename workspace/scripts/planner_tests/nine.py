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
V = {"flask_blue": ("flask", (0.485,-0.353)), "flask_red": ("flask", (0.571,-0.185)), "flask_empty": ("flask", (0.600,0.0)),
     "beaker_red": ("beaker", (0.571,0.185)), "beaker_green": ("beaker", (0.485,0.353))}
name="beaker_green"; kind,xy=V[name]
P.set_vessels([(k,np.array([v[1][0],v[1][1],0.0])) for k,v in V.items() if k!=name])
oth=[np.array(v[1]) for k,v in V.items() if k!=name]
segs,q,_=P.plan_pick(kind,xy,0.0,sg.HOME_Q,oth)
p0,a0=P.hand_pose(q)
print("RES start hand",np.round(p0,3),"yaw",np.round(a0,2),"tilt %.1f"%P.tilt_deg(q),"q",np.round(q,2),flush=True)
c0=P.reach.nearest_reachable(p0); print("RES nearest map cell",c0,"cellpos",np.round(rm.cell_pos(*c0),3),"mapQ",np.round(P.reach.Q[c0],2),flush=True)
print("RES joint distance start->map cell Q: %.2f"%np.max(np.abs(q-P.reach.Q[c0])),flush=True)
# map path
P._hold_h=sg.GRASP_HEIGHT[kind]
dest_pre=np.array([0.25,P.slot_y(1),0.36+0.081+0.006])-P.approach_vec(sg._unit2((0.25,P.slot_y(1))))*0.12
blocked=lambda c,qq: P.collides(qq,0.09,P._hold_h,cache=P._cell_geom(c,qq)) is not None
cells=P.reach.path(p0,dest_pre,blocked,extra_cost=P._plank_cost)
print("RES map path cells:",None if cells is None else len(cells),flush=True)
if cells:
    pts=[p0]+cells+[dest_pre]
    samples=[]
    for a_,b_ in zip(pts[:-1],pts[1:]):
        n=max(1,int(np.ceil(np.linalg.norm(b_-a_)/0.01)))
        for i in range(1,n+1):
            pt=a_+(b_-a_)*i/n; samples.append((pt,sg._unit2(pt[:2])))
    print("RES samples",len(samples),flush=True)
    prev=q
    for k,(p,a) in enumerate(samples):
        try:
            qq=P.chain([(p,a)],prev); prev=qq[-1]
        except sg.PlanError as e:
            print("RES chain fails at sample",k,"pos",np.round(p,3),"prev q",np.round(prev,2),flush=True)
            cm=P.reach.nearest_reachable(np.asarray(p,float),max_cells=1)
            print("RES   nearest map cell",cm,"pos",None if cm is None else np.round(rm.cell_pos(*cm),3),flush=True)
            if cm is not None:
                mq=P.reach.Q[cm]
                print("RES   joint dist prev->mapQ %.2f"%np.max(np.abs(prev-mq)),flush=True)
                for roll in (0.0,np.pi):
                    raw=P._ik_raw(p,sg.hand_quat(a,roll),mq,0.003,0.15)
                    print("RES   raw IK from mapQ roll",round(roll,2),"->",None if raw is None else "ok jump %.2f"%np.max(np.abs(raw-prev)),flush=True)
                    if raw is not None:
                        pp,rot=P._fk(raw); print("RES     pos err %.4f orient: z-lvl %.3f y-lvl %.3f"%(np.linalg.norm(pp-p),rot[2,2]+np.sin(np.radians(40)),rot[2,1]),flush=True)
            break
app.close()
