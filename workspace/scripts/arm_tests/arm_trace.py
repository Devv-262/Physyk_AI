# Fully instrumented pick-and-place: every step logs the controller phase, the 7 joints (position, velocity), what RMPFlow itself commanded
# (positions + velocities), and the hand / end-effector / RMPFlow target positions. One CSV per run in /out (mount ./logs there).
# Standalone: runs in its own throw-away Isaac Sim container (see README.md). Edit the run(...) list at the bottom.
#   obst="none"    no RMPFlow obstacles          obst="before"  registered BEFORE ctl.reset()  -> wiped (what the sim used to do)
#   obst="after"   registered after ctl.reset() -> active          yaw=(pick_yaw, place_yaw) hand yaw in degrees, cube_yaw cube turn in degrees
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "extra_args": ["--enable", "isaacsim.robot.manipulators.examples"]})
import numpy as np, math, json, csv
from pxr import UsdGeom, PhysxSchema
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller import PickPlaceController
DT=1/60.
world=World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT)
pc=world.get_physics_context(); pc.set_solver_type("TGS"); pc.enable_ccd(True)
api=PhysxSchema.PhysxSceneAPI.Apply(pc.get_current_physics_scene_prim()); api.CreateMinPositionIterationCountAttr(16); api.CreateMinVelocityIterationCountAttr(4)
stage=world.stage
gm=PhysicsMaterial(prim_path="/World/Physics/G", static_friction=1.05, dynamic_friction=0.95, restitution=0.0)
sm=PhysicsMaterial(prim_path="/World/Physics/S", static_friction=0.7, dynamic_friction=0.6, restitution=0.0)
table=world.scene.add(FixedCuboid(prim_path="/World/Table", name="table", position=np.array([0.45,0,-0.05]), scale=np.array([0.9,1.3,0.1]), physics_material=sm))
tray=world.scene.add(FixedCuboid(prim_path="/World/Tray", name="tray", position=np.array([0.42,-0.32,0.015]), scale=np.array([0.22,0.24,0.03]), physics_material=sm))
cube=world.scene.add(DynamicCuboid(prim_path="/World/Cube", name="cube", position=np.array([0.45,0.22,0.025]), scale=np.array([0.05]*3), mass=0.05, color=np.array([0.2,0.4,0.9]), physics_material=gm))
add_reference_to_stage(usd_path=get_assets_root_path()+"/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd", prim_path="/World/Franka")
stage.GetPrimAtPath("/World/Franka").GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
grip=ParallelGripper(end_effector_prim_path="/World/Franka/panda_rightfinger", joint_prim_names=["panda_finger_joint1","panda_finger_joint2"],
    joint_opened_positions=np.array([0.04,0.04]), joint_closed_positions=np.array([0.,0.]), action_deltas=np.array([0.01,0.01]))
robot=world.scene.add(SingleManipulator(prim_path="/World/Franka", name="franka", end_effector_prim_path="/World/Franka/panda_rightfinger", gripper=grip))
world.reset(); robot.initialize()
HOME=np.array([0.,-0.785,0.,-2.356,0.,1.571,0.785,0.04,0.04])
BASE=[0.008,0.005,0.2,0.04,0.05,0.05,0.0025,1,0.008,0.08]; MOTION={0,1,4,5,6,8,9}
EV=[BASE[i]*1.25 if i in MOTION else BASE[i] for i in range(10)]
def hand():
    t=UsdGeom.Xformable(stage.GetPrimAtPath("/World/Franka/panda_hand")).ComputeLocalToWorldTransform(0).ExtractTranslation(); return np.array([t[0],t[1],t[2]])
def ee():
    t=UsdGeom.Xformable(stage.GetPrimAtPath("/World/Franka/panda_rightfinger")).ComputeLocalToWorldTransform(0).ExtractTranslation(); return np.array([t[0],t[1],t[2]])

def run(tag, yaw, obst='before', gate=True, cube_yaw=0.0, pick=(0.45,0.22), place=(0.42,-0.32,0.055)):
    world.reset(); robot.set_joint_positions(HOME)
    cy=math.radians(cube_yaw)/2; cube.set_world_pose(position=np.array([pick[0],pick[1],0.025]), orientation=np.array([math.cos(cy),0,0,math.sin(cy)])); cube.set_linear_velocity(np.zeros(3)); cube.set_angular_velocity(np.zeros(3))
    for _ in range(20): world.step(render=True)
    ctl=PickPlaceController(name="c", gripper=robot.gripper, robot_articulation=robot, events_dt=list(EV))
    cs=ctl._cspace_controller; cap={}
    orig=cs.forward
    def wrapped(*a, **k):
        cap["tpos"]=np.array(k.get("target_end_effector_position")); r=orig(*a, **k); cap["act"]=r; return r
    cs.forward=wrapped
    pol=cs.get_motion_policy()
    def reg():
        pol.add_obstacle(table, static=True); pol.add_obstacle(tray, static=True)
    def counts(): 
        try: return (len(pol._static_obstacles), len(pol._dynamic_obstacles))
        except Exception as e: return str(e)[:40]
    if obst in ("before","both"): reg()
    c0=counts()
    ac=robot.get_articulation_controller(); ctl.reset(); robot.gripper.open()
    c1=counts()
    if obst in ("after","both"): reg()
    print("OBSTACLES", tag, "registered-mode", obst, "(static,dynamic) before reset:", c0, "after reset:", c1, "after registration:", counts(), flush=True)
    pp=np.array([pick[0],pick[1],0.025]); pl=np.array(place)
    out=[]; wait={0:0,5:0}; steps=0
    while not ctl.is_done() and steps<3000:
        ev=ctl.get_current_event(); tt=ctl._t; q=np.array(robot.get_joint_positions()).flatten(); cap.pop("act",None)
        p0,p1=yaw; psi=p0 if ev<5 else (p0+(p1-p0)*0.5*(1-math.cos(math.pi*tt)) if ev==5 else p1); r=math.radians(psi); h2=r/2
        act=ctl.forward(picking_position=pp, placing_position=pl, current_joint_positions=q,
            end_effector_offset=np.array([-0.005*math.sin(r), 0.005*math.cos(r), 0.0]), end_effector_orientation=np.array([0.0,-math.sin(h2),math.cos(h2),0.0]))
        ra=cap.get("act")           # what RMPFlow itself returned this step (None in phases 2,3,7)
        cmdp=[np.nan]*7; cmdv=[np.nan]*7
        if ra is not None:
            if ra.joint_positions is not None: cmdp=[np.nan if v is None else float(v) for v in np.array(ra.joint_positions,dtype=object).flatten()[:7]]
            if ra.joint_velocities is not None: cmdv=[np.nan if v is None else float(v) for v in np.array(ra.joint_velocities,dtype=object).flatten()[:7]]
        ac.apply_action(act)
        if ev in (0,1,2): ac.apply_action(robot.gripper.forward(action="open"))
        held=0
        if gate and ev in (0,5) and ctl.get_current_event()==ev+1:
            tgt=pp if ev==0 else pl; h=hand(); d=math.hypot(h[0]-tgt[0], h[1]-tgt[1])
            if d>0.015 and wait[ev]<600: wait[ev]+=1; ctl._event=ev; ctl._t=1.0-0.5*ctl._events_dt[ev]; held=1
        world.step(render=True); steps+=1
        qd=np.array(robot.get_joint_velocities()).flatten()[:7]; q2=np.array(robot.get_joint_positions()).flatten()[:7]; h=hand()
        tp=cap.get("tpos", np.array([np.nan]*3))
        out.append([steps, ev, tt, held, psi]+list(q2)+list(qd)+cmdp+cmdv+list(h)+list(ee())+list(tp[:3] if tp is not None and len(tp)>=3 else [np.nan]*3))
    for _ in range(40): world.step(render=True)
    cp,_=cube.get_world_pose(); err=math.hypot(cp[0]-pl[0], cp[1]-pl[1])
    hdr=["step","event","ev_t","held","psi"]+[f"q{i}" for i in range(1,8)]+[f"qd{i}" for i in range(1,8)]+[f"cmdq{i}" for i in range(1,8)]+[f"cmdqd{i}" for i in range(1,8)]+["hx","hy","hz","ex","ey","ez","tx","ty","tz"]
    with open(f"/out/trace_{tag}.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(hdr); w.writerows(out)
    print("DONE", json.dumps(dict(tag=tag, steps=steps, err_mm=round(1000*err,1), gate_waits=wait)), flush=True)
run("d_live_like_5to0_none",      (5,0),  obst="none",  cube_yaw=5.0)
run("e_live_like_5to0_obst",      (5,0),  obst="after", cube_yaw=5.0)
run("f_aligned_5to19_none",       (5,19), obst="none",  cube_yaw=5.0)
run("g_aligned_5to19_obst",       (5,19), obst="after", cube_yaw=5.0)
run("h_aligned_5to0_none_short",  (5,0),  obst="none",  cube_yaw=5.0, pick=(0.50,0.0))
run("i_aligned_5to0_obst_short",  (5,0),  obst="after", cube_yaw=5.0, pick=(0.50,0.0))
app.close()
