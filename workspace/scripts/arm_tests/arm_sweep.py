# Controlled arm-physics sweep: the same pick-and-place under different speed / damping / stiffness / arrival-gating / obstacle / hand-yaw /
# RMPFlow-parameter settings; one RESULT json line per run (success, error, peak joint speed and acceleration vs Franka's limits, frames at
# the velocity cap, joint path vs hand path). Standalone: own throw-away Isaac Sim container (see README.md). Edit CONFIGS / MOVES at the bottom.
# NOTE: obst=True registers the table/tray obstacles AFTER the controller reset (before it they are wiped).
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "extra_args": ["--enable", "isaacsim.robot.manipulators.examples"]})
import numpy as np, math, json, sys
from pxr import UsdGeom, PhysxSchema
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller import PickPlaceController

DT = 1/60.
world = World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT)
pc = world.get_physics_context(); pc.set_solver_type("TGS"); pc.enable_ccd(True)
api = PhysxSchema.PhysxSceneAPI.Apply(pc.get_current_physics_scene_prim()); api.CreateMinPositionIterationCountAttr(16); api.CreateMinVelocityIterationCountAttr(4)
stage = world.stage
grasp_mat = PhysicsMaterial(prim_path="/World/Physics/Grasp", static_friction=1.05, dynamic_friction=0.95, restitution=0.0)
surf_mat = PhysicsMaterial(prim_path="/World/Physics/Surf", static_friction=0.7, dynamic_friction=0.6, restitution=0.0)
table_obj = world.scene.add(FixedCuboid(prim_path="/World/Table", name="table", position=np.array([0.45,0,-0.05]), scale=np.array([0.9,1.3,0.1]), physics_material=surf_mat))
tray_obj = world.scene.add(FixedCuboid(prim_path="/World/Tray", name="tray", position=np.array([0.42,-0.32,0.015]), scale=np.array([0.22,0.24,0.03]), physics_material=surf_mat))
cube = world.scene.add(DynamicCuboid(prim_path="/World/Cube", name="cube", position=np.array([0.45,0.22,0.025]), scale=np.array([0.05]*3), mass=0.05, color=np.array([0.2,0.4,0.9]), physics_material=grasp_mat))
add_reference_to_stage(usd_path=get_assets_root_path()+"/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd", prim_path="/World/Franka")
stage.GetPrimAtPath("/World/Franka").GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
grip = ParallelGripper(end_effector_prim_path="/World/Franka/panda_rightfinger", joint_prim_names=["panda_finger_joint1","panda_finger_joint2"],
    joint_opened_positions=np.array([0.04,0.04]), joint_closed_positions=np.array([0.,0.]), action_deltas=np.array([0.01,0.01]))
robot = world.scene.add(SingleManipulator(prim_path="/World/Franka", name="franka", end_effector_prim_path="/World/Franka/panda_rightfinger", gripper=grip))
world.reset(); robot.initialize()
av = robot._articulation_view
KP0, KD0 = [float(x) for x in np.array(av.get_gains()[0]).flatten()[:1]], [float(x) for x in np.array(av.get_gains()[1]).flatten()[:1]]
HOME = np.array([0.,-0.785,0.,-2.356,0.,1.571,0.785,0.04,0.04])
VEL = np.array([2.175]*4+[2.61]*3); ACC = np.array([15,7.5,10,12.5,15,20,20.])
BASE = [0.008,0.005,0.2,0.04,0.05,0.05,0.0025,1,0.008,0.08]   # our events_dt before speed-up
NV   = [0.008,0.005,1,0.1,0.05,0.05,0.0025,1,0.008,0.08]      # NVIDIA Franka example default
MOTION = {0,1,4,5,6,8,9}
def hand():
    t = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Franka/panda_hand")).ComputeLocalToWorldTransform(0).ExtractTranslation(); return np.array([t[0],t[1],t[2]])

def run(name, events_dt, kd_scale=1.0, kp_scale=1.0, gate=False, obst=False, yaw=None, tray_obst=True, rmp=None, pick=(0.45,0.22), place=(0.42,-0.32,0.055), max_steps=3500):
    world.reset(); robot.set_joint_positions(HOME)
    kp = np.full((1,7), KP0[0]*kp_scale, dtype=np.float32); kd = np.full((1,7), KD0[0]*kd_scale, dtype=np.float32)
    av.set_gains(kps=kp, kds=kd, joint_indices=np.arange(7))
    cube.set_world_pose(position=np.array([pick[0],pick[1],0.025]), orientation=np.array([1.,0,0,0])); cube.set_linear_velocity(np.zeros(3)); cube.set_angular_velocity(np.zeros(3))
    for _ in range(20): world.step(render=True)
    ctl = PickPlaceController(name="c", gripper=robot.gripper, robot_articulation=robot, events_dt=list(events_dt))
    if rmp:
        import yaml, isaacsim.robot_motion.motion_generation as mg
        cfg = mg.interface_config_loader.load_supported_motion_policy_config("Franka","RMPflow")
        y = yaml.safe_load(open(cfg["rmpflow_config_path"]))
        for path, val in rmp.items():
            d = y
            for k in path.split(".")[:-1]: d = d[k]
            d[path.split(".")[-1]] = val
        out = "/tmp/rmp_override.yaml"; yaml.safe_dump(y, open(out, "w")); cfg["rmpflow_config_path"] = out
        rf = mg.lula.motion_policies.RmpFlow(**cfg); amp = mg.ArticulationMotionPolicy(robot, rf, DT)
        cs = mg.MotionPolicyController(name="cs", articulation_motion_policy=amp)
        pos, ori = robot.get_world_pose(); rf.set_robot_base_pose(robot_position=pos, robot_orientation=ori)
        ctl._cspace_controller = cs
    ac = robot.get_articulation_controller(); ctl.reset(); robot.gripper.open()
    if obst:
        pol = ctl._cspace_controller.get_motion_policy(); pol.add_obstacle(table_obj, static=True)
        if tray_obst: pol.add_obstacle(tray_obj, static=True)
    pp = np.array([pick[0],pick[1],0.025]); pl = np.array(place)
    qd_log=[]; hand_log=[]; q_log=[]; wait=0; ph5_end_dist=None; steps=0
    while not ctl.is_done() and steps < max_steps:
        ev = ctl.get_current_event()
        q = robot.get_joint_positions()
        if yaw is None:
            act = ctl.forward(picking_position=pp, placing_position=pl, current_joint_positions=q, end_effector_offset=np.array([0,0.005,0]))
        else:
            p0,p1 = yaw
            psi = p0 if ev<5 else (p0+(p1-p0)*0.5*(1-math.cos(math.pi*ctl._t)) if ev==5 else p1)
            r=math.radians(psi); h2=r/2
            act = ctl.forward(picking_position=pp, placing_position=pl, current_joint_positions=q,
                              end_effector_offset=np.array([-0.005*math.sin(r), 0.005*math.cos(r), 0.0]),
                              end_effector_orientation=np.array([0.0,-math.sin(h2),math.cos(h2),0.0]))
        ac.apply_action(act)
        if ev in (0,1,2): ac.apply_action(robot.gripper.forward(action="open"))
        if gate and ev in (0,5) and ctl.get_current_event()==ev+1:
            tgt = pp if ev==0 else pl; h = hand()
            d = math.hypot(h[0]-tgt[0], h[1]-tgt[1])
            if ev==5 and ph5_end_dist is None: ph5_end_dist = d
            if d > 0.015 and wait < 600:
                wait += 1; ctl._event = ev; ctl._t = 1.0 - 0.5*ctl._events_dt[ev]
        elif ev==5 and ctl.get_current_event()==6 and ph5_end_dist is None:
            h = hand(); ph5_end_dist = math.hypot(h[0]-pl[0], h[1]-pl[1])
        world.step(render=True); steps += 1
        qd_log.append(np.array(robot.get_joint_velocities()).flatten()[:7]); q_log.append(np.array(robot.get_joint_positions()).flatten()[:7]); hand_log.append(hand())
    for _ in range(40): world.step(render=True)
    cp,_ = cube.get_world_pose(); cp=np.array(cp)
    QD=np.array(qd_log).T; Q=np.array(q_log).T; H=np.array(hand_log)
    acc=np.diff(QD,axis=1)/DT
    jpath=float(np.sum(np.abs(np.diff(Q,axis=1))))
    hpath=float(np.sum(np.linalg.norm(np.diff(H,axis=0),axis=1)))
    ok = (math.hypot(cp[0]-pl[0], cp[1]-pl[1]) < 0.03) and cp[2] < 0.08
    jp_each=[round(float(np.sum(np.abs(np.diff(Q[i])))),2) for i in range(7)]
    res = dict(jp_each=jp_each, name=name, ok=bool(ok), err_mm=round(1000*math.hypot(cp[0]-pl[0], cp[1]-pl[1]),1), steps=steps,
               peak_v=[round(float(np.max(np.abs(QD[i]))/VEL[i]),2) for i in range(7)],
               frames_at_cap=int(np.sum(np.max(np.abs(QD)/VEL[:,None],axis=0)>0.97)),
               peak_a=[round(float(np.max(np.abs(acc[i]))/ACC[i]),2) for i in range(7)],
               pct_over_acc=round(100*float(np.mean(np.any(np.abs(acc)>ACC[:,None],axis=0))),1),
               joint_path_rad=round(jpath,2), hand_path_m=round(hpath,2), swivel_ratio=round(jpath/max(hpath,1e-6),1),
               ph5_end_dist_cm=None if ph5_end_dist is None else round(100*ph5_end_dist,1), gate_wait=wait)
    print("RESULT", json.dumps(res), flush=True); return res

def sp(base, f, f5=None):
    e = list(base)
    for i in MOTION: e[i] = base[i]*f
    if f5 is not None: e[5] = f5
    return e

CONFIGS = [
 ("NVIDIA default dt",                  NV, {}),
 ("ours x1.00",                         sp(BASE,1.0), {}),
 ("ours x1.25 (current)",               sp(BASE,1.25), {}),
 ("ours x1.50",                         sp(BASE,1.5), {}),
 ("ours x2.00",                         sp(BASE,2.0), {}),
 ("x1.25 + kd x0.33 (1500)",            sp(BASE,1.25), dict(kd_scale=0.327)),
 ("x1.25 + kd x2",                      sp(BASE,1.25), dict(kd_scale=2.0)),
 ("x1.25 + kp,kd x0.1 (Isaac Lab-like)", sp(BASE,1.25), dict(kp_scale=0.1, kd_scale=0.1)),
 ("x1.25 + arrival gating",             sp(BASE,1.25), dict(gate=True)),
 ("x1.25 + obstacles + yaw 5->19 + gating", sp(BASE,1.25), dict(obst=True, yaw=(5,19), gate=True)),
 ("  same, RMPFlow velocity cap 4 -> 2.0", sp(BASE,1.25), dict(obst=True, yaw=(5,19), gate=True, rmp={"rmp_params.joint_velocity_cap_rmp.max_velocity": 2.0})),
]
MOVES = [("long 49cm", dict(pick=(0.45,0.22), place=(0.42,-0.32,0.055))), ("short 30cm", dict(pick=(0.50,0.00), place=(0.42,-0.32,0.055)))]
for mname, mk in MOVES:
    for cname, ev, kw in CONFIGS:
        try: run(f"{mname} | {cname}", ev, **kw, **mk)
        except Exception as e: print("RESULT", json.dumps(dict(name=f"{mname} | {cname}", error=str(e)[:120])), flush=True)
app.close()
