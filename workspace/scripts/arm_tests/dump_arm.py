# Dumps the Franka arm's physics settings (drive gains, effort / velocity limits, armature, friction, solver iterations,
# link masses) exactly as the sim loads them. Standalone: runs in its own throw-away Isaac Sim container (see README.md).
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import numpy as np, math
from pxr import Usd, UsdPhysics, PhysxSchema, Sdf
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper

world = World(stage_units_in_meters=1.0, physics_dt=1/60., rendering_dt=1/60.)
pc = world.get_physics_context(); pc.set_solver_type("TGS"); pc.enable_ccd(True)
scene_prim = pc.get_current_physics_scene_prim(); api = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
api.CreateMinPositionIterationCountAttr(16); api.CreateMinVelocityIterationCountAttr(4)
stage = world.stage
add_reference_to_stage(usd_path=get_assets_root_path()+"/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd", prim_path="/World/Franka")
fp = stage.GetPrimAtPath("/World/Franka")
vs = fp.GetVariantSet("Gripper"); print("Gripper variants:", vs.GetVariantNames(), "default sel:", vs.GetVariantSelection())
vs.SetVariantSelection("AlternateFinger")
g = ParallelGripper(end_effector_prim_path="/World/Franka/panda_rightfinger", joint_prim_names=["panda_finger_joint1","panda_finger_joint2"],
    joint_opened_positions=np.array([0.04,0.04]), joint_closed_positions=np.array([0.,0.]), action_deltas=np.array([0.01,0.01]))
robot = world.scene.add(SingleManipulator(prim_path="/World/Franka", name="franka", end_effector_prim_path="/World/Franka/panda_rightfinger", gripper=g))
world.scene.add_default_ground_plane()
world.reset(); robot.initialize()
for _ in range(10): world.step(render=False)
av = robot._articulation_view
def arr(x): return np.round(np.array(x).flatten(), 4)
print("\n== articulation view (stage units; gains per RADIAN)")
print("dof names      :", robot.dof_names)
kps, kds = av.get_gains(); print("stiffness kp   :", arr(kps)); print("damping kd     :", arr(kds)); print("kd/kp (s)      :", arr(np.array(kds)/np.maximum(np.array(kps),1e-9)))
try: print("max efforts N·m:", arr(av.get_max_efforts()))
except Exception as e: print("max efforts err", e)
try: print("max joint vel  :", arr(av.get_max_joint_velocities()))
except Exception as e: print("max vel err", e)
try: print("joint limits lo:", arr(av.get_dof_limits()[...,0]), "\njoint limits hi:", arr(av.get_dof_limits()[...,1]))
except Exception as e: print("limits err", e)
try: print("armature       :", arr(av.get_armatures()))
except Exception as e: print("armature err", e)
try: print("joint friction :", arr(av.get_friction_coefficients()))
except Exception as e: print("friction err", e)
print("\n== USD per-joint drive / physx joint attributes")
for jn in [f"panda_joint{i}" for i in range(1,8)]+["panda_finger_joint1","panda_finger_joint2"]:
    p = None
    for prim in Usd.PrimRange(fp):
        if prim.GetName()==jn: p=prim; break
    if p is None: print(jn,"not found"); continue
    d = UsdPhysics.DriveAPI.Get(p,"angular"); out={}
    if not d: d = UsdPhysics.DriveAPI.Get(p,"linear")
    if d:
        out["stiff/deg"]=d.GetStiffnessAttr().Get(); out["damp/deg"]=d.GetDampingAttr().Get(); out["maxForce"]=d.GetMaxForceAttr().Get(); out["type"]=d.GetTypeAttr().Get()
    for a in p.GetAttributes():
        n=a.GetName()
        if any(k in n for k in ("maxJointVelocity","armature","jointFriction","frictionCoefficient")): out[n]=a.Get()
    print(f"{jn:20s}", out)
print("\n== articulation root / solver / sleep")
for prim in Usd.PrimRange(fp):
    if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
        a=PhysxSchema.PhysxArticulationAPI(prim)
        print(prim.GetPath(), {k:getattr(a,f"Get{k}Attr")().Get() for k in ("SolverPositionIterationCount","SolverVelocityIterationCount","SleepThreshold","StabilizationThreshold","EnabledSelfCollisions")})
print("\n== PhysX scene attributes")
for a in scene_prim.GetAttributes():
    if a.GetName().startswith("physx") or a.GetName().startswith("physics"): print(" ", a.GetName(), "=", a.Get())
print("\n== link masses (kg) and gravity flags")
tot=0
for prim in Usd.PrimRange(fp):
    if prim.HasAPI(UsdPhysics.MassAPI):
        m=UsdPhysics.MassAPI(prim).GetMassAttr().Get()
        if m: tot+=m; print(f"  {prim.GetName():22s} mass {m:.3f}  diagInertia {UsdPhysics.MassAPI(prim).GetDiagonalInertiaAttr().Get()}")
print("  total", round(tot,3))
print("\nphysics_dt", world.get_physics_dt(), "rendering_dt", world.get_rendering_dt())
app.close()
