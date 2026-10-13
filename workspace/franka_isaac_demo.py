#!/usr/bin/env python3
"""
Franka Panda — Isaac Sim 6.0.1 Realistic Demo
==============================================
- Official Franka Panda USD from NVIDIA Nucleus
- Wrist camera mounted on panda_link8
- Scene camera (third-person angled view)
- Full pick-and-place motion sequence
- WebRTC streaming on port 8211

Run inside container:
    /isaac-sim/python.sh /workspace/franka_isaac_demo.py
"""

import sys
import numpy as np

# ── 1. Boot SimulationApp FIRST ────────────────────────────────────────────────
from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": True,
    "width": 1920,
    "height": 1080,
    "renderer": "RaytracedLighting",
})

# ── 2. All other imports AFTER SimulationApp ────────────────────────────────────
import carb
import omni.usd
from pxr import UsdGeom, Gf, UsdPhysics, UsdLux, Sdf

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.core.utils.types import ArticulationAction

print("\n" + "=" * 65)
print("  Franka Panda — Isaac Sim 6.0.1 Demo")
print("  WebRTC → http://<HOST>:8211/streaming/webrtc-demo/")
print("=" * 65 + "\n")

# ── 3. World ────────────────────────────────────────────────────────────────────
world = World(
    stage_units_in_meters=1.0,
    physics_dt=1.0 / 60.0,
    rendering_dt=1.0 / 60.0,
)
world.scene.add_default_ground_plane()
stage = omni.usd.get_context().get_stage()

# ── 4. Lighting — RTX quality ───────────────────────────────────────────────────
dome_prim = stage.DefinePrim("/World/Lights/DomeLight", "DomeLight")
dome_prim.GetAttribute("intensity").Set(800.0)

key_prim = stage.DefinePrim("/World/Lights/KeyLight", "DistantLight")
key_prim.GetAttribute("intensity").Set(3000.0)
key_prim.GetAttribute("color").Set(Gf.Vec3f(1.0, 0.95, 0.85))
xf = UsdGeom.Xformable(key_prim)
xf.AddRotateXYZOp().Set(Gf.Vec3f(-45, 45, 0))

fill_prim = stage.DefinePrim("/World/Lights/FillLight", "SphereLight")
fill_prim.GetAttribute("intensity").Set(500.0)
fill_prim.GetAttribute("radius").Set(0.3)
fill_prim.GetAttribute("color").Set(Gf.Vec3f(0.55, 0.65, 1.0))
UsdGeom.Xformable(fill_prim).AddTranslateOp().Set(Gf.Vec3d(-1.5, 1.5, 2.5))

# ── 5. Franka Panda USD ─────────────────────────────────────────────────────────
assets_root = get_assets_root_path()
if assets_root is None:
    carb.log_error("Cannot reach NVIDIA Nucleus. Check network.")
    simulation_app.close()
    sys.exit(1)

franka_usd = assets_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
print(f"[INFO] Nucleus root : {assets_root}")
print(f"[INFO] Franka USD   : {franka_usd}")

ROBOT_PATH = "/World/Franka"
add_reference_to_stage(usd_path=franka_usd, prim_path=ROBOT_PATH)

# ── 6. Robot articulation wrapper ───────────────────────────────────────────────
gripper = ParallelGripper(
    end_effector_prim_path=ROBOT_PATH + "/panda_rightfinger",
    joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
    joint_opened_positions=np.array([0.04, 0.04]),
    joint_closed_positions=np.array([0.0,  0.0]),
    action_deltas=np.array([0.04, 0.04]),
)

robot = world.scene.add(
    SingleManipulator(
        prim_path=ROBOT_PATH,
        name="franka",
        end_effector_prim_name="panda_link8",
        gripper=gripper,
    )
)

# ── 7. Target objects ───────────────────────────────────────────────────────────
world.scene.add(DynamicCuboid(
    prim_path="/World/RedCube",
    name="red_cube",
    position=np.array([0.48, 0.0, 0.025]),
    scale=np.array([0.05, 0.05, 0.05]),
    color=np.array([0.88, 0.15, 0.10]),
))

world.scene.add(DynamicCuboid(
    prim_path="/World/BlueCube",
    name="blue_cube",
    position=np.array([0.42, 0.18, 0.025]),
    scale=np.array([0.05, 0.05, 0.05]),
    color=np.array([0.10, 0.35, 0.90]),
))

# ── 8. Scene Camera (third-person) ──────────────────────────────────────────────
scene_cam = stage.DefinePrim("/World/Cameras/SceneCamera", "Camera")
scene_cam.GetAttribute("focalLength").Set(28.0)
scene_cam.GetAttribute("horizontalAperture").Set(36.0)
scene_cam.GetAttribute("clippingRange").Set(Gf.Vec2f(0.01, 50.0))
sc_xf = UsdGeom.Xformable(scene_cam)
sc_xf.AddTranslateOp().Set(Gf.Vec3d(-1.1, -1.1, 1.5))
sc_xf.AddRotateXYZOp().Set(Gf.Vec3f(-32, 0, -45))
print("[INFO] Scene camera  → /World/Cameras/SceneCamera")

# ── 9. Wrist Camera (eye-in-hand on panda_link8) ────────────────────────────────
wrist_link_path = ROBOT_PATH + "/panda_link8"
wrist_cam = stage.DefinePrim(wrist_link_path + "/WristCamera", "Camera")
wrist_cam.GetAttribute("focalLength").Set(10.0)
wrist_cam.GetAttribute("horizontalAperture").Set(20.0)
wrist_cam.GetAttribute("clippingRange").Set(Gf.Vec2f(0.005, 4.0))
wc_xf = UsdGeom.Xformable(wrist_cam)
wc_xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.06))
wc_xf.AddRotateXYZOp().Set(Gf.Vec3f(-90, 0, 0))
print(f"[INFO] Wrist camera  → {wrist_link_path}/WristCamera")

# ── 10. Initialize ──────────────────────────────────────────────────────────────
world.reset()
for _ in range(10):
    world.step(render=True)

robot.initialize()
robot.gripper.open()
for _ in range(10):
    world.step(render=True)

num_dofs = robot.num_dof
print(f"[INFO] DOF count     : {num_dofs}")
print(f"[INFO] Joint names   : {robot.dof_names}\n")

# ── 11. Motion waypoints ────────────────────────────────────────────────────────
HOME   = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])

waypoints = [
    ("Home",               np.array([ 0.00, -0.785,  0.00, -2.356,  0.00, 1.571, 0.785, 0.04, 0.04]), 90),
    ("Pre-Grasp Approach", np.array([ 0.00,  0.150,  0.00, -1.750,  0.00, 1.920, 0.785, 0.04, 0.04]), 120),
    ("Descend to Object",  np.array([ 0.00,  0.450,  0.00, -1.450,  0.00, 1.900, 0.785, 0.04, 0.04]), 100),
    ("Close Gripper",      np.array([ 0.00,  0.450,  0.00, -1.450,  0.00, 1.900, 0.785, 0.00, 0.00]), 60),
    ("Lift",               np.array([ 0.00, -0.200,  0.00, -2.000,  0.00, 1.800, 0.785, 0.00, 0.00]), 120),
    ("Transport",          np.array([-0.80,  0.000,  0.00, -1.800,  0.00, 1.800, 0.785, 0.00, 0.00]), 120),
    ("Place Descent",      np.array([-0.80,  0.450,  0.00, -1.450,  0.00, 1.900, 0.785, 0.00, 0.00]), 100),
    ("Open Gripper",       np.array([-0.80,  0.450,  0.00, -1.450,  0.00, 1.900, 0.785, 0.04, 0.04]), 60),
    ("Return Home",        np.array([ 0.00, -0.785,  0.00, -2.356,  0.00, 1.571, 0.785, 0.04, 0.04]), 120),
]

# ── 12. Run simulation ──────────────────────────────────────────────────────────
print("=" * 65)
print("  🤖 Starting Franka pick-and-place sequence...")
print("=" * 65 + "\n")

total = 0
for name, joints, steps in waypoints:
    target = joints[:num_dofs]
    print(f"  ▶ {name}  ({steps} steps)")
    for s in range(steps):
        robot.apply_action(ArticulationAction(joint_positions=target))
        world.step(render=True)
        total += 1
        if s % 40 == 0:
            pos = robot.get_joint_positions()
            if pos is not None:
                print(f"    step {total:4d} | {np.round(pos[:7], 3)}")

print(f"\n  ✅ Sequence complete. Total steps: {total}")

# ── 13. Hold open so you can explore in the WebRTC viewer ──────────────────────
print("\n[INFO] Holding scene open for 5 minutes. Explore the WebRTC stream!")
print("[INFO] Press Ctrl+C to exit early.\n")
try:
    for _ in range(60 * 60 * 5):   # 5 min @ 60Hz
        world.step(render=True)
except KeyboardInterrupt:
    print("\n[INFO] Interrupted.")

simulation_app.close()
