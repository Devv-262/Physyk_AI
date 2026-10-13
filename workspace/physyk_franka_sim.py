#!/usr/bin/env python3
"""
Physyk AI — Real Isaac Sim 6.0.1 Franka Panda Simulation Engine
===============================================================
- Loads official Franka Panda USD directly from NVIDIA Nucleus Assets
- Configures physics, ground plane, lighting, and articulation
- Commands joint positions / gripper movements via Articulation API
- Logs real-time joint positions and end-effector telemetry
"""

import sys
import os
import argparse
import numpy as np

# 1. Initialize SimulationApp (Headless by default, enables Livestream extension if requested)
from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description="Physyk AI Franka Panda Simulation")
parser.add_argument("--livestream", action="store_true", help="Enable WebRTC/Livestream streaming")
parser.add_argument("--steps", type=int, default=300, help="Number of simulation steps to run")
args, _ = parser.parse_known_args()

sim_config = {
    "headless": True,
    "width": 1280,
    "height": 720,
}
simulation_app = SimulationApp(sim_config)

if args.livestream:
    from isaacsim.core.experimental.utils.app import enable_extension
    simulation_app.set_setting("/app/window/drawMouse", True)
    simulation_app.set_setting("/app/livestream/enabled", True)
    enable_extension("omni.kit.livestream.app")

# 2. Imports after SimulationApp is up
import carb
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import DistantLight, GroundPlane, Cube
from isaacsim.core.experimental.prims import Articulation, RigidPrim
from isaacsim.storage.native import get_assets_root_path

print("\n=======================================================", flush=True)
print("  [Physyk AI] Initializing Isaac Sim 6.0.1 Environment", flush=True)
print("=======================================================", flush=True)

# Verify NVIDIA Nucleus Assets path
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    print("[Physyk AI] Error: Assets root path unavailable", flush=True)
    simulation_app.close()
    sys.exit(1)

print(f"[Physyk AI] Connected to NVIDIA Nucleus: {assets_root_path}", flush=True)

# 3. Create Stage and Scene
stage_utils.create_new_stage()
GroundPlane("/World/GroundPlane", positions=[0, 0, 0])
distant_light = DistantLight("/World/DistantLight")
distant_light.set_intensities(500)

# Add Franka Panda robot USD from official NVIDIA assets
franka_usd = assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
print(f"[Physyk AI] Loading Franka Panda USD: {franka_usd}", flush=True)

stage_utils.add_reference_to_stage(
    usd_path=franka_usd,
    path="/World/Franka",
    variants=[("Gripper", "AlternateFinger"), ("Mesh", "Quality")],
)

# Target manipulation object (Cube for pick/place workcell)
cube = Cube(
    paths="/World/TargetCube",
    positions=np.array([0.45, 0.0, 0.025]),
    sizes=1.0,
    scales=np.array([0.05, 0.05, 0.05]),
    colors="red"
)

# Articulation Wrapper
robot = Articulation("/World/Franka")

# Default Home Pose (Ready state for Franka Panda)
# Joints: [joint1, joint2, joint3, joint4, joint5, joint6, joint7, finger1, finger2]
home_pose = [0.0, -0.568, 0.0, -2.811, 0.0, 3.037, 0.741, 0.04, 0.04]
robot.set_default_state(positions=[0.0, 0.0, 0.0], dof_positions=home_pose)

print(f"[Physyk AI] Robot loaded successfully with DOFs: {robot.dof_names}", flush=True)

# 4. Start Simulation Timeline
print("[Physyk AI] Starting physics simulation...", flush=True)
app_utils.play()
simulation_app.update()

robot.reset_to_default_state()
simulation_app.update()

# 5. Execute Simulation Trajectory & Dynamics
print("\n[Physyk AI] Executing joint motion sequence in Isaac Sim...", flush=True)

# Define a sequence of goal poses (e.g., Home -> Approach Cube -> Close Gripper -> Lift)
waypoints = [
    {"name": "Home Position", "joints": [0.0, -0.568, 0.0, -2.811, 0.0, 3.037, 0.741, 0.04, 0.04], "steps": 60},
    {"name": "Pre-Grasp Approach", "joints": [0.0, 0.2, 0.0, -1.8, 0.0, 2.0, 0.78, 0.04, 0.04], "steps": 80},
    {"name": "Descend to Object", "joints": [0.0, 0.5, 0.0, -1.5, 0.0, 2.0, 0.78, 0.04, 0.04], "steps": 80},
    {"name": "Grasp / Close Gripper", "joints": [0.0, 0.5, 0.0, -1.5, 0.0, 2.0, 0.78, 0.00, 0.00], "steps": 50},
    {"name": "Lift Object", "joints": [0.0, -0.2, 0.0, -2.0, 0.0, 1.8, 0.78, 0.00, 0.00], "steps": 80},
]

step_count = 0
for wp_idx, wp in enumerate(waypoints):
    target = wp["joints"]
    print(f"\n---> [Stage {wp_idx+1}/{len(waypoints)}] {wp['name']} (Duration: {wp['steps']} steps)", flush=True)
    
    for s in range(wp["steps"]):
        robot.set_dof_position_targets(target)
        simulation_app.update()
        step_count += 1
        
        if s % 25 == 0 or s == wp["steps"] - 1:
            curr_dofs = robot.get_dof_positions()
            if hasattr(curr_dofs, "numpy"):
                curr_dofs = curr_dofs.numpy()
            curr_str = np.array2string(curr_dofs.flatten()[:7], precision=3, suppress_small=True)
            gripper_str = np.array2string(curr_dofs.flatten()[7:9], precision=3, suppress_small=True)
            print(f"     Step {step_count:3d} | Arm Joints: {curr_str} | Gripper: {gripper_str}", flush=True)

print("\n=======================================================", flush=True)
print("  [Physyk AI] Isaac Sim Physics Run Completed Successfully!", flush=True)
print("=======================================================", flush=True)

simulation_app.close()
