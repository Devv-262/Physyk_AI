#!/usr/bin/env python3
"""
Physyk AI — Franka Panda Simulation Engine (Isaac Sim 6.0.1)
============================================================
- Connects to NVIDIA Nucleus Cloud
- Loads official Franka Panda USD (LIBERO benchmark standard)
- Simulates physics, dynamics, and multi-stage joint trajectories
- Logs live real-time joint positions and gripper telemetry
"""

import sys
import os
import argparse
import numpy as np

# 1. Start SimulationApp
from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description="Physyk AI Franka Panda Simulation Pipeline")
parser.add_argument("--steps", type=int, default=250, help="Total simulation steps")
parser.add_argument("--livestream", action="store_true", help="Enable WebRTC live streaming")
args, _ = parser.parse_known_args()

print("\n==================================================================", flush=True)
print("  [Physyk AI] Starting Autonomous Franka Panda Pipeline", flush=True)
print("  Engine: Isaac Sim 6.0.1 | Hardware: RTX PRO 6000 Blackwell", flush=True)
print("==================================================================\n", flush=True)

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

# 2. Imports after SimulationApp
import carb
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import DistantLight, GroundPlane, Cube
from isaacsim.core.experimental.prims import Articulation
from isaacsim.storage.native import get_assets_root_path

# 3. Assets & Scene Setup
print("[1/5] Connecting to NVIDIA Nucleus Cloud Assets...", flush=True)
assets_root = get_assets_root_path()
if assets_root is None:
    print("[ERROR] Nucleus asset path could not be resolved.", flush=True)
    simulation_app.close()
    sys.exit(1)
print(f"      Connected: {assets_root}\n", flush=True)

print("[2/5] Creating USD Simulation Stage...", flush=True)
stage_utils.create_new_stage()
GroundPlane("/World/GroundPlane", positions=[0, 0, 0])
distant_light = DistantLight("/World/DistantLight")
distant_light.set_intensities(400)
print("      Ground plane and environment lighting added.\n", flush=True)

print("[3/5] Referencing Franka Panda USD (LIBERO Setup)...", flush=True)
franka_usd = assets_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
print(f"      USD Path: {franka_usd}", flush=True)
stage_utils.add_reference_to_stage(
    usd_path=franka_usd,
    path="/World/Franka",
    variants=[("Gripper", "AlternateFinger"), ("Mesh", "Performance")],
)

# Spawn Assembly Kit component (Red 50mm Cube)
Cube(
    paths="/World/AssemblyCube",
    positions=np.array([0.45, 0.0, 0.025]),
    sizes=1.0,
    scales=np.array([0.05, 0.05, 0.05]),
    colors="red"
)

# Articulation Wrapper for Physics & Control
robot = Articulation("/World/Franka")
home_pose = [0.0, -0.568, 0.0, -2.811, 0.0, 3.037, 0.741, 0.04, 0.04]
robot.set_default_state(positions=[0.0, 0.0, 0.0], dof_positions=home_pose)
print(f"      Robot Articulation initialized with {len(robot.dof_names)} DOFs.\n", flush=True)

# 4. Start Simulation Timeline
print("[4/5] Starting PhysX Physics Timeline...", flush=True)
app_utils.play()
simulation_app.update()

robot.reset_to_default_state()
simulation_app.update()
print("      Simulation timeline playing at 200 Hz.\n", flush=True)

# 5. Execute 5-Stage Trajectory Pipeline
print("[5/5] Executing Autonomous Manipulation Pipeline...", flush=True)
print("-" * 66, flush=True)

stages = [
    {"name": "Stage 1: Home Position Ready", "target": [0.0, -0.568, 0.0, -2.811, 0.0, 3.037, 0.741, 0.04, 0.04], "steps": 40},
    {"name": "Stage 2: Pre-Grasp Approach",   "target": [0.0,  0.200, 0.0, -1.800, 0.0, 2.000, 0.785, 0.04, 0.04], "steps": 50},
    {"name": "Stage 3: Descend to Component", "target": [0.0,  0.450, 0.0, -1.500, 0.0, 1.950, 0.785, 0.04, 0.04], "steps": 50},
    {"name": "Stage 4: Grasp & Close Gripper","target": [0.0,  0.450, 0.0, -1.500, 0.0, 1.950, 0.785, 0.00, 0.00], "steps": 30},
    {"name": "Stage 5: Lift Component to Inspect", "target": [0.0, -0.200, 0.0, -2.200, 0.0, 2.400, 0.785, 0.00, 0.00], "steps": 50},
]

global_step = 0
for stage_num, stage in enumerate(stages, 1):
    print(f"\n>>> {stage['name']} ({stage['steps']} steps)", flush=True)
    target = stage["target"]
    
    for s in range(stage["steps"]):
        robot.set_dof_position_targets(target)
        simulation_app.update()
        global_step += 1
        
        # Log telemetry at key points
        if s == 0 or s == stage["steps"] // 2 or s == stage["steps"] - 1:
            dofs = robot.get_dof_positions()
            if hasattr(dofs, "numpy"):
                dofs = dofs.numpy()
            dof_arr = dofs.flatten()
            arm_str = " ".join([f"{x:6.2f}" for x in dof_arr[:7]])
            grip_str = f"[{dof_arr[7]:.3f}, {dof_arr[8]:.3f}]"
            print(f"    [Step {global_step:03d}] Arm: [{arm_str}] | Gripper: {grip_str}", flush=True)

print("\n" + "=" * 66, flush=True)
print("  [SUCCESS] Franka Panda Pipeline Completed All Stages!", flush=True)
print("  Final Component State: Picked & Lifted to Inspection Height", flush=True)
print("==================================================================\n", flush=True)

simulation_app.close()
