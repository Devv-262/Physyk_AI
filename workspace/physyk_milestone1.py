#!/usr/bin/env python3
"""
Physyk AI — Milestone 1: Franka Panda pick & place in real Isaac Sim 6.0.1
"""

from isaacsim import SimulationApp
import os
import sys

EXP_PATH = os.environ.get("EXP_PATH", "/isaac-sim/apps")
FULL_KIT  = os.path.join(EXP_PATH, "isaacsim.exp.full.kit")

sim = SimulationApp(
    launch_config={
        "headless": True,
        "width": 1280,
        "height": 720,
        "renderer": "RaytracedLighting",
    },
    experience=FULL_KIT,
)

# After SimulationApp starts, use carb for logging (print() is swallowed by Kit)
import carb
import numpy as np
import omni.timeline

from isaacsim.core.simulation_manager                   import SimulationManager
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents
from isaacsim.core.experimental.utils.stage             import add_reference_to_stage
from isaacsim.robot.experimental.manipulators.examples.franka.pick_place import FrankaPickPlace
from isaacsim.storage.native                            import get_assets_root_path

def log(msg):
    """Log via carb (appears in .log file) AND force-flush to stderr."""
    carb.log_warn(f"[Physyk M1] {msg}")
    print(f"[Physyk M1] {msg}", flush=True, file=sys.stderr)

assets_root = get_assets_root_path()
log(f"SimulationApp ready. Nucleus root: {assets_root}")

# ---- Scene layout ----
CUBE_START  = np.array([0.45, 0.00, 0.025])
CUBE_TARGET = np.array([0.00, 0.45, 0.14])

task = FrankaPickPlace(
    events_dt=[60, 40, 20, 40, 80, 20, 20],
    robot_name="Physyk-Panda",
)

task.setup_scene(
    cube_initial_position=CUBE_START,
    cube_initial_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
    cube_size=np.array([0.0515, 0.0515, 0.0515]),
    target_position=CUBE_TARGET,
    robot_path="/World/physyk_panda",
    cube_path="/World/assembly_cube_0",
)

log("Scene ready — Franka Panda loaded from Nucleus.")
log(f"  Cube start:  {CUBE_START}")
log(f"  Cube target: {CUBE_TARGET}")

add_reference_to_stage(
    usd_path=assets_root + "/Isaac/Environments/Grid/default_environment.usd",
    path="/World/Environment",
)
log("Grid environment loaded.")

# ---- Physics callback ----
step_count = 0

def on_physics_step(dt: float, context) -> None:
    global step_count
    step_count += 1

    if step_count % 50 == 0:
        try:
            ee_pos   = task.robot.end_effector_link.get_world_poses()[0].numpy()[0]
            cube_pos = task.cube.get_world_poses()[0].numpy()[0]
            log(
                f"step={step_count:4d} | "
                f"EE({ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}) | "
                f"Cube({cube_pos[0]:.3f},{cube_pos[1]:.3f},{cube_pos[2]:.3f})"
            )
        except Exception as e:
            log(f"step={step_count} | state read error: {e}")

    task.forward(ik_method="damped-least-squares")

    if task.is_done():
        try:
            cube_pos = task.cube.get_world_poses()[0].numpy()[0]
            log(f"COMPLETE. Final cube pos: {cube_pos.round(3)}")
        except Exception:
            log("COMPLETE.")

SimulationManager.register_callback(on_physics_step, IsaacEvents.POST_PHYSICS_STEP)

omni.timeline.get_timeline_interface().play()
log("Physics running. Pick & place in progress...")

while sim.is_running():
    sim.update()
    if task.is_done():
        break

log("Shutting down Isaac Sim.")
sim.close()
