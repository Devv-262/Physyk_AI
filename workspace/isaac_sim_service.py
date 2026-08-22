#!/usr/bin/env python3
"""
Physyk AI — Real Isaac Sim 6.0.1 Simulation & Multi-Camera Streaming Service
=============================================================================
- Franka Panda 9-DOF physics articulation in PhysX (60 Hz)
- Native NVIDIA Lula Inverse Kinematics Engine with Accurate 0.0584m Fingertip TCP Offset
- Static Unmovable Table & Tray (FixedCuboid)
- Separated Object Staging & Instant Object Reset on Table
- Calibrated 0.40 kg Rigid Cube Physics & Static Contacts
- HD 720p Multi-Perspective Cameras with Optimal Cinematic Framing
- Real-Time Calibrated Studio Lighting Controller
- High-Performance Zero-Lag MJPEG Video Streaming on port 8100
"""

import sys
import os
import time
import math
import base64
import threading
import queue
import urllib.request
import urllib.error
import json as jsonlib
import cv2
import numpy as np

# GR00T fine-tuning episode recorder — no-op unless RECORD_EPISODES=1 is set in the
# environment (see groot_episode_recorder.py). Purely additive: does not change any existing
# control flow when disabled (the default).
from groot_episode_recorder import EpisodeRecorder
episode_recorder = EpisodeRecorder(
    output_dir=os.environ.get("RECORD_EPISODES_DIR", "/workspace/finetune_data/raw_episodes")
)

# ── 1. Boot Isaac Sim SimulationApp ───────────────────────────────────────────
from isaacsim import SimulationApp

sim_config = {
    "headless": True,
    "width": 1280,
    "height": 720,
    "renderer": "RaytracedLighting",
    "anti_aliasing": 3,
    # Needed for isaacsim.robot.manipulators.examples.franka.controllers.PickPlaceController
    # (not enabled by default) — matches NVIDIA's own franka_pick_up.py demo config.
    "extra_args": ["--enable", "isaacsim.robot.manipulators.examples"],
}
simulation_app = SimulationApp(sim_config)

# ── 2. Imports after SimulationApp ─────────────────────────────────────────────
import carb
from pxr import Usd, UsdGeom, Gf, UsdPhysics, UsdLux, Sdf, PhysxSchema, UsdShade
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid, DynamicCuboid
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot_motion.motion_generation.interface_config_loader import load_supported_lula_kinematics_solver_config
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller import PickPlaceController
import omni.replicator.core as rep

# FastAPI for live camera streaming and simulation control
from fastapi import FastAPI, Response, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ── 3. Build Isaac Sim World & Stage ───────────────────────────────────────────
print("═══════════════════════════════════════════════════════════════════", flush=True)
print("  [Isaac Sim Service] Initializing Franka Panda Simulation Engine  ", flush=True)
print("═══════════════════════════════════════════════════════════════════", flush=True)

world = World(
    stage_units_in_meters=1.0,
    physics_dt=1.0 / 60.0,
    rendering_dt=1.0 / 60.0,
)
stage = world.stage
assets_root = get_assets_root_path()
if assets_root is None:
    print("[ERROR] Nucleus asset path could not be resolved.", flush=True)
    simulation_app.close()
    sys.exit(1)

# ── 3.0. PhysX Scene Tuning — real solver settings for stable rigid-body grasping ──
# (Previously left entirely at PhysX defaults despite doc comments claiming "calibrated"
#  physics; TGS solver + explicit iteration counts + CCD materially improve grasp stability
#  once joints are driven by real PD targets instead of being kinematically snapped — see
#  the execution loop below.)
try:
    physics_context = world.get_physics_context()
    physics_context.set_solver_type("TGS")
    physics_context.enable_ccd(True)
    scene_prim = physics_context.get_current_physics_scene_prim()
    physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
    physx_scene_api.CreateMinPositionIterationCountAttr(16)
    physx_scene_api.CreateMinVelocityIterationCountAttr(4)
    print("[Isaac Sim Service] PhysX scene tuned: TGS solver, CCD on, 16/4 solver iterations.", flush=True)
except Exception as e:
    print(f"[WARN] PhysX scene tuning failed, using engine defaults: {e}", flush=True)

# ── 3.0.1. Physics Materials — real friction for grasp contact and table/tray surfaces ──
# (Previously no PhysicsMaterial was assigned anywhere; grasp/slide behavior was governed
#  entirely by unset PhysX defaults.)
grasp_friction_material = PhysicsMaterial(
    prim_path="/World/Physics/GraspFrictionMaterial",
    static_friction=1.05,
    dynamic_friction=0.95,
    restitution=0.0,
)
surface_friction_material = PhysicsMaterial(
    prim_path="/World/Physics/SurfaceFrictionMaterial",
    static_friction=0.70,
    dynamic_friction=0.60,
    restitution=0.0,
)

# ── 3.1. Studio Lighting System with Calibrated Presets ────────────────────────
dome_light = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
overhead_light = UsdLux.DiskLight.Define(stage, "/World/Lights/OverheadLight")
key_light = UsdLux.DistantLight.Define(stage, "/World/Lights/KeyLight")
fill_light = UsdLux.SphereLight.Define(stage, "/World/Lights/FillLight")

UsdGeom.Xformable(overhead_light).AddTranslateOp().Set(Gf.Vec3d(0.46, 0.0, 1.85))
UsdGeom.Xformable(overhead_light).AddRotateXYZOp().Set(Gf.Vec3f(0, 0, 0))
overhead_light.CreateRadiusAttr(0.85)

xf = UsdGeom.Xformable(key_light)
xf.AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))

UsdGeom.Xformable(fill_light).AddTranslateOp().Set(Gf.Vec3d(1.10, -0.80, 1.30))
fill_light.CreateRadiusAttr(0.40)

current_lighting_preset = "studio"

def set_lighting_preset(preset: str):
    global current_lighting_preset
    preset = preset.lower().strip()
    
    if preset == "high_contrast":
        dome_light.GetIntensityAttr().Set(250.0)
        overhead_light.GetIntensityAttr().Set(3400.0)
        overhead_light.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))
        key_light.GetIntensityAttr().Set(2400.0)
        key_light.GetColorAttr().Set(Gf.Vec3f(0.95, 0.98, 1.0))
        fill_light.GetIntensityAttr().Set(250.0)
        if 'wrist_light' in globals(): wrist_light.GetIntensityAttr().Set(400.0)
    elif preset == "warm":
        dome_light.GetIntensityAttr().Set(500.0)
        overhead_light.GetIntensityAttr().Set(2200.0)
        overhead_light.GetColorAttr().Set(Gf.Vec3f(1.0, 0.88, 0.72))
        key_light.GetIntensityAttr().Set(1500.0)
        key_light.GetColorAttr().Set(Gf.Vec3f(1.0, 0.82, 0.65))
        fill_light.GetIntensityAttr().Set(450.0)
        if 'wrist_light' in globals(): wrist_light.GetIntensityAttr().Set(320.0)
    elif preset == "cyberpunk":
        dome_light.GetIntensityAttr().Set(450.0)
        overhead_light.GetIntensityAttr().Set(2000.0)
        overhead_light.GetColorAttr().Set(Gf.Vec3f(0.0, 0.90, 1.0))
        key_light.GetIntensityAttr().Set(2400.0)
        key_light.GetColorAttr().Set(Gf.Vec3f(1.0, 0.05, 0.70))
        fill_light.GetIntensityAttr().Set(1100.0)
        if 'wrist_light' in globals(): wrist_light.GetIntensityAttr().Set(500.0)
    elif preset == "neutral":
        dome_light.GetIntensityAttr().Set(1100.0)
        overhead_light.GetIntensityAttr().Set(1200.0)
        overhead_light.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))
        key_light.GetIntensityAttr().Set(900.0)
        key_light.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))
        fill_light.GetIntensityAttr().Set(450.0)
        if 'wrist_light' in globals(): wrist_light.GetIntensityAttr().Set(280.0)
    else: # "studio" default
        preset = "studio"
        dome_light.GetIntensityAttr().Set(700.0)
        overhead_light.GetIntensityAttr().Set(2400.0)
        overhead_light.GetColorAttr().Set(Gf.Vec3f(1.0, 0.98, 0.95))
        key_light.GetIntensityAttr().Set(1600.0)
        key_light.GetColorAttr().Set(Gf.Vec3f(1.0, 0.95, 0.90))
        fill_light.GetIntensityAttr().Set(550.0)
        if 'wrist_light' in globals(): wrist_light.GetIntensityAttr().Set(380.0)
        
    current_lighting_preset = preset
    print(f"[Isaac Sim Service] Lighting Preset changed to: {preset.upper()}", flush=True)

# ── 3.2. Static Workcell & Solid Manipulation Table (FixedCuboid) ──────────────
# Ground plane lowered below the table's own bottom face (table spans Z=[-0.10, 0.00] — see
# below). Previously left at its z_position=0 default, which is exactly coplanar with the
# table's TOP surface — the entire table body sat at or below the ground plane, causing
# z-fighting at that shared 0.00 plane (the reported "blurry" table) and making the table
# read as sunk into the ground instead of a raised object resting on it. Only the ground
# plane moves here — the table's own top surface stays at Z=0.00, so every downstream
# constant calibrated to that height (cube rest height, tray height, IK/pick-place targets,
# etc.) is untouched.
world.scene.add_default_ground_plane(z_position=-0.12)

# The default ground plane loads a fixed USD environment asset (Isaac/Environments/Grid/
# default_environment.usd) whose light-blue grid material isn't exposed as a color param on
# add_default_ground_plane() — it has to be retinted directly on the loaded material. Reported
# issue: that light blue is close enough to the blue cube's own color to be visually
# confusing in camera frames. Isaac Sim's stock grid material conventionally exposes a
# "diffuse_tint" shader input for exactly this kind of recolor (multiplies the baked grid
# texture rather than replacing it, so the tile pattern is kept, just darker/different hue).
# Defensive: walks the ground plane's prim tree for a Shader with that input; if the asset's
# material doesn't expose it (e.g. a different Isaac Sim version), this just logs and leaves
# the default look rather than crashing startup over a cosmetic tweak.
def _retint_ground_plane(prim_path: str, tint_rgb=(0.10, 0.16, 0.34)):
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        print(f"[Isaac Sim Service] WARN: ground plane prim '{prim_path}' not found — skipping retint.", flush=True)
        return
    tinted = 0
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)
        for attr_name in ("inputs:diffuse_tint", "inputs:diffuse_color_constant"):
            shader_input = shader.GetInput(attr_name.split(":")[1])
            if shader_input and shader_input.GetAttr().IsValid():
                shader_input.Set(Gf.Vec3f(*tint_rgb))
                tinted += 1
    if tinted:
        print(f"[Isaac Sim Service] Ground plane retinted (darker blue, {tinted} shader input(s) set).", flush=True)
    else:
        print("[Isaac Sim Service] WARN: no diffuse_tint/diffuse_color_constant input found on "
              "ground plane material — left at its default color.", flush=True)

_retint_ground_plane("/World/defaultGroundPlane")

# Fixed Static Work Table (Immovable, rock-solid surface at Z = 0.00)
work_table_obj = world.scene.add(
    FixedCuboid(
        prim_path="/World/WorkTable",
        name="work_table",
        position=np.array([0.45, 0.0, -0.05]),
        scale=np.array([0.90, 1.30, 0.10]),
        color=np.array([0.16, 0.18, 0.22]),
        physics_material=surface_friction_material,
    )
)

# Fixed Static Target Sorting Tray (Immovable receptacle on left side)
target_tray_obj = world.scene.add(
    FixedCuboid(
        prim_path="/World/TargetTray",
        name="target_tray",
        position=np.array([0.42, -0.32, 0.015]),
        scale=np.array([0.22, 0.24, 0.03]),
        color=np.array([0.95, 0.72, 0.12]),
        physics_material=surface_friction_material,
    )
)

# Active Physical Target Cubes (0.05 kg stable mass, placed flush on table Z = 0.025)
red_cube_obj = world.scene.add(
    DynamicCuboid(
        prim_path="/World/RedCube",
        name="red_cube",
        position=np.array([0.50, 0.00, 0.025]),
        scale=np.array([0.050, 0.050, 0.050]),
        color=np.array([0.92, 0.18, 0.18]),
        mass=0.05,
        physics_material=grasp_friction_material,
    )
)

blue_cube_obj = world.scene.add(
    DynamicCuboid(
        prim_path="/World/BlueCube",
        name="blue_cube",
        position=np.array([0.45, 0.22, 0.025]),
        scale=np.array([0.050, 0.050, 0.050]),
        color=np.array([0.18, 0.48, 0.95]),
        mass=0.05,
        physics_material=grasp_friction_material,
    )
)

green_cube_obj = world.scene.add(
    DynamicCuboid(
        prim_path="/World/GreenCube",
        name="green_cube",
        position=np.array([0.56, 0.10, 0.025]),
        scale=np.array([0.050, 0.050, 0.050]),
        color=np.array([0.15, 0.85, 0.35]),
        mass=0.05,
        physics_material=grasp_friction_material,
    )
)

# Tighter contact/rest offsets on the cubes give the solver a smaller interpenetration
# tolerance to resolve during a grasp, reducing jitter/popping now that the fingers are
# driven by real PD targets (see below) rather than being kinematically snapped closed.
for _cube_path in ("/World/RedCube", "/World/BlueCube", "/World/GreenCube"):
    try:
        _cube_prim = stage.GetPrimAtPath(_cube_path)
        _physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(_cube_prim)
        _physx_collision_api.CreateContactOffsetAttr(0.004)
        _physx_collision_api.CreateRestOffsetAttr(0.001)
    except Exception as e:
        print(f"[WARN] Could not tune collision offsets for {_cube_path}: {e}", flush=True)

# ── Dynamic Scene Objects Registry for Real-Time USD Queries ──────────────────
SCENE_OBJECTS = {
    "red_cube": {
        "path": "/World/RedCube",
        "name": "Red Cube",
        "keywords": ["red", "red cube", "red block", "crimson cube"],
        "default_pos": np.array([0.50, 0.00, 0.025]),
    },
    "blue_cube": {
        "path": "/World/BlueCube",
        "name": "Blue Cube",
        "keywords": ["blue", "blue cube", "blue block", "azure cube"],
        "default_pos": np.array([0.45, 0.22, 0.025]),
    },
    "green_cube": {
        "path": "/World/GreenCube",
        "name": "Green Cube",
        "keywords": ["green", "green cube", "emerald cube", "green block"],
        "default_pos": np.array([0.56, 0.10, 0.025]),
    },
    "target_tray": {
        "path": "/World/TargetTray",
        "name": "Target Tray",
        "keywords": ["tray", "bin", "box", "basket", "target", "place", "container", "sort", "drop"],
        "default_pos": np.array([0.42, -0.32, 0.02]),
    }
}

def get_live_object_pos(obj_key: str) -> np.ndarray:
    """Dynamically queries the live 3D world coordinates of any object from USD."""
    try:
        obj_info = SCENE_OBJECTS.get(obj_key)
        if obj_info:
            prim = stage.GetPrimAtPath(obj_info["path"])
            if prim.IsValid():
                xform = UsdGeom.Xformable(prim)
                mat = xform.ComputeLocalToWorldTransform(0)
                trans = mat.ExtractTranslation()
                return np.array([trans[0], trans[1], trans[2]], dtype=np.float32)
    except Exception as e:
        print(f"[USD Transform Query Error] {e}", flush=True)
    return SCENE_OBJECTS[obj_key]["default_pos"]

def get_end_effector_pos() -> np.ndarray:
    """Live world-space position of the gripper end effector (panda_rightfinger frame).
    Previously this was a hardcoded, never-updated placeholder in telemetry."""
    try:
        prim = stage.GetPrimAtPath("/World/Franka/panda_rightfinger")
        if prim.IsValid():
            xform = UsdGeom.Xformable(prim)
            mat = xform.ComputeLocalToWorldTransform(0)
            trans = mat.ExtractTranslation()
            return np.array([trans[0], trans[1], trans[2]], dtype=np.float32)
    except Exception as e:
        print(f"[USD Transform Query Error] {e}", flush=True)
    return np.array([0.45, 0.0, 0.40], dtype=np.float32)

def reset_scene_cubes():
    """Resets all 3 cubes to their designated staging spots on the table with zero velocity."""
    for key, data in SCENE_OBJECTS.items():
        if key == "target_tray": continue
        try:
            obj = world.scene.get_object(key)
            if obj is not None:
                def_pos = data["default_pos"]
                obj.set_world_pose(position=np.array([def_pos[0], def_pos[1], 0.025]), orientation=np.array([1.0, 0.0, 0.0, 0.0]))
                obj.set_linear_velocity(np.zeros(3))
                obj.set_angular_velocity(np.zeros(3))
        except Exception as e:
            print(f"[Reset Obj Error] {key}: {e}", flush=True)
    print("[Isaac Sim Service] All scene cubes reset to designated table coordinates.", flush=True)

# Reachable-workspace bounds for random cube placement — clear of the tray footprint
# (X=[0.31,0.53], Y=[-0.44,-0.20]) and comfortably within arm reach.
CUBE_RANDOM_X_RANGE = (0.28, 0.65)
CUBE_RANDOM_Y_RANGE = (-0.15, 0.55)
CUBE_MIN_SEPARATION = 0.10  # meters between cube centers, so they don't spawn overlapping

def randomize_scene_cubes():
    """Places all 3 cubes at genuinely random XY positions each call (np.random, not a fixed
    list) within the arm's reachable workspace, clear of the tray, with a minimum separation
    so they don't spawn on top of each other."""
    placed = []
    for key in SCENE_OBJECTS:
        if key == "target_tray":
            continue
        try:
            obj = world.scene.get_object(key)
            if obj is None:
                continue
            x, y = None, None
            for _attempt in range(50):
                cx = float(np.random.uniform(*CUBE_RANDOM_X_RANGE))
                cy = float(np.random.uniform(*CUBE_RANDOM_Y_RANGE))
                if all(math.hypot(cx - px, cy - py) >= CUBE_MIN_SEPARATION for px, py in placed):
                    x, y = cx, cy
                    break
            if x is None:  # 50 rejection-sampling attempts failed (very unlikely with 3 cubes)
                x, y = cx, cy
            placed.append((x, y))
            obj.set_world_pose(position=np.array([x, y, 0.025]), orientation=np.array([1.0, 0.0, 0.0, 0.0]))
            obj.set_linear_velocity(np.zeros(3))
            obj.set_angular_velocity(np.zeros(3))
        except Exception as e:
            print(f"[Randomize Obj Error] {key}: {e}", flush=True)
    print(f"[Isaac Sim Service] Cubes randomized to: {[(round(x,3), round(y,3)) for x,y in placed]}", flush=True)

# ── 3.3. Franka Panda Articulation ─────────────────────────────────────────────
print("[Isaac Sim Service] Spawning Franka Panda Robot...", flush=True)
franka_usd = assets_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
add_reference_to_stage(usd_path=franka_usd, prim_path="/World/Franka")

# NVIDIA's own validated Franka pick-and-place example
# (standalone_examples/deprecated/api/isaacsim.robot.manipulators/franka_pick_up.py)
# explicitly selects the "AlternateFinger" gripper variant before grasping — the default
# variant on this asset is apparently not the one intended for real physical grasping
# (consistent with panda_finger_joint2 reporting kp=kd=maxforce=0 by default, see the gain
# diagnostics below). Select it here, before robot.initialize(), so the driven-finger
# geometry/drive setup NVIDIA validated is what we're actually simulating.
try:
    franka_prim = stage.GetPrimAtPath("/World/Franka")
    gripper_variant_set = franka_prim.GetVariantSet("Gripper")
    available_variants = gripper_variant_set.GetVariantNames()
    print(f"[Isaac Sim Service] Franka 'Gripper' variants available: {available_variants}", flush=True)
    if "AlternateFinger" in available_variants:
        gripper_variant_set.SetVariantSelection("AlternateFinger")
        print("[Isaac Sim Service] Selected 'AlternateFinger' gripper variant (matches NVIDIA's validated pick-place demo).", flush=True)
except Exception as e:
    print(f"[WARN] Could not select Franka gripper variant, using default: {e}", flush=True)

# Use NVIDIA's own validated manipulator/gripper wrapper (isaacsim.robot.manipulators)
# instead of a bare Articulation view — this is what their official
# franka_pick_up.py pick-and-place demo builds on, and it's required to drive
# NVIDIA's PickPlaceController (real RMPFlow collision-aware motion + a validated
# 10-phase pick/place state machine) instead of our own hand-rolled waypoint
# interpolation, which repeated tuning attempts couldn't get to a reliable grasp.
franka_gripper = ParallelGripper(
    end_effector_prim_path="/World/Franka/panda_rightfinger",
    joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
    joint_opened_positions=np.array([0.04, 0.04]),
    joint_closed_positions=np.array([0.0, 0.0]),
    action_deltas=np.array([0.01, 0.01]),
)
robot = world.scene.add(
    SingleManipulator(
        prim_path="/World/Franka",
        name="franka_robot",
        end_effector_prim_path="/World/Franka/panda_rightfinger",
        gripper=franka_gripper,
    )
)

# Add dedicated Work Light attached to robot hand to illuminate the grasping zone
wrist_light = UsdLux.SphereLight.Define(stage, "/World/Franka/panda_hand/WristLight")
wrist_light.CreateIntensityAttr(380.0)
wrist_light.CreateRadiusAttr(0.05)
wrist_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 0.98))
UsdGeom.Xformable(wrist_light).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.06))

# Apply default studio lighting
set_lighting_preset("studio")

# ── 4. HD Multi-Perspective Cameras with Optimal Cinematic Framing ────────────
# Fix note: these 4 cameras previously all aimed at (~0.46, 0, ~0.15) — the middle of the
# table/cubes only — while the Franka's base sits at world origin (0,0,0), well outside that
# look_at point. Combined with a fairly tight ~47° FOV (default focal_length=24mm) and cameras
# positioned close in, the robot itself was mostly cropped out of frame (only its base
# visible) and everything read as zoomed-in on just the tray/cubes. The front camera was worst:
# positioned at X=1.20 — almost at the table's own far edge (table spans X=[0, 0.9]) — so its
# frame was dominated by the table's own dark top surface (color ~ near-black) filling half
# the shot, not a rendering bug.
# Fix: share one look_at point roughly between the robot body and the workspace (not just the
# workspace), pull every camera back further for real clearance, and widen the lens
# (focal_length 18mm vs the previous 24mm default) so the whole robot + table + tray fits.
#
# NOTE: these 4 cameras were briefly widened further (18mm -> 14mm, top 20mm -> 15mm) to
# chase a table-edge visibility complaint, but that complaint turned out to be specifically
# about the WRIST camera (see WristCamera below) — reverted back to 18mm/20mm.
#
# Then nudged the other direction (18mm -> 22mm, top 20mm -> 24mm): reported that these 4
# views read as too far/zoomed-out. Longer focal_length = narrower FOV = more zoomed in, at
# the same camera position — a mild bump, not the aggressive framing change further up this
# comment's history. Wrist camera is untouched by this — stays at its own separately-tuned
# 7mm (see WristCamera below).
RIG_LOOK_AT = (0.35, 0.0, 0.28)

# 1. Overview Camera (Isometric 3/4 view, rich cinematic framing of robot, table, cubes, and tray)
scene_cam = rep.create.camera(position=(1.85, -1.55, 1.30), look_at=RIG_LOOK_AT, focal_length=22.0)

# 2. Front Camera (Direct operator viewpoint, pulled back so the table's dark top surface
#    doesn't dominate the frame, and raised/re-aimed so the robot is actually in shot)
front_cam = rep.create.camera(position=(1.95, 0.0, 0.90), look_at=RIG_LOOK_AT, focal_length=22.0)

# 3. Side Camera (Profile view showing vertical height clearance and depth transit)
side_cam = rep.create.camera(position=(0.35, -1.75, 0.85), look_at=RIG_LOOK_AT, focal_length=22.0)

# 4. Top-Down Camera (Bird's eye planar layout — robot base through to the tray)
top_cam = rep.create.camera(position=(0.35, 0.0, 2.00), look_at=(0.35, 0.0, 0.05), focal_length=24.0)

# 5. Wrist Camera (Mounted on panda_hand, offset to the side so a grasped object sits beside
#    the frame instead of dead-center blocking it — see fix history below)
#
# BUG #1 (original): mounted 0.07m out along the hand's Z axis, but the actual fingertip
# grasp point (TCP) is only 0.0584m out (the same "0.0584m Fingertip TCP Offset" used
# elsewhere in this file for IK targeting). 0.07m > 0.0584m put the camera physically PAST
# where a held object sits — confirmed by capturing an actual frame mid-grasp: solid red,
# no scene visible, the object's near face pressed against the lens.
#
# BUG #2 (first fix attempt): pulling the camera back to a NEGATIVE Z offset (behind the hand
# origin, toward the arm) put it inside the robot's own solid hand geometry instead — confirmed
# by capturing a frame there too: solid black, camera embedded in the robot body.
#
# Real fix: the actual problem is the camera being COAXIAL with the object — centered on the
# exact line the gripper closes along, so anything grasped sits dead-center against the lens
# no matter the distance along that same line. Offsetting sideways (lateral X) puts a held
# object off to one side of the frame instead of filling it, while keeping the mount at a
# modest positive Z (short of the 0.0584m TCP point, so nothing is ever "inside" a held
# object) and widening the lens for extra context margin.
wrist_cam_prim = stage.DefinePrim("/World/Franka/panda_hand/WristCamera", "Camera")
# Widened 12mm -> 7mm: reported that cubes near the table edges were falling out of the
# wrist camera's frame during approach/transit — shorter focal_length = wider FOV at the
# same aperture (same technique used for the other 4 cameras above), without moving the
# mount position, so the coaxial-object framing fix in the comment below still holds.
wrist_cam_prim.GetAttribute("focalLength").Set(7.0)
wrist_cam_prim.GetAttribute("horizontalAperture").Set(20.955)
wrist_cam_prim.GetAttribute("clippingRange").Set(Gf.Vec2f(0.005, 10.0))
wc_xf = UsdGeom.Xformable(wrist_cam_prim)
wc_xf.AddTranslateOp().Set(Gf.Vec3d(0.05, 0.0, 0.02))
wc_xf.AddRotateXYZOp().Set(Gf.Vec3f(180, 0, 0))

# Attach HD 540p Replicator Render Products
CAM_RES = (960, 540)
rp_scene = rep.create.render_product(scene_cam, CAM_RES)
scene_annot = rep.AnnotatorRegistry.get_annotator("rgb")
scene_annot.attach([rp_scene])

# ── Perception camera plumbing (Cosmos-Reason2 integration, Phase A "camera plumbing" step
# — see cosmos_integration.md) ──────────────────────────────────────────────────────────
# Depth for the SAME overview camera/render product RGB already streams from, rather than a
# second render product (per the integration doc's own advice: two render products at this
# resolution is real extra GPU/FPS cost for no reason when one camera can carry both).
# distance_to_image_plane is z-depth along the optical axis (what the pinhole equations
# assume), not distance_to_camera (euclidean range, which biases estimates outward toward
# the image edges) — same distinction the integration doc calls out explicitly.
scene_depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
scene_depth_annot.attach([rp_scene])

# Intrinsics: computed analytically from this camera's own known creation parameters
# (standard pinhole projection), not queried from a live API — deliberate choice over
# guessing at a Replicator "camera params" annotator's exact name/schema for this Isaac Sim
# version without being able to verify it live first. horizontalAperture=20.955mm is USD's
# own Camera schema default (the same constant already used explicitly for WristCamera
# above) — scene_cam doesn't override it, so it applies here too.
# NOTE: must be kept in sync with scene_cam's own focal_length above (22.0) — this is a
# separate duplicate constant since rep.create.camera() doesn't expose a getter, so there's
# no single source of truth to read back from instead of hand-matching it here.
_SCENE_CAM_FOCAL_LENGTH_MM = 22.0
_SCENE_CAM_HORIZONTAL_APERTURE_MM = 20.955
_scene_cam_fx = (_SCENE_CAM_FOCAL_LENGTH_MM / _SCENE_CAM_HORIZONTAL_APERTURE_MM) * CAM_RES[0]
_scene_cam_fy = _scene_cam_fx  # square pixels, matches every other camera in this file
SCENE_CAM_INTRINSICS = {
    "fx": _scene_cam_fx, "fy": _scene_cam_fy,
    "cx": CAM_RES[0] / 2.0, "cy": CAM_RES[1] / 2.0,
    "width": CAM_RES[0], "height": CAM_RES[1],
}

# Extrinsics: this camera never moves (fixed position/look_at, set once above), so its world
# transform is computed once, analytically, from those exact same values via a standard
# look-at construction — USD/Replicator cameras look down their own local -Z axis, and this
# stage's world up-axis is Z (confirmed by this file's own top_cam, positioned high on Z and
# looking straight down). If SCENE_CAM_POSITION/SCENE_CAM_LOOK_AT below ever change, this
# must be recomputed alongside them — they're kept in one place for exactly that reason.
SCENE_CAM_POSITION = np.array([1.85, -1.55, 1.30])
SCENE_CAM_LOOK_AT = RIG_LOOK_AT  # (0.35, 0.0, 0.28) — same point scene_cam was created with


def _compute_lookat_world_transform(eye: np.ndarray, target: np.ndarray, world_up: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    """Standard look-at camera-to-world 4x4 transform. Camera looks down local -Z (USD
    convention); returns world = R @ local + t, i.e. columns of R are the camera's local
    X/Y/Z axes expressed in world space."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    rot = np.column_stack([right, up, -forward])  # camera-local X, Y, Z axes in world space
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = eye
    return transform


SCENE_CAM_WORLD_TRANSFORM = _compute_lookat_world_transform(SCENE_CAM_POSITION, np.array(SCENE_CAM_LOOK_AT))

# Latest depth frame + a timestamp, updated on the same telemetry cadence as everything else
# (see the main loop's "5. Update Telemetry" block) — plain in-process storage since nothing
# outside this process consumes it yet (no HTTP depth streaming added in this plumbing pass).
perception_camera_state = {
    "intrinsics": SCENE_CAM_INTRINSICS,
    "world_transform": SCENE_CAM_WORLD_TRANSFORM.tolist(),
    "depth_shape": None,
    "depth_min_m": None,
    "depth_max_m": None,
    "depth_center_m": None,
    "timestamp": 0.0,
}

# ── Cosmos-Reason2 provider, shadow mode (cosmos_integration.md Step 5) ──────────────────
# Wired in purely additively: get_object_pose() below always returns the same stage
# (ground-truth USD) pose in "stage"/"shadow" modes, so resolve_pick_place_targets()'s own
# real dispatch/decision logic (unmodified) still drives the robot exactly as before. The
# only new thing this adds is a real Cosmos grounding call per dispatched pick/place
# instruction, purely for logging a delta via /perception — zero effect on robot behavior
# unless PERCEPTION_MODE=vision is explicitly set.
from physyk_perception.provider import PerceptionProvider

_PERCEPTION_LABEL_MAP = {
    "red_cube": "red cube", "blue_cube": "blue cube",
    "green_cube": "green cube", "target_tray": "tray",
}
_PERCEPTION_WORKSPACE_AABB = ((-0.2, 1.2), (-0.8, 0.8), (-0.05, 0.6))
# Tight box around the actual WorkTable (center [0.45,0,-0.05], scale [0.90,1.30,0.10] ->
# spans X=[0,0.9], Y=[-0.65,0.65], top Z=0.00 — see work_table_obj above), with a little
# margin above for cube/tray height — used only to crop the grounding frame (see
# PerceptionProvider's crop_aabb docstring), never to reject a pose (that's the broader
# _PERCEPTION_WORKSPACE_AABB above).
_PERCEPTION_CROP_AABB = ((0.25, 0.72), (-0.42, 0.32), (-0.02, 0.12))
perception_provider = PerceptionProvider(
    get_rgb=lambda: scene_annot.get_data(),
    get_depth=lambda: scene_depth_annot.get_data(),
    intrinsics=SCENE_CAM_INTRINSICS,
    world_transform=SCENE_CAM_WORLD_TRANSFORM,
    stage_lookup=get_live_object_pos,
    label_map=_PERCEPTION_LABEL_MAP,
    mode=os.environ.get("PERCEPTION_MODE", "shadow"),
    workspace_aabb=_PERCEPTION_WORKSPACE_AABB,
    crop_aabb=_PERCEPTION_CROP_AABB,
)
print(f"[Perception] Cosmos provider initialized in '{perception_provider.mode}' mode", flush=True)

rp_front = rep.create.render_product(front_cam, CAM_RES)
front_annot = rep.AnnotatorRegistry.get_annotator("rgb")
front_annot.attach([rp_front])

rp_side = rep.create.render_product(side_cam, CAM_RES)
side_annot = rep.AnnotatorRegistry.get_annotator("rgb")
side_annot.attach([rp_side])

rp_top = rep.create.render_product(top_cam, CAM_RES)
top_annot = rep.AnnotatorRegistry.get_annotator("rgb")
top_annot.attach([rp_top])

rp_wrist = rep.create.render_product("/World/Franka/panda_hand/WristCamera", (960, 540))
wrist_annot = rep.AnnotatorRegistry.get_annotator("rgb")
wrist_annot.attach([rp_wrist])

# Reset and warmup physics
world.reset()
robot.initialize()

HOME_JOINTS = np.array([0.00, -0.785, 0.00, -2.356, 0.00, 1.571, 0.785, 0.04, 0.04])
robot.set_joint_positions(HOME_JOINTS)
reset_scene_cubes()

for _ in range(15):
    world.step(render=True)

print(f"[Isaac Sim Service] Robot DOFs ({robot.num_dof}): {robot.dof_names}", flush=True)

# ── 4.4. Finger Joint Drive Tuning — real PD gains for a firm, fast grasp close ────
# Now that fingers are driven by real position targets (not kinematically snapped), the
# stock drive gains determine whether the gripper can generate real squeeze force fast
# enough to hold an object before the arm starts lifting. Query current gains for
# diagnostics, then raise stiffness/damping/effort limit on the two finger DOFs only.
FINGER_JOINT_INDICES = np.array([7, 8])  # panda_finger_joint1, panda_finger_joint2
try:
    # SingleManipulator/SingleArticulation don't expose get_gains/set_gains directly —
    # they wrap a real (batched-view) Articulation internally as `_articulation_view`,
    # which does.
    articulation_view = robot._articulation_view
    all_kps, all_kds = articulation_view.get_gains()
    all_efforts = articulation_view.get_max_efforts()
    print(f"[Isaac Sim Service] Stock joint gains — kps: {np.round(np.array(all_kps).flatten(), 1)}", flush=True)
    print(f"[Isaac Sim Service] Stock joint gains — kds: {np.round(np.array(all_kds).flatten(), 1)}", flush=True)
    print(f"[Isaac Sim Service] Stock joint max efforts: {np.round(np.array(all_efforts).flatten(), 2)}", flush=True)

    # Only override gains if the "AlternateFinger" variant switch above did NOT already
    # give both fingers a real drive (i.e. only patch what's still broken). Blindly forcing
    # gains on both fingers regardless of the asset's own (validated-by-NVIDIA) defaults is
    # what caused the earlier overshoot/too-sluggish tuning failures.
    finger_kps_now = np.array(all_kps).flatten()[FINGER_JOINT_INDICES]
    finger_efforts_now = np.array(all_efforts).flatten()[FINGER_JOINT_INDICES]
    if np.any(finger_kps_now <= 0.0) or np.any(finger_efforts_now <= 0.0):
        finger_kps = np.array([[400.0, 400.0]], dtype=np.float32)
        finger_kds = np.array([[80.0, 80.0]], dtype=np.float32)
        finger_efforts = np.array([[20.0, 20.0]], dtype=np.float32)
        articulation_view.set_gains(kps=finger_kps, kds=finger_kds, joint_indices=FINGER_JOINT_INDICES)
        articulation_view.set_max_efforts(finger_efforts, joint_indices=FINGER_JOINT_INDICES)
        print("[Isaac Sim Service] One or both fingers still had no drive after variant selection — "
              "applied finger1's own stock gains (kp=400, kd=80) to both, effort raised to 20N.", flush=True)
    else:
        print("[Isaac Sim Service] Both fingers already have a real drive after variant selection — leaving stock gains as-is.", flush=True)
except Exception as e:
    print(f"[WARN] Could not tune finger joint drive gains, using stock values: {e}", flush=True)

# ── 4.5. Initialize NVIDIA Lula Kinematics Solver ──────────────────────────────
print("[Isaac Sim Service] Initializing Native Lula Kinematics Solver for Franka...", flush=True)
kinematics_config = load_supported_lula_kinematics_solver_config("Franka")
lula_solver = LulaKinematicsSolver(**kinematics_config)

# ── 4.6. NVIDIA's Validated Pick-Place Controller (RMPFlow cspace + gripper state machine) ──
# Replaces our own hand-rolled waypoint-interpolation grasp logic, which repeated tuning
# passes could not get to a reliable lift. PickPlaceController's cspace_controller is
# RMPFlow (real collision-aware motion generation — the Phase 2 goal) and its gripper
# close/open + timing (10-phase state machine) is NVIDIA's own validated sequencing.
print("[Isaac Sim Service] Initializing PickPlaceController (RMPFlow + validated grasp state machine)...", flush=True)
# Speed up pure-motion phases by 25% (larger events_dt = fewer physics steps per phase =
# faster, over the same travel distance = higher average speed) while leaving the three
# contact-critical phases at NVIDIA's validated timing:
#   phase 2 "settling before grasp", phase 3 "closing gripper", phase 7 "releasing gripper".
# Those are exactly the phases where earlier hand-tuning attempts caused overshoot/knock-away
# failures — rushing them risks reintroducing that. The 7 pure-motion phases (move above,
# descend to grasp, lift, transit, descend to place, ascend, return to staging) cover both
# the empty-gripper travel (0, 1) and the cube-in-gripper travel (4, 5, 6, 8, 9) equally, so
# both get the same +25% speed with no change to grasp reliability.
#
# NOTE: was briefly bumped to 2.25 (avg speed x1.5) but that made the gripper struggle with
# grasping / trajectory planning. 1.5 was then also still too fast, so dialed back further
# to 1.25 below.
#
# Separate fix (not a speed/transit issue): NVIDIA's PickPlaceController advances phases on a
# pure time budget, not on an actual "grip is tight" signal — phase 2 "settle" (dt=1, a single
# frame) and phase 3 "close grip" (dt=0.1, ~10 frames) were completing and handing off to
# phase 4 "lifting, keeping grip tight" before the jaws had actually converged/centered on the
# cube, so the arm would start moving while only holding an edge, which could slip mid-motion.
# Lengthened both so the gripper has real time to settle and fully close before lift begins.
_FRANKA_DEFAULT_EVENTS_DT = [0.008, 0.005, 0.2, 0.04, 0.05, 0.05, 0.0025, 1, 0.008, 0.08]
_MOTION_ONLY_PHASES = {0, 1, 4, 5, 6, 8, 9}  # excludes 2 (settle), 3 (close), 7 (release)
SPEEDUP_FACTOR = 1.25  # dt scaled by 1.25x on motion-only phases -> slower than the earlier 1.5
events_dt = [
    dt * SPEEDUP_FACTOR if i in _MOTION_ONLY_PHASES else dt
    for i, dt in enumerate(_FRANKA_DEFAULT_EVENTS_DT)
]
pick_place_controller = PickPlaceController(
    name="pick_place_controller",
    gripper=robot.gripper,
    robot_articulation=robot,
    events_dt=events_dt,
)
articulation_controller = robot.get_articulation_controller()

# ── 4.7. Register Real Collision Obstacles with RMPFlow (Phase 2: collision-aware planning) ──
# RMPFlow does NOT automatically know about scene geometry — obstacles must be explicitly
# registered, or it plans paths blind to the table/tray/other cubes (per
# isaacsim.robot_motion.motion_generation's WorldInterface.add_obstacle docs). The table and
# tray are permanent, always-on static obstacles. The three cubes are also registered so the
# arm routes around whichever ones it isn't currently grasping — but the *current pick target*
# must be temporarily disabled each task, or RMPFlow would refuse to let the gripper
# approach/enclose it at all.
rmpflow_motion_policy = pick_place_controller._cspace_controller.get_motion_policy()
CUBE_OBJECTS = {"red_cube": red_cube_obj, "blue_cube": blue_cube_obj, "green_cube": green_cube_obj}
try:
    rmpflow_motion_policy.add_obstacle(work_table_obj, static=True)
    rmpflow_motion_policy.add_obstacle(target_tray_obj, static=True)
    for cube_obj in CUBE_OBJECTS.values():
        rmpflow_motion_policy.add_obstacle(cube_obj, static=False)
    print("[Isaac Sim Service] Registered table, tray, and all 3 cubes as RMPFlow collision obstacles.", flush=True)
except Exception as e:
    print(f"[WARN] Could not register RMPFlow obstacles, planning will be collision-blind: {e}", flush=True)

# ── 5. Shared Global State & Command Queue ────────────────────────────────────
# Holds (request_id, instruction) tuples, not bare strings — the request_id is what lets a
# caller (the agentic orchestrator, or a manual GUI command) unambiguously identify which
# queued item is actually theirs once it's dequeued, even against a byte-identical duplicate
# instruction from someone else. See sim_telemetry["current_request_id"].
command_queue = queue.Queue()
lighting_queue = queue.Queue()
_request_id_lock = threading.Lock()
_next_request_id = 0

# Emergency stop — set by the /hard_stop route (GUI's red "Hard Reset" button). Checked at
# the very top of the sim loop every frame, so it preempts whatever's in progress (a
# mid-motion pick_place, an in-flight VLA chunk, or anything still sitting in the queue)
# instead of waiting for the current task to finish on its own. Drops the queue/plan, homes
# the arm, AND resets every cube to its default staging spot — a full known-good-state reset,
# not just a motion abort.
hard_stop_event = threading.Event()

def _drain_command_queue():
    """Empties command_queue without blocking — used by hard-stop so nothing queued before
    the stop request (including a stale duplicate of the very task being aborted) gets
    dequeued and executed right after the arm reaches home."""
    while True:
        try:
            command_queue.get_nowait()
        except queue.Empty:
            break

def _new_request_id() -> int:
    global _next_request_id
    with _request_id_lock:
        _next_request_id += 1
        return _next_request_id

latest_frames = {
    "scene_jpeg": None,
    "front_jpeg": None,
    "side_jpeg": None,
    "top_jpeg": None,
    "wrist_jpeg": None,
    "timestamp": 0.0
}
frame_lock = threading.Lock()

sim_telemetry = {
    "joints": HOME_JOINTS[:7].tolist(),
    "gripper": 0.04,
    "ee_pos": [0.45, 0.0, 0.40],
    "stage": "READY",
    "busy": False,
    "last_instruction": "System Ready",
    "lighting": current_lighting_preset,
    "fps": 30.0,
    # Live per-object positions (Phase 5 scene grounding — see `objects` update below and
    # `resolve_pick_place_targets`'s own get_live_object_pos, same USD query, exposed here so
    # the agentic orchestrator can ground LLM-named objects/destinations against real state
    # instead of guessing coordinates).
    "objects": {},
    # Real post-place verification (Phase 4/5 punch list) — set once per completed pick-place
    # task below; None until the first one finishes. No longer a hardcoded always-PASS stub.
    "last_task_success": None,
    "last_task_error_mm": None,
    # ID of the request currently being processed (see `_next_request_id` / command_queue
    # below) — set the moment it's dequeued, matching whatever /execute returned to the
    # caller that sent it. This is the only way a caller can unambiguously tell "the sim
    # started processing *this specific* dispatch" apart from "still busy with someone
    # else's" or "leftover match from my own earlier identical-text retry" — matching on
    # busy/last_instruction text alone can't distinguish either of those (confirmed as two
    # separate real bugs this session, both from concurrent manual + orchestrated use of this
    # one shared command queue).
    "current_request_id": 0,
    # Real GR00T VLA per-chunk summary (see the VLA control loop below) — None until the
    # first "vla:"-prefixed task actually runs a chunk through the model.
    "vla_reasoning": None,
    "vla_chunks_done": 0,
    # Real grasp-confirmation (gripper's actual achieved width vs GRASP_MIN_WIDTH_M) and
    # object-displacement (did it move a meaningful amount at all) signals — closes a real
    # gap where position-only post-place verification couldn't tell "landed wrong" apart
    # from "grasp failed and nothing really happened", especially for a no-explicit-
    # destination instruction. None until the first pick-place task's grasp phase runs.
    "last_grasp_confirmed": None,
    "last_object_displacement_mm": None,
}
telemetry_lock = threading.Lock()

# ── 6. Instruction → Pick/Place Target Resolution ──────────────────────────────
# The actual grasp motion/timing is now entirely NVIDIA's validated PickPlaceController
# (RMPFlow cspace + 10-phase gripper state machine) — this just resolves which object the
# instruction refers to and what pick/place XYZ positions to hand it, reusing the same
# live-USD-query object matching the old generate_dynamic_trajectory() used.
# EE_OFFSET matches NVIDIA's own franka_pick_up.py demo value for the panda_rightfinger
# end-effector frame.
EE_OFFSET = np.array([0.0, 0.005, 0.0])
CUBE_HALF_HEIGHT = 0.025
# Shared post-place verification tolerances — used by both the deterministic
# PickPlaceController path and the GR00T VLA path (see each's own verification block).
# XY tolerance is half a cube width; Z gets a looser tolerance since a dropped/settled
# cube can still be bouncing/rotating slightly when verification fires.
POSITION_XY_TOLERANCE_M = 0.03
POSITION_Z_TOLERANCE_M = 0.035
# Real grasp-confirmation threshold: cubes are 0.050m across (2*CUBE_HALF_HEIGHT), so a
# successful grasp should settle with the gripper's total opening (2x one finger's joint
# position) somewhere near that. A failed grasp closes down much further, toward the
# gripper's own mechanical minimum, since nothing stops the fingers early. Set well below
# cube width to tolerate a slightly off-center grasp, but well above "closed on nothing".
GRASP_MIN_WIDTH_M = 0.03
# Real "did the object actually move" threshold, independent of whether it landed on
# target — catches a failed grasp on a task with no explicit destination (where the
# intended and actual positions can coincide even with nothing having happened).
MEANINGFUL_DISPLACEMENT_M = 0.05
# TargetTray is a FixedCuboid at position Z=0.015 with scale Z=0.03 (see scene setup above),
# so its live-queried position is its CENTER (0.015), not its top surface (0.03). Using the
# center directly here undershoots the real resting height by the tray's half-thickness —
# the commanded place target ends up *below* the tray's solid top surface, so the arm keeps
# pushing the already-placed cube down against it (visible as vibration/force after a
# successful place) instead of stopping once the cube is actually resting on top.
TRAY_HALF_HEIGHT = 0.015

# Tray is a flat platform (not a walled bin), so multiple cubes just need distinct XY spots on
# it, not literal bin-packing. Previously every placement targeted the exact same fixed tray
# center regardless of what was already there — fine for the first cube, but the second cube
# placed got commanded into the SAME spot as the first, so the arm pushed it straight into the
# already-resting cube instead of recognizing the spot was taken. Three slots spread across the
# tray's X extent (tray spans X=[0.31,0.53]) fixes that.
TRAY_SLOTS = [np.array([0.35, -0.32]), np.array([0.42, -0.32]), np.array([0.49, -0.32])]
# Preference order for _pick_free_tray_slot: center first (index 1, which is also the
# tray's own center coordinate), then left, then right — was previously just 0,1,2 in
# array order, which meant slot 0 (the left EDGE) got picked for every single/first cube
# placed in an empty tray, never the actual center. Confirmed as the real cause of a
# reported "why does it always place near the edge, not the center" — not a controller
# precision issue, just this fixed iteration order.
TRAY_SLOT_PREFERENCE = [1, 0, 2]
TRAY_SLOT_OCCUPIED_RADIUS = 0.055  # a live cube within this of a slot counts as occupying it
TRAY_X_BOUNDS = (0.31, 0.53)
TRAY_Y_BOUNDS = (-0.44, -0.20)

# Relative spatial placement — resolved against another cube's LIVE position (queried fresh via
# get_live_object_pos), not a canned/precomputed coordinate. Direction convention calibrated
# against the front/operator camera (the one a human naturally judges left/right from): world
# +Y is that camera's screen-right (derived from cross(view_forward, world_up), then verified
# empirically — an initial guess had this backwards, confirmed by testing "right of" and
# visually checking the front camera frame, which showed the placed cube on-screen LEFT of the
# reference instead of right; signs below are corrected from that real test, not just theory).
# "in front of"/"behind" offset along X (toward/away from the front-camera/operator side).
# "on top of"/"above" stacks directly above (same XY, +1 cube-height in Z).
SPATIAL_OFFSET = 0.08  # meters, lateral/depth gap for a relative placement
SPATIAL_RELATIONS = [
    # (phrase, (dx, dy, is_stack))
    ("on top of", (0.0, 0.0, True)),
    ("stacked on", (0.0, 0.0, True)),
    ("stack on", (0.0, 0.0, True)),
    ("above", (0.0, 0.0, True)),
    ("to the right of", (0.0, SPATIAL_OFFSET, False)),
    ("right of", (0.0, SPATIAL_OFFSET, False)),
    ("to the left of", (0.0, -SPATIAL_OFFSET, False)),
    ("left of", (0.0, -SPATIAL_OFFSET, False)),
    ("in front of", (SPATIAL_OFFSET, 0.0, False)),
    ("behind", (-SPATIAL_OFFSET, 0.0, False)),
    ("next to", (0.0, SPATIAL_OFFSET, False)),
    ("beside", (0.0, SPATIAL_OFFSET, False)),
]


def _find_object_mentions(inst: str):
    """Every (string_index, object_key) for object keywords found in inst, sorted by position."""
    mentions = []
    for key, data in SCENE_OBJECTS.items():
        if key == "target_tray":
            continue
        for kw in data["keywords"]:
            idx = inst.find(kw)
            if idx != -1:
                mentions.append((idx, key))
    mentions.sort(key=lambda pair: pair[0])
    return mentions


def _pick_free_tray_slot(exclude_key: str) -> np.ndarray:
    """Returns the most-preferred tray slot not already occupied by another live cube —
    center first, then left, then right (see TRAY_SLOT_PREFERENCE)."""
    occupied = set()
    for key in CUBE_OBJECTS:
        if key == exclude_key:
            continue
        pos = get_live_object_pos(key)
        if TRAY_X_BOUNDS[0] <= pos[0] <= TRAY_X_BOUNDS[1] and TRAY_Y_BOUNDS[0] <= pos[1] <= TRAY_Y_BOUNDS[1]:
            dists = [math.hypot(pos[0] - s[0], pos[1] - s[1]) for s in TRAY_SLOTS]
            nearest = int(np.argmin(dists))
            if dists[nearest] < TRAY_SLOT_OCCUPIED_RADIUS:
                occupied.add(nearest)
    for i in TRAY_SLOT_PREFERENCE:
        if i not in occupied:
            return TRAY_SLOTS[i]
    return TRAY_SLOTS[0]  # all 3 slots taken (shouldn't happen with only 3 cubes total)


def resolve_pick_place_targets(instruction: str):
    """Returns (matched_target_key, picking_position, placing_position, target_name,
    reference_key_or_None) — reference_key is set when the instruction places the target
    relative to another cube (e.g. "right of the red cube"), so its obstacle status can be
    temporarily disabled too when stacking directly onto it."""
    inst = instruction.lower().strip()
    mentions = _find_object_mentions(inst)

    # Detect a spatial-relation phrase and the object mentioned AFTER it — that's the
    # reference object the placement is relative to, distinct from the pick target.
    relation_offset, reference_key = None, None
    for phrase, offset in SPATIAL_RELATIONS:
        idx = inst.find(phrase)
        if idx == -1:
            continue
        after = [(i, k) for i, k in mentions if i > idx]
        if after:
            relation_offset = offset
            reference_key = after[0][1]
            break

    # Pick target = first object mentioned that isn't the reference object.
    matched_target = next((k for _, k in mentions if k != reference_key), None)
    if matched_target is None:
        matched_target = mentions[0][1] if mentions else "red_cube"

    # Cosmos-Reason2 perception (cosmos_integration.md Steps 5 & 8): ground every object
    # this step cares about in ONE call, then let get_object_pose() decide what actually
    # drives the robot. In "stage"/"shadow" mode get_object_pose() ALWAYS returns the stage
    # (ground-truth USD) pose regardless of what Cosmos saw — this is what kept Step 5 a
    # zero-risk, purely-additive change. In "vision" mode it returns Cosmos's own deprojected
    # estimate, but only if it passed the workspace-bounds check and is within the sanity
    # gate of the stage pose — any failure (Cosmos down/timeout, bad grounding, no depth,
    # gate exceeded) falls back to the exact same stage pose automatically, so this call can
    # never leave target_pos/tray_pos undefined or wildly wrong, whichever mode is active.
    _perception_keys = {matched_target, "target_tray"}
    if reference_key:
        _perception_keys.add(reference_key)
    try:
        perception_provider.refresh(list(_perception_keys))
    except Exception as e:
        print(f"[Perception] Refresh failed (non-fatal, falling back to stage poses): {e}", flush=True)

    target_pos = perception_provider.get_object_pose(matched_target)
    tray_pos = perception_provider.get_object_pose("target_tray")
    target_name = SCENE_OBJECTS[matched_target]["name"]
    # "place"/"put" removed — both are generic verbs present in essentially every dispatched
    # instruction (including the agentic orchestrator's own "pick up the X and place it Y"
    # template), so this was silently true almost always, regardless of the actual destination
    # — confirmed as a real live bug: "pick up the green cube and place it red cube" (a
    # stacking destination with no relation phrase) got routed to a free tray slot instead of
    # anywhere near the red cube, because "place" alone satisfied this check. The remaining
    # words are specific enough to only fire on a genuinely tray-directed instruction.
    wants_tray = any(k in inst for k in ["tray", "bin", "sort", "transport", "drop", "basket", "container"])

    # Real live Z, not a hardcoded table-height constant — confirmed as a second real bug from
    # the same incident: picking_position used to always assume the target sits flat on the
    # table (CUBE_HALF_HEIGHT), which happens to be correct for that case since a table-resting
    # cube's live Z already equals CUBE_HALF_HEIGHT, but is wrong for a cube currently stacked
    # on another one (a real elevated Z) — the gripper was descending to grasp at completely
    # the wrong height and physically knocked over an already-correct stack while attempting to
    # pick the cube sitting on top of it.
    picking_position = np.array([target_pos[0], target_pos[1], target_pos[2]])

    if relation_offset is not None and reference_key is not None and reference_key != matched_target:
        ref_pos = perception_provider.get_object_pose(reference_key)
        dx, dy, is_stack = relation_offset
        if is_stack:
            # Confirmed as a real live bug: this used to always target ref_pos's OWN height +
            # one cube — correct only when nothing already sits on the reference. Re-stacking
            # onto a cube that already has a different cube on top of it (e.g. an existing
            # red->green stack, now asked to put blue "on top of the red cube") computed a
            # target that coincided with the already-occupied spot, causing a real collision
            # and a large, repeated position error (confirmed: 105mm, 3/3 attempts exhausted,
            # plan aborted). Fix: check every OTHER live cube for one sitting near ref's (x,y)
            # at roughly one-cube-height above it — i.e. something already stacked on ref —
            # and if found, stack on top of THAT instead, so the actual physical target is
            # always the current top of whatever pile is really there.
            occupant_pos, occupant_key = None, None
            for other_key in SCENE_OBJECTS:
                if other_key in (matched_target, reference_key) or "cube" not in other_key:
                    continue
                other_pos = perception_provider.get_object_pose(other_key)
                same_xy = (abs(other_pos[0] - ref_pos[0]) < CUBE_HALF_HEIGHT) and \
                          (abs(other_pos[1] - ref_pos[1]) < CUBE_HALF_HEIGHT)
                stacked_on_ref = other_pos[2] > ref_pos[2] + CUBE_HALF_HEIGHT * 0.75
                if same_xy and stacked_on_ref:
                    if occupant_pos is None or other_pos[2] > occupant_pos[2]:
                        occupant_pos, occupant_key = other_pos, other_key
            stack_base = occupant_pos if occupant_pos is not None else ref_pos
            placing_position = np.array([stack_base[0], stack_base[1], stack_base[2] + CUBE_HALF_HEIGHT * 2])
            if occupant_key is not None:
                print(f"[Planner] Stacking target 'on top of {SCENE_OBJECTS[reference_key]['name']}' "
                      f"is already occupied by '{SCENE_OBJECTS[occupant_key]['name']}' @ "
                      f"{np.round(occupant_pos, 3)} — stacking on top of that occupant instead of "
                      f"colliding with it (actual final order may not match the instruction's "
                      f"requested order).", flush=True)
                # Confirmed as the actual regression from the first version of this fix: the
                # physical target moved to sit above the occupant, but the caller only disables
                # RMPFlow obstacle status for `reference_key` — leaving the occupant (the cube
                # actually being placed onto) enabled as an obstacle to avoid. That fights the
                # arm's own placement motion at the exact same spot it's trying to reach —
                # observed live as erratic hovering/avoidance and knocked-over cubes, not a
                # clean placement. reference_key must point at whatever is actually being
                # disabled/targeted, which is now the occupant, not the originally-named cube.
                reference_key = occupant_key
        else:
            px, py = ref_pos[0] + dx, ref_pos[1] + dy
            if wants_tray:
                px = float(np.clip(px, TRAY_X_BOUNDS[0] + 0.03, TRAY_X_BOUNDS[1] - 0.03))
                py = float(np.clip(py, TRAY_Y_BOUNDS[0] + 0.03, TRAY_Y_BOUNDS[1] - 0.03))
            place_z = (tray_pos[2] + TRAY_HALF_HEIGHT + CUBE_HALF_HEIGHT) if wants_tray else CUBE_HALF_HEIGHT
            placing_position = np.array([px, py, place_z])
        print(f"[Planner] '{target_name}' relative to '{SCENE_OBJECTS[reference_key]['name']}' "
              f"@ {np.round(ref_pos, 3)} (stack={is_stack}) -> place {np.round(placing_position, 3)}", flush=True)
    elif wants_tray:
        slot_xy = _pick_free_tray_slot(matched_target)
        placing_position = np.array([slot_xy[0], slot_xy[1], tray_pos[2] + TRAY_HALF_HEIGHT + CUBE_HALF_HEIGHT])
        print(f"[Planner] Assessing '{target_name}' @ XYZ={np.round(target_pos, 3)} -> free tray slot {np.round(slot_xy, 3)}", flush=True)
    else:
        # No explicit place target — pick, then set back down at the same spot (still
        # exercises the full pick/place cycle for grasp-reliability testing).
        placing_position = picking_position.copy()
        print(f"[Planner] Assessing '{target_name}' @ XYZ={np.round(target_pos, 3)}", flush=True)

    return matched_target, picking_position, placing_position, target_name, reference_key

# Friendly names for PickPlaceController's 10 internal phases (see its docstring).
PICK_PLACE_EVENT_NAMES = {
    0: "Moving Above Target",
    1: "Descending to Grasp",
    2: "Settling Before Grasp",
    3: "Closing Gripper",
    4: "Lifting",
    5: "Transporting to Place XY",
    6: "Descending to Place Height",
    7: "Releasing Gripper",
    8: "Ascending",
    9: "Returning to Staging Position",
}

# ── 6.5. Real GR00T-N1.7 VLA Closed-Loop Client (Phase 3) ──────────────────────
# GR00T's torch/transformers/flash-attn build only exists in Isaac-GR00T/.venv, not in Isaac
# Sim's own bundled Python this file runs under — so real inference runs as a separate process
# (groot_policy_service.py, port 8300) and this is an HTTP client of it, same pattern as the
# Nemotron server. Real forward passes only: no keyword-matching fallback exists here.
GROOT_SERVER_URL = "http://localhost:8300"
VLA_ACTION_ORDER = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
# Fixed downward grasp orientation for state feedback — we don't have a real measured EE
# orientation telemetry channel, so this constant matches the fixed DOWNWARD_ROT convention
# every other part of the pipeline already assumes for a top-down grasp. VLA-predicted
# roll/pitch/yaw deltas are intentionally NOT applied to arm orientation for this first
# closed-loop pass — only xyz position + gripper are driven from the model, to avoid
# compounding orientation error on top of an already-approximate state input.
VLA_FIXED_RPY = np.array([3.14159, 0.0, 0.0], dtype=np.float32)
# Workspace safety clamp — VLA position deltas accumulate every step; without bounds an
# out-of-distribution prediction (this checkpoint isn't fine-tuned on our scene) could walk
# the target outside the reachable workspace or into the table.
VLA_XYZ_MIN = np.array([0.20, -0.45, 0.035], dtype=np.float32)
VLA_XYZ_MAX = np.array([0.70, 0.45, 0.45], dtype=np.float32)
# Same fixed downward end-effector orientation PickPlaceController itself defaults to
# (euler_angles_to_quat([0, pi, 0])) — reused here for consistency, not re-derived.
VLA_DOWNWARD_QUAT = euler_angles_to_quat(np.array([0, np.pi, 0]))
VLA_HOLD_FRAMES = 15   # physics frames to hold/converge on each chunk step's target (~0.25s)
VLA_MAX_CHUNKS = 4     # cap total re-queries per task so an out-of-distribution loop can't run forever


def encode_frame_b64(raw_rgb: np.ndarray) -> str:
    """JPEG-encodes a raw RGB annotator frame to a base64 string for the GR00T HTTP API."""
    bgr = cv2.cvtColor(raw_rgb[:, :, :3], cv2.COLOR_RGB2BGR)
    _, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def query_groot_server(instruction: str, scene_rgb: np.ndarray, wrist_rgb: np.ndarray,
                        ee_xyz: np.ndarray, gripper_width: float):
    """Calls the real GR00T-N1.7 policy server. Returns (horizon, 7) np.ndarray or None on error."""
    ee_pose6 = [float(ee_xyz[0]), float(ee_xyz[1]), float(ee_xyz[2]),
                float(VLA_FIXED_RPY[0]), float(VLA_FIXED_RPY[1]), float(VLA_FIXED_RPY[2])]
    payload = jsonlib.dumps({
        "instruction": instruction,
        "scene_image_b64": encode_frame_b64(scene_rgb),
        "wrist_image_b64": encode_frame_b64(wrist_rgb),
        "ee_pose": ee_pose6,
        "gripper_width": float(gripper_width),
    }).encode("utf-8")
    try:
        req = urllib.request.Request(
            f"{GROOT_SERVER_URL}/predict", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=15.0) as resp:
            data = jsonlib.loads(resp.read().decode("utf-8"))
            return np.array(data["action_chunk"], dtype=np.float32)
    except Exception as e:
        print(f"[GR00T VLA] Query failed: {e}", flush=True)
        return None

# ── 7. FastAPI App for Streaming & Control ─────────────────────────────────────
api_app = FastAPI(title="Isaac Sim Franka Stream Service", version="2.0.0")
api_app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@api_app.get("/", response_class=HTMLResponse)
def root_index():
    return """<!DOCTYPE html>
<html>
<head>
    <title>Isaac Sim Camera API Service</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0b0f19; color: #e2e8f0; padding: 30px; }
        h1 { color: #38bdf8; }
        .card { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 20px; margin-bottom: 20px; max-width: 800px; }
        a { color: #60a5fa; text-decoration: none; font-weight: bold; }
        a:hover { text-decoration: underline; }
        ul { line-height: 1.8; }
        .badge { background: #166534; color: #4ade80; padding: 3px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: bold; }
    </style>
</head>
<body>
    <div class="card">
        <h1>🦾 Isaac Sim Franka Simulation API <span class="badge">ONLINE</span></h1>
        <p>Simulation Engine: <strong>PhysX 60 Hz</strong> | Hardware: <strong>NVIDIA RTX PRO 6000 Blackwell</strong></p>
        <p>Web Interface: <a href="http://localhost:7860" target="_blank">http://localhost:7860 (Physyk GUI)</a></p>
    </div>
    <div class="card">
        <h3>🎥 Live Multi-Camera Streams (HD 720p MJPEG)</h3>
        <ul>
            <li><a href="/camera/stream" target="_blank">/camera/stream</a> — 3rd Person Isometric Overview</li>
            <li><a href="/camera/scene.stream" target="_blank">/camera/scene.stream</a> — 3rd Person Scene View</li>
            <li><a href="/camera/front.stream" target="_blank">/camera/front.stream</a> — Front Operator View</li>
            <li><a href="/camera/side.stream" target="_blank">/camera/side.stream</a> — Side Profile & Clearance View</li>
            <li><a href="/camera/top.stream" target="_blank">/camera/top.stream</a> — Top-Down Bird's Eye View</li>
            <li><a href="/camera/wrist.stream" target="_blank">/camera/wrist.stream</a> — Wrist Camera (End-Effector)</li>
        </ul>
        <h3>📊 Telemetry & Control APIs</h3>
        <ul>
            <li><a href="/health" target="_blank">/health</a> — Service Health & Simulation State</li>
            <li><a href="/state" target="_blank">/state</a> — Live 7-DOF Joint Positions & FPS Telemetry</li>
            <li><a href="/api/cameras" target="_blank">/api/cameras</a> — List of all 5 cameras</li>
            <li><a href="/api/lighting" target="_blank">/api/lighting</a> — List of all lighting presets</li>
        </ul>
    </div>
</body>
</html>"""

@api_app.get("/health")
def health():
    with telemetry_lock:
        return {
            "status": "ok",
            "busy": sim_telemetry["busy"],
            "stage": sim_telemetry["stage"],
            "cameras": 5,
            "lighting": current_lighting_preset
        }

@api_app.get("/state")
def get_state():
    with telemetry_lock:
        return JSONResponse(sim_telemetry)

@api_app.get("/perception/camera")
def get_perception_camera():
    """Cosmos-Reason2 integration, Phase A "camera plumbing" — real depth + intrinsics/
    extrinsics for the overview camera, exposed for verification. Not consumed by any
    decision-making path yet; resolve_pick_place_targets and everything it drives are
    completely untouched by this."""
    with telemetry_lock:
        return JSONResponse(perception_camera_state)

@api_app.get("/perception")
def get_perception():
    """Cosmos-Reason2 integration Step 5 — shadow-mode snapshot: the last grounding call's
    per-object stage pose, Cosmos-estimated pose, and delta in mm. `used` is always "stage"
    unless PERCEPTION_MODE=vision is set AND that object's estimate passed the sanity gate —
    see physyk_perception/provider.py. Purely informational; resolve_pick_place_targets and
    real robot dispatch are unaffected by this in stage/shadow mode."""
    return JSONResponse(perception_provider.snapshot())

@api_app.get("/api/cameras")
def get_cameras():
    return {
        "cameras": [
            {"id": "scene", "name": "3rd Person Isometric Overview", "stream": "/camera/stream"},
            {"id": "front", "name": "Front Operator View", "stream": "/camera/front.stream"},
            {"id": "side", "name": "Side Profile & Height Clearance", "stream": "/camera/side.stream"},
            {"id": "top", "name": "Top-Down Bird's Eye Planar", "stream": "/camera/top.stream"},
            {"id": "wrist", "name": "Wrist Camera (Gripper Close-up)", "stream": "/camera/wrist.stream"},
        ]
    }

@api_app.get("/api/lighting")
def get_lighting_presets():
    return {
        "current": current_lighting_preset,
        "presets": [
            {"id": "studio", "name": "Studio White (Default)"},
            {"id": "high_contrast", "name": "High-Contrast Inspection"},
            {"id": "warm", "name": "Warm Industrial"},
            {"id": "cyberpunk", "name": "Cyberpunk Neon"},
            {"id": "neutral", "name": "Neutral Daylight"}
        ]
    }

@api_app.post("/lighting")
async def change_lighting(req: Request):
    data = await req.json()
    preset = data.get("preset", "studio")
    lighting_queue.put(preset)
    with telemetry_lock:
        sim_telemetry["lighting"] = preset
    return {"status": "ok", "lighting": preset}

# Single Snapshot JPEGs (HD 720p)
@api_app.get("/camera/scene.jpg")
def get_scene_jpeg():
    with frame_lock:
        if latest_frames["scene_jpeg"] is not None:
            return Response(content=latest_frames["scene_jpeg"], media_type="image/jpeg")
    return Response(status_code=404)

@api_app.get("/camera/front.jpg")
def get_front_jpeg():
    with frame_lock:
        if latest_frames["front_jpeg"] is not None:
            return Response(content=latest_frames["front_jpeg"], media_type="image/jpeg")
    return Response(status_code=404)

@api_app.get("/camera/side.jpg")
def get_side_jpeg():
    with frame_lock:
        if latest_frames["side_jpeg"] is not None:
            return Response(content=latest_frames["side_jpeg"], media_type="image/jpeg")
    return Response(status_code=404)

@api_app.get("/camera/top.jpg")
def get_top_jpeg():
    with frame_lock:
        if latest_frames["top_jpeg"] is not None:
            return Response(content=latest_frames["top_jpeg"], media_type="image/jpeg")
    return Response(status_code=404)

@api_app.get("/camera/wrist.jpg")
def get_wrist_jpeg():
    with frame_lock:
        if latest_frames["wrist_jpeg"] is not None:
            return Response(content=latest_frames["wrist_jpeg"], media_type="image/jpeg")
    return Response(status_code=404)

# Continuous Live MJPEG Streams with Low Latency
def create_mjpeg_response(frame_key: str):
    def frame_generator():
        while True:
            with frame_lock:
                frame_data = latest_frames.get(frame_key)
            if frame_data is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            time.sleep(0.016)  # ~60 FPS update check
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@api_app.get("/camera/stream")
def get_scene_stream():
    return create_mjpeg_response("scene_jpeg")

@api_app.get("/camera/scene.stream")
def get_scene_stream_alias():
    return create_mjpeg_response("scene_jpeg")

@api_app.get("/camera/front.stream")
def get_front_stream():
    return create_mjpeg_response("front_jpeg")

@api_app.get("/camera/side.stream")
def get_side_stream():
    return create_mjpeg_response("side_jpeg")

@api_app.get("/camera/top.stream")
def get_top_stream():
    return create_mjpeg_response("top_jpeg")

@api_app.get("/camera/wrist.stream")
def get_wrist_stream():
    return create_mjpeg_response("wrist_jpeg")

@api_app.post("/execute")
async def execute_command(req: Request):
    data = await req.json()
    instruction = data.get("instruction", "").lower().strip()
    
    with telemetry_lock:
        if sim_telemetry["busy"]:
            return JSONResponse({"status": "error", "message": "Simulation is currently busy executing an action."}, status_code=409)
            
    print(f"[Isaac Sim API] Received Instruction: '{instruction}'", flush=True)
    req_id = _new_request_id()
    command_queue.put((req_id, instruction))
    return {"status": "accepted", "instruction": instruction, "request_id": req_id}

@api_app.post("/reset")
def reset_sim():
    req_id = _new_request_id()
    command_queue.put((req_id, "reset"))
    return {"status": "resetting", "request_id": req_id}

@api_app.post("/randomize")
def randomize_cubes():
    req_id = _new_request_id()
    command_queue.put((req_id, "randomize cube positions"))
    return {"status": "randomizing", "request_id": req_id}

@api_app.post("/hard_stop")
def hard_stop():
    """Emergency stop for the GUI's red 'Hard Reset' button — abort whatever's running (a
    mid-motion pick_place, an in-flight VLA chunk, anything still queued) and home the arm,
    without touching cube positions. Just sets a flag + drains the queue; the actual abort
    happens synchronously at the top of the sim loop on its next frame (see hard_stop_event
    below) since only that thread may safely touch the robot/controller/USD stage."""
    _drain_command_queue()
    hard_stop_event.set()
    return {"status": "stopping"}

# ── 8. Start Background Web Server Thread ──────────────────────────────────────
def run_fastapi():
    uvicorn.run(api_app, host="0.0.0.0", port=8100, log_level="warning")

server_thread = threading.Thread(target=run_fastapi, daemon=True)
server_thread.start()
print("[Isaac Sim Service] Streaming HTTP API started on port 8100", flush=True)

# ── 9. Main Simulation & Physics Stepping Loop ────────────────────────────────
print("═══════════════════════════════════════════════════════════════════", flush=True)
print("  [Isaac Sim Service] Physics Timeline Running at 60 Hz           ", flush=True)
print("  📡 Live HD Camera Streams: http://localhost:8100/                ", flush=True)
print("═══════════════════════════════════════════════════════════════════", flush=True)

active_pick_place = None  # dict: {picking_position, placing_position, matched_target} or None
active_vla = None  # dict: {instruction, chunk, step_idx, target_xyz} or None — Phase 3 GR00T loop
last_fps_time = time.time()
fps_counter = 0
held_object_key = None

# Helper to encode with clean, subtle, anti-aliased HUD and rich color fidelity
def encode_camera_frame(raw_rgb, label: str):
    if raw_rgb is None or raw_rgb.size == 0:
        return None
    bgr = cv2.cvtColor(raw_rgb[:, :, :3], cv2.COLOR_RGB2BGR)
    
    # Subtle, sleek, crisp watermark HUD
    cv2.putText(bgr, f"PHYSYK • {label.upper()}", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 240, 255), 1, cv2.LINE_AA)
    with telemetry_lock:
        st = sim_telemetry.get("stage", "READY")
    color = (100, 235, 140) if st == "READY" else (100, 200, 255)
    cv2.putText(bgr, f"STAGE: {st}", (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
    
    # High-quality 90% JPEG encoding
    _, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return encoded.tobytes()


def _verify_vla_task(instruction: str):
    """Real post-place verification for a GR00T VLA-driven task — this used to not exist at
    all: the VLA loop just ran its chunk budget and stopped, with no last_task_success/
    last_task_error_mm ever computed, so there was no way to know whether GR00T actually
    accomplished anything. Reuses the exact same resolution (resolve_pick_place_targets —
    safe to call again, it's pure planning logic with no side effects of its own) and
    tolerances (POSITION_XY/Z_TOLERANCE_M) the deterministic path already uses, so
    VLA-driven and PickPlaceController-driven actions are judged by the same real standard,
    not two different bars. This does not make GR00T reliable — it makes its real
    (un)reliability observable, which it wasn't before."""
    try:
        matched_target, picking_pos, placing_position, target_name, _ref = \
            resolve_pick_place_targets(instruction)
    except Exception as e:
        print(f"[GR00T VLA] Could not resolve a verification target for '{instruction}': {e}", flush=True)
        with telemetry_lock:
            sim_telemetry["last_task_success"] = None
            sim_telemetry["last_task_error_mm"] = None
        return
    final_pos = get_live_object_pos(matched_target)
    err_xy = float(np.hypot(final_pos[0] - placing_position[0], final_pos[1] - placing_position[1]))
    err_z = float(abs(final_pos[2] - placing_position[2]))
    position_ok = (err_xy <= POSITION_XY_TOLERANCE_M) and (err_z <= POSITION_Z_TOLERANCE_M)

    # Same real "did it actually move" check as the deterministic path — VLA has no
    # discrete grasp-close phase to hook a grasp-confirmation check onto the way
    # PickPlaceController does, but this displacement check is free either way.
    displacement_m = float(np.hypot(final_pos[0] - picking_pos[0], final_pos[1] - picking_pos[1]))
    intended_displacement_m = float(np.hypot(placing_position[0] - picking_pos[0], placing_position[1] - picking_pos[1]))
    moved_enough = (intended_displacement_m < MEANINGFUL_DISPLACEMENT_M) or \
        (displacement_m >= MEANINGFUL_DISPLACEMENT_M * 0.5)
    task_success = position_ok and moved_enough

    with telemetry_lock:
        sim_telemetry["last_task_success"] = task_success
        sim_telemetry["last_task_error_mm"] = round(err_xy * 1000.0, 1)
        sim_telemetry["last_object_displacement_mm"] = round(displacement_m * 1000.0, 1)
    fail_reasons = []
    if not position_ok:
        fail_reasons.append(f"missed target by {err_xy * 1000:.1f}mm")
    if not moved_enough:
        fail_reasons.append(f"barely moved ({displacement_m * 1000:.1f}mm, expected ~{intended_displacement_m * 1000:.1f}mm)")
    print(
        f"[GR00T VLA] Task Verification: {'PASS' if task_success else 'FAIL'} — "
        f"'{target_name}' landed {err_xy * 1000:.1f}mm (xy) / {err_z * 1000:.1f}mm (z) "
        f"from resolved target {np.round(placing_position, 3)} (actual {np.round(final_pos, 3)})"
        + (f" — {'; '.join(fail_reasons)}" if fail_reasons else ""), flush=True
    )

try:
    while simulation_app.is_running():
        # -1. Emergency Hard Stop — pre-empts everything else this frame. Aborts whatever
        #     task is active (pick_place or VLA), drops it without finishing its motion,
        #     opens the gripper, sends the arm home, AND returns every cube to its default
        #     staging spot — the button is meant as a full "get back to a known-good state"
        #     reset, not just a motion abort.
        if hard_stop_event.is_set():
            hard_stop_event.clear()
            active_pick_place = None
            active_vla = None
            held_object_key = None
            _drain_command_queue()  # catches anything queued between /hard_stop and this frame
            with telemetry_lock:
                sim_telemetry["stage"] = "HARD STOP — HOMING & RESETTING PROPS"
            pick_place_controller.reset()
            robot.gripper.open()  # don't leave a mid-grasp gripper physically closed across a hard stop
            reset_scene_cubes()
            articulation_controller.apply_action(ArticulationAction(joint_positions=HOME_JOINTS))
            with telemetry_lock:
                sim_telemetry["busy"] = False
                sim_telemetry["stage"] = "READY"
                sim_telemetry["last_instruction"] = "Hard reset — ready for new commands"
                sim_telemetry["current_request_id"] = 0
                sim_telemetry["vla_reasoning"] = None
                sim_telemetry["vla_chunks_done"] = 0
            print("[Isaac Sim Service] HARD STOP — task aborted, arm homed, cubes reset to default.", flush=True)
            continue

        # 0. Process Dynamic Lighting Updates
        if not lighting_queue.empty():
            preset_name = lighting_queue.get_nowait()
            set_lighting_preset(preset_name)

        # 1. Process Pending Commands
        if active_pick_place is None and active_vla is None and not command_queue.empty():
            req_id, cmd = command_queue.get_nowait()
            with telemetry_lock:
                sim_telemetry["busy"] = True
                sim_telemetry["last_instruction"] = cmd
                sim_telemetry["current_request_id"] = req_id

            if "reset" in cmd:
                # Full reset: cubes back to fixed staging spots AND robot home. Distinct from
                # "home" below — these used to be the same branch, which meant asking the
                # robot to go home silently teleported every cube back to a fixed spot too.
                with telemetry_lock:
                    sim_telemetry["stage"] = "HOMING & RESETTING PROPS"
                reset_scene_cubes()
                held_object_key = None
                pick_place_controller.reset()
                robot.gripper.open()  # same fix as hard_stop — don't carry a closed gripper across resets
                articulation_controller.apply_action(ArticulationAction(joint_positions=HOME_JOINTS))
                with telemetry_lock:
                    sim_telemetry["busy"] = False
                    sim_telemetry["stage"] = "READY"
                    sim_telemetry["vla_reasoning"] = None
                    sim_telemetry["vla_chunks_done"] = 0
            elif "home" in cmd:
                # Robot only — cube positions are left exactly where they are.
                with telemetry_lock:
                    sim_telemetry["stage"] = "RETURNING TO HOME POSE"
                held_object_key = None
                pick_place_controller.reset()
                robot.gripper.open()
                articulation_controller.apply_action(ArticulationAction(joint_positions=HOME_JOINTS))
                with telemetry_lock:
                    sim_telemetry["busy"] = False
                    sim_telemetry["stage"] = "READY"
                    sim_telemetry["vla_reasoning"] = None
                    sim_telemetry["vla_chunks_done"] = 0
            elif "randomize" in cmd or "shuffle" in cmd or "random" in cmd:
                with telemetry_lock:
                    sim_telemetry["stage"] = "RANDOMIZING CUBE POSITIONS"
                randomize_scene_cubes()
                with telemetry_lock:
                    sim_telemetry["busy"] = False
                    sim_telemetry["stage"] = "READY"
                    sim_telemetry["vla_reasoning"] = None
                    sim_telemetry["vla_chunks_done"] = 0
            elif cmd.strip().lower().startswith("vla:") or cmd.strip().lower().startswith("vla "):
                # Real GR00T-N1.7 closed-loop mode — separate from the reliable
                # PickPlaceController path above, opt-in via a "vla:" prefix so the default
                # instruction path stays on the validated, deterministic controller.
                vla_instruction = cmd.split(":", 1)[-1].strip() if ":" in cmd else cmd[4:].strip()
                active_vla = {"instruction": vla_instruction, "chunk": None, "step_idx": 0}
                with telemetry_lock:
                    sim_telemetry["stage"] = f"GR00T VLA — Querying Model"
            else:
                with telemetry_lock:
                    sim_telemetry["stage"] = "ASSESSING SCENE"
                    sim_telemetry["vla_reasoning"] = None
                    sim_telemetry["vla_chunks_done"] = 0
                try:
                    # Perception grounding + (in vision mode) actually driving target_pos/
                    # tray_pos/ref_pos now happens INSIDE resolve_pick_place_targets itself
                    # (cosmos_integration.md Steps 5 & 8) — a single call there covers both
                    # the shadow-mode delta log and vision-mode's real substitution, so it's
                    # not duplicated here.
                    matched_target, picking_position, placing_position, target_name, reference_key = resolve_pick_place_targets(cmd)

                    pick_place_controller.reset()
                    # pick_place_controller.reset() only resets the state machine's phase/timer —
                    # it does NOT physically command the gripper open. If the PREVIOUS task ended
                    # via the early grasp-fail abort below (rather than completing phase 7
                    # "release"), the gripper stays physically closed from that failed attempt.
                    # Without this, the arm approaches this new task with jaws already shut, so
                    # phase 3 "close grip" is a no-op and the grasp check reads ~0mm again —
                    # guaranteed failure, cascading into failure streaks. Explicitly re-open here
                    # so every new task always starts from a known-good open-gripper state.
                    robot.gripper.open()
                    # The cube being picked must NOT be an obstacle to itself — disable it for
                    # the duration of this task, keep every other cube as a real obstacle so
                    # the arm routes around them instead of clipping through (as happened
                    # during Phase 1 tuning, before obstacles were registered at all). If this
                    # is a "stack on top of X" placement, X must be disabled too — otherwise
                    # RMPFlow treats the intended landing spot as an obstacle to avoid touching.
                    for key, cube_obj in CUBE_OBJECTS.items():
                        try:
                            if key == matched_target or key == reference_key:
                                rmpflow_motion_policy.disable_obstacle(cube_obj)
                            else:
                                rmpflow_motion_policy.enable_obstacle(cube_obj)
                        except Exception as e:
                            print(f"[WARN] Obstacle enable/disable failed for {key}: {e}", flush=True)
                    active_pick_place = {
                        "picking_position": picking_position,
                        "placing_position": placing_position,
                        "matched_target": matched_target,
                        "target_name": target_name,
                        "reference_key": reference_key,
                        "grasp_checked": False,
                    }
                    episode_recorder.start_episode(cmd)
                except Exception as e:
                    print(f"[Planner Error] {e}", flush=True)
                    with telemetry_lock:
                        sim_telemetry["busy"] = False
                        sim_telemetry["stage"] = "READY"

        # 2. Advance NVIDIA's validated Pick-Place state machine (RMPFlow cspace controller
        #    + gripper open/close), replacing the earlier hand-rolled waypoint interpolation.
        if active_pick_place is not None:
            target_name = active_pick_place["target_name"]
            event_idx = pick_place_controller.get_current_event()
            stage_name = f"{PICK_PLACE_EVENT_NAMES.get(event_idx, f'Phase {event_idx}')} — {target_name}"
            with telemetry_lock:
                sim_telemetry["stage"] = stage_name

            if event_idx == 3:
                held_object_key = active_pick_place["matched_target"]
            elif event_idx == 4 and not active_pick_place["grasp_checked"]:
                # Real grasp-confirmation, checked exactly once per task at the moment the
                # gripper has finished closing (transition into "Lifting") — a signal that
                # didn't exist before. Position-only post-place verification can't tell a
                # successful-but-wrong-spot placement from a grasp that failed outright and
                # left the object sitting roughly where it started (especially for an
                # instruction with no explicit destination, where "didn't move" and
                # "succeeded" can look identical position-wise). The gripper's actual
                # achieved width is a direct physical signal instead: PhysX stops the
                # fingers early if something solid is between them, so a real grasp settles
                # near cube-width, while a missed grasp closes down near the gripper's own
                # mechanical minimum.
                active_pick_place["grasp_checked"] = True
                grasp_dofs = robot.get_joint_positions()
                if grasp_dofs is not None and grasp_dofs.ndim > 1:
                    grasp_dofs = grasp_dofs.flatten()
                achieved_width = float(grasp_dofs[7] * 2.0) if grasp_dofs is not None and len(grasp_dofs) > 7 else 0.0
                grasp_confirmed = achieved_width >= GRASP_MIN_WIDTH_M
                with telemetry_lock:
                    sim_telemetry["last_grasp_confirmed"] = grasp_confirmed
                print(
                    f"[Isaac Sim Service] Grasp check for '{target_name}': "
                    f"{'CONFIRMED' if grasp_confirmed else 'FAILED'} — gripper settled at "
                    f"{achieved_width * 1000:.1f}mm (need >= {GRASP_MIN_WIDTH_M * 1000:.0f}mm)",
                    flush=True
                )
                if not grasp_confirmed:
                    # Real early-abort — this is what actually closes "the agent doesn't
                    # realize the cube wasn't picked up until it finishes the entire
                    # pipeline, then retries": previously a failed grasp still ran phases
                    # 5-9 (transport, descend, release, ascend, return — all with nothing
                    # actually held) before verification ever ran, so the orchestrator's
                    # poll loop had no way to find out until the whole ~30-40s sequence
                    # finished. The grasp is already known to have failed right here — stop
                    # the state machine and report it immediately instead of pointlessly
                    # finishing the rest of the motion.
                    print(f"[Isaac Sim Service] Grasp failed for '{target_name}' — aborting "
                          f"early instead of completing phases 5-9 with nothing held.", flush=True)
                    # Re-open immediately rather than leaving the gripper physically closed —
                    # see the matching comment at task dispatch above. Doing it here too (not
                    # just at the next task's start) means a live /state read between tasks
                    # also shows a physically-consistent open gripper, not a stale closed one.
                    robot.gripper.open()
                    pick_place_controller.reset()
                    final_pos = get_live_object_pos(active_pick_place["matched_target"])
                    start_pos = active_pick_place["picking_position"]
                    displacement_m = float(np.hypot(final_pos[0] - start_pos[0], final_pos[1] - start_pos[1]))
                    with telemetry_lock:
                        sim_telemetry["last_task_success"] = False
                        sim_telemetry["last_task_error_mm"] = None
                        sim_telemetry["last_object_displacement_mm"] = round(displacement_m * 1000.0, 1)
                    episode_recorder.end_episode(False)  # early grasp-fail abort — drop the episode
                    try:
                        rmpflow_motion_policy.enable_obstacle(CUBE_OBJECTS[active_pick_place["matched_target"]])
                        ref_key = active_pick_place.get("reference_key")
                        if ref_key is not None:
                            rmpflow_motion_policy.enable_obstacle(CUBE_OBJECTS[ref_key])
                    except Exception as e:
                        print(f"[WARN] Could not re-enable obstacle after early grasp-fail abort: {e}", flush=True)
                    active_pick_place = None
                    held_object_key = None
                    with telemetry_lock:
                        sim_telemetry["busy"] = False
                        sim_telemetry["stage"] = "READY"
            elif event_idx == 7:
                held_object_key = None

        # Captured before this block can clear active_pick_place back to None below, so the
        # episode recorder's per-frame tap (in section 4, after raw camera frames are
        # captured) knows whether a pick_place task was actually running this frame.
        _recording_active_this_frame = active_pick_place is not None

        if active_pick_place is not None:
            current_joint_positions = robot.get_joint_positions()
            actions = pick_place_controller.forward(
                picking_position=active_pick_place["picking_position"],
                placing_position=active_pick_place["placing_position"],
                current_joint_positions=current_joint_positions,
                end_effector_offset=EE_OFFSET,
            )
            articulation_controller.apply_action(actions)

            if pick_place_controller.is_done():
                print(f"[Isaac Sim Service] Task Execution Complete: {sim_telemetry['last_instruction']}", flush=True)
                # Real post-place verification (was a hardcoded always-PASS stub elsewhere —
                # see PLAN.md Part 3). Compares the object's actual final live position against
                # the resolved placing_position it was sent to. XY tolerance is half a cube
                # width; Z gets a looser tolerance since a dropped/settled cube can still be
                # bouncing/rotating slightly when this fires.
                final_pos = get_live_object_pos(active_pick_place["matched_target"])
                target_pos = active_pick_place["placing_position"]
                start_pos = active_pick_place["picking_position"]
                err_xy = float(np.hypot(final_pos[0] - target_pos[0], final_pos[1] - target_pos[1]))
                err_z = float(abs(final_pos[2] - target_pos[2]))
                position_ok = (err_xy <= POSITION_XY_TOLERANCE_M) and (err_z <= POSITION_Z_TOLERANCE_M)

                # Real "did it actually move" + real grasp-confirmation checks, alongside the
                # existing position check — closes a real gap: position-only verification
                # can't tell "landed wrong" apart from "grasp failed, nothing really
                # happened", especially for a no-explicit-destination instruction where the
                # intended and actual positions can trivially coincide. A confirmed-failed
                # grasp forces this to fail regardless of where the object ended up,
                # including a coincidental position-tolerance match.
                displacement_m = float(np.hypot(final_pos[0] - start_pos[0], final_pos[1] - start_pos[1]))
                intended_displacement_m = float(np.hypot(target_pos[0] - start_pos[0], target_pos[1] - start_pos[1]))
                moved_enough = (intended_displacement_m < MEANINGFUL_DISPLACEMENT_M) or \
                    (displacement_m >= MEANINGFUL_DISPLACEMENT_M * 0.5)
                with telemetry_lock:
                    grasp_confirmed = sim_telemetry.get("last_grasp_confirmed")
                task_success = position_ok and moved_enough and (grasp_confirmed is not False)

                with telemetry_lock:
                    sim_telemetry["last_task_success"] = task_success
                    sim_telemetry["last_task_error_mm"] = round(err_xy * 1000.0, 1)
                    sim_telemetry["last_object_displacement_mm"] = round(displacement_m * 1000.0, 1)
                # Flush (on success) or drop (on failure) the just-finished episode, using
                # Physyk's own real verification above as the ground-truth label.
                episode_recorder.end_episode(task_success)
                fail_reasons = []
                if not position_ok:
                    fail_reasons.append(f"missed target by {err_xy * 1000:.1f}mm")
                if not moved_enough:
                    fail_reasons.append(f"barely moved ({displacement_m * 1000:.1f}mm, expected ~{intended_displacement_m * 1000:.1f}mm)")
                if grasp_confirmed is False:
                    fail_reasons.append("grasp not confirmed (gripper closed on nothing)")
                print(
                    f"[Isaac Sim Service] Task Verification: {'PASS' if task_success else 'FAIL'} — "
                    f"'{active_pick_place['target_name']}' landed {err_xy * 1000:.1f}mm (xy) / "
                    f"{err_z * 1000:.1f}mm (z) from target {np.round(target_pos, 3)} "
                    f"(actual {np.round(final_pos, 3)})"
                    + (f" — {'; '.join(fail_reasons)}" if fail_reasons else ""), flush=True
                )
                # Re-enable the just-placed cube (and, if this was a stacking placement, the
                # reference cube it was placed onto) as obstacles now that the task is done.
                try:
                    rmpflow_motion_policy.enable_obstacle(CUBE_OBJECTS[active_pick_place["matched_target"]])
                    ref_key = active_pick_place.get("reference_key")
                    if ref_key is not None:
                        rmpflow_motion_policy.enable_obstacle(CUBE_OBJECTS[ref_key])
                except Exception as e:
                    print(f"[WARN] Could not re-enable obstacle after task: {e}", flush=True)
                active_pick_place = None
                held_object_key = None
                with telemetry_lock:
                    sim_telemetry["busy"] = False
                    sim_telemetry["stage"] = "READY"

        # 2.5. Advance the real GR00T-N1.7 closed-loop VLA control mode (Phase 3). Observe
        #      (live camera + EE pose) -> real model forward pass -> 16-step delta chunk ->
        #      drive the arm through RMPFlow one delta-step at a time, re-observing every
        #      chunk. Genuinely separate code path from the PickPlaceController above — no
        #      keyword matching, no scripted waypoints; whatever xyz it predicts is what
        #      drives the arm (gripper too, clamped to a safe range).
        if active_vla is not None:
            if active_vla["chunk"] is None:
                if active_vla.get("chunks_done", 0) >= VLA_MAX_CHUNKS:
                    print(f"[GR00T VLA] Reached max chunk budget ({VLA_MAX_CHUNKS}) — ending task.", flush=True)
                    _verify_vla_task(active_vla["instruction"])
                    active_vla = None
                    with telemetry_lock:
                        sim_telemetry["busy"] = False
                        sim_telemetry["stage"] = "READY"
                else:
                    with telemetry_lock:
                        sim_telemetry["stage"] = f"GR00T VLA — Querying Model ({active_vla['instruction']})"
                    scene_rgb = scene_annot.get_data()
                    wrist_rgb = wrist_annot.get_data()
                    ee_xyz = get_end_effector_pos()
                    dofs = robot.get_joint_positions()
                    if dofs is not None and dofs.ndim > 1:
                        dofs = dofs.flatten()
                    gripper_width = float(dofs[7] * 2.0) if dofs is not None and len(dofs) > 7 else 0.04
                    chunk = None
                    if scene_rgb is not None and scene_rgb.size > 0 and wrist_rgb is not None and wrist_rgb.size > 0:
                        chunk = query_groot_server(active_vla["instruction"], scene_rgb, wrist_rgb, ee_xyz, gripper_width)
                    if chunk is None:
                        print("[GR00T VLA] No usable chunk (camera not ready or server error) — ending task.", flush=True)
                        # Only verify if at least one chunk actually ran — with zero real
                        # motion attempted, there's nothing meaningful to check yet.
                        if active_vla.get("chunks_done", 0) > 0:
                            _verify_vla_task(active_vla["instruction"])
                        active_vla = None
                        with telemetry_lock:
                            sim_telemetry["busy"] = False
                            sim_telemetry["stage"] = "READY"
                    else:
                        active_vla["chunk"] = chunk
                        active_vla["step_idx"] = 0
                        active_vla["hold_counter"] = 0
                        active_vla["target_xyz"] = ee_xyz.astype(np.float32).copy()
                        active_vla["gripper_target"] = gripper_width
                        active_vla["chunks_done"] = active_vla.get("chunks_done", 0) + 1
                        # GR00T is a DiT action head, not a language model — it has no natural-
                        # language "reasoning" to show. This is the honest substitute: a real
                        # summary of what the model actually predicted for this chunk (net
                        # displacement direction/magnitude + gripper intent), computed from the
                        # real action_chunk tensor, not narration. Surfaced via /state so the UI
                        # can show "what GR00T is doing" per chunk instead of a silent black box.
                        net_xyz = chunk[:, :3].sum(axis=0)
                        move_mag_cm = float(np.linalg.norm(net_xyz) * 100.0)
                        move_dir = []
                        if abs(net_xyz[0]) > 0.005: move_dir.append(("forward" if net_xyz[0] > 0 else "backward"))
                        if abs(net_xyz[1]) > 0.005: move_dir.append(("left" if net_xyz[1] > 0 else "right"))
                        if abs(net_xyz[2]) > 0.005: move_dir.append(("up" if net_xyz[2] > 0 else "down"))
                        gripper_end = float(chunk[-1, 6])
                        gripper_intent = "closing (grasp)" if gripper_end < gripper_width - 0.005 else (
                            "opening (release)" if gripper_end > gripper_width + 0.005 else "holding")
                        vla_reasoning = (
                            f"Chunk {active_vla['chunks_done']}/{VLA_MAX_CHUNKS}: predicted net move "
                            f"{move_mag_cm:.1f}cm {'/'.join(move_dir) if move_dir else 'in place'}, "
                            f"gripper {gripper_intent}."
                        )
                        with telemetry_lock:
                            sim_telemetry["vla_reasoning"] = vla_reasoning
                            sim_telemetry["vla_chunks_done"] = active_vla["chunks_done"]
                        print(f"[GR00T VLA] Chunk {active_vla['chunks_done']}/{VLA_MAX_CHUNKS} received, shape {chunk.shape} — {vla_reasoning}", flush=True)
            else:
                chunk = active_vla["chunk"]
                idx = active_vla["step_idx"]
                with telemetry_lock:
                    sim_telemetry["stage"] = (
                        f"GR00T VLA — Step {idx + 1}/{len(chunk)} "
                        f"(chunk {active_vla['chunks_done']}/{VLA_MAX_CHUNKS}) — {active_vla['instruction']}"
                    )

                if active_vla["hold_counter"] == 0:
                    delta = chunk[idx]  # order: x, y, z, roll, pitch, yaw, gripper (see VLA_ACTION_ORDER)
                    new_target = active_vla["target_xyz"] + delta[:3]
                    active_vla["target_xyz"] = np.clip(new_target, VLA_XYZ_MIN, VLA_XYZ_MAX).astype(np.float32)
                    active_vla["gripper_target"] = float(np.clip(delta[6], 0.0, 0.04))

                cspace_actions = pick_place_controller._cspace_controller.forward(
                    target_end_effector_position=active_vla["target_xyz"],
                    target_end_effector_orientation=VLA_DOWNWARD_QUAT,
                )
                articulation_controller.apply_action(cspace_actions)
                g = active_vla["gripper_target"]
                robot.gripper.apply_action(ArticulationAction(joint_positions=np.array([g, g], dtype=np.float32)))

                active_vla["hold_counter"] += 1
                if active_vla["hold_counter"] >= VLA_HOLD_FRAMES:
                    active_vla["hold_counter"] = 0
                    active_vla["step_idx"] += 1
                    if active_vla["step_idx"] >= len(chunk):
                        active_vla["chunk"] = None  # triggers a fresh observe+query next iteration

        # 3. Step PhysX Simulation & Rendering
        world.step(render=True)
        fps_counter += 1
        
        # 4. Capture & Encode All 5 Camera Annotators
        rep.orchestrator.step()
        
        raw_scene = scene_annot.get_data()
        raw_front = front_annot.get_data()
        raw_side = side_annot.get_data()
        raw_top = top_annot.get_data()
        raw_wrist = wrist_annot.get_data()

        # GR00T fine-tuning episode recorder tap — no-op unless RECORD_EPISODES=1 (see top of
        # file). Uses the same raw (pre-HUD-overlay) frames as the JPEG/stream encoding below,
        # and current live EE/gripper state, without altering anything else in this section.
        if _recording_active_this_frame and raw_scene is not None and raw_scene.size > 0 \
                and raw_wrist is not None and raw_wrist.size > 0:
            _dofs_for_recording = robot.get_joint_positions()
            if _dofs_for_recording is not None and _dofs_for_recording.ndim > 1:
                _dofs_for_recording = _dofs_for_recording.flatten()
            _gripper_w = float(_dofs_for_recording[7] * 2.0) if _dofs_for_recording is not None and len(_dofs_for_recording) > 7 else 0.04
            episode_recorder.log_frame(raw_scene, raw_wrist, get_end_effector_pos(), _gripper_w)

        f_scene = encode_camera_frame(raw_scene, "Isometric Overview")
        f_front = encode_camera_frame(raw_front, "Front View")
        f_side = encode_camera_frame(raw_side, "Side Profile")
        f_top = encode_camera_frame(raw_top, "Top-Down")
        f_wrist = encode_camera_frame(raw_wrist, "Wrist Gripper")
        
        with frame_lock:
            if f_scene: latest_frames["scene_jpeg"] = f_scene
            if f_front: latest_frames["front_jpeg"] = f_front
            if f_side: latest_frames["side_jpeg"] = f_side
            if f_top: latest_frames["top_jpeg"] = f_top
            if f_wrist: latest_frames["wrist_jpeg"] = f_wrist
            latest_frames["timestamp"] = time.time()

        # 5. Update Telemetry
        if fps_counter % 5 == 0:
            dofs = robot.get_joint_positions()
            if dofs is not None:
                if dofs.ndim > 1:
                    dofs = dofs.flatten()
                ee_pos = get_end_effector_pos()
                live_objects = {
                    key: {"name": info["name"], "position": np.round(get_live_object_pos(key), 4).tolist()}
                    for key, info in SCENE_OBJECTS.items()
                }
                with telemetry_lock:
                    sim_telemetry["joints"] = np.round(dofs[:7], 4).tolist()
                    sim_telemetry["gripper"] = float(np.round(dofs[7], 4)) if len(dofs) > 7 else 0.04
                    sim_telemetry["ee_pos"] = np.round(ee_pos, 4).tolist()
                    sim_telemetry["objects"] = live_objects

            # Perception camera plumbing: real depth stats from the overview camera's depth
            # annotator, same cadence as the rest of telemetry. Plumbing only — nothing reads
            # this to make a decision yet (that's later, once Isaac Sim integration is
            # actually approved); this just proves the depth channel is real and live.
            try:
                depth_frame = scene_depth_annot.get_data()
            except Exception as e:
                depth_frame = None
                print(f"[WARN] Depth annotator read failed: {e}", flush=True)
            if depth_frame is not None and depth_frame.size > 0:
                finite = depth_frame[np.isfinite(depth_frame)]
                cy, cx = depth_frame.shape[0] // 2, depth_frame.shape[1] // 2
                with telemetry_lock:
                    perception_camera_state["depth_shape"] = list(depth_frame.shape)
                    perception_camera_state["depth_min_m"] = float(finite.min()) if finite.size else None
                    perception_camera_state["depth_max_m"] = float(finite.max()) if finite.size else None
                    perception_camera_state["depth_center_m"] = float(depth_frame[cy, cx])
                    perception_camera_state["timestamp"] = time.time()

        # Calculate real FPS
        now = time.time()
        if now - last_fps_time >= 1.0:
            with telemetry_lock:
                sim_telemetry["fps"] = round(fps_counter / (now - last_fps_time), 1)
            fps_counter = 0
            last_fps_time = now

except Exception as e:
    print(f"[Isaac Sim Service] Exception occurred: {e}", flush=True)
finally:
    print("[Isaac Sim Service] Shutting down simulation...", flush=True)
    simulation_app.close()
