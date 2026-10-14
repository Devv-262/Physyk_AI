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
import collections
import urllib.request
import urllib.error
import json as jsonlib
import cv2
import re
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
    "headless": os.environ.get("ISAAC_SIM_HEADLESS", "0") == "1",
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
from isaacsim.core.api.objects import FixedCuboid, DynamicCuboid, VisualCuboid
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
# Scene backdrop: "lab" (default) = a life-science lab (floor, walls, bench body, shelving — all
# visual-only, see build_lab_background below); "grid" = the original blue grid ground. In lab mode the
# physics ground plane moves down to the real floor level (its grid visuals are hidden); cubes that
# fall off the bench land there.
BACKGROUND = os.environ.get("PHYSYK_BACKGROUND", "lab").lower()
LAB_FLOOR_Z = -0.85   # bench-top (Z=0) to floor: a standard 85 cm lab bench
world.scene.add_default_ground_plane(z_position=(LAB_FLOOR_Z if BACKGROUND == "lab" else -0.12))

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

if BACKGROUND != "lab":
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

def build_lab_background():
    """Visual-only life-science-lab backdrop. Nothing here has physics or collision (VisualCuboid), so the
    task physics, planner obstacles and perception ROI are untouched; the work table itself is not
    modified. Colours are deliberately NEUTRAL (white / greys): the colour detector looks for saturated
    red, green and blue, and saturated background pixels near the table could be mistaken for cubes."""
    # Hide the stock blue-grid visuals; the (invisible) collision ground stays where it was placed.
    try:
        UsdGeom.Imageable(stage.GetPrimAtPath("/World/defaultGroundPlane")).MakeInvisible()
    except Exception as e:
        print(f"[WARN] Could not hide the default ground plane visuals: {e}", flush=True)

    def box(name, center, size, color):
        VisualCuboid(prim_path=f"/World/Lab/{name}", name=f"lab_{name}",
                     position=np.array(center, dtype=float), scale=np.array(size, dtype=float),
                     color=np.array(color, dtype=float))

    WHITE, OFFWHITE = (0.93, 0.94, 0.95), (0.88, 0.89, 0.90)
    LIGHTGREY, MIDGREY, DARKGREY = (0.74, 0.76, 0.78), (0.52, 0.54, 0.57), (0.16, 0.17, 0.19)
    STEEL = (0.68, 0.70, 0.73)
    fz, top = LAB_FLOOR_Z, 2.55                 # floor height, ceiling height of the room (world Z)
    wall_h, wall_zc = top - fz, (top + fz) / 2.0

    # --- room: floor + back wall (behind the robot) + side walls -----------------------------------
    box("floor",      (0.5, 0.0, fz - 0.01),    (4.6, 4.6, 0.02),  (0.70, 0.72, 0.74))
    box("wall_back",  (-1.25, 0.0, wall_zc),    (0.06, 4.6, wall_h), OFFWHITE)
    box("wall_right", (0.5, 2.1, wall_zc),      (4.6, 0.06, wall_h), OFFWHITE)
    box("wall_left",  (0.5, -2.1, wall_zc),     (4.6, 0.06, wall_h), OFFWHITE)
    box("skirt_back", (-1.21, 0.0, fz + 0.06),  (0.03, 4.6, 0.12), DARKGREY)
    box("skirt_right",(0.5, 2.06, fz + 0.06),   (4.6, 0.03, 0.12), DARKGREY)

    # --- one continuous bench: the (unchanged) work table in front, extended behind the robot --------
    # The robot stands at X=0, on the table's near edge; a same-tone extension behind it makes the base sit
    # on the bench instead of overhanging it, and leaves room for a rack there later.
    box("bench_top_ext", (-0.475, 0.0, -0.05), (0.95, 1.30, 0.10), (0.16, 0.18, 0.22))   # same tone as the table top
    bx0, bx1 = -0.93, 0.86                                                              # body spans the whole bench
    bcx, blen = (bx0 + bx1) / 2.0, bx1 - bx0
    bh = (-0.10 - fz) - 0.01                                                            # body height under the slab
    box("bench_body",    (bcx, 0.0, (-0.10 + fz) / 2.0 - 0.005), (blen, 1.20, bh), WHITE)
    box("bench_toekick", (bcx, 0.0, fz + 0.05),                  (blen - 0.04, 1.16, 0.10), DARKGREY)
    for i, y in enumerate((-0.40, 0.0, 0.40)):                  # drawer fronts + handles on the +X face (front)
        box(f"drawer_x_{i}", (bx1 + 0.003, y, -0.40), (0.006, 0.36, 0.55), LIGHTGREY)
        box(f"handle_x_{i}", (bx1 + 0.012, y, -0.20), (0.012, 0.18, 0.02), STEEL)
    for i, x in enumerate((-0.70, -0.25, 0.20, 0.65)):          # ... and on the -Y face (the side the overview camera sees)
        box(f"drawer_y_{i}", (x, -0.603, -0.40), (0.38, 0.006, 0.55), LIGHTGREY)
        box(f"handle_y_{i}", (x, -0.612, -0.20), (0.18, 0.012, 0.02), STEEL)

    # --- backdrop furniture (all neutral) -----------------------------------------------------------
    box("wallcab_back",  (-1.07, 0.0, 1.55), (0.32, 2.2, 0.60), WHITE)                # wall cabinets above the shelves
    box("shelf_back_lo", (-1.05, 0.0, 0.85), (0.36, 2.2, 0.03), STEEL)
    box("shelf_back_hi", (-1.05, 0.0, 1.12), (0.36, 2.2, 0.03), STEEL)
    for i, (y, w) in enumerate(((-0.8, 0.30), (-0.35, 0.22), (0.15, 0.34), (0.7, 0.26))):  # reagent-box silhouettes
        box(f"rbox_lo_{i}", (-1.05, y, 0.85 + 0.02 + 0.09), (0.22, w, 0.18), LIGHTGREY if i % 2 else OFFWHITE)
        box(f"rbox_hi_{i}", (-1.05, y + 0.1, 1.12 + 0.02 + 0.07), (0.20, w * 0.8, 0.14), MIDGREY if i % 2 else LIGHTGREY)
    box("fumehood",      (0.2, 2.0, 0.35), (1.3, 0.2, 2.1), WHITE)                    # fume-hood block on the right wall
    box("fumehood_glass",(0.2, 1.895, 0.85), (1.1, 0.01, 0.8), DARKGREY)
    box("fumehood_sill", (0.2, 1.88, 0.42),  (1.2, 0.05, 0.03), STEEL)
    box("fumehood_ctrl", (0.2, 1.895, -0.15), (1.1, 0.01, 0.25), MIDGREY)
    print("[Isaac Sim Service] Lab background built (visual-only: floor, walls, bench body, shelving, fume hood).", flush=True)

if BACKGROUND == "lab":
    build_lab_background()

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

def get_hand_pos() -> np.ndarray:
    """World position of the hand's centre (panda_hand frame): the point between the fingers. The end-effector frame
    (panda_rightfinger) sits ~2.5 cm to one side of it, so arrival at a pick/place xy is judged on this one."""
    try:
        prim = stage.GetPrimAtPath("/World/Franka/panda_hand")
        t = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0).ExtractTranslation()
        return np.array([t[0], t[1], t[2]], dtype=np.float32)
    except Exception:
        return get_end_effector_pos()


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

if os.environ.get("PHYSYK_RANDOM_SEED"):
    np.random.seed(int(os.environ["PHYSYK_RANDOM_SEED"]))      # same sequence of "randomize" layouts on every start (for A/B tests)
    print(f"[Isaac Sim Service] numpy random seed fixed to {os.environ['PHYSYK_RANDOM_SEED']}.", flush=True)


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
CUBE_RANDOM_Y_RANGE = (-0.15, 0.42)   # y 0.54 (radius 0.70) was beyond the downward-gripper reach: the hand stopped 10 cm above the cube
CUBE_MAX_RADIUS_M = 0.66
CUBE_MIN_RADIUS_M = float(os.environ.get("PHYSYK_MIN_RADIUS", "0.38"))  #   # closer than this the arm folds until joint 6 reaches its limit (3.7525 rad) and the grasp fails (measured: a pick at 0.33 m)
CUBE_MIN_SEPARATION = 0.10  # meters between cube centers, so they don't spawn overlapping

def randomize_scene_cubes():
    """Places all 3 cubes at genuinely random XY positions each call (np.random, not a fixed
    list) within the arm's reachable workspace, clear of the tray, with a minimum separation
    so they don't spawn on top of each other. Each cube also gets a random turn about Z."""
    placed, yaws = [], []
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
                if CUBE_MIN_RADIUS_M <= math.hypot(cx, cy) <= CUBE_MAX_RADIUS_M and all(math.hypot(cx - px, cy - py) >= CUBE_MIN_SEPARATION for px, py in placed):
                    x, y = cx, cy
                    break
            if x is None:  # 50 rejection-sampling attempts failed (very unlikely with 3 cubes)
                x, y = cx, cy
            placed.append((x, y))
            yaw = float(np.random.uniform(-np.pi / 2, np.pi / 2))        # random turn about Z (a cube repeats every 90 deg)
            obj.set_world_pose(position=np.array([x, y, 0.025]),
                               orientation=np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]))
            yaws.append(round(float(np.degrees(yaw)), 1))
            obj.set_linear_velocity(np.zeros(3))
            obj.set_angular_velocity(np.zeros(3))
        except Exception as e:
            print(f"[Randomize Obj Error] {key}: {e}", flush=True)
    print(f"[Isaac Sim Service] Cubes randomized to: {[(round(x,3), round(y,3)) for x,y in placed]}, yaw (deg) {yaws}", flush=True)

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

# (The side-profile and top-down cameras were removed: two views — overview and wrist — are
#  enough, and every extra 960x540 render product costs frame rate.)

# 3. Wrist Camera (Mounted on panda_hand, offset to the side so a grasped object sits beside
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

# ── Perception camera plumbing (Cosmos integration, Phase A "camera plumbing" step
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
# stage's world up-axis is Z (confirmed by this file's own cameras, which sit high on +Z and look down at
# the table). If SCENE_CAM_POSITION/SCENE_CAM_LOOK_AT below ever change, this
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

# ── Cosmos provider, shadow mode (cosmos_integration.md Step 5) ──────────────────
# Wired in purely additively: get_object_pose() below always returns the same stage
# (ground-truth USD) pose in "stage"/"shadow" modes, so resolve_pick_place_targets()'s own
# real dispatch/decision logic (unmodified) still drives the robot exactly as before. The
# only new thing this adds is a real Cosmos grounding call per dispatched pick/place
# instruction, purely for logging a delta via /perception — zero effect on robot behavior
# unless PERCEPTION_MODE=vision is explicitly set.
from physyk_perception.provider import PerceptionProvider, PerceptionUnavailable
from physyk_perception.overlay_layout import layout_labels

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
    # Table and tray never move — in pure-perception mode the tray is this calibration constant
    # (matches where the FixedCuboid is created above), not read from the stage or detected.
    fixed_poses={"target_tray": np.array([0.42, -0.32, 0.015])},
)
print(f"[Perception] Provider initialized: mode='{perception_provider.mode}' detector='{perception_provider.detector}'", flush=True)


def decision_pos(obj_key):
    """Pose the planner should act on. 'shadow'/'stage': simulator ground truth (deterministic
    path). 'pure': camera-derived only — raises PerceptionUnavailable if the object isn't seen."""
    return perception_provider.get_object_pose(obj_key)


def fresh_decision_pos(obj_key):
    """Re-localize from the CURRENT camera frame, then return decision_pos. Used for post-place
    verification so 'did it land there?' is answered by looking, not by reading the stage."""
    if perception_provider.mode == "pure":   # deterministic mode reads the stage; no detector call
        try:
            perception_provider.refresh(list(_PERCEPTION_LABEL_MAP))
        except Exception as e:
            print(f"[Perception] Fresh refresh failed (non-fatal): {e}", flush=True)
    return decision_pos(obj_key)


perception_log = collections.deque(maxlen=40)   # newest last; entries carry an annotated JPEG
_perception_log_id = 0


def log_perception_event(event: str, note: str = ""):
    """Records what the camera saw right now — an annotated overview frame (boxes + estimated
    XYZ) plus the per-object estimates — for the GUI's Perception Log page. Pure mode only.
    Never raises: logging must not be able to break a task."""
    global _perception_log_id
    if perception_provider.mode != "pure":
        return
    try:
        raw = scene_annot.get_data()
        if raw is None or raw.size == 0:
            return
        bgr = cv2.cvtColor(raw[:, :, :3], cv2.COLOR_RGB2BGR)
        bgr = draw_perception_overlay(bgr)
        ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            return
        snap = perception_provider.snapshot()
        objs = {}
        for key, e in (snap.get("objects") or {}).items():
            objs[key] = {"seen": e.get("vision_pos_m") is not None,
                         "camera_xyz_m": [round(v, 4) for v in e["vision_pos_m"]] if e.get("vision_pos_m") else None,
                         "yaw_deg": None if e.get("yaw_deg") is None else round(e["yaw_deg"], 1),
                         "score_error_mm": None if e.get("delta_mm") is None else round(e["delta_mm"], 1)}
        _perception_log_id += 1
        perception_log.append({"id": _perception_log_id, "t": time.strftime("%H:%M:%S"), "event": event,
                               "note": note, "objects": objs, "jpeg": enc.tobytes()})
    except Exception as e:
        print(f"[Perception] log_perception_event failed (non-fatal): {e}", flush=True)


def telemetry_object_view():
    """Objects as the rest of the system (orchestrator, Nemotron prompt, guardrails) is allowed
    to see them. Pure mode: only what the camera currently localizes — anything not seen is
    absent. Otherwise: simulator ground truth, as before."""
    if perception_provider.mode == "pure":
        view = {}
        snap_objects = perception_provider.snapshot().get("objects", {})
        for key, info in SCENE_OBJECTS.items():
            entry = snap_objects.get(key)
            if entry and entry.get("vision_pos_m") is not None:
                view[key] = {"name": info["name"], "position": np.round(entry["vision_pos_m"], 4).tolist(),
                             "source": "camera"}
        for key in perception_provider.fixed_poses:
            view[key] = {"name": SCENE_OBJECTS[key]["name"],
                         "position": np.round(perception_provider.fixed_poses[key], 4).tolist(),
                         "source": "fixed calibration"}
        return view
    return {key: {"name": info["name"], "position": np.round(get_live_object_pos(key), 4).tolist()}
            for key, info in SCENE_OBJECTS.items()}

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
    # Arm damping: stock Franka drive (kp 22918, kd 4584), which is what the cube pick-and-place was validated with.
    # A lower kd makes the arm faster but jerkier. Set PHYSYK_ARM_KD=<value> to experiment (0 = stock).
    ARM_KD = float(os.environ.get("PHYSYK_ARM_KD", "0"))
    if ARM_KD > 0:
        articulation_view.set_gains(kds=np.full((1, 7), ARM_KD, dtype=np.float32), joint_indices=np.arange(7))
        print(f"[Isaac Sim Service] Arm joint damping set to kd={ARM_KD:.0f} (stock 4584; PHYSYK_ARM_KD=0 keeps stock).", flush=True)
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
# 1.875 (+50%) was tried and failed to pick the cube, so this is back at 1.25. Only the motion
# phases in _MOTION_ONLY_PHASES scale — settle/close/release keep their original timing.
# Override with PHYSYK_SPEEDUP=<factor> to experiment without editing code.
SPEEDUP_FACTOR = float(os.environ.get("PHYSYK_SPEEDUP", "1.25"))
# Per-group speed-ups on top of the motion-only set (settle / close / release never scale):
#   free-space moves   0 above target, 5 transport, 9 return      PHYSYK_SPEEDUP_TRANSIT (default = PHYSYK_SPEEDUP)
#   vertical moves     1 descend to grasp, 4 lift, 6 descend to place, 8 ascend    PHYSYK_SPEEDUP_VERTICAL (default = PHYSYK_SPEEDUP)
# Defaults equal the validated 1.25 baseline: 1.5 / 1.75 made the open-loop controller outrun the arm (missed grasps,
# wandering motion). Raise these only together with a check of the grasp success rate.
_PHASE_SPEEDUP = {i: SPEEDUP_FACTOR for i in _MOTION_ONLY_PHASES}
for _i in (0, 5, 9):
    _PHASE_SPEEDUP[_i] = float(os.environ.get("PHYSYK_SPEEDUP_TRANSIT", SPEEDUP_FACTOR))
for _i in (1, 4, 6, 8):
    _PHASE_SPEEDUP[_i] = float(os.environ.get("PHYSYK_SPEEDUP_VERTICAL", SPEEDUP_FACTOR))
events_dt = [dt * _PHASE_SPEEDUP.get(i, 1.0) for i, dt in enumerate(_FRANKA_DEFAULT_EVENTS_DT)]
print(f"[Isaac Sim Service] Phase speed-ups: {_PHASE_SPEEDUP}", flush=True)
pick_place_controller = PickPlaceController(
    name="pick_place_controller",
    gripper=robot.gripper,
    robot_articulation=robot,
    events_dt=events_dt,
)
articulation_controller = robot.get_articulation_controller()

# ── 4.7. RMPFlow collision obstacles (optional, off by default) ─────────────────────────────────────────────────
# RMPFlow only avoids what is registered with it. NOTE: PickPlaceController.reset() clears every registered obstacle
# (measured: 2 registered -> 0 after reset), so registering once at startup does nothing once the first task resets the
# controller. The obstacles are therefore re-registered after EVERY reset (_controller_reset below).
#   PHYSYK_ARM_OBSTACLES=off     (default) no avoidance: the arm travels well above the cubes and descends vertically, the
#                                hand yaw is picked to keep the fingers clear of neighbours, and in measurements active
#                                obstacles were the only setup in which the arm ever swung joints 1-6 to the velocity cap
#   PHYSYK_ARM_OBSTACLES=static  table + tray
#   PHYSYK_ARM_OBSTACLES=all     table + tray + the cubes (the cube being picked / stacked onto is disabled per task)
# PHYSYK_CUBE_OBSTACLES=1 is the old spelling of "all".
rmpflow_motion_policy = pick_place_controller._cspace_controller.get_motion_policy()
CUBE_OBJECTS = {"red_cube": red_cube_obj, "blue_cube": blue_cube_obj, "green_cube": green_cube_obj}
ARM_OBSTACLE_MODE = os.environ.get("PHYSYK_ARM_OBSTACLES", "all" if os.environ.get("PHYSYK_CUBE_OBSTACLES", "0") == "1" else "off").lower()
if ARM_OBSTACLE_MODE not in ("off", "static", "all"):
    ARM_OBSTACLE_MODE = "off"
CUBE_OBSTACLES = ARM_OBSTACLE_MODE == "all"


def _register_arm_obstacles():
    """(Re-)registers the RMPFlow obstacles for the configured mode. Called after every controller reset."""
    if ARM_OBSTACLE_MODE == "off":
        return
    try:
        rmpflow_motion_policy.add_obstacle(work_table_obj, static=True)
        rmpflow_motion_policy.add_obstacle(target_tray_obj, static=True)
        if CUBE_OBSTACLES:
            for cube_obj in CUBE_OBJECTS.values():
                rmpflow_motion_policy.add_obstacle(cube_obj, static=False)
    except Exception as e:
        print(f"[WARN] Could not register RMPFlow obstacles, planning will be collision-blind: {e}", flush=True)


def _controller_reset():
    """pick_place_controller.reset() + re-register the obstacles that reset just cleared."""
    pick_place_controller.reset()
    _register_arm_obstacles()


print(f"[Isaac Sim Service] RMPFlow obstacles: mode '{ARM_OBSTACLE_MODE}' (registered after every controller reset).", flush=True)

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
    "scene_raw_jpeg": None,   # overview frame WITHOUT the perception overlay (what Cosmos should see)
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
    # Why the last task never started ("cannot plan a sideways grasp ...", "no free slot ...") and whether that is a
    # refusal that retrying cannot fix. The agent shows the text to Nemotron and does not blindly retry a refusal.
    "last_task_error": None,
    "last_task_refused": False,
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
RELEASE_HEIGHT_MARGIN_M = 0.004   # open the gripper when the cube centre is this close to its placing height
# ── Joint trace (PHYSYK_JOINT_TRACE=1): one CSV row per step of a pick/place: every arm joint's position, commanded target,
# velocity and tracking error, the controller phase, the hand and the carried cube. For finding overshoot / oscillation.
JOINT_TRACE = None
if os.environ.get("PHYSYK_JOINT_TRACE", "1") != "0":
    JOINT_TRACE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "joint_trace.csv"), "w", buffering=1)
    JOINT_TRACE.write("wall_s,task,event,ev_t,psi_deg," + ",".join(f"q{i+1}" for i in range(7)) + ",finger_w,"
                      + ",".join(f"cmd{i+1}" for i in range(7)) + "," + ",".join(f"qd{i+1}" for i in range(7)) + ","
                      + ",".join(f"err{i+1}" for i in range(7)) + ",ee_x,ee_y,ee_z,cube_x,cube_y,cube_z,hand_x,hand_y,hand_z\n")
_trace_t0 = time.time()
_trace_task = [0]
_trace_prev = {"qd": None, "n": 0, "last_print": 0.0}
ANOMALY_ERR_RAD = 0.06       # commanded-vs-actual error on a joint
ANOMALY_DQD_STEP = 0.6       # velocity change within one step (rad/s): a kick / jerk


def _trace_joints(event, ev_t, psi, actions, task):
    try:
        q = np.asarray(robot.get_joint_positions(), float).flatten()
        qd = np.asarray(robot.get_joint_velocities(), float).flatten()
        cmd = getattr(actions, "joint_positions", None)
        cmd = [np.nan] * 9 if cmd is None else [np.nan if v is None else float(v) for v in np.asarray(cmd, dtype=object).flatten()]
        cmd = (cmd + [np.nan] * 9)[:9]
        err = [cmd[i] - q[i] if np.isfinite(cmd[i]) else np.nan for i in range(7)]
        ee = get_end_effector_pos()
        cube = get_live_object_pos(task["matched_target"])
        row = [time.time() - _trace_t0, task.get("target_name", "?").replace(",", " "), event, ev_t, psi]
        row += list(q[:7]) + [float(q[7] * 2.0) if len(q) > 7 else np.nan] + cmd[:7] + list(qd[:7]) + err
        row += list(ee) + list(cube) + list(get_hand_pos())
        # anomaly lines in the sim log (joints 2-6 are the ones that swing the arm): large tracking error or a velocity kick
        qd7 = np.array(qd[:7])
        prev = _trace_prev["qd"]
        bad = []
        for j in range(1, 6):
            if np.isfinite(err[j]) and abs(err[j]) > ANOMALY_ERR_RAD:
                bad.append(f"j{j+1} err {err[j]:+.2f}")
            if prev is not None and abs(qd7[j] - prev[j]) > ANOMALY_DQD_STEP:
                bad.append(f"j{j+1} kick {qd7[j] - prev[j]:+.2f} rad/s")
        _trace_prev["qd"] = qd7
        if bad and time.time() - _trace_prev["last_print"] > 0.3 and _trace_prev["n"] < 400:
            _trace_prev["last_print"] = time.time(); _trace_prev["n"] += 1
            print(f"[Joint anomaly] {task.get('target_name', '?')} phase {event} t={ev_t:.2f}: " + ", ".join(bad)
                  + f" | hand z {get_hand_pos()[2]:.3f} q " + " ".join(f"{v:.2f}" for v in q[:7]), flush=True)
        JOINT_TRACE.write(",".join(x if isinstance(x, str) else (str(int(x)) if isinstance(x, (int, np.integer)) else f"{x:.5f}") for x in row) + "\n")
    except Exception as e:
        print(f"[WARN] joint trace failed: {e}", flush=True)


RELEASE_MIN_DESCENT_M = 0.03      # the cube must start the lowering phase at least this far above its placing height
RELEASE_STALL_FRAMES = 6          # ...or when it has not descended RELEASE_STALL_DZ_M over this many frames in phase 6
RELEASE_STALL_DZ_M = 0.0004
CARRY_CLEARANCE_M = 0.04          # carried cube's bottom must clear the highest other cube's top by this much
ARRIVE_XY_M = 0.015              # hand within this of the target xy before the approach / transport phase ends
ARRIVE_WAIT_MAX_FRAMES = 400     # give up waiting after this many extra frames
LIFT_WAIT_MAX_FRAMES = 150        # give up waiting for that after this many frames
RELEASE_HOLD_FRAMES = 14          # frames the arm holds still while the fingers open
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


FREE_SPOT_CLEARANCE_M = 0.15   # a free spot is at least this far (centre to centre) from every other cube
FREE_SPOT_TRAY_MARGIN_M = 0.10  # ...and this far outside the tray's edge: the open hand is ~0.11 m wide


def _tray_distance(x: float, y: float) -> float:
    """Distance (m) from a table point to the tray's footprint (0 inside it)."""
    dx = max(TRAY_X_BOUNDS[0] - x, 0.0, x - TRAY_X_BOUNDS[1])
    dy = max(TRAY_Y_BOUNDS[0] - y, 0.0, y - TRAY_Y_BOUNDS[1])
    return math.hypot(dx, dy)


def _free_table_spot(moving_key: str, from_pos) -> np.ndarray:
    """A table spot inside the arm's reach (CUBE_RANDOM_*_RANGE) with real room for the hand around it: at least
    FREE_SPOT_CLEARANCE_M from every other cube (never back onto the stack the cube is leaving) and
    FREE_SPOT_TRAY_MARGIN_M from the tray. The nearest such spot to the cube's current position wins; if the table is too
    crowded for that, the spot with the most room is used."""
    others = []
    for key in CUBE_OBJECTS:
        if key == moving_key:
            continue
        try:
            others.append(np.asarray(decision_pos(key), float)[:2])
        except PerceptionUnavailable:
            continue
    good, roomiest, best_room = [], None, -1.0
    for x in np.arange(CUBE_RANDOM_X_RANGE[0], CUBE_RANDOM_X_RANGE[1] + 1e-6, 0.03):
        for y in np.arange(CUBE_RANDOM_Y_RANGE[0], CUBE_RANDOM_Y_RANGE[1] + 1e-6, 0.03):
            if not (CUBE_MIN_RADIUS_M <= math.hypot(x, y) <= CUBE_MAX_RADIUS_M):
                continue                                   # outside what the downward hand can grasp (too close folds the wrist to its limit)
            cube_room = min([math.hypot(x - o[0], y - o[1]) for o in others] or [1.0])
            tray_room = _tray_distance(x, y)
            d = math.hypot(x - from_pos[0], y - from_pos[1])
            if cube_room >= FREE_SPOT_CLEARANCE_M and tray_room >= FREE_SPOT_TRAY_MARGIN_M:
                good.append((d, np.array([x, y])))
            room = min(cube_room / FREE_SPOT_CLEARANCE_M, tray_room / FREE_SPOT_TRAY_MARGIN_M)
            if room > best_room:
                best_room, roomiest = room, np.array([x, y])
    if good:
        return min(good, key=lambda g: g[0])[1]
    if roomiest is None or best_room < 0.6:
        raise RuntimeError("there is no free spot on the table with enough room around it to move it to")
    return roomiest


# ── Grasp orientation (wrist rotation) ─────────────────────────────────────────────────────────────────────────
# The hand points straight down; rotating it about world Z (the last wrist joint, joint 7) turns the finger axis. At yaw 0
# the fingers open along world Y. For a cube turned by theta, closing the fingers on a pair of faces needs yaw = theta
# (mod 90 deg: a cube is symmetric). Between the two perpendicular choices (and +-90) the one whose open fingers sweep
# no neighbouring cube (or the tray rim) is used, so a tight pair of cubes is gripped from the free side instead of being
# knocked over.
FINGER_SWEEP_HALF_M = 0.075     # open fingers reach this far from the cube centre along the finger axis
FINGER_PAD_HALF_M = 0.045       # neighbour cube half-width (0.025) + finger pad half-width (~0.02) across the finger axis
MAX_GRIP_YAW_DEG = 100.0


def _wrap90(deg: float) -> float:
    return ((deg + 45.0) % 90.0) - 45.0


def _cube_yaw_deg(key: str) -> float:
    """Current yaw of a cube about Z, degrees. Pure mode: the camera's estimate; otherwise the simulator's pose."""
    try:
        if perception_provider.mode == "pure":
            y = (perception_provider.snapshot().get("objects", {}).get(key) or {}).get("yaw_deg")
            return float(y) if y is not None else 0.0
        q = UsdGeom.Xformable(stage.GetPrimAtPath(SCENE_OBJECTS[key]["path"])).ComputeLocalToWorldTransform(0).RemoveScaleShear().ExtractRotationQuat()
        w, (x, y, z) = q.GetReal(), tuple(q.GetImaginary())
        return float(np.degrees(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))))
    except Exception:
        return 0.0


def _finger_hazard(psi_deg: float, xy, neighbours, near_table: bool = True) -> float:
    """How badly the open fingers at yaw psi (hand centred on xy) would hit neighbouring cubes or the tray rim (0 = clear)."""
    r = np.radians(psi_deg)
    f = np.array([-np.sin(r), np.cos(r)])          # finger axis (world)
    p = np.array([np.cos(r), np.sin(r)])           # across the fingers
    hazard = 0.0
    for n in neighbours:
        d = np.asarray(n, float)[:2] - np.asarray(xy, float)[:2]
        along, across = abs(float(d @ f)), abs(float(d @ p))
        if along < FINGER_SWEEP_HALF_M and across < FINGER_PAD_HALF_M:
            hazard += (FINGER_SWEEP_HALF_M - along) + (FINGER_PAD_HALF_M - across)
    if near_table:
        for sign in (-1.0, 1.0):
            tip = np.asarray(xy, float)[:2] + sign * f * 0.05
            if TRAY_X_BOUNDS[0] - 0.01 <= tip[0] <= TRAY_X_BOUNDS[1] + 0.01 and TRAY_Y_BOUNDS[0] - 0.01 <= tip[1] <= TRAY_Y_BOUNDS[1] + 0.01:
                hazard += 0.05
    return hazard


def _choose_grip_yaw(theta_deg: float, xy, neighbours, prefer_deg: float = 0.0, near_table: bool = True) -> float:
    """Hand yaw (deg) aligned with a cube turned by theta_deg: the face-aligned choice with the least finger hazard, then
    the smallest turn from prefer_deg."""
    base = _wrap90(theta_deg)
    cands = [c for c in (base - 90.0, base, base + 90.0) if abs(c) <= MAX_GRIP_YAW_DEG]
    return min(cands, key=lambda c: (round(_finger_hazard(c, xy, neighbours, near_table), 4), abs(c - prefer_deg), abs(c)))


def _grip_quat(psi_deg: float) -> np.ndarray:
    """Gripper-down orientation turned psi about world Z: Rz(psi) * Ry(pi) as (w, x, y, z); psi = 0 is the stock
    euler_angles_to_quat([0, pi, 0])."""
    h = np.radians(psi_deg) / 2.0
    return np.array([0.0, -np.sin(h), np.cos(h), 0.0])


def _plan_grip_yaws(matched_key, picking_position, placing_position):
    """(yaw at the pick, yaw at the place), degrees, chosen from the cubes' current positions and turns."""
    others = {}
    for k in CUBE_OBJECTS:
        if k == matched_key:
            continue
        try:
            others[k] = np.asarray(decision_pos(k), float)
        except PerceptionUnavailable:
            continue
    theta = _cube_yaw_deg(matched_key)
    psi_pick = _choose_grip_yaw(theta, picking_position, [p for p in others.values()])
    # Destination: a stack takes the turn of the cube underneath (so the pile is square); the table, the tray and free
    # spots are world-aligned. Cubes underneath are not neighbours of the placed cube.
    below = [k for k, p in others.items() if math.hypot(p[0] - placing_position[0], p[1] - placing_position[1]) < 0.035
             and placing_position[2] - p[2] > 0.035]
    theta_place = _cube_yaw_deg(below[0]) if below else 0.0
    place_neighbours = [p for k, p in others.items() if k not in below]
    psi_place = _choose_grip_yaw(theta_place, placing_position, place_neighbours, prefer_deg=psi_pick, near_table=not below)
    return psi_pick, psi_place


def _find_object_mentions(inst: str):
    """Every (string_index, object_key) for object keywords found in inst, sorted by position. Where two
    objects' keywords overlap the same words, the LONGEST keyword wins — so "red beaker" resolves to the beaker
    and not to the red cube's bare keyword "red"."""
    hits = []   # (start, end, key)
    for key, data in SCENE_OBJECTS.items():
        if key == "target_tray":
            continue
        for kw in data["keywords"]:
            idx = inst.find(kw)
            if idx != -1:
                hits.append((idx, idx + len(kw), key))
    hits.sort(key=lambda h: (-(h[1] - h[0]), h[0]))          # longest first
    chosen = []
    for start, end, key in hits:
        if any(start < c[1] and c[0] < end and c[2] != key for c in chosen):
            continue                                         # overlaps a longer match for a different object
        chosen.append((start, end, key))
    mentions = sorted(((start, key) for start, _end, key in chosen), key=lambda pair: pair[0])
    return mentions


def _pick_free_tray_slot(exclude_key: str) -> np.ndarray:
    """Returns the most-preferred tray slot not already occupied by another live cube —
    center first, then left, then right (see TRAY_SLOT_PREFERENCE)."""
    occupied = set()
    for key in CUBE_OBJECTS:
        if key == exclude_key:
            continue
        try:
            pos = decision_pos(key)
        except PerceptionUnavailable:
            continue   # a cube the camera can't see can't be judged as occupying a slot
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
        if not mentions:
            raise RuntimeError("no known object is mentioned in the instruction")
        matched_target = mentions[0][1]

    # Cosmos perception (cosmos_integration.md Steps 5 & 8): ground every object
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

    log_perception_event(f"Pre-pick detection", f"Instruction: {instruction} — target {matched_target}")
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
    elif any(k in inst for k in ("free spot", "clear spot", "empty spot", "on the table")):
        # "Move it out of the way": any reachable table spot clear of every other cube (and the tray).
        spot = _free_table_spot(matched_target, picking_position)
        placing_position = np.array([spot[0], spot[1], CUBE_HALF_HEIGHT])
        print(f"[Planner] '{target_name}' -> free table spot {np.round(spot, 3)}", flush=True)
    elif wants_tray:
        slot_xy = _pick_free_tray_slot(matched_target)
        # TRAY_SLOTS are absolute for the default tray spot — shift by how far the tray actually
        # is from it (0 in deterministic mode; the camera's measurement in pure mode).
        default_tray = SCENE_OBJECTS["target_tray"]["default_pos"]
        slot_xy = np.array([slot_xy[0] + (tray_pos[0] - default_tray[0]),
                            slot_xy[1] + (tray_pos[1] - default_tray[1])])
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
            <li><a href="/camera/wrist.stream" target="_blank">/camera/wrist.stream</a> — Wrist Camera (End-Effector)</li>
        </ul>
        <h3>📊 Telemetry & Control APIs</h3>
        <ul>
            <li><a href="/health" target="_blank">/health</a> — Service Health & Simulation State</li>
            <li><a href="/state" target="_blank">/state</a> — Live 7-DOF Joint Positions & FPS Telemetry</li>
            <li><a href="/api/cameras" target="_blank">/api/cameras</a> — List of all 3 cameras</li>
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
            "cameras": 3,
            "lighting": current_lighting_preset
        }

@api_app.get("/state")
def get_state():
    with telemetry_lock:
        return JSONResponse(sim_telemetry)

@api_app.get("/perception/camera")
def get_perception_camera():
    """Cosmos integration, Phase A "camera plumbing" — real depth + intrinsics/
    extrinsics for the overview camera, exposed for verification. Not consumed by any
    decision-making path yet; resolve_pick_place_targets and everything it drives are
    completely untouched by this."""
    with telemetry_lock:
        return JSONResponse(perception_camera_state)

@api_app.get("/perception")
def get_perception():
    """Cosmos integration Step 5 — shadow-mode snapshot: the last grounding call's
    per-object stage pose, Cosmos-estimated pose, and delta in mm. `used` is always "stage"
    unless PERCEPTION_MODE=vision is set AND that object's estimate passed the sanity gate —
    see physyk_perception/provider.py. Purely informational; resolve_pick_place_targets and
    real robot dispatch are unaffected by this in stage/shadow mode."""
    return JSONResponse(perception_provider.snapshot())

@api_app.post("/perception/debug_dump")
async def perception_debug_dump():
    """Saves the raw overview RGB + depth to workspace/perception_calib/dump_{rgb,depth}.npy on the next frame."""
    _dump_frame_requested[0] = True
    return JSONResponse({"status": "requested"})

@api_app.post("/perception/mode")
async def set_perception_mode(req: Request):
    """Live switch between the deterministic path and the perception path (the GUI's
    "Drive arm from camera" switch). 'shadow' = simulator ground-truth poses drive the robot,
    perception runs alongside and only logs a delta. 'vision' = the camera-derived poses drive
    the robot (with the provider's sanity gate + automatic fallback to ground truth).
    Takes effect on the very next dispatched pick/place — no restart."""
    body = await req.json()
    mode = body.get("mode")
    if mode not in ("shadow", "vision", "pure"):
        return JSONResponse({"error": "mode must be 'shadow', 'vision' or 'pure'"}, status_code=400)
    if mode == "pure":
        perception_provider.detector = "color"   # pure mode needs the fast local detector
    perception_provider.mode = mode
    print(f"[Perception] Mode switched to '{mode}' via API", flush=True)
    _mode_switch_pending_log[0] = mode
    # The sim loop (which owns the camera annotators) refreshes + logs on its next iteration.
    return JSONResponse({"status": "ok", "mode": mode})

@api_app.get("/api/cameras")
def get_cameras():
    return {
        "cameras": [
            {"id": "scene", "name": "3rd Person Isometric Overview", "stream": "/camera/stream"},
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

@api_app.get("/camera/scene_raw.jpg")
def get_scene_raw_jpeg():
    """Overview frame with no perception overlay — this is what the vision-language model (Cosmos)
    is shown, so drawn boxes/labels can't bias what it 'sees'."""
    with frame_lock:
        data = latest_frames["scene_raw_jpeg"] or latest_frames["scene_jpeg"]
    if data is not None:
        return Response(content=data, media_type="image/jpeg")
    return Response(status_code=404)

@api_app.get("/perception/log")
def get_perception_log():
    """Newest-first perception events (no image bytes) — the Perception Log page reads this."""
    return JSONResponse([{k: v for k, v in e.items() if k != "jpeg"} for e in reversed(list(perception_log))])

@api_app.get("/perception/log/{entry_id}.jpg")
def get_perception_log_image(entry_id: int):
    for e in list(perception_log):
        if e["id"] == entry_id:
            return Response(content=e["jpeg"], media_type="image/jpeg")
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
    uvicorn.run(api_app, host=os.environ.get("ISAAC_SIM_BIND", "127.0.0.1"), port=8100, log_level="warning")

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
# cv2.putText's Hershey fonts only draw ASCII — anything else (em dashes in the stage names, bullets,
# arrows...) renders as "???". Everything drawn onto a frame goes through this first.
_HUD_CHAR_MAP = str.maketrans({
    "\u2014": "-", "\u2013": "-", "\u2022": "-", "\u00b7": "-", "\u2192": "->", "\u2190": "<-",
    "\u2026": "...", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u00d7": "x",
    "\u00b0": " deg", "\u2265": ">=", "\u2264": "<=", "\u2713": "OK", "\u00a0": " ",
})


def hud_text(value) -> str:
    """ASCII-safe text for cv2.putText: common punctuation is mapped to ASCII equivalents, accents are
    stripped, and anything that still can't be drawn is dropped rather than shown as '?'."""
    import unicodedata
    text = str(value).translate(_HUD_CHAR_MAP)
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


_OVERLAY_BGR = {"red_cube": (60, 60, 255), "green_cube": (90, 230, 90), "blue_cube": (255, 170, 60),
                "target_tray": (60, 220, 255)}


_FIXED_EXTENTS = {"target_tray": (0.11, 0.12, 0.015)}   # half extents (m) of the tray FixedCuboid (scale 0.22 x 0.24 x 0.03)


def draw_perception_overlay(bgr):
    """Bounding boxes + camera-estimated XYZ for every object the color detector currently sees,
    plus a banner naming which pose source is driving the arm. Drawn on the overview frame only."""
    mode = perception_provider.mode
    snap = perception_provider.snapshot()
    banner = {"pure": "PERCEPTION: CAMERA ONLY", "vision": "PERCEPTION: CAMERA + SIM FALLBACK",
              "shadow": "DETERMINISTIC (perception shadowing)", "stage": "DETERMINISTIC"}.get(mode, mode)
    banner_color = (100, 235, 140) if mode == "pure" else (200, 200, 200)
    cv2.putText(bgr, hud_text(banner), (14, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.5, banner_color, 1, cv2.LINE_AA)
    items = []   # (key, padded box, label text)
    for key, entry in (snap.get("objects") or {}).items():
        box = entry.get("bbox_px")
        pos = entry.get("vision_pos_m")
        if not box or pos is None:
            continue
        x0, y0, x1, y1 = box
        name = SCENE_OBJECTS[key]["name"].upper() if key in SCENE_OBJECTS else key
        items.append((key, (x0 - 3, y0 - 3, x1 + 3, y1 + 3),
                      hud_text(f"{name} ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")))
    # Fixed-pose objects (the tray) are not detected, so they have no mask: draw the box of their known footprint.
    from physyk_perception.deproject import project_world_to_pixel
    for key, fp in perception_provider.fixed_poses.items():
        info = SCENE_OBJECTS.get(key)
        if info is None or key not in _FIXED_EXTENTS:
            continue
        hx, hy, hz = _FIXED_EXTENTS[key]
        px = [project_world_to_pixel((fp[0] + dx, fp[1] + dy, fp[2] + dz), SCENE_CAM_INTRINSICS, SCENE_CAM_WORLD_TRANSFORM)
              for dx in (-hx, hx) for dy in (-hy, hy) for dz in (-hz, hz)]
        px = [p for p in px if p is not None]
        if len(px) < 4:
            continue
        us, vs = [p[0] for p in px], [p[1] for p in px]
        items.append((key, (int(min(us)) - 3, int(min(vs)) - 3, int(max(us)) + 3, int(max(vs)) + 3),
                      hud_text(f"{info['name'].upper()} ({fp[0]:.2f}, {fp[1]:.2f}, {fp[2]:.2f}) FIXED")))
    if not items:
        return bgr
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.38
    sizes = []
    for _, _, text in items:
        (tw, th), base = cv2.getTextSize(text, font, scale, 1)
        sizes.append((tw, th + base))
    # Labels go in a column beside the cluster (stacked cubes' boxes nearly coincide, so a label
    # above each box would collide); leader lines tie each one back to its box.
    h_img, w_img = bgr.shape[:2]
    placed = layout_labels([b for _, b, _ in items], sizes, w_img, h_img)
    for (key, box, text), lay in zip(items, placed):
        color = _OVERLAY_BGR.get(key, (255, 255, 255))
        cv2.rectangle(bgr, (box[0], box[1]), (box[2], box[3]), color, 1, cv2.LINE_AA)
        lx = lay["label_x"] - 3 if lay["side"] == "right" else lay["label_x"] + sizes[items.index((key, box, text))][0] + 3
        cv2.line(bgr, lay["anchor"], (lx, lay["label_mid_y"]), color, 1, cv2.LINE_AA)
        cv2.putText(bgr, text, lay["text_xy"], font, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(bgr, text, lay["text_xy"], font, scale, color, 1, cv2.LINE_AA)
    return bgr


def encode_camera_frame(raw_rgb, label: str, overlay: bool = False):
    if raw_rgb is None or raw_rgb.size == 0:
        return None
    bgr = cv2.cvtColor(raw_rgb[:, :, :3], cv2.COLOR_RGB2BGR)
    if overlay:
        try:
            bgr = draw_perception_overlay(bgr)
        except Exception as e:   # a drawing bug must never take the camera stream down
            print(f"[WARN] Perception overlay failed: {e}", flush=True)
    
    # Subtle, sleek, crisp watermark HUD
    cv2.putText(bgr, hud_text(f"PHYSYK - {label.upper()}"), (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 240, 255), 1, cv2.LINE_AA)
    with telemetry_lock:
        st = sim_telemetry.get("stage", "READY")
    color = (100, 235, 140) if st == "READY" else (100, 200, 255)
    cv2.putText(bgr, hud_text(f"STAGE: {st}"), (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
    
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

# ── Frame-cost knobs ─────────────────────────────────────────────────────────────────────────────────
# The loop used to call rep.orchestrator.step() every frame, which renders ALL camera products a second time
# (world.step(render=True) has already rendered them and the annotators already hold that frame). Skipping it
# took the loop from ~6.9 to ~30 iterations/s with identical picks (verified: fresh frames, deterministic and
# camera-only picks succeed) — and since the controller counts its phases per loop iteration, a pick went from
# ~127 s to ~26 s of wall time.
#   PHYSYK_SKIP_ORCH=0      restore the original second render per frame
#   PHYSYK_PROFILE=1        print where each loop iteration's time goes, every 5 s
PROFILE = bool(os.environ.get("PHYSYK_PROFILE"))
SKIP_ORCH_STEP = os.environ.get("PHYSYK_SKIP_ORCH", "1") != "0"
ENCODE_EVERY = max(1, int(os.environ.get("PHYSYK_ENCODE_EVERY", "2")))   # JPEG-encode the streams every Nth iteration
_enc_iter = 0
_prof = collections.defaultdict(float)
_prof_state = {"n": 0, "t": time.time()}


def _prof_add(name, seconds):
    if PROFILE:
        _prof[name] += seconds


def _prof_report():
    if not PROFILE:
        return
    _prof_state["n"] += 1
    now = time.time()
    if now - _prof_state["t"] >= 5.0:
        n = max(_prof_state["n"], 1)
        parts = ", ".join(f"{k}={1000 * v / n:.0f}ms" for k, v in sorted(_prof.items(), key=lambda kv: -kv[1]))
        print(f"[PROFILE] {n / (now - _prof_state['t']):.1f} it/s | per-iteration: {parts}", flush=True)
        _prof.clear()
        _prof_state.update(n=0, t=now)

_mode_switch_pending_log = [None]   # set by the API thread; the sim loop writes the log entry
_dump_frame_requested = [False]     # set by POST /perception/debug_dump; the sim loop saves the raw frames
_last_live_perception = 0.0   # throttles the idle live-perception refresh in the loop below
_last_telemetry_t = 0.0       # wall-clock fallback for the telemetry block (see "5. Update Telemetry")

try:
    while simulation_app.is_running():
        _t_iter0 = time.perf_counter()
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
            _controller_reset()
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
                sim_telemetry["last_task_error"] = None
                sim_telemetry["last_task_refused"] = False

            if "reset" in cmd:
                # Full reset: cubes back to fixed staging spots AND robot home. Distinct from
                # "home" below — these used to be the same branch, which meant asking the
                # robot to go home silently teleported every cube back to a fixed spot too.
                with telemetry_lock:
                    sim_telemetry["stage"] = "HOMING & RESETTING PROPS"
                reset_scene_cubes()
                held_object_key = None
                _controller_reset()
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
                _controller_reset()
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

                    _controller_reset()
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
                    for key, cube_obj in (CUBE_OBJECTS.items() if CUBE_OBSTACLES else ()):
                        try:
                            if key == matched_target or key == reference_key:
                                rmpflow_motion_policy.disable_obstacle(cube_obj)
                            else:
                                rmpflow_motion_policy.enable_obstacle(cube_obj)
                        except Exception as e:
                            print(f"[WARN] Obstacle enable/disable failed for {key}: {e}", flush=True)
                    _psi_pick, _psi_place = _plan_grip_yaws(matched_target, picking_position, placing_position)
                    print(f"[Planner] Hand yaw: pick {_psi_pick:.0f} deg (cube turned {_cube_yaw_deg(matched_target):.0f} deg), "
                          f"place {_psi_place:.0f} deg", flush=True)
                    active_pick_place = {
                        "psi_pick": _psi_pick, "psi_place": _psi_place,
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
                        sim_telemetry["last_task_error"] = str(e)
                        sim_telemetry["last_task_refused"] = True
                        # A task that never started must not leave the PREVIOUS task's success
                        # flag standing (e.g. pure mode: target not visible to the camera).
                        sim_telemetry["last_task_success"] = False
                        sim_telemetry["last_task_error_mm"] = None
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
                    _controller_reset()
                    start_pos = active_pick_place["picking_position"]
                    try:
                        final_pos = fresh_decision_pos(active_pick_place["matched_target"])
                        displacement_m = float(np.hypot(final_pos[0] - start_pos[0], final_pos[1] - start_pos[1]))
                    except PerceptionUnavailable:
                        displacement_m = None
                    with telemetry_lock:
                        sim_telemetry["last_task_success"] = False
                        sim_telemetry["last_task_error_mm"] = None
                        sim_telemetry["last_object_displacement_mm"] = (
                            round(displacement_m * 1000.0, 1) if displacement_m is not None else None)
                    episode_recorder.end_episode(False)  # early grasp-fail abort — drop the episode
                    try:
                        if not CUBE_OBSTACLES:
                            raise StopIteration     # cubes are not registered in this mode: nothing to re-enable
                        rmpflow_motion_policy.enable_obstacle(CUBE_OBJECTS[active_pick_place["matched_target"]])
                        ref_key = active_pick_place.get("reference_key")
                        if ref_key is not None:
                            rmpflow_motion_policy.enable_obstacle(CUBE_OBJECTS[ref_key])
                    except StopIteration:
                        pass
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

        # Release on contact. Phase 6 lowers the cube on a fixed schedule; if the cube meets the surface early (it sits
        # a few mm low in the fingers, the table/stack surface is a little higher than assumed, or the arm overshoots) the
        # still-closed fingers would keep pushing and drag it sideways. So: open the gripper as soon as the cube reaches its
        # placing height, or stops descending while the schedule is still lowering it.
        _hold_release = False
        if active_pick_place is not None:
            _ev = pick_place_controller.get_current_event()
            if _ev == 6 and not active_pick_place.get("released"):
                try:
                    _cz = float(get_live_object_pos(active_pick_place["matched_target"])[2])   # physical contact sensing: stage pose
                except Exception:
                    _cz = None
                if _cz is not None:
                    _hist = active_pick_place.setdefault("z_hist", [])
                    _hist.append(_cz)
                    _place_z = active_pick_place["placing_position"][2]
                    # Only a cube that is genuinely being LOWERED onto the surface can touch it: it must have started this
                    # phase clearly above its placing height. (A cube carried low, e.g. before the top of a 3-high stack,
                    # is already "at height" without touching anything - releasing it there drops it.)
                    if "descent_from" not in active_pick_place:
                        active_pick_place["descent_from"] = _cz
                    _lowering = active_pick_place["descent_from"] > _place_z + RELEASE_MIN_DESCENT_M
                    _at_height = _lowering and _cz <= _place_z + RELEASE_HEIGHT_MARGIN_M
                    _stalled = (_lowering and len(_hist) > RELEASE_STALL_FRAMES and pick_place_controller._t > 0.25
                                and _hist[-RELEASE_STALL_FRAMES - 1] - _cz < RELEASE_STALL_DZ_M
                                and _cz < _place_z + 0.05)
                    if _at_height or _stalled:
                        active_pick_place["released"] = True
                        active_pick_place["release_hold"] = RELEASE_HOLD_FRAMES
                        print(f"[Isaac Sim Service] Contact release for '{active_pick_place['target_name']}': cube z "
                              f"{_cz * 1000:.1f} mm, placing z {active_pick_place['placing_position'][2] * 1000:.1f} mm "
                              f"({'reached height' if _at_height else 'stopped descending'}).", flush=True)
            if active_pick_place.get("release_hold", 0) > 0:
                # Fingers open, arm holds still, THEN the controller continues from phase 7 (release) -> 8 (ascend).
                articulation_controller.apply_action(robot.gripper.forward(action="open"))
                active_pick_place["release_hold"] -= 1
                _hold_release = True
                if active_pick_place["release_hold"] == 0:
                    pick_place_controller._event, pick_place_controller._t = 7, 0.0
        # Lift clearance. The state machine is open-loop and the arm lags it, so a cube can still be low when transport starts.
        # Hold at the end of the lift until the carried cube's bottom clears the highest other cube (the top of a stack)
        # by CARRY_CLEARANCE_M, or the wait times out.
        if (active_pick_place is not None and pick_place_controller.get_current_event() == 4
                and pick_place_controller._t > 0.85 and active_pick_place.get("lift_wait", 0) < LIFT_WAIT_MAX_FRAMES):
            try:
                _cz_now = float(get_live_object_pos(active_pick_place["matched_target"])[2])
                _top = max([float(get_live_object_pos(k)[2]) for k in CUBE_OBJECTS if k != active_pick_place["matched_target"]] or [0.0])
                _need = min(0.30, _top + CUBE_HALF_HEIGHT + CARRY_CLEARANCE_M + CUBE_HALF_HEIGHT)
                if _cz_now < _need:
                    active_pick_place["lift_wait"] = active_pick_place.get("lift_wait", 0) + 1
                    pick_place_controller._t = 0.85          # stay at the end of the lift (arm keeps climbing)
            except Exception:
                pass
        if active_pick_place is not None and not _hold_release:
            current_joint_positions = robot.get_joint_positions()
            _ev = pick_place_controller.get_current_event()
            _psi0, _psi1 = active_pick_place["psi_pick"], active_pick_place["psi_place"]
            if _ev < 5:
                _psi = _psi0                                   # approach, grasp, lift: aligned with the cube
            elif _ev == 5:                                     # transport: turn to the destination's alignment
                _psi = _psi0 + (_psi1 - _psi0) * 0.5 * (1.0 - np.cos(np.pi * pick_place_controller._t))
            else:
                _psi = _psi1
            _r = np.radians(_psi)
            _ee_offset = np.array([-EE_OFFSET[1] * np.sin(_r) + EE_OFFSET[0] * np.cos(_r),
                                   EE_OFFSET[1] * np.cos(_r) + EE_OFFSET[0] * np.sin(_r), EE_OFFSET[2]])
            actions = pick_place_controller.forward(
                picking_position=active_pick_place["picking_position"],
                placing_position=active_pick_place["placing_position"],
                current_joint_positions=current_joint_positions,
                end_effector_offset=_ee_offset,
                end_effector_orientation=_grip_quat(_psi),
            )
            articulation_controller.apply_action(actions)
            if JOINT_TRACE is not None:
                _trace_joints(_ev, pick_place_controller._t, _psi, actions, active_pick_place)
            # Arrival gating. The state machine is open-loop: phases 0 (move above the cube) and 5 (transport) last only ~100 and
            # ~16 steps, far less than the arm needs, so the arm used to do the whole move DURING the next phase (descending
            # diagonally while still travelling, arriving late and fast). Hold at the end of those two phases until the hand is
            # within ARRIVE_XY_M of the target xy (or the wait times out), so travel happens at the carry height and the
            # descent is a straight vertical move.
            if _ev in (0, 5) and pick_place_controller.get_current_event() == _ev + 1:
                _tgt = active_pick_place["picking_position"] if _ev == 0 else active_pick_place["placing_position"]
                _hand_now = get_hand_pos()
                _dist = math.hypot(_hand_now[0] - _tgt[0], _hand_now[1] - _tgt[1])
                _wk = f"arrive_wait_{_ev}"
                if _dist > ARRIVE_XY_M and active_pick_place.get(_wk, 0) < ARRIVE_WAIT_MAX_FRAMES:
                    active_pick_place[_wk] = active_pick_place.get(_wk, 0) + 1
                    pick_place_controller._event = _ev
                    pick_place_controller._t = 1.0 - 0.5 * pick_place_controller._events_dt[_ev]
                elif active_pick_place.get(_wk, 0) > 0:
                    print(f"[Isaac Sim Service] {'Approach' if _ev == 0 else 'Transport'} arrived after {active_pick_place[_wk]} extra frames "
                          f"(hand {_dist * 1000:.0f} mm from target xy).", flush=True)
            if _ev in (0, 1, 2):
                # Fingers fully open (0.04 each) for the whole approach and settle: never go down onto a cube half-closed.
                articulation_controller.apply_action(robot.gripper.forward(action="open"))

            if pick_place_controller.is_done():
                print(f"[Isaac Sim Service] Task Execution Complete: {sim_telemetry['last_instruction']}", flush=True)
                # Real post-place verification (was a hardcoded always-PASS stub elsewhere —
                # see PLAN.md Part 3). Compares the object's actual final live position against
                # the resolved placing_position it was sent to. XY tolerance is half a cube
                # width; Z gets a looser tolerance since a dropped/settled cube can still be
                # bouncing/rotating slightly when this fires.
                target_pos = active_pick_place["placing_position"]
                start_pos = active_pick_place["picking_position"]
                try:
                    # Deterministic mode: the stage's true pose. Pure mode: a fresh look through
                    # the overview camera — the check is made by seeing where the cube ended up.
                    final_pos = fresh_decision_pos(active_pick_place["matched_target"])
                except PerceptionUnavailable:
                    # Can't see the cube after the place (occluded/knocked out of view): success
                    # cannot be established by looking, so it is NOT claimed.
                    final_pos = np.array([np.nan, np.nan, np.nan])
                log_perception_event("Post-place verification",
                                     f"{active_pick_place['target_name']}: intended {np.round(target_pos, 3).tolist()}, "
                                     f"seen at {np.round(final_pos, 3).tolist()}")
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
                    sim_telemetry["last_task_error_mm"] = round(err_xy * 1000.0, 1) if np.isfinite(err_xy) else None
                    sim_telemetry["last_object_displacement_mm"] = round(displacement_m * 1000.0, 1) if np.isfinite(displacement_m) else None
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
                    if not CUBE_OBSTACLES:
                        raise StopIteration         # cubes are not registered in this mode: nothing to re-enable
                    rmpflow_motion_policy.enable_obstacle(CUBE_OBJECTS[active_pick_place["matched_target"]])
                    ref_key = active_pick_place.get("reference_key")
                    if ref_key is not None:
                        rmpflow_motion_policy.enable_obstacle(CUBE_OBJECTS[ref_key])
                except StopIteration:
                    pass
                except Exception as e:
                    print(f"[WARN] Could not re-enable obstacle after task: {e}", flush=True)
                active_pick_place = None
                held_object_key = None
                with telemetry_lock:
                    sim_telemetry["busy"] = False
                    sim_telemetry["stage"] = "READY"

        # Idle and holding nothing: keep the fingers wide open (0.04 each), so the next grasp never starts half-closed.
        if active_pick_place is None and active_vla is None and held_object_key is None:
            articulation_controller.apply_action(robot.gripper.forward(action="open"))

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
        #    NOTE: physics-only sub-stepping (several world.step(render=False) per rendered frame) was tried to
        #    speed picks up and does NOT work in this Isaac Sim build — render=False does not advance the arm,
        #    so the controller ran ahead of a stationary robot (after 100 controller steps the arm had moved
        #    ~0.04 rad at 8 sub-steps vs ~0.6 rad at 1) and the pick missed. Speed has to come from making
        #    each frame cheaper (see the camera work below), not from stepping physics separately.
        _t_step0 = time.perf_counter()
        world.step(render=True)
        _prof_add("world.step(render)", time.perf_counter() - _t_step0)
        fps_counter += 1
        
        # 4. Capture & Encode the 2 Camera Annotators (overview, wrist)
        _t_orch0 = time.perf_counter()
        if not SKIP_ORCH_STEP:
            rep.orchestrator.step()
        _prof_add("orchestrator.step", time.perf_counter() - _t_orch0)

        _t_get0 = time.perf_counter()
        raw_scene = scene_annot.get_data()
        raw_wrist = wrist_annot.get_data()
        if _dump_frame_requested[0]:
            # Debug aid: save the raw overview RGB + depth (exactly what perception sees) for offline analysis.
            _dump_frame_requested[0] = False
            try:
                _dump_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "perception_calib")
                os.makedirs(_dump_dir, exist_ok=True)
                np.save(os.path.join(_dump_dir, "dump_rgb.npy"), np.asarray(raw_scene))
                np.save(os.path.join(_dump_dir, "dump_depth.npy"), np.asarray(scene_depth_annot.get_data()))
                print(f"[Perception] Raw overview RGB+depth saved to {_dump_dir}", flush=True)
            except Exception as _e:
                print(f"[WARN] Frame dump failed: {_e}", flush=True)
        _prof_add("annotator get_data", time.perf_counter() - _t_get0)

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

        # The JPEG streams only need ~half the loop rate; encoding every Nth iteration frees time for physics steps.
        _enc_iter += 1
        if _enc_iter % ENCODE_EVERY == 0:
            _t_enc0 = time.perf_counter()
            _pure = perception_provider.mode == "pure"
            f_scene = encode_camera_frame(raw_scene, "Isometric Overview", overlay=_pure)
            f_scene_raw = encode_camera_frame(raw_scene, "Isometric Overview") if _pure else f_scene
            f_wrist = encode_camera_frame(raw_wrist, "Wrist Gripper")
            _prof_add("jpeg encode", time.perf_counter() - _t_enc0)

            with frame_lock:
                if f_scene: latest_frames["scene_jpeg"] = f_scene
                if f_scene_raw: latest_frames["scene_raw_jpeg"] = f_scene_raw
                if f_wrist: latest_frames["wrist_jpeg"] = f_wrist
                latest_frames["timestamp"] = time.time()

        # 5. Update Telemetry
        #    Every 5th iteration as before — but fps_counter is reset each second, so below ~5 iterations/s
        #    (a power-capped laptop GPU, heavy load) that condition never fired and telemetry, the object
        #    list and the live perception refresh all froze. Also fire on a 0.5 s wall-clock schedule.
        if fps_counter % 5 == 0 or (time.time() - _last_telemetry_t) >= 0.5:
            _last_telemetry_t = time.time()
            dofs = robot.get_joint_positions()
            if dofs is not None:
                if dofs.ndim > 1:
                    dofs = dofs.flatten()
                ee_pos = get_end_effector_pos()
                live_objects = telemetry_object_view()
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

            if _mode_switch_pending_log[0] is not None:
                _new_mode, _mode_switch_pending_log[0] = _mode_switch_pending_log[0], None
                try:
                    perception_provider.refresh(list(_PERCEPTION_LABEL_MAP))
                except Exception as e:
                    print(f"[Perception] Refresh after mode switch failed (non-fatal): {e}", flush=True)
                log_perception_event(f"Mode switched to {_new_mode}", "Initial camera detection after the switch")
                _last_live_perception = time.time()

            # Live perception readout: while idle, re-localize the cubes from the camera every
            # ~2 s so /perception (and the GUI panel) shows detected-vs-stage delta continuously,
            # not only right after a dispatch. Color detector only — it costs a few ms; the
            # Cosmos detector is a network/VLM call and must never be polled like this.
            # Pure mode only — with the switch off, the deterministic path runs exactly as before
            # (no extra per-frame perception work).
            _live_period = 1.0 if sim_telemetry["busy"] else 0.5
            if (perception_provider.mode == "pure" and perception_provider.detector == "color"
                    and time.time() - _last_live_perception >= _live_period):
                _last_live_perception = time.time()
                try:
                    perception_provider.refresh(list(_PERCEPTION_LABEL_MAP))
                except Exception as e:
                    print(f"[Perception] Live refresh failed (non-fatal): {e}", flush=True)

        # Calculate real FPS
        now = time.time()
        if now - last_fps_time >= 1.0:
            with telemetry_lock:
                sim_telemetry["fps"] = round(fps_counter / (now - last_fps_time), 1)
            fps_counter = 0
            last_fps_time = now
        _prof_add("iteration TOTAL", time.perf_counter() - _t_iter0)
        _prof_report()

except Exception as e:
    print(f"[Isaac Sim Service] Exception occurred: {e}", flush=True)
finally:
    print("[Isaac Sim Service] Shutting down simulation...", flush=True)
    simulation_app.close()
