#!/usr/bin/env python3
"""
Comprehensive Rigorous System Verification Test Suite
======================================================
Tests all physical assets, neural model weights, geometry, and agent pipeline.
"""

import os
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np

# Ensure paths
GR00T_DIR = "/opt/dlami/nvme/physyk/workspace/Isaac-GR00T"
if GR00T_DIR not in sys.path:
    sys.path.insert(0, GR00T_DIR)

WORKSPACE_DIR = "/opt/dlami/nvme/physyk/workspace"
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

PASSED = 0
FAILED = 0

def check(name: str, condition: bool, details: str = ""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ✅ [PASS] {name} {details}")
    else:
        FAILED += 1
        print(f"  ❌ [FAIL] {name} {details}")


print("=" * 80)
print("  PHYSIK AI — RIGOROUS SYSTEM VERIFICATION SUITE")
print("=" * 80)

# ─── TEST 1: Physical Franka Panda URDF & Meshes ──────────────────────────────
print("\n[1/5] Verifying Franka Panda URDF & 3D Mesh Assets...")
urdf_file = "/opt/dlami/nvme/physyk/assets/franka_panda/panda_arm_hand.urdf"
check("URDF file exists on disk", os.path.isfile(urdf_file), f"({os.path.getsize(urdf_file)} bytes)")

tree = ET.parse(urdf_file)
root = tree.getroot()
links = root.findall(".//link")
joints = root.findall(".//joint")
check("URDF has 12 links", len(links) == 12, f"(found {len(links)})")
check("URDF has 11 joints", len(joints) == 11, f"(found {len(joints)})")

# Check meshes
mesh_elements = root.findall(".//mesh")
missing_meshes = []
for elem in mesh_elements:
    rel_path = elem.attrib.get("filename")
    full_path = os.path.join("/opt/dlami/nvme/physyk/assets/franka_panda", rel_path)
    if not os.path.isfile(full_path) or os.path.getsize(full_path) == 0:
        missing_meshes.append(rel_path)

check("All 22 visual & collision 3D meshes exist", len(missing_meshes) == 0, f"(missing: {missing_meshes})")


# ─── TEST 2: LIBERO Benchmark BDDL Suite ─────────────────────────────────────
print("\n[2/5] Verifying LIBERO Benchmark Task Definitions...")
libero_dir = "/opt/dlami/nvme/physyk/assets/libero/libero/libero/bddl_files"
bddl_files = []
for r, d, files in os.walk(libero_dir):
    for f in files:
        if f.endswith(".bddl"):
            bddl_files.append(os.path.join(r, f))

check("130 LIBERO task BDDL files stored locally", len(bddl_files) == 130, f"(found {len(bddl_files)} files)")


# ─── TEST 3: NVIDIA GR00T N1.7 Model Weights & CUDA Tensors ──────────────────
print("\n[3/5] Verifying GR00T N1.7 Weights & GPU Tensors...")
model_dir = "/opt/dlami/nvme/physyk/models/GR00T-N1.7-3B"
weight_file1 = os.path.join(model_dir, "model-00001-of-00002.safetensors")
weight_file2 = os.path.join(model_dir, "model-00002-of-00002.safetensors")

check("GR00T weight shard 1 exists", os.path.isfile(weight_file1), f"({os.path.getsize(weight_file1) / (1024**3):.2f} GB)")
check("GR00T weight shard 2 exists", os.path.isfile(weight_file2), f"({os.path.getsize(weight_file2) / (1024**3):.2f} GB)")

# Test reading safetensors metadata and tensors
import torch
from safetensors.torch import load_file

check("PyTorch CUDA available", torch.cuda.is_available(), f"({torch.cuda.get_device_name(0)})")

# Inspect model weights structure
print("   Inspecting safetensors tensors in shard 2...")
tensors_shard2 = load_file(weight_file2, device="cpu")
tensor_names = list(tensors_shard2.keys())
check("Safetensors file is valid & readable", len(tensor_names) > 0, f"(contains {len(tensor_names)} weight tensors)")
print(f"   Sample weight tensor: '{tensor_names[0]}' -> shape {tensors_shard2[tensor_names[0]].shape}")


# ─── TEST 4: GR00T Policy Service & Modality Formatting ───────────────────────
print("\n[4/5] Testing GR00T N1.7 Policy Service & Modality Mapping...")
from groot_policy_service import GR00TN17PolicyService

policy = GR00TN17PolicyService(model_id=model_dir, device="cuda")
check("Policy service initialized", policy is not None)
check("Embodiment tag is libero_sim", policy.embodiment_tag == "libero_sim")

# Generate test multimodal observation
obs = policy.format_observation(
    scene_image=np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
    wrist_image=np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
    ee_pose=np.array([0.55, 0.00, 0.85, 0.0, 3.1415, 0.0]),
    gripper_width=0.08,
    instruction="grasp the golden gear",
)
check("Observation formatted with video/state/annotation", "video" in obs and "state" in obs and "annotation" in obs)
check("Proprioception vector is 8-DOF", obs["state"].shape == (8,))

action_chunk = policy.predict_action_chunk(obs, horizon=40)
check("Action chunk generated with shape (40, 7)", action_chunk.shape == (40, 7))


# ─── TEST 5: NeMo Agent Brain + ROS 2 RGB-D Unprojection ─────────────────────
print("\n[5/5] Testing NeMo Agentic Brain & ROS 2 RGB-D Bridge...")
from physyk_agent_orchestrator import NeMoRoboticsAgent
from physyk_ros2_rgbd_bridge import ROS2RGBDBridge, unproject_pixel_to_3d

bridge = ROS2RGBDBridge()
# Verify unprojection math
p_world = unproject_pixel_to_3d(128.0, 128.0, depth=0.55, intrinsics=bridge.intrinsics, extrinsics=bridge.extrinsics)
diff = np.abs(p_world - np.array([0.50, 0.00, 0.85]))
check("Pinhole unprojection accurate to <1mm", np.max(diff) < 0.001, f"(result: {p_world.tolist()})")

agent = NeMoRoboticsAgent(policy_service=policy)
plan = agent.decompose_task("pick up the golden gear and place it in the kit tray")
check("NeMo Planner decomposed task into 5 atomic sub-goals", len(plan) == 5)

# Verify guardrail catches out-of-bounds target
safe, msg = agent.rails.validate_cartesian_target(np.array([2.5, 0.0, 0.85]))
check("NeMo Guardrail rejects dangerous out-of-bound target", safe is False)

safe_valid, msg_valid = agent.rails.validate_cartesian_target(np.array([0.55, 0.15, 0.84]))
check("NeMo Guardrail approves valid workcell target", safe_valid is True)

# Format joint trajectory
traj_msg = bridge.format_joint_trajectory_msg(action_chunk, dt=0.02)
check("ROS 2 JointTrajectory formatted with 40 points", traj_msg["num_points"] == 40)


print("\n" + "=" * 80)
print(f"  VERIFICATION COMPLETE: {PASSED} PASSED, {FAILED} FAILED")
print("=" * 80)

if FAILED > 0:
    sys.exit(1)
