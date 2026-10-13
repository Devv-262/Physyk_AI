#!/usr/bin/env python3
"""
Interactive Agent CLI Test Harness
==================================
Allows you to type natural language instructions directly into a CLI prompt
and watch the NeMo Cognitive Brain, Guardrails, and GR00T N1.7 VLA execute them.
"""

import os
import sys
import time
import numpy as np

WORKSPACE_DIR = "/opt/dlami/nvme/physyk/workspace"
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from physyk_agent_orchestrator import NeMoRoboticsAgent
from physyk_ros2_rgbd_bridge import ROS2RGBDBridge


def print_banner(title: str):
    print("\n" + "=" * 75)
    print(f"  🤖 {title}")
    print("=" * 75)


def run_prompt(agent: NeMoRoboticsAgent, bridge: ROS2RGBDBridge, prompt: str):
    print_banner(f"EXECUTING COMMAND: \"{prompt}\"")
    
    # 1. Perception Step
    print("\n[Step 1: Visual Perception & RGB-D Scene Scan]")
    world = agent.scan_workspace()
    print(f"  • Detected {len(world['detected_objects'])} workcell objects:")
    for name, data in world["detected_objects"].items():
        print(f"    - {name:15s} @ XYZ={data['position']} (Reachable: {data['reachable']}, Status: {data['status']})")

    # 2. Cognitive Decomposition
    print("\n[Step 2: NeMo Cognitive Planner Decomposition]")
    subgoals = agent.decompose_task(prompt)
    print(f"  • Generated {len(subgoals)} atomic sub-goals:")
    for idx, sg in enumerate(subgoals):
        print(f"    {idx+1}. [{sg['action_type']:20s}] Target: {sg.get('target_object', 'N/A'):15s} -> {sg['instruction']}")

    # 3. Guardrail Validation & GR00T N1.7 Execution
    print("\n[Step 3: Guardrail Safety Validation & GR00T N1.7 Motor Action Dispatch]")
    for idx, sg in enumerate(subgoals):
        print(f"\n  ▶ Step {idx+1}/{len(subgoals)}: {sg['subgoal_id']}")
        
        # Synthetic camera frame
        scene_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
        wrist_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
        
        result = agent.execute_subgoal(sg, scene_rgb=scene_rgb, wrist_rgb=wrist_rgb)
        
        if result["status"] == "SUCCESS":
            action_chunk = np.zeros((40, 7), dtype=np.float32)
            traj = bridge.format_joint_trajectory_msg(action_chunk, dt=0.02)
            print(f"    ✅ Guardrail Check: PASSED (Target XYZ={result['target_pos']})")
            print(f"    ✅ GR00T N1.7 VLA: Generated {result['action_chunk_horizon']}-step action chunk")
            print(f"    ✅ ROS 2 Trajectory: Formatted {traj['num_points']} trajectory points ({traj['total_duration_sec']}s duration)")
            print(f"    • State Update: EEF={np.round(agent.current_ee_pose[:3], 3).tolist()}, Gripper={agent.current_gripper:.2f}m, Held='{agent.held_object}'")
        else:
            print(f"    🛡️ Guardrail Check: REJECTED! Reason: {result.get('reason')}")
            return

    print("\n" + "-" * 75)
    print(f"  ✅ Task Finished: All {len(subgoals)} steps executed through GR00T N1.7.")
    print("-" * 75)


def main():
    print_banner("PHYSIK AI — INTERACTIVE ROBOT AGENT CLI")
    print("  Models: NVIDIA NeMo Agent (System 2) + GR00T N1.7 VLA (System 1)")
    print("  Embodiment: Franka Emika Panda (7-DOF Arm + 2-Finger Gripper)")
    print("\n  Commands you can try:")
    print("    • 'pick up the golden gear and place it in the kit tray'")
    print("    • 'assemble kit a'")
    print("    • 'pick up the emerald cube and put it in the inspection area'")
    print("    • 'pick up the crimson shaft and place it in the tray'")
    print("    • Type 'exit' or 'quit' to close.")
    print("=" * 75)

    agent = NeMoRoboticsAgent()
    bridge = ROS2RGBDBridge()

    if len(sys.argv) > 1:
        # Command passed as argument
        user_input = " ".join(sys.argv[1:])
        run_prompt(agent, bridge, user_input)
        return

    # Interactive Loop
    while True:
        try:
            print("\n🤖 Enter instruction for Franka Panda:")
            user_input = input("physyk-agent> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting interactive test CLI.")
                break
            run_prompt(agent, bridge, user_input)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
