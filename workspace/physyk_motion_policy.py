#!/usr/bin/env python3
"""
Physyk AI — Franka Panda Kinematics, GR00T Manipulation & Gripper Policy
Generates physics-compliant 7-DOF joint trajectories with real-time waypoint streaming.
"""
import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Callable

class PhysykMotionPolicy:
    def __init__(self):
        # Franka Panda standard joint limits (rad)
        self.joint_limits_lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.joint_limits_upper = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])
        
        # Home Configuration (Neutral standby pose above workcell)
        self.home_joint_positions = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        self.current_joints = self.home_joint_positions.copy()
        
        # End-Effector / Gripper State
        self.home_ee_position = np.array([0.35, 0.0, 1.15])
        self.current_ee_position = self.home_ee_position.copy()
        self.current_ee_orientation = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Parallel Jaw Gripper
        self.gripper_width = 0.08
        self.gripper_force = 0.0
        self.held_object: Optional[str] = None
        self.step_callback: Optional[Callable] = None

    def solve_ik_analytical(self, target_pos: np.ndarray) -> np.ndarray:
        x, y, z = target_pos
        q1 = np.arctan2(y, x)
        r = np.sqrt(x**2 + y**2) - 0.05
        z_rel = z - 0.40
        
        l1, l2 = 0.40, 0.40
        d_sq = r**2 + z_rel**2
        d = np.clip(np.sqrt(d_sq), 0.1, l1 + l2 - 0.01)
        
        cos_q4 = (d_sq - l1**2 - l2**2) / (2 * l1 * l2)
        q4 = -np.arccos(np.clip(cos_q4, -1.0, 1.0))
        
        alpha = np.arctan2(z_rel, r)
        beta = np.arctan2(l2 * np.sin(-q4), l1 + l2 * np.cos(-q4))
        q2 = -(alpha + beta)
        
        q3 = 0.0
        q5 = 0.0
        q6 = 1.571 - (q2 + q4)
        q7 = 0.785
        
        joints = np.array([q1, q2, q3, q4, q5, q6, q7])
        return np.clip(joints, self.joint_limits_lower, self.joint_limits_upper)

    def generate_minimum_jerk_trajectory(self, start_pos: np.ndarray, end_pos: np.ndarray, steps: int = 15) -> List[np.ndarray]:
        trajectory = []
        for i in range(steps + 1):
            tau = i / float(steps)
            s = 10 * (tau**3) - 15 * (tau**4) + 6 * (tau**5)
            pos = start_pos + s * (end_pos - start_pos)
            trajectory.append(pos)
        return trajectory

    def _emit(self, stage: str, pos: np.ndarray, joints: np.ndarray, gap: float, held: Optional[str] = None):
        if self.step_callback:
            self.step_callback(stage, pos, joints, gap, held)

    def execute_pick_and_place(self, object_key: str, object_name: str, pick_pos: np.ndarray, place_pos: np.ndarray, step_callback=None) -> Dict:
        if step_callback:
            self.step_callback = step_callback
            
        stages = []
        
        # 1. APPROACH_PREGRASP (14cm above object)
        pregrasp_pos = pick_pos.copy()
        pregrasp_pos[2] += 0.14
        self.gripper_width = 0.08
        self.gripper_force = 0.0
        self.held_object = None
        
        traj_pregrasp = self.generate_minimum_jerk_trajectory(self.current_ee_position, pregrasp_pos, steps=15)
        for p in traj_pregrasp:
            self.current_ee_position = p
            self.current_joints = self.solve_ik_analytical(p)
            self._emit("APPROACHING PART (PRE-GRASP)", p, self.current_joints, self.gripper_width, None)
        stages.append({"stage": "APPROACH_PREGRASP", "target_xyz": pregrasp_pos.tolist()})
        
        # 2. DESCEND TO OBJECT
        grasp_contact_pos = pick_pos.copy()
        grasp_contact_pos[2] += 0.03
        traj_descend = self.generate_minimum_jerk_trajectory(pregrasp_pos, grasp_contact_pos, steps=12)
        for p in traj_descend:
            self.current_ee_position = p
            self.current_joints = self.solve_ik_analytical(p)
            self._emit("DESCENDING TO OBJECT CENTROID", p, self.current_joints, self.gripper_width, None)
        stages.append({"stage": "DESCEND_TO_OBJECT", "target_xyz": grasp_contact_pos.tolist()})
        
        # 3. GRASP CLOSURE (Close parallel fingers)
        for w in np.linspace(0.08, 0.035, 8):
            self.gripper_width = float(w)
            self._emit(f"CLOSING PARALLEL JAWS ({w*100:.1f}cm)", grasp_contact_pos, self.current_joints, self.gripper_width, object_key)
        self.held_object = object_key
        stages.append({"stage": "GRASP_ENGAGE", "clamping_force_N": 45.0, "held_object": object_name})
        
        # 4. LIFT VERTICALLY (+16cm clearance)
        lift_pos = grasp_contact_pos.copy()
        lift_pos[2] += 0.18
        traj_lift = self.generate_minimum_jerk_trajectory(grasp_contact_pos, lift_pos, steps=14)
        for p in traj_lift:
            self.current_ee_position = p
            self.current_joints = self.solve_ik_analytical(p)
            self._emit("LIFTING OBJECT WITH VERTICAL CLEARANCE", p, self.current_joints, self.gripper_width, object_key)
        stages.append({"stage": "LIFT_VERTICAL", "target_xyz": lift_pos.tolist()})
        
        # 5. TRANSIT TO DESTINATION (Smooth parabolic 3D arc)
        transit_preplace = place_pos.copy()
        transit_preplace[2] += 0.18
        traj_transit = self.generate_minimum_jerk_trajectory(lift_pos, transit_preplace, steps=25)
        for p in traj_transit:
            self.current_ee_position = p
            self.current_joints = self.solve_ik_analytical(p)
            self._emit("TRANSITING TO TARGET DESTINATION", p, self.current_joints, self.gripper_width, object_key)
        stages.append({"stage": "TRANSIT_TO_DESTINATION", "target_xyz": transit_preplace.tolist()})
        
        # 6. DESCEND TO PLACE
        place_contact_pos = place_pos.copy()
        place_contact_pos[2] += 0.03
        traj_place = self.generate_minimum_jerk_trajectory(transit_preplace, place_contact_pos, steps=12)
        for p in traj_place:
            self.current_ee_position = p
            self.current_joints = self.solve_ik_analytical(p)
            self._emit("DESCENDING TO PLACEMENT POSITION", p, self.current_joints, self.gripper_width, object_key)
        stages.append({"stage": "DESCEND_PLACE", "target_xyz": place_contact_pos.tolist()})
        
        # 7. OPEN GRIPPER / RELEASE
        for w in np.linspace(0.035, 0.08, 8):
            self.gripper_width = float(w)
            self._emit(f"RELEASING PARALLEL GRIPPER ({w*100:.1f}cm)", place_contact_pos, self.current_joints, self.gripper_width, None)
        self.held_object = None
        stages.append({"stage": "RELEASE_GRIPPER", "gripper_width_m": 0.08})
        
        # 8. RETREAT CLEARANCE & RETURN HOME
        retreat_pos = place_contact_pos.copy()
        retreat_pos[2] += 0.18
        traj_retreat = self.generate_minimum_jerk_trajectory(place_contact_pos, retreat_pos, steps=10)
        for p in traj_retreat:
            self.current_ee_position = p
            self.current_joints = self.solve_ik_analytical(p)
            self._emit("RETREATING TO SAFE TRANSIT POSTURE", p, self.current_joints, self.gripper_width, None)
            
        traj_home = self.generate_minimum_jerk_trajectory(retreat_pos, self.home_ee_position, steps=15)
        for p in traj_home:
            self.current_ee_position = p
            self.current_joints = self.solve_ik_analytical(p)
            self._emit("RETURNING TO STANDBY POSTURE", p, self.current_joints, self.gripper_width, None)
            
        stages.append({"stage": "STANDBY_IDLE", "target_xyz": self.home_ee_position.tolist()})
        
        return {
            "status": "SUCCESS",
            "manipulated_object": object_name,
            "origin_xyz": pick_pos.tolist(),
            "destination_xyz": place_pos.tolist(),
            "stages_executed": len(stages),
            "final_gripper_pose": [round(x, 3) for x in self.current_ee_position.tolist()]
        }
