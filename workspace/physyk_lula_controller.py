#!/usr/bin/env python3
"""
Physyk AI — Lula Kinematics & Collision Controller
==================================================
Wraps Omniverse Isaac Sim's native Lula solver to generate collision-free
joint trajectories for the Franka Panda robot.
"""

import numpy as np
import logging

try:
    from omni.isaac.motion_generation import RmpFlow, ArticulationKinematicsSolver
    from omni.isaac.motion_generation import lula
    ISAAC_AVAILABLE = True
except ImportError:
    ISAAC_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="[Lula-Controller] %(message)s")
log = logging.getLogger(__name__)


class PhysykLulaController:
    def __init__(
        self, 
        robot_articulation, 
        end_effector_frame_name="panda_hand",
        urdf_path="/opt/dlami/nvme/physyk/assets/franka_panda/panda_arm_hand.urdf"
    ):
        """
        Initializes the Lula RmpFlow solver for Franka.
        Requires an active Isaac Sim environment and an Articulation view of the robot.
        """
        self.robot = robot_articulation
        self.end_effector_name = end_effector_frame_name
        self.urdf_path = urdf_path
        
        if not ISAAC_AVAILABLE:
            log.warning("Omniverse Isaac Sim modules not found. Lula controller will run in dummy mode.")
            self.solver = None
            return

        log.info("Initializing Lula RmpFlow for Franka Panda...")
        
        # In a real Omniverse Isaac Sim environment, franka configs are built-in.
        # We load the default RmpFlow configuration for Franka.
        self.rmp_flow = RmpFlow(
            robot_description_path="franka/rmpflow/robot_descriptor.yaml",
            urdf_path="franka/rmpflow/franka.urdf", # Isaac Sim internal default
            rmpflow_config_path="franka/rmpflow/franka_rmpflow_common.yaml",
            end_effector_frame_name=self.end_effector_name,
            maximum_substep_size=0.00334
        )
        
        # Attach the kinematic solver to the robot articulation
        self.solver = ArticulationKinematicsSolver(
            self.robot, 
            self.rmp_flow, 
            self.end_effector_name
        )
        
        log.info("✅ Lula Controller initialized successfully.")

    def compute_safe_trajectory(self, target_position: np.ndarray, target_orientation: np.ndarray = None):
        """
        Computes a single safe joint position action towards the target pose,
        avoiding registered collision meshes (like the table).
        
        Args:
            target_position: (3,) array [X, Y, Z] in meters
            target_orientation: (4,) array [w, x, y, z] quaternion
            
        Returns:
            np.ndarray: The computed 7-DOF joint angles, or None if unreachable.
        """
        if not ISAAC_AVAILABLE or self.solver is None:
            # Dummy mode fallback for offline testing
            return self._dummy_ik(target_position)

        self.rmp_flow.set_end_effector_target(
            target_position=target_position, 
            target_orientation=target_orientation
        )
        
        # Calculate kinematics using Lula
        action, success = self.solver.compute_kinematics_and_resolve_collision(
            target_position=target_position,
            target_orientation=target_orientation
        )
        
        if success:
            return action.joint_positions
        else:
            log.error(f"Lula IK failed to find a collision-free path to {target_position}")
            return None

    def _dummy_ik(self, target_position: np.ndarray):
        """Fallback mock for testing without Omniverse running."""
        log.info(f"[Dummy Lula] Moving to {target_position}")
        # Return a mock 7-DOF action
        return np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.57, 0.785])
