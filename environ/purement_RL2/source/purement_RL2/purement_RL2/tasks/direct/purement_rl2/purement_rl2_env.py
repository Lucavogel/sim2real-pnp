# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, quat_error_magnitude, quat_mul
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR 

from .purement_rl2_env_cfg import PurementRl2EnvCfg



class PurementRl2Env(DirectRLEnv):
    cfg: PurementRl2EnvCfg

    def __init__(self, cfg: PurementRl2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Get all joint indices and ee body index
        self._joint_ids, _ = self.robot.find_joints(".*")
        self._ee_body_idx, _ = self.robot.find_bodies(self.cfg.ee_link_name)
        
        # Start joint positions for reset
        self._start_joint_pos = torch.tensor(
            self.cfg.start_joint_pos, 
            device=self.device
        ).repeat(self.num_envs, 1)
        
        # Target cartesian pose (will be updated after first forward kinematics)
        self._target_ee_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_ee_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._target_ee_quat[:, 0] = 1.0  # default quaternion (w=1)
        
        # Workspace limits
        self._workspace_limits = torch.tensor(
            [
                self.cfg.workspace_x_limits,
                self.cfg.workspace_y_limits,
                self.cfg.workspace_z_limits,
            ],
            device=self.device,
        )  # Shape: (3, 2) -> [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        
        # Store previous actions for action rate penalty
        self._previous_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        
        # Buffers for current state
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        self.ee_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_quat_w = torch.zeros(self.num_envs, 4, device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # --- objets Ur10.py (répliqués) ---
        stand_cfg = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd",
            scale=(2.0, 2.0, 2.0),
        )
        stand_cfg.func("/World/envs/env_0/Objects/Stand", stand_cfg, translation=(0.0, 0.0, 1.03))

        table_cfg = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        )
        table_cfg.func(
            "/World/envs/env_0/Objects/Table",
            table_cfg,
            translation=(1.0, 0.08, 1.05),
            orientation=(0.0, 0.0, 0.0, 0.7071068),
        )

        # -------------------------------

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # add articulation to scene
        self.scene.articulations["robot"] = self.robot

        # lights
        light_distant = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_distant.func("/World/lightDistant", light_distant, translation=(1.0, 0.0, 10.0))

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # Actions are cartesian position commands (delta_x, delta_y, delta_z)
        # We interpret these as desired changes to the end-effector position
        
        # Desired ee position = current position + scaled action
        desired_ee_pos = self.ee_pos_w + self.actions * self.cfg.action_scale
        
        # Clamp desired position to workspace limits
        desired_ee_pos = torch.clamp(
            desired_ee_pos,
            min=self._workspace_limits[:, 0],  # [x_min, y_min, z_min]
            max=self._workspace_limits[:, 1],  # [x_max, y_max, z_max]
        )
        
        # Simplified approach: Use proportional control in Cartesian space
        # This is a basic form of operational space control
        # For a more accurate solution, you would:
        # 1. Compute the Jacobian matrix
        # 2. Use pseudoinverse to get joint velocities from cartesian velocities
        # 3. Integrate to get joint positions
        
        # For now, use a simple proportional controller
        # Map cartesian error to joint space using a gain
        pos_error = desired_ee_pos - self.ee_pos_w
        
        # Simple heuristic: distribute the error across joints
        # This is NOT proper IK but allows the RL agent to learn a policy
        # The agent will learn to generate actions that achieve the desired ee motion
        
        # Proportional gain for converting position error to joint displacement
        p_gain = 2.0
        
        # Distribute error across joints (simplified)
        # In a real implementation, you'd use the Jacobian transpose or pseudoinverse
        joint_delta = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)
        
        # Simple mapping: influence joints based on their contribution to end-effector motion
        # Shoulder and elbow joints mainly affect position
        joint_delta[:, 0] = pos_error[:, 0] * p_gain  # shoulder pan affects x,y
        joint_delta[:, 1] = -pos_error[:, 2] * p_gain  # shoulder lift affects z
        joint_delta[:, 2] = pos_error[:, 2] * p_gain   # elbow affects z
        joint_delta[:, 3] = pos_error[:, 1] * p_gain * 0.5  # wrist affects y
        
        # Compute target joint positions
        joint_pos_target = self.joint_pos[:, self._joint_ids] + joint_delta
        
        # Set joint position targets
        self.robot.set_joint_position_target(joint_pos_target, joint_ids=self._joint_ids)

    def _get_observations(self) -> dict:
        # Observation: current joint positions and velocities
        obs = torch.cat(
            (
                self.joint_pos[:, self._joint_ids],
                self.joint_vel[:, self._joint_ids],
                self.ee_pos_w,
                self.ee_quat_w,
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_position_tracking,
            self.cfg.rew_scale_position_tracking_exp,
            self.cfg.rew_scale_reached_goal,
            self.cfg.rew_scale_orientation_error,
            self.cfg.rew_scale_joint_vel,
            self.cfg.rew_scale_action_rate,
            self.cfg.rew_scale_action,
            self.cfg.position_exp_alpha,
            self.cfg.goal_reached_threshold,
            self.ee_pos_w,
            self.ee_quat_w,
            self._target_ee_pos,
            self._target_ee_quat,
            self.joint_vel[:, self._joint_ids],
            self.actions,
            self._previous_actions,
            self.reset_terminated,
        )
        # Update previous actions
        self._previous_actions[:] = self.actions
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Reset only when goal position is reached
        pos_error = torch.norm(self.ee_pos_w - self._target_ee_pos, dim=-1)
        goal_reached = pos_error < self.cfg.goal_reached_threshold
        
        # Check if end-effector is outside workspace limits
        outside_workspace = (
            (self.ee_pos_w[:, 0] < self._workspace_limits[0, 0]) |  # x < x_min
            (self.ee_pos_w[:, 0] > self._workspace_limits[0, 1]) |  # x > x_max
            (self.ee_pos_w[:, 1] < self._workspace_limits[1, 0]) |  # y < y_min
            (self.ee_pos_w[:, 1] > self._workspace_limits[1, 1]) |  # y > y_max
            (self.ee_pos_w[:, 2] < self._workspace_limits[2, 0]) |  # z < z_min
            (self.ee_pos_w[:, 2] > self._workspace_limits[2, 1])    # z > z_max
        )
        
        # Out of bounds only for extreme joint velocities (safety)
        joint_vel_violation = torch.any(
            torch.abs(self.joint_vel[:, self._joint_ids]) > self.cfg.max_joint_vel, 
            dim=1
        )
        
        # Reset when goal is reached OR joint velocity violation OR outside workspace (safety)
        out_of_bounds = goal_reached | joint_vel_violation | outside_workspace
        
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Initialize joints at the START position (with small random noise)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_pos[:, self._joint_ids] = self._start_joint_pos[env_ids] + sample_uniform(
            self.cfg.initial_joint_pos_range[0],
            self.cfg.initial_joint_pos_range[1],
            (len(env_ids), len(self._joint_ids)),
            self.device,
        )
        
        joint_vel = torch.zeros_like(joint_pos)

        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # After reset, compute target ee position from a different joint configuration
        # For now, use predefined target or compute from forward kinematics
        # We'll set a target that's reachable
        if self.episode_length_buf[0] == 0:  # First reset
            # Compute target from forward kinematics at a different configuration
            target_joint_config = torch.tensor(
                [-1.712, -1.712, 1.712, 0.0, 0.0, 0.0],
                device=self.device
            ).repeat(len(env_ids), 1)
            
            # Temporarily set joints to target config to get FK
            temp_joint_pos = joint_pos.clone()
            temp_joint_pos[:, self._joint_ids] = target_joint_config
            self.robot.write_joint_state_to_sim(temp_joint_pos, joint_vel, None, env_ids)
            
            # Let physics update once to compute forward kinematics
            # In practice, you might want to use an FK solver directly
            # For now, we'll set a predefined target
            
        # Set predefined target (adjust based on your robot's workspace)
        self._target_ee_pos[env_ids] = torch.tensor(
            self.cfg.target_ee_pos, device=self.device
        ).repeat(len(env_ids), 1)
        
        # Ensure target is within workspace limits
        self._target_ee_pos[env_ids] = torch.clamp(
            self._target_ee_pos[env_ids],
            min=self._workspace_limits[:, 0],
            max=self._workspace_limits[:, 1],
        )
        
        self._target_ee_quat[env_ids] = torch.tensor(
            self.cfg.target_ee_quat, device=self.device
        ).repeat(len(env_ids), 1)
        
        # Reset previous actions
        self._previous_actions[env_ids] = 0.0


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_position_tracking: float,
    rew_scale_position_tracking_exp: float,
    rew_scale_reached_goal: float,
    rew_scale_orientation_error: float,
    rew_scale_joint_vel: float,
    rew_scale_action_rate: float,
    rew_scale_action: float,
    position_exp_alpha: float,
    goal_reached_threshold: float,
    ee_pos: torch.Tensor,
    ee_quat: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    joint_vel: torch.Tensor,
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    # Reward for being alive
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    
    # Penalty for termination
    rew_termination = rew_scale_terminated * reset_terminated.float()
    
    # 1. Position tracking error (keypoint tracking): r_track = -||p_ee - p_goal||²
    pos_error_norm = torch.norm(ee_pos - target_pos, dim=-1)
    rew_position_tracking = rew_scale_position_tracking * torch.square(pos_error_norm)
    
    # 2. Exponential position tracking (keypoint tracking exp): r_exp = exp(-α||p_ee - p_goal||²)
    # Met un poids plus fort quand on est proche (pour stabiliser la précision)
    pos_error_squared = torch.square(pos_error_norm)
    rew_position_tracking_exp = rew_scale_position_tracking_exp * torch.exp(-position_exp_alpha * pos_error_squared)
    
    # 3. Positive reward for reaching the goal position
    # Donne une récompense positive quand l'effecteur est proche de la cible
    goal_reached = (pos_error_norm < goal_reached_threshold).float()
    rew_reached_goal = rew_scale_reached_goal * goal_reached
    
    # 4. Orientation error (quaternion distance)
    # quat_error = 1 - |q1 · q2| where q1 and q2 are unit quaternions
    quat_dot = torch.abs(torch.sum(ee_quat * target_quat, dim=-1))
    quat_dot = torch.clamp(quat_dot, 0.0, 1.0)
    orientation_error = 1.0 - quat_dot
    rew_orientation = rew_scale_orientation_error * orientation_error
    
    # 5. Joint velocity penalty (for smooth movement)
    rew_joint_vel = rew_scale_joint_vel * torch.sum(torch.square(joint_vel), dim=-1)
    
    # 6. Action rate penalty (pénalise les changements rapides d'action, pour la douceur du mouvement)
    action_rate = torch.sum(torch.square(actions - previous_actions), dim=-1)
    rew_action_rate = rew_scale_action_rate * action_rate
    
    # 7. Action magnitude penalty (pénalise les grandes valeurs d'action, effort énergétique)
    action_magnitude = torch.sum(torch.square(actions), dim=-1)
    rew_action = rew_scale_action * action_magnitude
    
    total_reward = (
        rew_alive 
        + rew_termination 
        + rew_position_tracking 
        + rew_position_tracking_exp 
        + rew_reached_goal
        + rew_orientation 
        + rew_joint_vel 
        + rew_action_rate 
        + rew_action
    )
    return total_reward