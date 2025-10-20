# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.universal_robots import UR10_CFG

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils


@configclass
class PurementRl2EnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition
    action_space = 3  # 3D Cartesian position (x, y, z) for end-effector
    observation_space = 18  # ee_pos (3) + ee_quat (4) + target_pos (3) + target_quat (4) + joint_vel (6)
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = UR10_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg.init_state.pos = (0.0, 0.0, 1.03)  # <-- ajout

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # custom parameters/scales
    # - end-effector link name
    ee_link_name = "ee_link"  # name of the end-effector body in the UR10
    
    # - start and target joint positions (for reset)
    start_joint_pos = [0.0, -1.57, 1.57, 0.0, 0.0, 0.0]  # starting configuration
    
    # - workspace limits (cartesian boundaries for end-effector)
    # Define the reachable workspace as a 3D bounding box [min, max] for each axis
    workspace_x_limits = [-1.2, 1.5]  # x-axis limits [m]
    workspace_y_limits = [-1.8, 1.8]  # y-axis limits [m]
    workspace_z_limits = [-2.0, 2.0]  # z-axis limits [m]
    
    # - target cartesian pose (position + quaternion wxyz)
    # These will be computed from forward kinematics in the environment
    target_ee_pos = [1.0, 0.5, 0.5]  # target position in world frame [m]
    target_ee_quat = [1.0, 0.0, 0.0, 0.0]  # target orientation (w, x, y, z)
    
    # - action scale
    action_scale = 0.05  # scaling for cartesian position commands [m]
    
    # - reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -5.0
    # Position tracking rewards
    rew_scale_position_tracking = -1.0  # r_track = -||p_ee - p_goal||² (keypoint tracking)
    rew_scale_position_tracking_exp = 1.0  # r_exp = exp(-α||p_ee - p_goal||²) (keypoint tracking exp)
    position_exp_alpha = 4.0  # α parameter for exponential reward (higher = more weight when close)
    rew_scale_reached_goal = 100.0  # positive reward when reaching the goal position
    goal_reached_threshold = 0.05  # distance threshold to consider goal reached [m]
    # Other rewards
    rew_scale_orientation_error = -1.0  # penalty for orientation error
    rew_scale_joint_vel = -0.01  # penalty for high velocities
    rew_scale_action_rate = -0.01  # penalty for large action changes (smoothness)
    rew_scale_action = -0.005  # penalty for large action values (energy efficiency)
    
    # - reset states/conditions
    initial_joint_pos_range = [-0.1, 0.1]  # joint position sample range on reset [rad]
    max_joint_vel = 15.0  # reset if joint velocity exceeds this value [rad/s]
    max_position_error = 2.0  # reset if ee position error exceeds this [m]