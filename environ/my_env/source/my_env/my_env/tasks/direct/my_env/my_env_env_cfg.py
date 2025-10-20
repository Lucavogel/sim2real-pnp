# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.universal_robots import UR10_CFG 

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class MyEnvEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition
    action_space = 6  # 6 joints for UR10
    observation_space = 12  # joint positions (6) + joint velocities (6)
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = UR10_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg.init_state.pos = (0.0, 0.0, 1.03)  # <-- ajout

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # custom parameters/scales
    # - target joint positions (default pose)
    target_joint_pos = [-1.712, -1.712, 1.712, 0.0, 0.0, 0.0]  # target configuration
    
    # - action scale
    action_scale = 0.5  # scaling for joint position commands
    
    # - reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_joint_pos_error = -1.0  # penalty for deviating from target position
    rew_scale_joint_vel = -0.01  # penalty for high velocities
    
    # - reset states/conditions
    initial_joint_pos_range = [-0.2, 0.2]  # joint position sample range on reset [rad]
    max_joint_vel = 15.0  # reset if joint velocity exceeds this value [rad/s]