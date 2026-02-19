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
import isaaclab.sim as sim_utils


@configclass
class PurementRl2EnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    action_space = 3
    observation_space = 20
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # Chemin vers votre USD - ne pas utiliser comme asset cfg
    env_usd_path: str = "/home/ajin/work2/sim2real-pnp/environ/my_env/source/env_v2.usd"

    # Robot config - le chemin DOIT pointer vers le robot dans UN environnement spécifique
    robot_cfg: ArticulationCfg = UR10_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",  # Chemin simplifié
    )
    # Permet de surcharger avec une regex précise vers le robot dans le USD cloné
    # Exemple: "/World/envs/.*/Origin2/Table/Robot" si votre USD a ce chemin
    robot_prim_regex: str = "/World/envs/.*/Origin2/Table/Robot"
    
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4,
        env_spacing=5.0,
        replicate_physics=True
    )
    
    # custom parameters (rest stays the same)
    ee_link_name = "ee_link"
    start_joint_pos = [0.0, -1.57, 1.57, 0.0, 0.0, 0.0]
    
    workspace_x_limits = [-1.2, 1.5]
    workspace_y_limits = [-1.8, 1.8]
    workspace_z_limits = [0.0, 2.0]
    
    target_ee_pos = [1.0, 0.5, 0.5]
    target_ee_quat = [1.0, 0.0, 0.0, 0.0]
    
    action_scale = 0.01  # scaling factor for actions
    
    # reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -5.0
    rew_scale_position_tracking = -1.0
    rew_scale_position_tracking_exp = 1.0
    position_exp_alpha = 4.0
    rew_scale_reached_goal = 100.0
    goal_reached_threshold = 0.05
    rew_scale_orientation_error = -1.0
    rew_scale_joint_vel = -0.01
    rew_scale_action_rate = -0.01
    rew_scale_action = -0.005
    
    # reset conditions
    initial_joint_pos_range = [-0.1, 0.1]
    max_joint_vel = 15.0
    max_position_error = 2.0

    # termination gating (to avoid instant resets)
    # Require a minimum number of steps before failures can terminate the episode
    min_steps_before_termination: int = 50
    # Toggle which failure modes actually terminate
    terminate_on_far_position: bool = False
    terminate_on_fast_joint: bool = True

    # Force a reset every environment step (episodes of length 1)
    # Useful when you want independent single-step rollouts per iteration.
    reset_every_step: bool = False

    # Optionally force a truncation after N steps (approximate "reset every rollout")
    # Set to a positive integer (e.g., 256 or 512) or None to disable.
    force_truncate_every_n_steps: int | None = None