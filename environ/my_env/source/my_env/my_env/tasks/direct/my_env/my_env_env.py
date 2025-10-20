# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # <-- ajout

from .my_env_env_cfg import MyEnvEnvCfg


class MyEnvEnv(DirectRLEnv):
    cfg: MyEnvEnvCfg

    def __init__(self, cfg: MyEnvEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Get all joint indices for UR10
        self._joint_ids, _ = self.robot.find_joints(".*")
        
        # Target joint positions
        self._target_joint_pos = torch.tensor(
            self.cfg.target_joint_pos, 
            device=self.device
        ).repeat(self.num_envs, 1)

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

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
        # Apply actions as joint position targets
        target_pos = self._target_joint_pos + self.actions * self.cfg.action_scale
        self.robot.set_joint_position_target(target_pos, joint_ids=self._joint_ids)

    def _get_observations(self) -> dict:
        # Observation: current joint positions and velocities
        obs = torch.cat(
            (
                self.joint_pos[:, self._joint_ids],
                self.joint_vel[:, self._joint_ids],
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_joint_pos_error,
            self.cfg.rew_scale_joint_vel,
            self.joint_pos[:, self._joint_ids],
            self.joint_vel[:, self._joint_ids],
            self._target_joint_pos,
            self.reset_terminated,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Reset if any joint velocity is too high
        out_of_bounds = torch.any(
            torch.abs(self.joint_vel[:, self._joint_ids]) > self.cfg.max_joint_vel, 
            dim=1
        )
        
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Sample random initial joint positions around target
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_pos[:, self._joint_ids] = self._target_joint_pos[env_ids] + sample_uniform(
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


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_joint_pos_error: float,
    rew_scale_joint_vel: float,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    target_joint_pos: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    # Reward for being alive
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    
    # Penalty for termination
    rew_termination = rew_scale_terminated * reset_terminated.float()
    
    # Penalty for joint position error (distance from target)
    pos_error = torch.sum(torch.square(joint_pos - target_joint_pos), dim=-1)
    rew_joint_pos = rew_scale_joint_pos_error * pos_error
    
    # Penalty for high joint velocities
    rew_joint_vel = rew_scale_joint_vel * torch.sum(torch.square(joint_vel), dim=-1)
    
    total_reward = rew_alive + rew_termination + rew_joint_pos + rew_joint_vel
    return total_reward