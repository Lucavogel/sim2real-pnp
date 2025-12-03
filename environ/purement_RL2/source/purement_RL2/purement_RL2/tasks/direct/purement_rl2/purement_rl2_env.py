# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
import omni.usd
import isaaclab.sim as sim_utils
from collections.abc import Sequence
from pxr import UsdGeom

from .purement_rl2_env_cfg import PurementRl2EnvCfg
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, quat_error_magnitude, quat_mul
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR 



class PurementRl2Env(DirectRLEnv):
    cfg: PurementRl2EnvCfg

    def __init__(self, cfg: PurementRl2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Get UR arm joint indices - matches all 6 UR joints
        # Pattern matches: shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint
        arm_joint_ids_list, _ = self.robot.find_joints(".*_joint")
        # Convert to tensor for proper CUDA indexing
        self._arm_joint_ids = torch.tensor(arm_joint_ids_list, device=self.device, dtype=torch.long)
        self._smoothed_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        
        ee_body_idx_list, _ = self.robot.find_bodies(self.cfg.ee_link_name)
        # Convert to int (it's a single body)
        self._ee_body_idx = ee_body_idx_list[0]
        
        # Initialize start positions for ALL joints
        # Create a default pose (zeros or current pose)
        default_joint_pos = self.robot.data.default_joint_pos[0].clone()  # Shape: (num_joints,)
        
        # Override arm joints with your config values
        if len(self._arm_joint_ids) == len(self.cfg.start_joint_pos):
            default_joint_pos[self._arm_joint_ids] = torch.tensor(
                self.cfg.start_joint_pos, device=self.device
            )
        else:
            # If lengths don't match, pad or truncate
            config_pos = torch.tensor(self.cfg.start_joint_pos, device=self.device)
            min_len = min(len(self._arm_joint_ids), len(config_pos))
            default_joint_pos[self._arm_joint_ids[:min_len]] = config_pos[:min_len]
    
        self._start_joint_pos = default_joint_pos.repeat(self.num_envs, 1)
        
        # Target cartesian pose
        self._target_ee_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_ee_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._target_ee_quat[:, 0] = 1.0
        
        # Workspace limits
        self._workspace_limits = torch.tensor(
            [
                self.cfg.workspace_x_limits,
                self.cfg.workspace_y_limits,
                self.cfg.workspace_z_limits,
            ],
            device=self.device,
        )
        
        # Store previous actions
        self._previous_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        
        # Buffers for current state
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        self.ee_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_quat_w = torch.zeros(self.num_envs, 4, device=self.device) 
        
        # Track per-episode step counts to gate early terminations
        self._episode_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    

    def _setup_scene(self):
        """Sets up the scene with robot and environment"""
        stage = omni.usd.get_context().get_stage()

        # 1. Charger votre USD complet dans env_0 EN PREMIER (avant la config de scène)
        print(f"[SETUP] Loading USD into env_0...")
        env_prim_path = "/World/envs/env_0"
        
        # Créer le prim parent si nécessaire
        parent_prim = stage.DefinePrim("/World/envs", "Xform")
        
        # Définir env_0 et ajouter la référence USD
        env_prim = stage.DefinePrim(env_prim_path, "Xform")
        references = env_prim.GetReferences()
        references.AddReference(
            assetPath=self.cfg.env_usd_path,
            primPath="/World",  # Charge tout ce qui est sous /World dans le USD
        )
        
        print(f"[SETUP] USD loaded into {env_prim_path}")
        print(f"[SETUP] Robot should be at: {env_prim_path}/Origin2/Table/Robot")
        print(f"[SETUP] Other objects from env_v2.usd should also be present")

        # 2. Cloner les environnements pour créer les autres envs
        print(f"[SETUP] Cloning env_0 to create {self.scene.num_envs} environments...")
        self.scene.clone_environments(copy_from_source=True)

        # 3. Configurer le robot en WRAPPANT les robots existants, sans spawn
        #    -> prim_path doit matcher tous les robots dans tous les envs.
        #    Selon comment tes envs sont nommés, adapte le pattern.
        #    Utilise la regex configurable depuis le .cfg pour éviter les chemins en dur.
        robot_cfg = self.cfg.robot_cfg.replace(
            prim_path=self.cfg.robot_prim_regex,
            spawn=None,  # IMPORTANT : ne pas respawn le robot, il vient du USD
        )

        self.robot = Articulation(cfg=robot_cfg)
        self.scene.articulations["robot"] = self.robot
        print("[SETUP] Robot articulation configured")

        # Diagnostics: vérifier que le nombre d'instances trouvées == num_envs
        try:
            num_instances = int(self.robot.num_instances)
        except Exception:
            num_instances = -1
        if num_instances != self.scene.num_envs:
            print(
                f"[WARN] Articulation instances found ({num_instances}) do not match scene environments ({self.scene.num_envs})."
            )
            print(f"[WARN] Regex used for robot prims: {self.cfg.robot_prim_regex}")
            # Essayer dister les prims candidats sous env_0
            try:
                import re

                pattern = re.compile(self.cfg.robot_prim_regex)
                matches = []
                for prim in stage.Traverse():
                    path = str(prim.GetPath())
                    if pattern.fullmatch(path):
                        matches.append(path)
                if matches:
                    print("[HINT] Matched prims:")
                    for p in matches:
                        print(f"  - {p}")
                else:
                    print("[HINT] No prim matched the current regex. Here are some candidates containing 'Robot':")
                    for prim in stage.Traverse():
                        path = str(prim.GetPath())
                        if "Robot" in path or "robot" in path or "UR10" in path or "ur10" in path:
                            print(f"  - {path} ({prim.GetTypeName()})")
            except Exception as e:
                print(f"[WARN] Failed to run prim path diagnostics: {e}")

        # 4. Ground plane (global)
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(),
        )

        # 5. Lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(1.0, 1.0, 1.0))
        light_cfg.func("/World/Light", light_cfg)

        print(f"[SETUP] Scene setup complete! num_envs={self.scene.num_envs}")
        
        # 6. Filter collisions
        if self.sim.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

    

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        actions = torch.clamp(actions, -1.0, 1.0)

        # Lissage (filtre exponentiel)
        alpha = 0.2  # plus petit = plus lisse / plus lent
        self._smoothed_actions = (1.0 - alpha) * self._smoothed_actions + alpha * actions

        self.actions = self._smoothed_actions.clone()
        # self.actions = actions.clone()
        
        # Increment episode steps counter for all envs each sim step
        self._episode_steps += 1
    def _apply_action(self) -> None:
        """Applique les actions (commandes cartésiennes) au robot"""
        # Actions = déplacements cartésiens souhaités (delta_x, delta_y, delta_z)
        # On suppose action_space = 3
        # clamp déjà fait dans _pre_physics_step

        # Réduire fortement l'échelle d'action
        action_scale = self.cfg.action_scale  # configure-le petit (ex: 0.01) dans le YAML
        desired_ee_pos = self.ee_pos_w + self.actions * action_scale

        # Clamper dans l'espace de travail
        desired_ee_pos = torch.clamp(
            desired_ee_pos,
            min=self._workspace_limits[:, 0],
            max=self._workspace_limits[:, 1],
        )

        # Erreur de position
        pos_error = desired_ee_pos - self.ee_pos_w           # [num_envs, 3]
        pos_error_norm = torch.norm(pos_error, dim=-1, keepdim=True)  # [num_envs, 1]

        # Limiter la "vitesse" cartésienne par step
        max_cart_step = 0.01  # 1 cm par step par exemple
        scaling = torch.clamp(max_cart_step / (pos_error_norm + 1e-6), max=1.0)
        safe_pos_error = pos_error * scaling

        # Mapping hyper simple: même delta pour tous les joints (pas du vrai IK, mais plus stable)
        # → on réduit ENCORE le gain
        k_joint = 0.02  # très petit
        delta_joint = k_joint * safe_pos_error.sum(dim=-1, keepdim=True)  # [num_envs, 1]

        # Récupère les pos actuelles sur les joints de bras
        current_arm_pos = self.joint_pos[:, self._arm_joint_ids]

        # Limite le delta de joint par step
        max_joint_step = 0.05  # radian max par step
        delta_joint = torch.clamp(delta_joint, -max_joint_step, max_joint_step)

        joint_pos_target = current_arm_pos + delta_joint

        self.robot.set_joint_position_target(joint_pos_target, joint_ids=self._arm_joint_ids)


    def _get_observations(self) -> dict:
        """Récupère les observations de l'environnement"""
        # Mettre à jour les positions/orientations de l'end-effector
        self.ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_idx, :]
        self.ee_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx, :]
        
        # Use only arm joint velocities (6 joints for UR)
        arm_joint_vel = self.joint_vel[:, self._arm_joint_ids]
        
        # Observation: [ee_pos(3), ee_quat(4), target_pos(3), target_quat(4), joint_vel(6)]
        obs = torch.cat(
            [
                self.ee_pos_w,                    # 3
                self.ee_quat_w,                   # 4
                self._target_ee_pos,              # 3
                self._target_ee_quat,             # 4
                arm_joint_vel,                    # 6 (UR arm joints)
            ],
            dim=-1,
        )
        
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Calcule les récompenses"""
        # Use only arm joint velocities
        arm_joint_vel = self.joint_vel[:, self._arm_joint_ids]
        
        return compute_rewards(
            rew_scale_alive=self.cfg.rew_scale_alive,
            rew_scale_terminated=self.cfg.rew_scale_terminated,
            rew_scale_position_tracking=self.cfg.rew_scale_position_tracking,
            rew_scale_position_tracking_exp=self.cfg.rew_scale_position_tracking_exp,
            rew_scale_reached_goal=self.cfg.rew_scale_reached_goal,
            rew_scale_orientation_error=self.cfg.rew_scale_orientation_error,
            rew_scale_joint_vel=self.cfg.rew_scale_joint_vel,
            rew_scale_action_rate=self.cfg.rew_scale_action_rate,
            rew_scale_action=self.cfg.rew_scale_action,
            position_exp_alpha=self.cfg.position_exp_alpha,
            goal_reached_threshold=self.cfg.goal_reached_threshold,
            ee_pos=self.ee_pos_w,
            ee_quat=self.ee_quat_w,
            target_pos=self._target_ee_pos,
            target_quat=self._target_ee_quat,
            joint_vel=arm_joint_vel,
            actions=self.actions,
            previous_actions=self._previous_actions,
            reset_terminated=self.reset_terminated,
        )

    # def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
    #     # Pour le moment, on NE TERMINE JAMAIS à cause des erreurs.
    #     terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    #     truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    #     return terminated, truncated
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Conditions de fin d'épisode (succès / échec / état invalide)."""

        # Mode forcé: reset à chaque step (épisodes de longueur 1)
        if self.cfg.reset_every_step:
            terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            truncated = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            return terminated, truncated

        # 1) Erreur de position de l'end-effector
        pos_error = torch.norm(self.ee_pos_w - self._target_ee_pos, dim=1)

        # 2) Vitesse des joints du bras
        arm_joint_vel = self.joint_vel[:, self._arm_joint_ids]
        max_joint_vel = torch.max(torch.abs(arm_joint_vel), dim=1).values

        # 3) États invalides (NaN / Inf) → reset obligatoire
        pos_nan = torch.isnan(self.joint_pos).any(dim=1)
        pos_inf = torch.isinf(self.joint_pos).any(dim=1)
        vel_nan = torch.isnan(self.joint_vel).any(dim=1)
        vel_inf = torch.isinf(self.joint_vel).any(dim=1)
        invalid_state = pos_nan | pos_inf | vel_nan | vel_inf

        # 4) Succès : on atteint la cible
        success = pos_error < self.cfg.goal_reached_threshold

        # 5) Échecs permissifs (valeurs à fixer dans ton .cfg) + gate par nombre de steps
        far_failure_raw = pos_error > self.cfg.max_position_error
        fast_failure_raw = max_joint_vel > self.cfg.max_joint_vel

        # Gate failures until a minimum number of steps have elapsed
        gate = self._episode_steps >= self.cfg.min_steps_before_termination
        far_failure = (far_failure_raw & gate) if self.cfg.terminate_on_far_position else torch.zeros_like(gate)
        fast_failure = (fast_failure_raw & gate) if self.cfg.terminate_on_fast_joint else torch.zeros_like(gate)

        # 6) Terminated = succès ou échec ou état invalide
        terminated = invalid_state | success | far_failure | fast_failure

        # 7) Truncated: timeout géré par DirectRLEnv, plus optionnellement par un N fixe
        truncated = torch.zeros_like(terminated)
        if self.cfg.force_truncate_every_n_steps is not None and self.cfg.force_truncate_every_n_steps > 0:
            truncated_by_n = self._episode_steps >= self.cfg.force_truncate_every_n_steps
            truncated = truncated | truncated_by_n

        return terminated, truncated


    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset les environnements spécifiés"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # Convert to tensor if it's not already
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        
        # Debug print
        print(f"[RESET] num_envs: {self.num_envs}")
        print(f"[RESET] robot.num_instances: {self.robot.num_instances}")
        print(f"[RESET] robot.data.joint_pos shape: {self.robot.data.joint_pos.shape}")
        print(f"[RESET] env_ids: {env_ids}")
        print(f"[RESET] _start_joint_pos shape: {self._start_joint_pos.shape}")
        
        # Create noise for arm joints only
        arm_joint_noise = sample_uniform(
            self.cfg.initial_joint_pos_range[0],
            self.cfg.initial_joint_pos_range[1],
            (len(env_ids), len(self._arm_joint_ids)),
            device=self.device,
        )
        
        # Get current joint positions as base (this ensures correct shape)
        joint_pos = self.robot.data.joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        
        # Set arm joints to start position + noise
        joint_pos[:, self._arm_joint_ids] = self._start_joint_pos[env_ids][:, self._arm_joint_ids] + arm_joint_noise
        
        print(f"[RESET] joint_pos shape: {joint_pos.shape}, joint_vel shape: {joint_vel.shape}")
        
        # Apply resets
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        
        # Reset previous actions
        self._previous_actions[env_ids] = 0.0
        # Reset episode steps for these envs
        self._episode_steps[env_ids] = 0
        
        # Reset target
        self._target_ee_pos[env_ids] = torch.tensor(
            self.cfg.target_ee_pos, device=self.device
        ).repeat(len(env_ids), 1)
        self._target_ee_quat[env_ids] = torch.tensor(
            self.cfg.target_ee_quat, device=self.device
        ).repeat(len(env_ids), 1)


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
    """Calcule toutes les récompenses"""
    # Position tracking error
    pos_error = torch.norm(ee_pos - target_pos, dim=1)
    rew_position = rew_scale_position_tracking * pos_error ** 2
    rew_position_exp = rew_scale_position_tracking_exp * torch.exp(-position_exp_alpha * pos_error ** 2)
    
    # Goal reached bonus
    goal_reached = pos_error < goal_reached_threshold
    rew_goal = rew_scale_reached_goal * goal_reached.float()
    
    # Orientation error
    quat_error = quat_error_magnitude(ee_quat, target_quat)
    rew_orientation = rew_scale_orientation_error * quat_error
    
    # Joint velocity penalty
    rew_joint_vel = rew_scale_joint_vel * torch.sum(joint_vel ** 2, dim=1)
    
    # Action smoothness
    rew_action_rate = rew_scale_action_rate * torch.sum((actions - previous_actions) ** 2, dim=1)
    
    # Action magnitude penalty
    rew_action = rew_scale_action * torch.sum(actions ** 2, dim=1)
    
    # Alive bonus
    rew_alive = rew_scale_alive * (~reset_terminated).float()
    
    # Terminated penalty
    rew_terminated = rew_scale_terminated * reset_terminated.float()
    
    # Total reward
    total_reward = (
        rew_alive
        + rew_terminated
        + rew_position
        + rew_position_exp
        + rew_goal
        + rew_orientation
        + rew_joint_vel
        + rew_action_rate
        + rew_action
    )
    
    return total_reward