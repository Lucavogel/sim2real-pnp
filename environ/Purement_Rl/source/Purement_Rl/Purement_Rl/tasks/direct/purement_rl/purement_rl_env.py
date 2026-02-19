from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
import isaaclab.sim as sim_utils

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_from_euler_xyz

from .purement_rl_env_cfg import PurementRlEnvCfg


class PurementRlEnv(DirectRLEnv):
    cfg: PurementRlEnvCfg

    def __init__(self, cfg: PurementRlEnvCfg, render_mode: str | None = None, **kwargs):
        print("[INFO] PurementRlEnv: VERSION FIX 4 (Decimation 2, No Loop, No Auto-Term)")
        super().__init__(cfg, render_mode, **kwargs)

        # dt (pour convertir erreur position -> vitesse)
        self.dt = float(getattr(getattr(self.cfg, "sim", None), "dt", 0.01))
        # self.step_dt est déjà défini par DirectRLEnv comme property (dt * decimation)

        # Trajectory phase increment per RL step: match time between RL step and dataset.
        # Default: round(step_dt / traj_dt). Can be overridden via cfg.traj_phase_inc.
        try:
            cfg_inc = int(getattr(self.cfg, "traj_phase_inc", 0))
        except Exception:
            cfg_inc = 0
        if cfg_inc > 0:
            self._traj_phase_inc = cfg_inc
        else:
            traj_dt = float(getattr(self.cfg, "traj_dt", 0.0) or 0.0)
            step_dt = float(getattr(self, "step_dt", 0.0) or 0.0)
            if traj_dt > 0.0 and step_dt > 0.0:
                self._traj_phase_inc = max(1, int(round(step_dt / traj_dt)))
            else:
                self._traj_phase_inc = 1
        print(
            f"[INFO] traj_dt={float(getattr(self.cfg, 'traj_dt', 0.0)):.6f} | step_dt={float(getattr(self, 'step_dt', 0.0)):.6f} | traj_phase_inc={self._traj_phase_inc}"
        )

        # vitesse max (rad/s): float ou (dof,)
        self.max_joint_vel_rad_s = getattr(self.cfg, "max_joint_vel_rad_s", 1.0)

        # ✅ SOLUCE 3: warmup reward steps (ignore r_track/r_guide au début d'épisode)
        self.reward_warmup_steps = int(getattr(self.cfg, "reward_warmup_steps", 0))

        # ----------------------------
        # Joints contrôlés
        # ----------------------------
        self._joint_dof_idx, _ = self.robot.find_joints(self.cfg.joint_names)

        # ----------------------------
        # EE (outil)
        # ----------------------------
        self._ee_body_id, _ = self.robot.find_bodies(["wrist_3_link"])
        if isinstance(self._ee_body_id, (list, tuple)):
            self._ee_body_id = int(self._ee_body_id[0])
        elif torch.is_tensor(self._ee_body_id):
            self._ee_body_id = int(self._ee_body_id.item()) if self._ee_body_id.numel() == 1 else int(
                self._ee_body_id[0].item()
            )
        else:
            self._ee_body_id = int(self._ee_body_id)

        # ----------------------------
        # Charger dataset
        # ----------------------------
        data = np.load(self.cfg.traj_path, allow_pickle=True)
        if "paths" not in data.files:
            raise KeyError(f"Dataset: clé 'paths' introuvable. Keys dispo: {data.files}")

        q_paths = data["paths"]

        ee_key_candidates = ["ee_pos", "ee_positions", "eef_pos", "eef_positions", "ee", "eef", "ee_ref_pos"]
        ee_key = next((k for k in ee_key_candidates if k in data.files), None)
        if ee_key is None:
            raise KeyError(
                f"Dataset: aucune clé EE trouvée. Cherché {ee_key_candidates}. Keys dispo: {data.files}"
            )
        ee_list = data[ee_key]

        assert len(q_paths) > 0, "Dataset vide"
        assert len(q_paths) == len(ee_list), "paths et ee_pos doivent avoir la même longueur"

        self.num_trajs = len(q_paths)
        dof = len(self._joint_dof_idx)

        # ----------------------------
        # Padding dataset (GPU)
        # ----------------------------
        traj_lens = []
        max_len = 0
        for i, (q_i, ee_i) in enumerate(zip(q_paths, ee_list)):
            assert q_i.ndim == 2 and q_i.shape[1] == dof, f"paths[{i}] shape invalide: {q_i.shape}"
            assert ee_i.ndim == 2 and ee_i.shape[1] == 3, f"{ee_key}[{i}] shape invalide: {ee_i.shape}"
            assert q_i.shape[0] == ee_i.shape[0], f"T mismatch traj {i}"
            T = q_i.shape[0]
            traj_lens.append(T)
            max_len = max(max_len, T)

        self.max_len = int(max_len)
        self.traj_lens = torch.tensor(traj_lens, device=self.device, dtype=torch.long)

        q_pad = torch.zeros((self.num_trajs, self.max_len, dof), device=self.device, dtype=torch.float32)
        ee_pad = torch.zeros((self.num_trajs, self.max_len, 3), device=self.device, dtype=torch.float32)

        for i, (q_i, ee_i) in enumerate(zip(q_paths, ee_list)):
            q_t = torch.as_tensor(q_i, device=self.device, dtype=torch.float32)
            ee_t = torch.as_tensor(ee_i, device=self.device, dtype=torch.float32)
            T = q_t.shape[0]
            q_pad[i, :T] = q_t
            ee_pad[i, :T] = ee_t
            if T < self.max_len:
                q_pad[i, T:] = q_t[-1]
                ee_pad[i, T:] = ee_t[-1]

        self.q_paths_pad = q_pad
        self.ee_paths_pad = ee_pad

        # ----------------------------
        # Buffers par env
        # ----------------------------
        num_envs = self.scene.num_envs

        self.traj_id = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.traj_phase = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.env_traj_len = torch.ones(num_envs, dtype=torch.long, device=self.device)

        self.q_target = torch.zeros((num_envs, dof), device=self.device, dtype=torch.float32)
        self.qd_target = torch.zeros_like(self.q_target)

        # Cache refs (évite relecture multiple / step)
        self._ref_cache_step = -1
        self._q_ref_noisy_cache = torch.zeros((num_envs, dof), device=self.device, dtype=torch.float32)
        self._ee_des_noisy_cache = torch.zeros((num_envs, 3), device=self.device, dtype=torch.float32)

        # ----------------------------
        # Bruit sur la cible ee_des (par timestep)
        # ----------------------------
        self.ee_des_noise_std_m = float(getattr(self.cfg, "ee_des_noise_std_m", 0.0))
        self.ee_des_noise_xy_only = bool(getattr(self.cfg, "ee_des_noise_xy_only", True))
        self.ee_des_noise_in_observation = bool(getattr(self.cfg, "ee_des_noise_in_observation", True))
        self.ee_des_noise_in_reward = bool(getattr(self.cfg, "ee_des_noise_in_reward", False))

        # Optional: freeze z of the desired EE target (planar XY tracking)
        self.freeze_ee_des_z = bool(getattr(self.cfg, "freeze_ee_des_z", False))
        self.ee_des_z_fixed = torch.zeros((num_envs,), device=self.device, dtype=torch.float32)

        # Cache noisy target for observation so it's consistent within a step
        self._ee_des_obs_cache_step = -1
        self._ee_des_obs_cache = torch.zeros((num_envs, 3), device=self.device, dtype=torch.float32)

        # RMSE
        self.ee_err_sq_sum = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self.ee_err_count = torch.zeros(num_envs, device=self.device, dtype=torch.float32)

        # Reward shaping helpers
        self.prev_e_used = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self._last_reward_terms: dict[str, torch.Tensor] = {}

        # ----------------------------
        # SUCCESS METRIC (hold)
        # ----------------------------
        self.success_hold = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.success_episode = torch.zeros(num_envs, device=self.device, dtype=torch.float32)  # 0/1 (épisode)

        # ----------------------------
        # ACTIONS + LATENCE (1..3 steps)
        # ----------------------------
        self.action_dim = dof
        self.actions_raw = torch.zeros((num_envs, dof), device=self.device, dtype=torch.float32)

        # Action delay is expressed in SIMULATION steps (physics dt), not in RL steps.
        # This models real communication/processing delay of ~20-60ms when dt~1/60.
        self.action_delay_min = int(getattr(self.cfg, "action_delay_min", 1))
        self.action_delay_max = int(getattr(self.cfg, "action_delay_max", 3))
        self._action_delay_counter = torch.zeros((num_envs,), device=self.device, dtype=torch.long)
        self._pending_action = torch.zeros((num_envs, dof), device=self.device, dtype=torch.float32)
        self._action_applied = torch.zeros((num_envs, dof), device=self.device, dtype=torch.float32)
        self._sim_step_in_control_cycle = 0

        # Used to avoid jumping the trajectory phase immediately after a reset.
        self._just_reset = torch.zeros((num_envs,), device=self.device, dtype=torch.bool)

        # smoothness penalty uses delayed action (ce qui est vraiment appliqué)
        self.prev_actions_applied = torch.zeros((num_envs, dof), device=self.device, dtype=torch.float32)

        # Reward smoothness term (user spec): based on NN output action history (a_t - a_{t-1}).
        self.prev_actions_raw = torch.zeros((num_envs, dof), device=self.device, dtype=torch.float32)

        # ----------------------------
        # VITESSE OT (EE) - buffers
        # ----------------------------
        self.prev_ee_pos_local = torch.zeros((num_envs, 3), device=self.device, dtype=torch.float32)
        self.ee_vel_local = torch.zeros((num_envs, 3), device=self.device, dtype=torch.float32)
        self.prev_ee_vel_local = torch.zeros((num_envs, 3), device=self.device, dtype=torch.float32)
        self.ee_accel_local = torch.zeros((num_envs, 3), device=self.device, dtype=torch.float32)
        self.ee_speed = torch.zeros((num_envs,), device=self.device, dtype=torch.float32)

        # Print / log vitesse (garanti)
        self._print_ee_speed_every = int(getattr(self.cfg, "print_ee_speed_every", 20))
        self._speed_log_path = str(getattr(self.cfg, "ee_speed_log_path", "/tmp/ee_speed_log.txt"))

        # ----------------------------
        # CAMERA intrinsics
        # ----------------------------
        self.cam_fx = float(self.cfg.cam_fx)
        self.cam_fy = float(self.cfg.cam_fy)
        self.cam_cx = float(self.cfg.cam_cx)
        self.cam_cy = float(self.cfg.cam_cy)

        self.cam_pixel_noise_std = float(self.cfg.cam_pixel_noise_std)
        self.cam_depth_noise_std = float(self.cfg.cam_depth_noise_std)
        self.cam_xyz_jitter_std_m = float(getattr(self.cfg, "cam_xyz_jitter_std_m", 0.0))

        self.cam_extrinsic_bias_max_m = float(getattr(self.cfg, "cam_extrinsic_bias_max_m", 0.0))
        self.resample_cam_extrinsic_bias_each_episode = bool(
            getattr(self.cfg, "resample_cam_extrinsic_bias_each_episode", False)
        )
        self.cam_extrinsic_bias = torch.zeros((num_envs, 3), device=self.device, dtype=torch.float32)

        # Par défaut: erreur de calibration fixe (tirée une seule fois au démarrage).
        if (not self.resample_cam_extrinsic_bias_each_episode) and (self.cam_extrinsic_bias_max_m > 0.0):
            self.cam_extrinsic_bias[:] = (
                (2.0 * torch.rand((num_envs, 3), device=self.device) - 1.0) * self.cam_extrinsic_bias_max_m
            )

        cam_pos = torch.tensor(self.cfg.cam_pos_local, device=self.device, dtype=torch.float32)
        cam_rpy = self.cfg.cam_rpy_local
        self.R_cam_from_local = self._rpy_to_R(
            roll=float(cam_rpy[0]),
            pitch=float(cam_rpy[1]),
            yaw=float(cam_rpy[2]),
            device=self.device,
            dtype=torch.float32,
        )
        self.t_cam_from_local = cam_pos  # (3,)

        self.R_local_from_cam = self.R_cam_from_local.transpose(0, 1)
        self.t_local_from_cam_nominal = -(self.R_local_from_cam @ self.t_cam_from_local)

        self.use_camera_error_for_reward = bool(getattr(self.cfg, "use_camera_error_for_reward", False))

        # ----------------------------
        # Denoising caméra (EMA) + cache (évite 2 tirages de bruit/step)
        # ----------------------------
        self.use_camera_filter = bool(getattr(self.cfg, "use_camera_filter", False))
        self.cam_filter_alpha = float(getattr(self.cfg, "cam_filter_alpha", 1.0))
        self.cam_filter_alpha = float(max(0.0, min(1.0, self.cam_filter_alpha)))

        self._global_step = 0
        self._ee_meas_cache_step = -1
        self._ee_meas_raw_cache = torch.zeros((num_envs, 3), device=self.device, dtype=torch.float32)
        self._ee_meas_used_cache = torch.zeros((num_envs, 3), device=self.device, dtype=torch.float32)
        self.ee_meas_filt = torch.zeros((num_envs, 3), device=self.device, dtype=torch.float32)

        # ----------------------------
        # Domain rand physique (hooks)
        # ----------------------------
        self._warned_phys_rand = False
        self._warned_phys_rand_apply = False

        self.resample_damping_each_episode = bool(getattr(self.cfg, "resample_damping_each_episode", False))
        self.resample_payload_each_episode = bool(getattr(self.cfg, "resample_payload_each_episode", False))
        self.resample_friction_each_episode = bool(getattr(self.cfg, "resample_friction_each_episode", False))

        self.damping_scale = torch.ones((num_envs,), device=self.device, dtype=torch.float32)
        self.payload_mass = torch.zeros((num_envs,), device=self.device, dtype=torch.float32)
        self.friction_static = torch.ones((num_envs,), device=self.device, dtype=torch.float32)
        self.friction_dynamic = torch.ones((num_envs,), device=self.device, dtype=torch.float32)

        # Par défaut: on tire UNE fois au démarrage (valeurs fixes par env sur toute l'exécution).
        if bool(getattr(self.cfg, "randomize_damping", False)) and (not self.resample_damping_each_episode):
            lo, hi = self.cfg.damping_scale_range
            self.damping_scale[:] = lo + (hi - lo) * torch.rand((num_envs,), device=self.device)

        if bool(getattr(self.cfg, "randomize_payload", False)) and (not self.resample_payload_each_episode):
            mlo, mhi = self.cfg.payload_mass_range_kg
            self.payload_mass[:] = mlo + (mhi - mlo) * torch.rand((num_envs,), device=self.device)

        # Friction: by default sample ONCE per env and keep fixed for entire run.
        if bool(getattr(self.cfg, "randomize_friction", False)) and (not self.resample_friction_each_episode):
            slo, shi = self.cfg.friction_static_range
            dlo, dhi = self.cfg.friction_dynamic_range
            self.friction_static[:] = slo + (shi - slo) * torch.rand((num_envs,), device=self.device)
            self.friction_dynamic[:] = dlo + (dhi - dlo) * torch.rand((num_envs,), device=self.device)

        # One-time warning flag for friction application
        self._warned_friction_apply = False

        # IDs / nominal values for physics randomization
        self.payload_apply_to_all_links = bool(getattr(self.cfg, "payload_apply_to_all_links", False))
        self.payload_all_links_mode = str(getattr(self.cfg, "payload_all_links_mode", "distribute"))

        self._payload_body_id = None
        self._nominal_rigid_body_masses = None
        self._nominal_dof_damping = None
        self._nominal_payload_mass = None

        # Cache body id for payload (best effort)
        # If payload is applied to all links, we don't need a specific body id.
        if not self.payload_apply_to_all_links:
            try:
                body_id, _ = self.robot.find_bodies([str(getattr(self.cfg, "payload_body_name", "wrist_3_link"))])
                if isinstance(body_id, (list, tuple)):
                    body_id = int(body_id[0])
                elif torch.is_tensor(body_id):
                    body_id = int(body_id.item()) if body_id.numel() == 1 else int(body_id[0].item())
                else:
                    body_id = int(body_id)
                self._payload_body_id = body_id
            except Exception:
                self._payload_body_id = None

        # Read nominal damping / mass from PhysX view (if available)
        try:
            view = getattr(self.robot, "root_physx_view", None)
            if view is not None:
                # Nominal DOF damping
                if hasattr(view, "get_dof_damping"):
                    self._nominal_dof_damping = view.get_dof_damping().clone()
                elif hasattr(view, "dof_damping"):
                    self._nominal_dof_damping = view.dof_damping.clone()

                # Nominal payload mass / masses
                if bool(getattr(self.cfg, "randomize_payload", False)):
                    # If applying to all links, cache the full mass matrix if possible.
                    if self.payload_apply_to_all_links:
                        if hasattr(view, "get_rigid_body_masses"):
                            self._nominal_rigid_body_masses = view.get_rigid_body_masses().clone()
                        elif hasattr(view, "get_body_masses"):
                            self._nominal_rigid_body_masses = view.get_body_masses().clone()
                        elif hasattr(view, "rigid_body_masses"):
                            self._nominal_rigid_body_masses = view.rigid_body_masses.clone()
                    else:
                        if self._payload_body_id is not None:
                            if hasattr(view, "get_rigid_body_masses"):
                                masses = view.get_rigid_body_masses().clone()
                                # expected shape (N, num_bodies)
                                self._nominal_payload_mass = masses[:, self._payload_body_id].clone()
                            elif hasattr(view, "get_body_masses"):
                                masses = view.get_body_masses().clone()
                                self._nominal_payload_mass = masses[:, self._payload_body_id].clone()
                            elif hasattr(view, "rigid_body_masses"):
                                masses = view.rigid_body_masses.clone()
                                self._nominal_payload_mass = masses[:, self._payload_body_id].clone()
        except Exception:
            # Best effort only; we may still be able to apply later.
            pass

        # Best-effort: apply initial (possibly fixed) randomization at startup.
        try:
            all_env_ids = torch.arange(num_envs, device=self.device, dtype=torch.long)
            self._apply_physics_randomization_best_effort(all_env_ids)
        except Exception:
            pass

        self._debug_every = 50

        # init speed log
        try:
            with open(self._speed_log_path, "w") as f:
                f.write("# step, ee_speed_env0_mps, vx, vy, vz\n")
        except Exception as e:
            print(f"[WARN] Impossible d'initialiser le log vitesse: {self._speed_log_path} ({e})")

        print(f"[INFO] dt={self.dt} | max_joint_vel_rad_s={self.max_joint_vel_rad_s}")
        print(f"[INFO] EE speed print every={self._print_ee_speed_every} | log={self._speed_log_path}")
        print(f"[INFO] reward_warmup_steps={self.reward_warmup_steps}")

    def _get_ee_meas_local(self) -> torch.Tensor:
        """Return camera-measured EE position in local frame.

        Samples Gaussian noise once per env-step and reuses it across obs/reward.
        If enabled, applies an EMA filter to reduce Gaussian noise.
        """
        if self._ee_meas_cache_step == self._global_step:
            return self._ee_meas_used_cache

        ee_true = self._get_ee_pos_local_true()
        ee_meas_raw, _, _ = self._camera_measure_xyz_local_from_true(ee_true)
        self._ee_meas_raw_cache = ee_meas_raw

        if self.use_camera_filter and self.cam_filter_alpha > 0.0:
            a = self.cam_filter_alpha
            self.ee_meas_filt = (1.0 - a) * self.ee_meas_filt + a * ee_meas_raw
            ee_used = self.ee_meas_filt
        else:
            ee_used = ee_meas_raw

        self._ee_meas_used_cache = ee_used
        self._ee_meas_cache_step = self._global_step
        return ee_used

    # ------------------------------------------------------------------ #
    # SCÈNE
    # ------------------------------------------------------------------ #
    def _setup_scene(self):
        decor_cfg = self.cfg.env_asset_cfg
        decor_cfg.func("/World/envs/env_0/Origin2", decor_cfg)

        self.robot = Articulation(self.cfg.robot_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.scene.articulations["robot"] = self.robot
        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _lookahead_phase(self) -> torch.Tensor:
        k = int(self.cfg.lookahead_steps)
        # Clamp (default) or wrap (loop) to avoid out of bounds
        phase = self.traj_phase + k
        env_len = torch.clamp(self.env_traj_len, min=1)
        if str(getattr(self.cfg, "traj_end_mode", "hold")) == "loop":
            return torch.remainder(phase, env_len)
        return torch.clamp(phase, max=env_len - 1)

    def _get_noisy_refs(self, phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (q_ref, ee_des) from the dataset.

        Kept as a separate function because multiple parts of the env call it per step.
        `phase` is expected to be a (num_envs,) long tensor (already clamped).
        """
        # Cache per env-step: _get_noisy_refs() is used in pre_step/obs/reward/dones.
        # Without caching, we would re-index the dataset multiple times for the same step.
        if self._ref_cache_step == self._global_step:
            return self._q_ref_noisy_cache, self._ee_des_noisy_cache

        q_ref = self.q_paths_pad[self.traj_id, phase]
        ee_des = self.ee_paths_pad[self.traj_id, phase]

        # Optional: remove vertical command by freezing target z.
        if self.freeze_ee_des_z:
            ee_des_n = ee_des.clone()
            ee_des_n[:, 2] = self.ee_des_z_fixed
            ee_des = ee_des_n

        # Optional: add Gaussian noise to the target ee_des for REWARD/DONES only.
        if self.ee_des_noise_in_reward and (self.ee_des_noise_std_m > 0.0):
            ee_des = self._apply_ee_des_noise(ee_des)

        self._q_ref_noisy_cache = q_ref
        self._ee_des_noisy_cache = ee_des
        self._ref_cache_step = self._global_step
        return q_ref, ee_des

    def _get_ee_des_for_observation(self, phase: torch.Tensor) -> torch.Tensor:
        """Return ee_des as seen by the agent (optionally noisy), cached per env-step."""
        if self._ee_des_obs_cache_step == self._global_step:
            return self._ee_des_obs_cache

        # Start from clean target (or reward-noisy target if configured that way)
        _, ee_des = self._get_noisy_refs(phase)

        if self.ee_des_noise_in_observation and (self.ee_des_noise_std_m > 0.0):
            ee_des = self._apply_ee_des_noise(ee_des)

        self._ee_des_obs_cache = ee_des
        self._ee_des_obs_cache_step = self._global_step
        return ee_des

    def _apply_ee_des_noise(self, ee_des: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian noise to desired EE target.

        Ensures compatibility with planar tasks: when `freeze_ee_des_z` is enabled, Z is not jittered.
        """
        if self.ee_des_noise_std_m <= 0.0:
            return ee_des

        # If planar Z is frozen, never jitter Z (even if ee_des_noise_xy_only=False).
        xy_only = bool(self.ee_des_noise_xy_only) or bool(self.freeze_ee_des_z)
        if xy_only:
            ee_des_n = ee_des.clone()
            ee_des_n[:, 0:2] = ee_des_n[:, 0:2] + torch.randn_like(ee_des_n[:, 0:2]) * self.ee_des_noise_std_m
            return ee_des_n
        return ee_des + torch.randn_like(ee_des) * self.ee_des_noise_std_m

    def _rpy_to_R(self, roll: float, pitch: float, yaw: float, device, dtype) -> torch.Tensor:
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)

        Rx = torch.tensor([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], device=device, dtype=dtype)
        Ry = torch.tensor([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], device=device, dtype=dtype)
        Rz = torch.tensor([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], device=device, dtype=dtype)
        return Rz @ Ry @ Rx

    def _local_to_cam(self, p_local: torch.Tensor) -> torch.Tensor:
        t_eff = self.t_cam_from_local + self.cam_extrinsic_bias
        return (p_local @ self.R_cam_from_local.transpose(0, 1)) + t_eff

    def _cam_to_local(self, p_cam: torch.Tensor) -> torch.Tensor:
        t_eff = self.t_cam_from_local + self.cam_extrinsic_bias
        t_local_eff = -(t_eff @ self.R_local_from_cam.transpose(0, 1))
        return (p_cam @ self.R_local_from_cam.transpose(0, 1)) + t_local_eff

    def _project_cam_to_pixel(self, p_cam: torch.Tensor):
        X = p_cam[:, 0]
        Y = p_cam[:, 1]
        Z = torch.clamp(p_cam[:, 2], min=1e-6)
        u = self.cam_fx * (X / Z) + self.cam_cx
        v = self.cam_fy * (Y / Z) + self.cam_cy
        return u, v, Z

    def _unproject_pixel_to_cam(self, u: torch.Tensor, v: torch.Tensor, Z: torch.Tensor):
        X = (u - self.cam_cx) * (Z / self.cam_fx)
        Y = (v - self.cam_cy) * (Z / self.cam_fy)
        return torch.stack([X, Y, Z], dim=-1)

    def _get_ee_pos_local_true(self) -> torch.Tensor:
        d = self.robot.data
        if hasattr(d, "body_state_w"):
            ee_state = d.body_state_w[:, self._ee_body_id, :]
            ee_pos_w = ee_state[:, 0:3]
        else:
            ee_pos_w = d.body_pos_w[:, self._ee_body_id, :]

        # Transformation: World -> Robot Base Frame
        # Robot base is at env_origin + [0, 0, 0.725] and rotated by 180 deg (PI) around Z.
        origin = self.scene.env_origins[:, :3]
        base_offset = torch.tensor([0.0, 0.0, 0.725], device=self.device)
        
        # Position relative to robot base (in World Orientation)
        p_rel_world = ee_pos_w - (origin + base_offset)

        # Apply inverse rotation (RotZ(-PI) = RotZ(PI))
        # x' = -x, y' = -y
        p_local = p_rel_world.clone()
        p_local[:, 0] = -p_rel_world[:, 0]
        p_local[:, 1] = -p_rel_world[:, 1]
        
        return p_local

    def _camera_measure_xyz_local_from_true(self, p_local_true: torch.Tensor):
        p_cam_true = self._local_to_cam(p_local_true)
        u, v, Z = self._project_cam_to_pixel(p_cam_true)

        noise_uv = (
            torch.randn((p_local_true.shape[0], 2), device=self.device, dtype=torch.float32) * self.cam_pixel_noise_std
        )
        uv = torch.stack([u, v], dim=-1) + noise_uv

        if self.cam_depth_noise_std > 0.0:
            Z_meas = Z + torch.randn_like(Z) * self.cam_depth_noise_std
            Z_meas = torch.clamp(Z_meas, min=1e-6)
        else:
            Z_meas = Z

        p_cam_meas = self._unproject_pixel_to_cam(uv[:, 0], uv[:, 1], Z_meas)
        p_local_meas = self._cam_to_local(p_cam_meas)

        if self.cam_xyz_jitter_std_m > 0.0:
            p_local_meas = p_local_meas + torch.randn_like(p_local_meas) * self.cam_xyz_jitter_std_m

        return p_local_meas, uv, Z_meas

    def _apply_physics_randomization_best_effort(self, env_ids_t: torch.Tensor):
        # Sampling policy:
        # - If resample_*_each_episode=True, sample on every reset.
        # - Else, keep the already-sampled per-env values (fixed for entire run).
        if bool(getattr(self.cfg, "randomize_damping", False)):
            if self.resample_damping_each_episode:
                lo, hi = self.cfg.damping_scale_range
                self.damping_scale[env_ids_t] = lo + (hi - lo) * torch.rand((len(env_ids_t),), device=self.device)
        else:
            self.damping_scale[env_ids_t] = 1.0

        if bool(getattr(self.cfg, "randomize_payload", False)):
            if self.resample_payload_each_episode:
                mlo, mhi = self.cfg.payload_mass_range_kg
                self.payload_mass[env_ids_t] = mlo + (mhi - mlo) * torch.rand((len(env_ids_t),), device=self.device)
        else:
            self.payload_mass[env_ids_t] = 0.0

        if bool(getattr(self.cfg, "randomize_friction", False)):
            if self.resample_friction_each_episode:
                slo, shi = self.cfg.friction_static_range
                dlo, dhi = self.cfg.friction_dynamic_range
                self.friction_static[env_ids_t] = slo + (shi - slo) * torch.rand((len(env_ids_t),), device=self.device)
                self.friction_dynamic[env_ids_t] = dlo + (dhi - dlo) * torch.rand((len(env_ids_t),), device=self.device)
        else:
            self.friction_static[env_ids_t] = 1.0
            self.friction_dynamic[env_ids_t] = 1.0

        # Apply to simulation (best effort). If we cannot find the right API, we keep the sampled values for logging.
        try:
            view = getattr(self.robot, "root_physx_view", None)
            if view is None:
                raise RuntimeError("root_physx_view is None")

            # ----------------------------
            # Joint damping scaling
            # ----------------------------
            if bool(getattr(self.cfg, "randomize_damping", False)) and (self._nominal_dof_damping is not None):
                # expected nominal shape: (N, dof) or (dof,)
                nominal = self._nominal_dof_damping
                if nominal.dim() == 1:
                    nominal_env = nominal.view(1, -1).repeat(self.scene.num_envs, 1)
                else:
                    nominal_env = nominal
                scale = self.damping_scale[env_ids_t].view(-1, 1)
                new_damping = nominal_env[env_ids_t] * scale

                if hasattr(view, "set_dof_damping"):
                    view.set_dof_damping(new_damping, env_ids=env_ids_t)
                elif hasattr(view, "dof_damping"):
                    view.dof_damping[env_ids_t] = new_damping

            # ----------------------------
            # Payload mass randomization (additive)
            # ----------------------------
            if bool(getattr(self.cfg, "randomize_payload", False)):
                if self.payload_apply_to_all_links:
                    # Apply a total additive mass per env, distributed across all links by default.
                    if self._nominal_rigid_body_masses is None:
                        if hasattr(view, "get_rigid_body_masses"):
                            self._nominal_rigid_body_masses = view.get_rigid_body_masses().clone()
                        elif hasattr(view, "get_body_masses"):
                            self._nominal_rigid_body_masses = view.get_body_masses().clone()
                        elif hasattr(view, "rigid_body_masses"):
                            self._nominal_rigid_body_masses = view.rigid_body_masses.clone()

                    if self._nominal_rigid_body_masses is not None:
                        masses = self._nominal_rigid_body_masses.clone()
                        num_bodies = int(masses.shape[1]) if masses.dim() == 2 else 0
                        if num_bodies > 0:
                            mode = str(self.payload_all_links_mode).lower().strip()
                            if mode not in ("distribute", "per_link"):
                                mode = "distribute"

                            if mode == "per_link":
                                add = self.payload_mass[env_ids_t].view(-1, 1).expand(-1, num_bodies)
                            else:
                                # distribute (default): treat payload_mass as TOTAL extra mass to distribute
                                add = (self.payload_mass[env_ids_t].view(-1, 1) / float(num_bodies)).expand(
                                    -1, num_bodies
                                )

                            masses[env_ids_t, :] = torch.clamp(masses[env_ids_t, :] + add, min=1e-6)

                            if hasattr(view, "set_rigid_body_masses"):
                                view.set_rigid_body_masses(masses, env_ids=env_ids_t)
                            elif hasattr(view, "set_body_masses"):
                                view.set_body_masses(masses, env_ids=env_ids_t)
                            elif hasattr(view, "rigid_body_masses"):
                                view.rigid_body_masses[env_ids_t, :] = masses[env_ids_t, :]

                else:
                    # Original behavior: apply additive mass to one body.
                    if self._payload_body_id is not None:
                        if self._nominal_payload_mass is None:
                            # Try reading masses lazily (in case view wasn't ready at __init__)
                            if hasattr(view, "get_rigid_body_masses"):
                                masses = view.get_rigid_body_masses().clone()
                                self._nominal_payload_mass = masses[:, self._payload_body_id].clone()
                            elif hasattr(view, "get_body_masses"):
                                masses = view.get_body_masses().clone()
                                self._nominal_payload_mass = masses[:, self._payload_body_id].clone()
                            elif hasattr(view, "rigid_body_masses"):
                                masses = view.rigid_body_masses.clone()
                                self._nominal_payload_mass = masses[:, self._payload_body_id].clone()

                        if self._nominal_payload_mass is not None:
                            new_mass = self._nominal_payload_mass[env_ids_t] + self.payload_mass[env_ids_t]
                            new_mass = torch.clamp(new_mass, min=1e-6)

                            if hasattr(view, "set_rigid_body_masses"):
                                # expected full array shape (N, num_bodies)
                                masses = None
                                if hasattr(view, "get_rigid_body_masses"):
                                    masses = view.get_rigid_body_masses().clone()
                                elif hasattr(view, "rigid_body_masses"):
                                    masses = view.rigid_body_masses.clone()
                                if masses is not None:
                                    masses[env_ids_t, self._payload_body_id] = new_mass
                                    view.set_rigid_body_masses(masses, env_ids=env_ids_t)
                            elif hasattr(view, "set_body_masses"):
                                masses = None
                                if hasattr(view, "get_body_masses"):
                                    masses = view.get_body_masses().clone()
                                if masses is not None:
                                    masses[env_ids_t, self._payload_body_id] = new_mass
                                    view.set_body_masses(masses, env_ids=env_ids_t)
                            elif hasattr(view, "rigid_body_masses"):
                                view.rigid_body_masses[env_ids_t, self._payload_body_id] = new_mass

            # ----------------------------
            # Friction randomization (robot shapes) - best effort
            # ----------------------------
            if bool(getattr(self.cfg, "randomize_friction", False)):
                try:
                    props = None
                    if hasattr(view, "get_rigid_shape_material_properties"):
                        props = view.get_rigid_shape_material_properties()
                        if torch.is_tensor(props):
                            props = props.clone()
                    elif hasattr(view, "rigid_shape_material_properties"):
                        props = view.rigid_shape_material_properties
                        if torch.is_tensor(props):
                            props = props.clone()

                    if not torch.is_tensor(props):
                        raise RuntimeError("No tensor material properties available")

                    # Expected: (N, num_shapes, K) with K>=2 => [static, dynamic, ...]
                    if props.dim() == 2:
                        props = props.unsqueeze(0).repeat(self.scene.num_envs, 1, 1)
                    if props.dim() != 3 or props.shape[-1] < 2:
                        raise RuntimeError(f"Unexpected material props shape: {tuple(props.shape)}")

                    num_shapes = props.shape[1]
                    s = self.friction_static[env_ids_t].view(-1, 1).expand(-1, num_shapes)
                    d = self.friction_dynamic[env_ids_t].view(-1, 1).expand(-1, num_shapes)

                    # Ensure static >= dynamic (common PhysX expectation)
                    s = torch.maximum(s, d)

                    props[env_ids_t, :, 0] = s
                    props[env_ids_t, :, 1] = d

                    if hasattr(view, "set_rigid_shape_material_properties"):
                        view.set_rigid_shape_material_properties(props, env_ids=env_ids_t)
                    elif hasattr(view, "rigid_shape_material_properties"):
                        view.rigid_shape_material_properties = props
                except Exception:
                    if not getattr(self, "_warned_friction_apply", False):
                        print(
                            "[WARN] Randomisation friction: impossible d'appliquer via PhysX view (best-effort). "
                            "Les valeurs sont échantillonnées mais la simu peut ne pas être modifiée.",
                            flush=True,
                        )
                        self._warned_friction_apply = True

        except Exception:
            if not self._warned_phys_rand_apply:
                print(
                    "[WARN] Randomisation damping/payload: impossible d'appliquer via PhysX view (best-effort). "
                    "Les valeurs sont échantillonnées mais la simu peut ne pas être modifiée.",
                    flush=True,
                )
                self._warned_phys_rand_apply = True

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Global step counter (used for sensor cache)
        self._global_step += 1
        self._ee_meas_cache_step = -1
        self._ref_cache_step = -1
        self._ee_des_obs_cache_step = -1

        # Reset sim-step counter within the decimation loop (DirectRLEnv calls _apply_action multiple times).
        self._sim_step_in_control_cycle = 0

        self.actions_raw = torch.clamp(actions, -1.0, 1.0)

        # New command arrives now, but will be applied after a random delay (in simulation steps).
        self._pending_action.copy_(self.actions_raw)
        dmin = int(self.action_delay_min)
        dmax = int(self.action_delay_max)
        if dmax < dmin:
            dmax = dmin
        self._action_delay_counter = torch.randint(dmin, dmax + 1, (self.scene.num_envs,), device=self.device)

        # Advance dataset phase (but do not advance on the very first step after a reset).
        inc = int(getattr(self, "_traj_phase_inc", 1))
        if inc != 0:
            if getattr(self, "_just_reset", None) is not None and self._just_reset.numel() == self.traj_phase.numel():
                mask = ~self._just_reset
                if torch.any(mask):
                    self.traj_phase[mask] += inc
                # Clear flag after first post-reset step.
                self._just_reset.zero_()
            else:
                self.traj_phase += inc
        if str(getattr(self.cfg, "traj_end_mode", "hold")) == "loop":
            env_len = torch.clamp(self.env_traj_len, min=1)
            self.traj_phase = torch.remainder(self.traj_phase, env_len)

        # Targets are computed in _apply_action() at every simulation step, so nothing more to do here.

    def _apply_action(self) -> None:
        # Apply delayed action at SIMULATION step frequency.
        # For a delay of k sim steps: keep previous applied action for k sim steps.
        if self._action_delay_counter.numel() > 0:
            still_waiting = self._action_delay_counter > 0
            # Decrement counters (clamp at 0)
            self._action_delay_counter = torch.clamp(self._action_delay_counter - 1, min=0)
            # Once counter reached 0, switch to the pending command
            ready_now = self._action_delay_counter == 0
            # Update applied action only for envs that are ready now and were waiting before
            to_update = ready_now & still_waiting
            if torch.any(to_update):
                self._action_applied[to_update] = self._pending_action[to_update]

        # Compute control targets using CURRENT applied action (dt-level).
        phase_la = self._lookahead_phase()
        q_ref, _ = self._get_noisy_refs(phase_la)

        q_des = q_ref + self._action_applied * float(self.cfg.action_scale)

        if torch.is_tensor(self.max_joint_vel_rad_s):
            vmax = self.max_joint_vel_rad_s.to(device=self.device, dtype=torch.float32)
        elif isinstance(self.max_joint_vel_rad_s, (list, tuple, np.ndarray)):
            vmax = torch.tensor(self.max_joint_vel_rad_s, device=self.device, dtype=torch.float32)
        else:
            vmax = torch.full((q_des.shape[1],), float(self.max_joint_vel_rad_s), device=self.device, dtype=torch.float32)

        q = self.robot.data.joint_pos[:, self._joint_dof_idx]
        dt = max(float(self.dt), 1e-8)
        qd_cmd = (q_des - q) / dt
        qd_cmd = torch.clamp(qd_cmd, -vmax, +vmax)

        self.qd_target = qd_cmd
        self.q_target = q + qd_cmd * dt

        self.robot.set_joint_position_target(self.q_target, joint_ids=self._joint_dof_idx)
        self.robot.set_joint_velocity_target(self.qd_target, joint_ids=self._joint_dof_idx)

        self._sim_step_in_control_cycle += 1

    def _get_observations(self) -> dict:
        q = self.robot.data.joint_pos[:, self._joint_dof_idx]
        qd = self.robot.data.joint_vel[:, self._joint_dof_idx]

        phase_la = self._lookahead_phase()
        q_ref, _ = self._get_noisy_refs(phase_la)
        ee_des = self._get_ee_des_for_observation(phase_la)
        err_q = q_ref - q

        ee_true = self._get_ee_pos_local_true()
        ee_meas = self._get_ee_meas_local()

        e_true = ee_des - ee_true
        e_meas = ee_des - ee_meas

        e_true_norm = torch.norm(e_true, dim=-1)
        e_meas_norm = torch.norm(e_meas, dim=-1, keepdim=True)

        self.ee_err_sq_sum += (e_true_norm ** 2)
        self.ee_err_count += 1.0

        good = e_true_norm < float(self.cfg.success_ee_thresh_m)
        self.success_hold = torch.where(good, self.success_hold + 1, torch.zeros_like(self.success_hold))
        achieved = self.success_hold >= int(self.cfg.success_hold_steps)
        self.success_episode = torch.maximum(self.success_episode, achieved.float())

        obs = torch.cat(
            (
                q,
                qd,
                q_ref,
                err_q,
                ee_meas,
                ee_des,
                e_meas,
                e_meas_norm,
                getattr(self, "_action_applied", self.actions_raw),
            ),
            dim=-1,
        )
        return {"policy": obs}

    # ------------------------------------------------------------------ #
    # REWARD + SOLUCE 3: warmup masking
    # ------------------------------------------------------------------ #
    def _get_rewards(self) -> torch.Tensor:
        phase_la = self._lookahead_phase()

        # ---- Vitesse EE (ton code inchangé)
        ee_true_now = self._get_ee_pos_local_true()
        v = (ee_true_now - self.prev_ee_pos_local) / max(self.step_dt, 1e-8)
        self.ee_vel_local = v
        self.ee_speed = torch.norm(v, dim=-1)

        # Accélération EE (stabilité OT)
        a = (v - self.prev_ee_vel_local) / max(self.step_dt, 1e-8)
        self.ee_accel_local = a
        self.prev_ee_vel_local = v.clone()

        self.prev_ee_pos_local = ee_true_now.clone()

        step0 = int(self.episode_length_buf[0].item())
        if self._print_ee_speed_every > 0 and (step0 % self._print_ee_speed_every) == 0:
            s0 = float(self.ee_speed[0].item())
            vx0, vy0, vz0 = self.ee_vel_local[0].tolist()
            print(f"[EE SPEED env0] {s0:.6f} m/s | vxyz=[{vx0:.6f}, {vy0:.6f}, {vz0:.6f}]", flush=True)
            try:
                with open(self._speed_log_path, "a") as f:
                    f.write(f"{step0},{s0:.8f},{vx0:.8f},{vy0:.8f},{vz0:.8f}\n")
            except Exception:
                pass

        # ---- Reward (per user spec):
        # r_t = r_track + r_guide - p_action - p_smooth
        ee_true = ee_true_now
        ee_meas = self._get_ee_meas_local()
        q_ref, ee_des = self._get_noisy_refs(phase_la)

        # Tracking reward: minimize position error.
        # By default, use true EE error for reward (recommended for stable learning), but
        # allow using camera-measured EE error if requested.
        # e = ||Target_xyz - EE_xyz||
        # r_track = exp(-e^2 / sigma_pos)
        use_cam_for_reward = bool(getattr(self.cfg, "use_camera_error_for_reward", False))
        ee_for_reward = ee_meas if use_cam_for_reward else ee_true
        e_vec = ee_des - ee_for_reward
        e = torch.norm(e_vec, dim=-1)
        sigma_pos = max(float(getattr(self.cfg, "sigma_pos", 0.02)), 1e-8)
        r_track = torch.exp(-(e * e) / sigma_pos)

        # Guidance reward: keep joints close to the MoveIt/reference joints.
        # e_q = ||q_ref - q_current||
        # r_guide = exp(-e_q^2 / sigma_joint)
        q = self.robot.data.joint_pos[:, self._joint_dof_idx]
        e_q = torch.norm(q_ref - q, dim=-1)
        sigma_joint = max(float(getattr(self.cfg, "sigma_joint", 0.4)), 1e-8)
        r_guide = torch.exp(-(e_q * e_q) / sigma_joint)

        # Action penalty: minimal effort (based on NN output a_t).
        a_t = self.actions_raw
        p_action = float(getattr(self.cfg, "lambda_action", 0.0)) * torch.sum(a_t * a_t, dim=-1)

        # Smoothness / jerk penalty: prevent shaking (based on NN output delta).
        da = a_t - self.prev_actions_raw
        p_smooth = float(getattr(self.cfg, "lambda_smooth", 0.0)) * torch.sum(da * da, dim=-1)
        # Avoid penalizing the first step after reset.
        p_smooth = torch.where(self.episode_length_buf == 0, torch.zeros_like(p_smooth), p_smooth)
        self.prev_actions_raw.copy_(a_t)
        # ----
        scale_track = float(getattr(self.cfg, "rew_scale_track", 1.0))
        scale_guide = float(getattr(self.cfg, "rew_scale_guide", 1.0))

        

        # Cache terms for logging/debug
        self._last_reward_terms = {
            "r_track": r_track.detach(),
            "r_guide": r_guide.detach(),
            "p_action": p_action.detach(),
            "p_smooth": p_smooth.detach(),
            "e": e.detach(),
            "e_q": e_q.detach(),
        }

        return (scale_track * r_track) + (scale_guide * r_guide) - p_action - p_smooth

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        out_of_bounds = torch.any(
            torch.abs(self.robot.data.joint_pos[:, self._joint_dof_idx]) > 2 * math.pi,
            dim=1,
        )

        max_ee_error = 4.0 + 0.05 * self.episode_length_buf
        phase_la = self._lookahead_phase()
        ee_true = self._get_ee_pos_local_true()
        _, ee_des = self._get_noisy_refs(phase_la)
        e_true_norm = torch.norm(ee_des - ee_true, dim=-1)
        ee_exploded = e_true_norm > max_ee_error

        # Hard constraints on vertical motion (optional)
        terminated_ee_speed_z = torch.zeros_like(time_out, dtype=torch.bool)
        if bool(getattr(self.cfg, "terminate_on_ee_speed_z_over", False)):
            max_ee_speed_z = float(getattr(self.cfg, "max_ee_speed_z_m_s", 0.0))
            if max_ee_speed_z > 0.0:
                vz = self.ee_vel_local[:, 2]
                terminated_ee_speed_z = torch.abs(vz) > max_ee_speed_z

        terminated_ee_z_error = torch.zeros_like(time_out, dtype=torch.bool)
        if bool(getattr(self.cfg, "terminate_on_ee_z_error_over", False)):
            max_ee_z_error = float(getattr(self.cfg, "max_ee_z_error_m", 0.0))
            if max_ee_z_error > 0.0:
                ez = (ee_des - ee_true)[:, 2]
                terminated_ee_z_error = torch.abs(ez) > max_ee_z_error
        
        traj_end_mode = str(getattr(self.cfg, "traj_end_mode", "hold"))

        # Trajectory finished flag (meaningful only for hold/reset modes)
        traj_finished = self.traj_phase >= (torch.clamp(self.env_traj_len, min=1) - 1)
        if traj_end_mode == "loop":
            traj_finished = torch.zeros_like(time_out, dtype=torch.bool)

        # Default (hold): terminate only when finished AND success met
        success_met = self.success_hold >= int(self.cfg.success_hold_steps)

        terminated = out_of_bounds | ee_exploded | terminated_ee_speed_z | terminated_ee_z_error
        if traj_end_mode == "hold":
            terminated = terminated | (traj_finished & success_met)
        elif traj_end_mode == "reset":
            # Truncate episode when the trajectory ends, regardless of success
            time_out = time_out | traj_finished
        else:
            # Unknown mode -> behave like hold (but without auto-termination)
            pass
        

        return terminated, time_out

    def _get_extras(self) -> dict:
        success_rate = torch.mean(self.success_episode).detach()
        ee_speed_mean = torch.mean(self.ee_speed).detach()
        ee_speed_max = torch.max(self.ee_speed).detach()
        log = {
            "success_rate": success_rate,
            "ee_speed_mean": ee_speed_mean,
            "ee_speed_max": ee_speed_max,
        }
        if getattr(self, "_last_reward_terms", None):
            for k, v in self._last_reward_terms.items():
                # log mean values only (scalars)
                try:
                    log[k + "_mean"] = torch.mean(v).detach()
                except Exception:
                    pass
        return {"log": log}

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # Mark for next step: avoid advancing traj_phase immediately after reset.
        if getattr(self, "_just_reset", None) is not None:
            self._just_reset[env_ids_t] = True

        super()._reset_idx(env_ids)

        # Trajectory selection at reset.
        # Training often wants resampling; deployment often wants restarting the same mission.
        resample_traj = bool(getattr(self.cfg, "resample_traj_id_on_reset", True))
        if resample_traj:
            new_traj_id = torch.randint(0, self.num_trajs, (len(env_ids_t),), device=self.device)
            self.traj_id[env_ids_t] = new_traj_id
        else:
            # Keep existing traj_id but ensure it's in bounds.
            self.traj_id[env_ids_t] = torch.clamp(self.traj_id[env_ids_t], 0, self.num_trajs - 1)

        self.traj_phase[env_ids_t] = 0
        traj_id_sel = self.traj_id[env_ids_t]
        self.env_traj_len[env_ids_t] = self.traj_lens[traj_id_sel]

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(self.robot.data.default_joint_vel[env_ids])

        root_state = self.robot.data.default_root_state[env_ids].clone()

        zeros_t = torch.zeros(len(env_ids_t), dtype=torch.long, device=self.device)
        q0 = self.q_paths_pad[traj_id_sel, zeros_t]
        joint_pos[:, self._joint_dof_idx] = q0

        base_offset = torch.tensor([0.0, 0.0, 0.725], device=self.device)
        root_state[:, :3] = self.scene.env_origins[env_ids_t] + base_offset

        rx = torch.zeros(len(env_ids), device=self.device)
        ry = torch.zeros(len(env_ids), device=self.device)
        rz = torch.full((len(env_ids),), math.pi, device=self.device)
        root_state[:, 3:7] = quat_from_euler_xyz(rx, ry, rz)

        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.q_target[env_ids_t] = q0
        self.qd_target[env_ids_t].zero_()

        self.ee_err_sq_sum[env_ids_t] = 0.0
        self.ee_err_count[env_ids_t] = 0.0

        self.success_hold[env_ids_t] = 0
        self.success_episode[env_ids_t] = 0.0

        # Reset action delay buffers (delay is re-sampled at each new action command)
        self._action_delay_counter[env_ids_t].zero_()
        self._pending_action[env_ids_t].zero_()
        self._action_applied[env_ids_t].zero_()
        self.prev_actions_applied[env_ids_t].zero_()

        if self.cam_extrinsic_bias_max_m <= 0.0:
            # Désactivé => force à zéro à chaque reset
            self.cam_extrinsic_bias[env_ids_t].zero_()
        elif self.resample_cam_extrinsic_bias_each_episode:
            # Ancien comportement: re-tirage à chaque épisode
            self.cam_extrinsic_bias[env_ids_t] = (
                (2.0 * torch.rand((len(env_ids_t), 3), device=self.device) - 1.0) * self.cam_extrinsic_bias_max_m
            )
        else:
            # Biais fixe: ne rien faire au reset
            pass

        self._apply_physics_randomization_best_effort(env_ids_t)

        # init vitesse EE : prev = current (évite un spike)
        ee_now = self._get_ee_pos_local_true()
        if self.freeze_ee_des_z:
            # Fix z target to current EE z at reset (planar XY task without initial z mismatch).
            self.ee_des_z_fixed[env_ids_t] = ee_now[env_ids_t, 2]
        self.prev_ee_pos_local[env_ids_t] = ee_now[env_ids_t]
        self.ee_vel_local[env_ids_t].zero_()
        self.prev_ee_vel_local[env_ids_t].zero_()
        self.ee_accel_local[env_ids_t].zero_()
        self.ee_speed[env_ids_t].zero_()

        # init filtre caméra : démarre au niveau de la première mesure (évite un transitoire)
        try:
            ee_meas0, _, _ = self._camera_measure_xyz_local_from_true(ee_now)
            self.ee_meas_filt[env_ids_t] = ee_meas0[env_ids_t]
        except Exception:
            # Best effort: if camera path isn't ready yet, keep zeros.
            pass

        # invalidate caches for these envs
        self._ref_cache_step = -1

