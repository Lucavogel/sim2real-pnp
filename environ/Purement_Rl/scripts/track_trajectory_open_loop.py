"""
Script simplifié pour le suivi de trajectoire en boucle ouverte (sans politique apprise).
Il envoie des actions nulles pour suivre la trajectoire de référence définie dans le fichier .npz.
"""

import argparse
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

# ==============================================================================
# Helper Classes (Copied from isaaclab_deploy_policy_npz.py)
# ==============================================================================
class _DeployLogger:
    """Log deployment rollout signals to a NPZ file (post-deployment plotting)."""

    def __init__(self, env_unwrapped, env_ids: list[int], dt: float, log_every: int = 1):
        import numpy as np

        self.env = env_unwrapped
        self.env_ids = [int(i) for i in env_ids]
        self.dt = float(dt)
        self.log_every = max(1, int(log_every))

        self._np = np
        self._t: list[float] = []
        self._traj_id: list[list[int]] = []
        self._traj_phase: list[list[int]] = []

        self._ee_true: list[list[list[float]]] = []
        self._ee_meas: list[list[list[float]]] = []
        self._ee_des_clean: list[list[list[float]]] = []
        self._ee_des_obs: list[list[list[float]]] = []

        self._ee_speed: list[list[float]] = []

        self._q_meas: list[list[list[float]]] = []
        self._qd_meas: list[list[list[float]]] = []
        self._q_ref: list[list[list[float]]] = []
        self._q_target: list[list[list[float]]] = []
        self._qd_target: list[list[list[float]]] = []

        self._e_true_clean: list[list[list[float]]] = []
        self._e_true_obs: list[list[list[float]]] = []
        self._e_meas_clean: list[list[list[float]]] = []

    def _to_py3(self, x):
        return [float(x[0]), float(x[1]), float(x[2])]

    @staticmethod
    def _to_py_list_1d(x):
        return [float(v) for v in x]

    def update(self, step: int):
        if (step % self.log_every) != 0:
            return

        import torch
        env = self.env
        ids = self.env_ids

        # Time
        self._t.append(float(step) * self.dt)

        # traj_id / traj_phase
        traj_id = []
        traj_phase = []
        if hasattr(env, "traj_id"):
            traj_id = [int(env.traj_id[i].item()) for i in ids]
        else:
            traj_id = [-1 for _ in ids]
        if hasattr(env, "traj_phase"):
            traj_phase = [int(env.traj_phase[i].item()) for i in ids]
        else:
            traj_phase = [-1 for _ in ids]
        self._traj_id.append(traj_id)
        self._traj_phase.append(traj_phase)

        # True EE (local)
        ee_true = env._get_ee_pos_local_true()
        ee_true_sel = ee_true[ids, :]

        # EE speed
        ee_speed_sel = None
        if hasattr(env, "ee_speed"):
            try:
                ee_speed_sel = env.ee_speed[ids].detach().cpu().numpy().tolist()
            except Exception:
                ee_speed_sel = None

        # Measured EE
        ee_meas_sel = None
        if hasattr(env, "_get_ee_meas_local"):
            try:
                ee_meas = env._get_ee_meas_local()
                ee_meas_sel = ee_meas[ids, :]
            except Exception:
                ee_meas_sel = None

        # Desired EE
        ee_des_clean_sel = None
        try:
            if hasattr(env, "ee_paths_pad") and hasattr(env, "traj_id") and hasattr(env, "traj_phase"):
                phase = env.traj_phase.clone()
                if hasattr(env, "_lookahead_phase"):
                    phase = env._lookahead_phase()
                # Clamp safety
                if hasattr(env, "env_traj_len"):
                    env_len = torch.clamp(env.env_traj_len, min=1)
                    phase = torch.clamp(phase, max=env_len - 1)

                ee_des = env.ee_paths_pad[env.traj_id, phase]
                if bool(getattr(env, "freeze_ee_des_z", False)) and hasattr(env, "ee_des_z_fixed"):
                    ee_des_n = ee_des.clone()
                    ee_des_n[:, 2] = env.ee_des_z_fixed
                    ee_des = ee_des_n
                ee_des_clean_sel = ee_des[ids, :]
        except Exception:
            ee_des_clean_sel = None

        # Desired EE (observed)
        ee_des_obs_sel = None
        try:
            if hasattr(env, "_lookahead_phase") and hasattr(env, "_get_ee_des_for_observation"):
                phase_la = env._lookahead_phase()
                ee_des_obs = env._get_ee_des_for_observation(phase_la)
                ee_des_obs_sel = ee_des_obs[ids, :]
        except Exception:
            ee_des_obs_sel = None

        # Errors
        e_true_clean_sel = None
        e_true_obs_sel = None
        e_meas_clean_sel = None
        if ee_des_clean_sel is not None:
            e_true_clean_sel = ee_des_clean_sel - ee_true_sel
            if ee_meas_sel is not None:
                e_meas_clean_sel = ee_des_clean_sel - ee_meas_sel
        if ee_des_obs_sel is not None:
            e_true_obs_sel = ee_des_obs_sel - ee_true_sel

        # Append
        self._ee_true.append([self._to_py3(ee_true_sel[k]) for k in range(len(ids))])

        if ee_speed_sel is not None:
            self._ee_speed.append([float(v) for v in ee_speed_sel])
        else:
            self._ee_speed.append([float("nan") for _ in ids])

        if ee_meas_sel is not None:
            self._ee_meas.append([self._to_py3(ee_meas_sel[k]) for k in range(len(ids))])
        else:
            self._ee_meas.append([[float("nan"), float("nan"), float("nan")] for _ in ids])

        if ee_des_clean_sel is not None:
            self._ee_des_clean.append([self._to_py3(ee_des_clean_sel[k]) for k in range(len(ids))])
        else:
            self._ee_des_clean.append([[float("nan"), float("nan"), float("nan")] for _ in ids])

        if ee_des_obs_sel is not None:
            self._ee_des_obs.append([self._to_py3(ee_des_obs_sel[k]) for k in range(len(ids))])
        else:
            self._ee_des_obs.append([[float("nan"), float("nan"), float("nan")] for _ in ids])

        if e_true_clean_sel is not None:
            self._e_true_clean.append([self._to_py3(e_true_clean_sel[k]) for k in range(len(ids))])
        else:
            self._e_true_clean.append([[float("nan"), float("nan"), float("nan")] for _ in ids])

        if e_true_obs_sel is not None:
            self._e_true_obs.append([self._to_py3(e_true_obs_sel[k]) for k in range(len(ids))])
        else:
            self._e_true_obs.append([[float("nan"), float("nan"), float("nan")] for _ in ids])

        if e_meas_clean_sel is not None:
            self._e_meas_clean.append([self._to_py3(e_meas_clean_sel[k]) for k in range(len(ids))])
        else:
            self._e_meas_clean.append([[float("nan"), float("nan"), float("nan")] for _ in ids])

        # Joint signals
        q_meas_sel = None
        qd_meas_sel = None
        q_ref_sel = None
        q_target_sel = None
        qd_target_sel = None

        try:
            if hasattr(env, "robot") and hasattr(env.robot, "data") and hasattr(env, "_joint_dof_idx"):
                jidx = env._joint_dof_idx
                q_all = env.robot.data.joint_pos[:, jidx]
                qd_all = env.robot.data.joint_vel[:, jidx]
                q_meas_sel = q_all[ids, :]
                qd_meas_sel = qd_all[ids, :]
        except Exception:
            q_meas_sel = None
            qd_meas_sel = None

        try:
            if hasattr(env, "q_paths_pad") and hasattr(env, "traj_id") and hasattr(env, "traj_phase"):
                phase = env.traj_phase.clone()
                if hasattr(env, "_lookahead_phase"):
                    phase = env._lookahead_phase()
                if hasattr(env, "env_traj_len"):
                    import torch
                    env_len = torch.clamp(env.env_traj_len, min=1)
                    phase = torch.clamp(phase, max=env_len - 1)
                q_ref = env.q_paths_pad[env.traj_id, phase]
                q_ref_sel = q_ref[ids, :]
        except Exception:
            q_ref_sel = None

        try:
            if hasattr(env, "q_target"):
                q_target_sel = env.q_target[ids, :]
            if hasattr(env, "qd_target"):
                qd_target_sel = env.qd_target[ids, :]
        except Exception:
            q_target_sel = None
            qd_target_sel = None

        if q_meas_sel is not None:
            self._q_meas.append([self._to_py_list_1d(q_meas_sel[k]) for k in range(len(ids))])
        else:
            self._q_meas.append([[float("nan")] for _ in ids])

        if qd_meas_sel is not None:
            self._qd_meas.append([self._to_py_list_1d(qd_meas_sel[k]) for k in range(len(ids))])
        else:
            self._qd_meas.append([[float("nan")] for _ in ids])

        if q_ref_sel is not None:
            self._q_ref.append([self._to_py_list_1d(q_ref_sel[k]) for k in range(len(ids))])
        else:
            self._q_ref.append([[float("nan")] for _ in ids])

        if q_target_sel is not None:
            self._q_target.append([self._to_py_list_1d(q_target_sel[k]) for k in range(len(ids))])
        else:
            self._q_target.append([[float("nan")] for _ in ids])

        if qd_target_sel is not None:
            self._qd_target.append([self._to_py_list_1d(qd_target_sel[k]) for k in range(len(ids))])
        else:
            self._qd_target.append([[float("nan")] for _ in ids])

    def save(self, out_path: str):
        import os
        from datetime import datetime
        from pathlib import Path

        np = self._np

        out = Path(out_path).expanduser()
        if out.suffix.lower() != ".npz":
            out.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            out = out / f"deploy_log_{stamp}.npz"
        else:
            out.parent.mkdir(parents=True, exist_ok=True)

        t = np.asarray(self._t, dtype=np.float32)
        traj_id = np.asarray(self._traj_id, dtype=np.int32)
        traj_phase = np.asarray(self._traj_phase, dtype=np.int32)

        ee_true_local = np.asarray(self._ee_true, dtype=np.float32)
        ee_meas_local = np.asarray(self._ee_meas, dtype=np.float32)
        ee_des_clean_local = np.asarray(self._ee_des_clean, dtype=np.float32)
        ee_des_obs_local = np.asarray(self._ee_des_obs, dtype=np.float32)

        ee_speed = np.asarray(self._ee_speed, dtype=np.float32)

        q_meas = np.asarray(self._q_meas, dtype=np.float32)
        qd_meas = np.asarray(self._qd_meas, dtype=np.float32)
        q_ref = np.asarray(self._q_ref, dtype=np.float32)
        q_target = np.asarray(self._q_target, dtype=np.float32)
        qd_target = np.asarray(self._qd_target, dtype=np.float32)

        e_true_clean_local = np.asarray(self._e_true_clean, dtype=np.float32)
        e_true_obs_local = np.asarray(self._e_true_obs, dtype=np.float32)
        e_meas_clean_local = np.asarray(self._e_meas_clean, dtype=np.float32)

        # Useful norms
        e_true_clean_norm = np.linalg.norm(e_true_clean_local, axis=-1)
        e_true_obs_norm = np.linalg.norm(e_true_obs_local, axis=-1)
        e_meas_clean_norm = np.linalg.norm(e_meas_clean_local, axis=-1)

        meta = {
            "dt": float(self.dt),
            "log_every": int(self.log_every),
            "env_ids": np.asarray(self.env_ids, dtype=np.int32),
            "cwd": os.getcwd(),
        }

        np.savez_compressed(
            str(out),
            t=t,
            traj_id=traj_id,
            traj_phase=traj_phase,
            ee_true_local=ee_true_local,
            ee_meas_local=ee_meas_local,
            ee_des_clean_local=ee_des_clean_local,
            ee_des_obs_local=ee_des_obs_local,
            ee_speed=ee_speed,
            q_meas=q_meas,
            qd_meas=qd_meas,
            q_ref=q_ref,
            q_target=q_target,
            qd_target=qd_target,
            e_true_clean_local=e_true_clean_local,
            e_true_obs_local=e_true_obs_local,
            e_meas_clean_local=e_meas_clean_local,
            e_true_clean_norm=e_true_clean_norm,
            e_true_obs_norm=e_true_obs_norm,
            e_meas_clean_norm=e_meas_clean_norm,
            **meta,
        )
        return str(out)

# ==============================================================================
# Trajectory Visualizer Class (from isaaclab_deploy_policy_npz.py)
# ==============================================================================
class _TrajectoryVisualizer:
    """Draw desired vs followed EE trajectories as USD curves in IsaacLab."""

    def __init__(self, env_unwrapped, env_id: int, max_points: int = 2000):
        # Delayed imports: require Kit to be running.
        import omni.usd  # type: ignore
        from pxr import Gf, UsdGeom  # type: ignore

        self._omni_usd = omni.usd
        self._UsdGeom = UsdGeom
        self._Gf = Gf

        self.env = env_unwrapped
        self.env_id = int(env_id)
        self.max_points = int(max_points)

        self._stage = omni.usd.get_context().get_stage()
        self._root_path = f"/World/DebugTraj/env_{self.env_id}"

        # Create a small namespace to keep things tidy.
        UsdGeom.Xform.Define(self._stage, self._root_path)

        self._curve_desired = self._create_curve(
            f"{self._root_path}/desired",
            color_rgb=(0.1, 0.8, 0.1),
            width=0.006,
        )
        self._curve_followed = self._create_curve(
            f"{self._root_path}/followed",
            color_rgb=(0.9, 0.2, 0.2),
            width=0.008,
        )

        self._last_traj_id: int | None = None
        self._followed_points_w: list[tuple[float, float, float]] = []

    def _create_curve(self, prim_path: str, color_rgb: tuple[float, float, float], width: float):
        UsdGeom = self._UsdGeom
        Gf = self._Gf

        curve = UsdGeom.BasisCurves.Define(self._stage, prim_path)
        curve.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
        curve.CreateCurveVertexCountsAttr().Set([0])
        curve.CreatePointsAttr().Set([])
        curve.CreateWidthsAttr().Set([float(width)])

        pv = curve.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant)
        pv.Set([Gf.Vec3f(*[float(c) for c in color_rgb])])
        return curve

    def _set_curve_points(self, curve, points_w: list[tuple[float, float, float]]):
        Gf = self._Gf
        pts = [Gf.Vec3f(float(x), float(y), float(z)) for (x, y, z) in points_w]
        curve.CreateCurveVertexCountsAttr().Set([len(pts)])
        curve.CreatePointsAttr().Set(pts)

    def _env_origin_w(self):
        origin = self.env.scene.env_origins[self.env_id, :3]
        return origin

    @staticmethod
    def _quat_to_rot_wxyz(q):
        """Quaternion to rotation matrix assuming q=(w,x,y,z)."""
        import torch

        w, x, y, z = q
        ww = w * w
        xx = x * x
        yy = y * y
        zz = z * z
        wx = w * x
        wy = w * y
        wz = w * z
        xy = x * y
        xz = x * z
        yz = y * z
        return torch.stack(
            [
                torch.stack([ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)]),
                torch.stack([2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)]),
                torch.stack([2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz]),
            ]
        )

    @staticmethod
    def _quat_to_rot_xyzw(q):
        """Quaternion to rotation matrix assuming q=(x,y,z,w)."""
        import torch

        x, y, z, w = q
        ww = w * w
        xx = x * x
        yy = y * y
        zz = z * z
        wx = w * x
        wy = w * y
        wz = w * z
        xy = x * y
        xz = x * z
        yz = y * z
        return torch.stack(
            [
                torch.stack([ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)]),
                torch.stack([2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)]),
                torch.stack([2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz]),
            ]
        )

    def _maybe_redraw_desired(self):
        traj_id = int(self.env.traj_id[self.env_id].item())
        if self._last_traj_id == traj_id:
            return

        self._last_traj_id = traj_id
        self._followed_points_w.clear()

        traj_len = int(self.env.env_traj_len[self.env_id].item())
        traj_len = max(1, traj_len)

        import torch

        origin = self._env_origin_w()

        desired_raw = self.env.ee_paths_pad[traj_id, :traj_len, :].clone()
        if bool(getattr(self.env, "freeze_ee_des_z", False)) and hasattr(self.env, "ee_des_z_fixed"):
            desired_raw[:, 2] = self.env.ee_des_z_fixed[self.env_id]

        d = self.env.robot.data
        if hasattr(d, "body_state_w"):
             ee_w_now_all = d.body_state_w[:, self.env._ee_body_id, 0:3]
        else:
             ee_w_now_all = d.body_pos_w[:, self.env._ee_body_id, 0:3]
        ee_w_now = ee_w_now_all[self.env_id]

        cand_env_w = desired_raw + origin
        cand_world_w = desired_raw

        cand_root_w = None
        cand_root_w_alt = None
        try:
            if hasattr(d, "root_state_w"):
                rs = d.root_state_w[self.env_id]
                root_pos_w = rs[0:3]
                root_quat = rs[3:7]
            elif hasattr(d, "root_pos_w") and hasattr(d, "root_quat_w"):
                root_pos_w = d.root_pos_w[self.env_id]
                root_quat = d.root_quat_w[self.env_id]
            else:
                root_pos_w = None
                root_quat = None

            if root_pos_w is not None and root_quat is not None:
                q = root_quat.to(dtype=torch.float32)
                R_wxyz = self._quat_to_rot_wxyz(q)
                R_xyzw = self._quat_to_rot_xyzw(q)
                cand_root_w = (desired_raw @ R_wxyz.T) + root_pos_w
                cand_root_w_alt = (desired_raw @ R_xyzw.T) + root_pos_w
        except Exception:
            pass

        def _dist_first(cand: torch.Tensor | None) -> float:
            if cand is None or cand.numel() == 0:
                return float("inf")
            return float(torch.norm(cand[0] - ee_w_now).item())

        candidates: list[tuple[str, torch.Tensor]] = [("env", cand_env_w), ("world", cand_world_w)]
        if cand_root_w is not None:
            candidates.append(("root_wxyz", cand_root_w))
        if cand_root_w_alt is not None:
            candidates.append(("root_xyzw", cand_root_w_alt))
        
        best_name, best = min(candidates, key=lambda kv: _dist_first(kv[1]))

        desired_w = best.detach().cpu().numpy()
        desired_pts = [(float(p[0]), float(p[1]), float(p[2])) for p in desired_w]
        self._set_curve_points(self._curve_desired, desired_pts)
        self._set_curve_points(self._curve_followed, [])

    def update(self):
        self._maybe_redraw_desired()

        d = self.env.robot.data
        if hasattr(d, "body_state_w"):
             ee_w_now_all = d.body_state_w[:, self.env._ee_body_id, 0:3]
        else:
             ee_w_now_all = d.body_pos_w[:, self.env._ee_body_id, 0:3]
        
        ee_w = ee_w_now_all[self.env_id].detach().cpu().numpy()
        p = (float(ee_w[0]), float(ee_w[1]), float(ee_w[2]))

        self._followed_points_w.append(p)
        if len(self._followed_points_w) > self.max_points:
            self._followed_points_w = self._followed_points_w[-self.max_points :]

        self._set_curve_points(self._curve_followed, self._followed_points_w)

def main():
    # --------------------------------------------------------------------------
    # 1. Parsing des arguments
    # --------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Suivi de trajectoire sans modèle (Open-Loop).")
    
    # Arguments spécifiques à la tâche
    parser.add_argument("--task", type=str, default="Template-Purement-Rl-Direct-v0", help="Nom de la tâche Gym.")
    parser.add_argument("--npz", type=str, required=True, help="Chemin vers le fichier de trajectoire (.npz).")
    parser.add_argument("--num_envs", type=int, default=1, help="Nombre d'environnements.")
    parser.add_argument("--steps", type=int, default=1000, help="Nombre de pas de simulation.")
    parser.add_argument("--viz", action="store_true", default=False, help="Activer la visualisation de la trajectoire.")
    parser.add_argument("--log-npz", type=str, default="deploy_log.npz", help="Fichier de log (.npz) pour plotter ensuite.")
    parser.add_argument("--log-every", type=int, default=1, help="Logger tous les N steps.")
    parser.add_argument("--log-envs", type=str, default="0", help="Env indices to log (e.g. '0,1').")

    # Arguments IsaacLab
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    # --------------------------------------------------------------------------
    # 2. Lancement de l'application Isaac Sim
    # --------------------------------------------------------------------------
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Imports après le lancement de l'app (nécessaire pour Isaac Sim)
    import gymnasium as gym
    import torch
    import Purement_Rl.tasks  # Enregistrement des environnements
    from Purement_Rl.tasks.direct.purement_rl.purement_rl_env_cfg import PurementRlEnvCfg
    
    # --------------------------------------------------------------------------
    # 3. Configuration de l'environnement
    # --------------------------------------------------------------------------
    env_cfg = PurementRlEnvCfg()
    
    # Configuration de base
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.traj_path = str(Path(args_cli.npz).expanduser().resolve())
    
    # Pour le suivi strict de la trajectoire (sans lookahead)
    env_cfg.lookahead_steps = 0 
    
    # --------------------------------------------------------------------------
    # Forcer une seule trajectoire répétée en boucle
    # --------------------------------------------------------------------------
    # "resample_traj_id_on_reset = False" signifie qu'au reset, on garde le même ID.
    if hasattr(env_cfg, "resample_traj_id_on_reset"):
        env_cfg.resample_traj_id_on_reset = False
    
    # "traj_end_mode = 'reset'" signifie qu'à la fin de la trajectoire, l'épisode reset
    # (et comme resample=False, ça repart sur la même).
    env_cfg.traj_end_mode = "reset"

    # Activer le bruit sur la trajectoire pour un test réaliste
    if hasattr(env_cfg, "use_noisy_traj_ref"):
        env_cfg.use_noisy_traj_ref = True
    if hasattr(env_cfg, "use_traj_ref_filter"):
        env_cfg.use_traj_ref_filter = True
    
    # Augmenter le domain randomization pour créer des erreurs en open-loop
    if hasattr(env_cfg, "randomize_damping"):
        env_cfg.randomize_damping = True
        env_cfg.damping_scale_range = (0.5, 1.5)  # ±50% variation
    
    if hasattr(env_cfg, "randomize_payload"):
        env_cfg.randomize_payload = True
        env_cfg.payload_mass_range_kg = (0.0, 2.0)  # 0-2 kg charge aléatoire (doublé)
    
    if hasattr(env_cfg, "action_delay_min") and hasattr(env_cfg, "action_delay_max"):
        env_cfg.action_delay_min = 3
        env_cfg.action_delay_max = 6  # Délai action encore augmenté
    
    # Réduire les gains PD pour rendre le contrôle moins parfait en open-loop
    if hasattr(env_cfg, "robot") and hasattr(env_cfg.robot, "actuators"):
        for actuator_name, actuator_cfg in env_cfg.robot.actuators.items():
            if hasattr(actuator_cfg, "stiffness"):
                actuator_cfg.stiffness = 5000.0  # Réduit de 10000 à 5000
            if hasattr(actuator_cfg, "damping"):
                actuator_cfg.damping = 500.0  # Réduit de 1000 à 500
        
    # S'assurer que le device est correct
    if getattr(args_cli, "device", None) is not None:
        env_cfg.sim.device = args_cli.device

    # --------------------------------------------------------------------------
    # 4. Création de l'environnement
    # --------------------------------------------------------------------------
    print(f"[INFO] Création de l'environnement : {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # Reset initial
    obs, _ = env.reset()
    
    # Récupération des infos utiles
    # gym.make returns a wrapper, likely need unwrapped for device/num_envs
    # But standard gym env might not have device attribute directly on top level
    
    unwrapped_env = env.unwrapped
    device = getattr(unwrapped_env, "device", torch.device("cpu"))
    num_envs = getattr(unwrapped_env, "num_envs", args_cli.num_envs)
    
    # Détermination de la dimension d'action
    if hasattr(unwrapped_env, "action_dim"):
        action_dim = unwrapped_env.action_dim
    else:
        # Fallback si action_dim n'est pas directement accessible, try action_space
        action_space = env.action_space
        if hasattr(action_space, "shape"):
            action_dim = action_space.shape[0]
        else:
             action_dim = 6 # Default fallback
        
    print(f"[INFO] Démarrage de la simulation pour {args_cli.steps} steps.")
    print(f"[INFO] Mode Open-Loop: Actions = 0 (Suivi de référence pure).")

    # --------------------------------------------------------------------------
    # 5. Initialisation du logger
    # --------------------------------------------------------------------------
    logger = None
    if args_cli.log_npz is not None:
        try:
            env_ids = [int(s) for s in str(args_cli.log_envs).split(",") if str(s).strip() != ""]
            if not env_ids:
                env_ids = [0]
            # dt : best effort
            dt = getattr(unwrapped_env, "step_dt", 0.02)
            logger = _DeployLogger(unwrapped_env, env_ids=env_ids, dt=dt, log_every=args_cli.log_every)
            print(f"[INFO] Logging enabled: {args_cli.log_npz} (envs={env_ids})")
        except Exception as e:
            print(f"[WARN] Failed to init logger: {e}")

    # --------------------------------------------------------------------------
    # 6. Initialisation du visualiseur de trajectoire
    # --------------------------------------------------------------------------
    viz = None
    if args_cli.viz:
        try:
            viz = _TrajectoryVisualizer(unwrapped_env, env_id=0, max_points=2000)
            print("[INFO] Trajectory visualization enabled (green=desired, red=followed)")
        except Exception as exc:
            print(f"[WARN] Trajectory visualization disabled (failed to init USD curves): {exc}")
            viz = None

    # --------------------------------------------------------------------------
    # 7. Boucle de simulation
    # --------------------------------------------------------------------------
    try:
        step = 0
        while simulation_app.is_running():
            if step >= args_cli.steps:
                break
            
            # Action Nulle : Le robot suit la trajectoire de référence (q_target = q_ref + 0)
            actions = torch.zeros((num_envs, action_dim), device=device)
            
            # Step
            obs, rew, terminated, truncated, info = env.step(actions)
            
            # Update logger
            if logger is not None:
                logger.update(step)

            # Update trajectory visualization
            if viz is not None:
                try:
                    viz.update()
                except Exception as exc:
                    print(f"[WARN] Trajectory visualization update failed: {exc}")
                    viz = None

            if step % 100 == 0:
                print(f"[Step {step}] Running...")

            # Gestion des resets automatiques faite par l'environnement
            
            step += 1
            
    except KeyboardInterrupt:
        print("[INFO] Interruption utilisateur.")
    finally:
        if logger is not None and args_cli.log_npz:
            saved_path = logger.save(args_cli.log_npz)
            print(f"[INFO] Log saved to: {saved_path}")
            print(f"[INFO] To plot, run: python plot_deploy_log.py --npz {saved_path} --show")

        env.close()
        simulation_app.close()
        print("[INFO] Simulation terminée.")

if __name__ == "__main__":
    main()
