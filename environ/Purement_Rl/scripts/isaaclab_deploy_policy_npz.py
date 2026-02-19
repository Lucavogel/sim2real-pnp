from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

from isaaclab.app import AppLauncher


class _DeployLogger:
    """Log deployment rollout signals to a NPZ file (post-deployment plotting).

    Logs are stored in env-local coordinates (consistent with the task env):
    - ee_true_local: true EE position
    - ee_des_clean_local: desired EE from dataset (with optional Z-freeze)
    - ee_des_obs_local: desired EE as observed by the agent (may include noise)
    - ee_meas_local: camera-measured EE (if enabled)
    - e_true_clean_local / e_true_obs_local / e_meas_clean_local
    Also stores traj_id, traj_phase, and time.
    """

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
        # x is shape (D,)
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

        # EE speed (best-effort)
        ee_speed_sel = None
        if hasattr(env, "ee_speed"):
            try:
                ee_speed_sel = env.ee_speed[ids].detach().cpu().numpy().tolist()
            except Exception:
                ee_speed_sel = None

        # Measured EE (local) if available
        ee_meas_sel = None
        if hasattr(env, "_get_ee_meas_local"):
            try:
                ee_meas = env._get_ee_meas_local()
                ee_meas_sel = ee_meas[ids, :]
            except Exception:
                ee_meas_sel = None

        # Desired EE (clean dataset target, local)
        ee_des_clean_sel = None
        try:
            if hasattr(env, "ee_paths_pad") and hasattr(env, "traj_id") and hasattr(env, "traj_phase"):
                phase = env.traj_phase.clone()
                # Match env behavior: use lookahead phase when available.
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

        # Desired EE as seen by agent (may include observation noise)
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

        # Append (as python floats)
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

        # Joint signals (best-effort): measured q/qd + desired q_ref + command targets q_target/qd_target
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
        # One curve with N vertices.
        curve.CreateCurveVertexCountsAttr().Set([0])
        curve.CreatePointsAttr().Set([])
        # A single width value => constant width.
        curve.CreateWidthsAttr().Set([float(width)])

        # Constant displayColor.
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

        # Desired points stored in the env dataset buffer.
        desired_raw = self.env.ee_paths_pad[traj_id, :traj_len, :].clone()  # (T, 3)
        if bool(getattr(self.env, "freeze_ee_des_z", False)) and hasattr(self.env, "ee_des_z_fixed"):
            desired_raw[:, 2] = self.env.ee_des_z_fixed[self.env_id]

        # Current EE (world) is our reference for frame selection.
        # FIX: Get EE world position directly from robot data, do not rely on local transform + origin which is often wrong.
        d = self.env.robot.data
        if hasattr(d, "body_state_w"):
             ee_w_now_all = d.body_state_w[:, self.env._ee_body_id, 0:3]
        else:
             ee_w_now_all = d.body_pos_w[:, self.env._ee_body_id, 0:3]
        ee_w_now = ee_w_now_all[self.env_id]

        # Candidate 1: dataset is in env-local frame (most common in this env).
        cand_env_w = desired_raw + origin

        # Candidate 2: dataset is already in world frame.
        cand_world_w = desired_raw

        # Candidate 3/4: dataset is in robot-root frame (needs root pose transform).
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
                # Try both quaternion conventions; pick the one that best matches the current EE.
                q = root_quat.to(dtype=torch.float32)
                R_wxyz = self._quat_to_rot_wxyz(q)
                R_xyzw = self._quat_to_rot_xyzw(q)
                cand_root_w = (desired_raw @ R_wxyz.T) + root_pos_w
                cand_root_w_alt = (desired_raw @ R_xyzw.T) + root_pos_w
        except Exception:
            pass

        # Select the frame that makes the first waypoint closest to the current EE.
        def _dist_first(cand: torch.Tensor | None) -> float:
            if cand is None or cand.numel() == 0:
                return float("inf")
            # Compare first point of trajectory with current EE
            return float(torch.norm(cand[0] - ee_w_now).item())

        candidates: list[tuple[str, torch.Tensor]] = [("env", cand_env_w), ("world", cand_world_w)]
        if cand_root_w is not None:
            candidates.append(("root_wxyz", cand_root_w))
        if cand_root_w_alt is not None:
            candidates.append(("root_xyzw", cand_root_w_alt))
        
        # Force candidate "root_wxyz" priority if the robot frame logic is standard.
        # But keeping the min() logic is safer if everything is correct.
        best_name, best = min(candidates, key=lambda kv: _dist_first(kv[1]))
        
        # print(f"[DEBUG] Trajectory Visualizer selected frame: {best_name} (dist={_dist_first(best):.4f})")

        desired_w = best.detach().cpu().numpy()
        desired_pts = [(float(p[0]), float(p[1]), float(p[2])) for p in desired_w]
        self._set_curve_points(self._curve_desired, desired_pts)
        self._set_curve_points(self._curve_followed, [])

    def update(self):
        # Redraw desired if the env switched trajectories.
        self._maybe_redraw_desired()

        # Followed path: use true WORLD position from robot data
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


def _load_policy(policy_path: str, device: str | torch.device):
    """Load a TorchScript policy (preferred) or a plain torch module."""
    # 1) TorchScript
    try:
        module = torch.jit.load(policy_path, map_location=device)
        module.eval()
        return module
    except Exception:
        pass

    # 2) Eager module / state dict (best-effort)
    obj = torch.load(policy_path, map_location=device)
    if isinstance(obj, torch.nn.Module):
        obj.eval()
        return obj

    raise RuntimeError(
        "Unsupported policy format. Provide a TorchScript policy (policy.pt) exported from RSL-RL play script."\
        f"\nGot: {policy_path}"\
        f"\nType loaded: {type(obj)}"
    )


def _extract_obs(obs):
    """Handle either dict observations (DirectRLEnv) or tensor (wrapped)."""
    if isinstance(obs, dict):
        if "policy" not in obs:
            raise KeyError(f"Observation dict missing 'policy'. Keys: {list(obs.keys())}")
        return obs["policy"]
    return obs


def main():
    parser = argparse.ArgumentParser(
        description="Deploy Purement_Rl inside IsaacLab using the task env (no RTDE)."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Template-Purement-Rl-Direct-v0",
        help="Gym task id (default: Template-Purement-Rl-Direct-v0).",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Path to exported TorchScript policy (policy.pt). Required unless --no-policy is set.",
    )
    parser.add_argument(
        "--no-policy",
        action="store_true",
        default=False,
        help="Run without a policy (open-loop): send zero residual actions => track dataset q_ref.",
    )
    parser.add_argument("--npz", type=str, required=True, help="Path to mission dataset (.npz).")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of envs (default: 1).")
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Max steps to run (0 = run while simulator is running).",
    )
    parser.add_argument("--real-time", action="store_true", default=False, help="Sleep to match env.step_dt.")

    # Logging (post-deployment plotting)
    parser.add_argument(
        "--log-npz",
        type=str,
        default=None,
        help="If set, save a rollout log as .npz (path to file.npz or output directory).",
    )
    parser.add_argument(
        "--log-envs",
        type=str,
        default="0",
        help="Comma-separated env indices to log (default: '0').",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Log every N env steps (default: 1).",
    )

    # Debug
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Print rollout diagnostics (actions, dones, traj phase).",
    )
    parser.add_argument(
        "--debug-every",
        type=int,
        default=50,
        help="Debug print interval in env steps (default: 50).",
    )

    # Visualization (in-sim)
    parser.add_argument(
        "--viz-traj",
        action="store_true",
        default=False,
        help="Draw desired (green) and followed (red) EE trajectories in IsaacLab.",
    )
    parser.add_argument(
        "--viz-env",
        type=int,
        default=0,
        help="Which environment index to visualize (default: 0).",
    )
    parser.add_argument(
        "--viz-max-points",
        type=int,
        default=2000,
        help="Max points kept for the followed trajectory curve (default: 2000).",
    )

    # Optional overrides to match your RTDE script behavior.
    parser.add_argument("--lookahead", type=int, default=None, help="Override cfg.lookahead_steps")
    parser.add_argument("--action-scale", type=float, default=None, help="Override cfg.action_scale")
    parser.add_argument(
        "--traj-end-mode",
        type=str,
        default=None,
        choices=["hold", "loop", "reset"],
        help="Override cfg.traj_end_mode",
    )
    parser.add_argument(
        "--no-traj-noise",
        action="store_true",
        default=False,
        help="Disable dataset ref noise/filter (use_noisy_traj_ref=False, use_traj_ref_filter=False).",
    )

    # ------------------------------------------------------------
    # Domain randomization (sim2real noise model)
    # ------------------------------------------------------------
    parser.add_argument(
        "--no-domain-rand",
        action="store_true",
        default=False,
        help="Disable domain randomization / noise model for testing.",
    )
    parser.add_argument(
        "--dr-print",
        action="store_true",
        default=False,
        help="Print sampled DR parameters for env0 (extrinsic bias, damping, payload).",
    )

    # IsaacLab / Kit launcher args
    AppLauncher.add_app_launcher_args(parser)

    args_cli, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args

    if (not args_cli.no_policy) and (args_cli.policy is None):
        parser.error("--policy is required unless --no-policy is set")

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Delayed imports (after Kit is up)
    import gymnasium as gym  # noqa: E402

    import Purement_Rl.tasks  # noqa: F401, E402 (register envs)
    from Purement_Rl.tasks.direct.purement_rl.purement_rl_env_cfg import PurementRlEnvCfg  # noqa: E402

    # Build env cfg
    env_cfg = PurementRlEnvCfg()

    # Deployment default: do not loop trajectories.
    # When the trajectory ends, reset the episode and teleport back to the start pose.
    env_cfg.traj_end_mode = "reset"
    if hasattr(env_cfg, "resample_traj_id_on_reset"):
        env_cfg.resample_traj_id_on_reset = False

    # To start exactly at the first waypoint, default to no lookahead in deployment.
    # (You can still pass --lookahead to override.)
    env_cfg.lookahead_steps = 0
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.traj_path = str(Path(args_cli.npz).expanduser().resolve())

    # Deployment sanity: disable strict Z-constraint terminations.
    # These are useful during training to enforce planar motion, but during playback they can
    # cause immediate resets every step (looks like "no movement").
    if hasattr(env_cfg, "terminate_on_ee_speed_z_over"):
        env_cfg.terminate_on_ee_speed_z_over = False
    if hasattr(env_cfg, "terminate_on_ee_z_error_over"):
        env_cfg.terminate_on_ee_z_error_over = False

    # NOTE: action delay is part of the sim2real domain randomization (1..3 simulation steps).
    # Keep the task defaults here so playback matches training.

    if args_cli.lookahead is not None:
        env_cfg.lookahead_steps = int(args_cli.lookahead)
    if args_cli.action_scale is not None:
        env_cfg.action_scale = float(args_cli.action_scale)
    if args_cli.traj_end_mode is not None:
        env_cfg.traj_end_mode = str(args_cli.traj_end_mode)

    if args_cli.no_traj_noise:
        env_cfg.use_noisy_traj_ref = False
        env_cfg.use_traj_ref_filter = False

    # Domain randomization toggle (testing)
    if args_cli.no_domain_rand:
        # Visual target jitter
        if hasattr(env_cfg, "ee_des_noise_std_m"):
            env_cfg.ee_des_noise_std_m = 0.0

        # Camera extrinsic bias
        if hasattr(env_cfg, "cam_extrinsic_bias_max_m"):
            env_cfg.cam_extrinsic_bias_max_m = 0.0
        if hasattr(env_cfg, "resample_cam_extrinsic_bias_each_episode"):
            env_cfg.resample_cam_extrinsic_bias_each_episode = False

        # Physics (damping + payload)
        if hasattr(env_cfg, "randomize_damping"):
            env_cfg.randomize_damping = False
        if hasattr(env_cfg, "randomize_payload"):
            env_cfg.randomize_payload = False

        # Action latency
        if hasattr(env_cfg, "action_delay_min"):
            env_cfg.action_delay_min = 1
        if hasattr(env_cfg, "action_delay_max"):
            env_cfg.action_delay_max = 1

    # Device override (AppLauncher provides --device; env_cfg expects sim.device)
    if getattr(args_cli, "device", None) is not None:
        env_cfg.sim.device = args_cli.device

    # Create env
    env = gym.make(args_cli.task, cfg=env_cfg)

    def _print_dr_env0(prefix: str = "[DR]"):
        if not args_cli.dr_print:
            return
        uw = env.unwrapped if hasattr(env, "unwrapped") else None
        if uw is None:
            return
        try:
            # Best-effort: only print what exists.
            parts = []
            if hasattr(uw, "cam_extrinsic_bias"):
                b = uw.cam_extrinsic_bias[0].detach().cpu().numpy().tolist()
                parts.append(f"cam_bias_m={b}")
            if hasattr(uw, "damping_scale"):
                ds = float(uw.damping_scale[0].item())
                parts.append(f"damping_scale={ds:.3f}")
            if hasattr(uw, "payload_mass"):
                pm = float(uw.payload_mass[0].item())
                parts.append(f"payload_mass_kg={pm:.3f}")
            if hasattr(uw, "action_delay_min") and hasattr(uw, "action_delay_max"):
                parts.append(f"action_delay_steps=[{int(uw.action_delay_min)},{int(uw.action_delay_max)}]")
            if parts:
                print(prefix + " env0 | " + " | ".join(parts), flush=True)
        except Exception:
            return

    viz = None
    if args_cli.viz_traj:
        try:
            viz = _TrajectoryVisualizer(env.unwrapped, env_id=args_cli.viz_env, max_points=args_cli.viz_max_points)
        except Exception as exc:
            print(f"[WARN] Trajectory visualization disabled (failed to init USD curves): {exc}")
            viz = None

    # Load policy (optional)
    device = env.unwrapped.device if hasattr(env, "unwrapped") else torch.device("cpu")
    policy = None
    if not args_cli.no_policy:
        policy = _load_policy(str(Path(args_cli.policy).expanduser().resolve()), device)

    # Reset
    obs, _ = env.reset()
    obs_t = _extract_obs(obs)
    _print_dr_env0(prefix="[DR] after reset")

    dt = float(getattr(env.unwrapped, "step_dt", 1.0 / 50.0))

    logger = None
    if args_cli.log_npz is not None:
        try:
            env_ids = [int(s) for s in str(args_cli.log_envs).split(",") if str(s).strip() != ""]
            if not env_ids:
                env_ids = [0]
            logger = _DeployLogger(env.unwrapped, env_ids=env_ids, dt=dt, log_every=int(args_cli.log_every))
            print(
                f"[INFO] Deployment logging enabled: env_ids={env_ids} every={int(args_cli.log_every)} step(s)",
                flush=True,
            )
        except Exception as exc:
            print(f"[WARN] Failed to init logger; disabling --log-npz. Reason: {exc}", flush=True)
            logger = None

    step = 0
    try:
        while simulation_app.is_running():
            if args_cli.steps and step >= args_cli.steps:
                break

            start_time = time.time()

            if args_cli.no_policy:
                # Open-loop: env uses actions as residual around q_ref (dataset).
                # Zero residual => track q_ref directly.
                uw = env.unwrapped if hasattr(env, "unwrapped") else None
                if uw is not None and hasattr(uw, "actions_raw") and torch.is_tensor(uw.actions_raw):
                    actions = torch.zeros_like(uw.actions_raw)
                else:
                    num_envs = int(getattr(getattr(uw, "scene", None), "num_envs", int(args_cli.num_envs)))
                    action_space = getattr(env, "action_space", None)
                    action_dim = int(getattr(uw, "action_dim", getattr(action_space, "shape", (6,))[0]))
                    actions = torch.zeros((num_envs, action_dim), device=device, dtype=torch.float32)
            else:
                with torch.inference_mode():
                    actions = policy(obs_t)

            # Ensure (num_envs, action_dim)
            if actions.ndim == 1:
                actions = actions.unsqueeze(0)

            actions = torch.clamp(actions, -1.0, 1.0)

            if args_cli.debug and (args_cli.debug_every > 0) and (step % int(args_cli.debug_every) == 0):
                try:
                    a_mean = float(torch.mean(torch.abs(actions)).item())
                    a_max = float(torch.max(torch.abs(actions)).item())
                    print(f"[DEBUG] step={step} | mean|a|={a_mean:.4f} | max|a|={a_max:.4f}", flush=True)
                    uw = env.unwrapped if hasattr(env, "unwrapped") else None
                    if uw is not None and hasattr(uw, "traj_phase"):
                        tp0 = int(uw.traj_phase[0].item())
                        tid0 = int(uw.traj_id[0].item()) if hasattr(uw, "traj_id") else -1
                        print(f"[DEBUG] traj_id0={tid0} traj_phase0={tp0}", flush=True)
                except Exception as exc:
                    print(f"[DEBUG] failed to print action stats: {exc}", flush=True)

            obs, _, terminated, truncated, info = env.step(actions)
            obs_t = _extract_obs(obs)

            # If the env auto-resets internally, the sampled params may change; print periodically.
            if args_cli.dr_print and (args_cli.debug_every > 0) and (step % int(args_cli.debug_every) == 0):
                _print_dr_env0(prefix="[DR]")

            if args_cli.debug and (args_cli.debug_every > 0) and (step % int(args_cli.debug_every) == 0):
                try:
                    # terminated/truncated can be bool or array-like
                    def _count_true(x):
                        if torch.is_tensor(x):
                            return int(torch.sum(x).item())
                        if isinstance(x, (list, tuple)):
                            return int(sum(bool(v) for v in x))
                        return int(bool(x))

                    t_n = _count_true(terminated)
                    tr_n = _count_true(truncated)
                    print(f"[DEBUG] terminated={t_n} truncated={tr_n}", flush=True)
                    if isinstance(info, dict) and "log" in info:
                        # IsaacLab extras often appear under info['log']
                        log = info.get("log", {})
                        if isinstance(log, dict):
                            ee_speed = log.get("ee_speed_mean", None)
                            if ee_speed is not None:
                                try:
                                    ee_speed_v = float(ee_speed)
                                    print(f"[DEBUG] ee_speed_mean={ee_speed_v:.6f}", flush=True)
                                except Exception:
                                    pass
                except Exception as exc:
                    print(f"[DEBUG] failed to print dones/info: {exc}", flush=True)

            if viz is not None:
                try:
                    viz.update()
                except Exception as exc:
                    # Best-effort: don't kill rollout if viz fails.
                    print(f"[WARN] Trajectory visualization update failed: {exc}")
                    viz = None

            if logger is not None:
                try:
                    logger.update(step)
                except Exception as exc:
                    print(f"[WARN] Logger update failed: {exc}", flush=True)
                    logger = None

            step += 1

            if args_cli.real_time:
                sleep_time = dt - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    finally:
        if logger is not None and args_cli.log_npz is not None:
            try:
                saved = logger.save(args_cli.log_npz)
                print(f"[INFO] Saved deployment log: {saved}", flush=True)
            except Exception as exc:
                print(f"[WARN] Failed to save deployment log: {exc}", flush=True)
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
