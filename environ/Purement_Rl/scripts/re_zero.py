#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import torch

from isaaclab.app import AppLauncher

# ------------------------------------------------------------
# Args + Isaac Lab app
# ------------------------------------------------------------
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)

parser.add_argument("--usd", type=str, required=True)
parser.add_argument("--npz", type=str, required=True)

parser.add_argument("--robot_prim", type=str, default="/Root/ur10/root_joint")  # <= IMPORTANT
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--dt", type=float, default=0.01)
parser.add_argument("--hold_steps", type=int, default=2)
parser.add_argument("--print_every", type=int, default=20)
parser.add_argument("--episode_len", type=int, default=0)
parser.add_argument("--loop", action="store_true")

parser.add_argument("--stiffness", type=float, default=800.0)
parser.add_argument("--damping", type=float, default=80.0)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ------------------------------------------------------------
# Imports Isaac / USD
# ------------------------------------------------------------
import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg


UR10_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def load_dataset(npz_path: str):
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"❌ NPZ introuvable: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    q_paths = data["paths"]
    ee_pos = data["ee_pos"]
    return q_paths, ee_pos


def make_robot(robot_prim: str) -> Articulation:
    cfg = ArticulationCfg(
        prim_path=robot_prim,
        actuators={
            "ur10": ImplicitActuatorCfg(
                joint_names_expr=UR10_JOINTS,
                stiffness=args_cli.stiffness,
                damping=args_cli.damping,
            )
        },
    )
    return Articulation(cfg)


def dump_robot_api(robot: Articulation):
    # On affiche quelques attributs “probables” selon versions
    print("\n[DEBUG] --- Robot API quick dump ---")
    for name in ["num_joints", "num_dof", "joint_names", "is_initialized"]:
        if hasattr(robot, name):
            try:
                v = getattr(robot, name)
                v = v() if callable(v) else v
                print(f"[DEBUG] {name}: {v}")
            except Exception as e:
                print(f"[DEBUG] {name}: <error {e}>")
        else:
            print(f"[DEBUG] {name}: <missing>")

    # Liste des méthodes utiles présentes
    useful = [
        "set_joint_position_target",
        "set_joint_velocity_target",
        "set_joint_effort_target",
        "write_data_to_sim",
        "write_joint_state_to_sim",
        "update",
        "reset",
        "get_joint_positions",
        "get_joint_velocities",
        "get_joint_efforts",
    ]
    present = [m for m in useful if hasattr(robot, m)]
    print("[DEBUG] methods present:", present)

    # data / _data
    print("[DEBUG] has robot.data:", hasattr(robot, "data"))
    print("[DEBUG] has robot._data:", hasattr(robot, "_data"))
    print("[DEBUG] ----------------------------\n")


def read_joint_pos(robot: Articulation):
    """
    Fallback multi-versions pour lire q.
    Retourne np.ndarray (6,) ou None si introuvable.
    """
    # 1) robot.data.joint_pos
    try:
        if hasattr(robot, "data") and hasattr(robot.data, "joint_pos"):
            q = robot.data.joint_pos
            # q peut être (num_envs, dof)
            q0 = q[0] if hasattr(q, "__getitem__") else q
            if torch.is_tensor(q0):
                return q0.detach().cpu().numpy()
            return np.array(q0)
    except Exception:
        pass

    # 2) robot._data.joint_pos
    try:
        if hasattr(robot, "_data") and hasattr(robot._data, "joint_pos"):
            q = robot._data.joint_pos
            q0 = q[0] if hasattr(q, "__getitem__") else q
            if torch.is_tensor(q0):
                return q0.detach().cpu().numpy()
            return np.array(q0)
    except Exception:
        pass

    # 3) méthodes get_joint_positions()
    try:
        if hasattr(robot, "get_joint_positions"):
            q = robot.get_joint_positions()
            if torch.is_tensor(q):
                q0 = q[0] if q.ndim == 2 else q
                return q0.detach().cpu().numpy()
            return np.array(q)
    except Exception:
        pass

    return None


def main():
    rng = np.random.default_rng(args_cli.seed)

    print(f"[INFO] Loading USD: {args_cli.usd}")
    omni.usd.get_context().open_stage(args_cli.usd)

    sim_cfg = sim_utils.SimulationCfg(dt=args_cli.dt, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    # IMPORTANT: play + reset pour que la physique s'exécute
    try:
        sim.play()
    except Exception:
        pass
    sim.reset()

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(args_cli.robot_prim)
    if not prim.IsValid():
        raise RuntimeError(f"❌ robot_prim invalide: {args_cli.robot_prim}")

    q_paths, ee_pos = load_dataset(args_cli.npz)
    print(f"[INFO] Dataset OK: N={len(q_paths)} trajectoires (variable T_i)")

    print(f"[INFO] Binding robot at prim: {args_cli.robot_prim}")
    robot = make_robot(args_cli.robot_prim)

    dump_robot_api(robot)

    # --- run episodes ---
    print("[INFO] Running. Ctrl+C pour arrêter.")

    try:
        while simulation_app.is_running():
            traj_id = int(rng.integers(0, len(q_paths)))
            q_traj = q_paths[traj_id].astype(np.float32)[:, :6]
            T_i = q_traj.shape[0]
            episode_len = args_cli.episode_len if args_cli.episode_len > 0 else T_i
            episode_len = min(episode_len, T_i)

            print(f"\n[EPISODE] traj_id={traj_id} | T_i={T_i} | episode_len={episode_len} | hold_steps={args_cli.hold_steps}")

            sim.reset()
            try:
                robot.reset()
            except Exception:
                pass

            # init target
            if not hasattr(robot, "set_joint_position_target"):
                raise RuntimeError("❌ set_joint_position_target() absent => API trop différente.")

            device = sim.device
            step_count = 0

            for t in range(episode_len):
                q_des = torch.tensor(q_traj[t], device=device).unsqueeze(0)
                robot.set_joint_position_target(q_des)

                for _ in range(args_cli.hold_steps):
                    if hasattr(robot, "write_data_to_sim"):
                        robot.write_data_to_sim()

                    sim.step()

                    if hasattr(robot, "update"):
                        robot.update(sim.dt)

                    step_count += 1
                    if args_cli.print_every > 0 and (step_count % args_cli.print_every == 0):
                        q_now = read_joint_pos(robot)
                        if q_now is None:
                            # au moins on print ce qu'on envoie
                            print(f"step={step_count:06d} | t={t:04d} | SENT q_des={np.round(q_traj[t], 3).tolist()} | q_now=<unavailable>")
                        else:
                            print(f"step={step_count:06d} | t={t:04d} | q_now={np.round(q_now, 3).tolist()}")

                    if not simulation_app.is_running():
                        break

                if not simulation_app.is_running():
                    break

            if not args_cli.loop:
                break

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
