#!/usr/bin/env python3
"""
Script IsaacLab simplifié pour exécuter des trajectoires pré-calculées
SANS ROS Bridge - Supporte dataset.npz MoveIt V2 (clé "paths") + fallback legacy (traj_000...)

Usage:
  ./isaaclab.sh -p Ur10_trajectory_executor.py --dataset dataset.npz --num_envs 1

Notes:
- Simulation dt=0.01 (100 Hz physics)
- Dataset = 25 Hz (dt=0.04) => on tient chaque waypoint 4 steps à 100 Hz
"""

import argparse
from pathlib import Path
import os

from isaaclab.app import AppLauncher

# Argparse
parser = argparse.ArgumentParser(description="UR10 avec trajectoires pré-calculées (dataset.npz)")
parser.add_argument("--num_envs", type=int, default=1, help="Nombre d'environnements")
parser.add_argument("--dataset", type=str, default=None,
                    help="Chemin vers dataset.npz (défaut: dataset.npz dans le dossier du script)")
parser.add_argument("--apply_delta", action="store_true",
                    help="Appliquer corrections delta aléatoires (LEGACY)")
parser.add_argument("--delta_std", type=float, default=0.0,
                    help="Écart-type perturbation delta (radians)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Si dataset n'est pas spécifié, chercher dataset.npz dans le dossier du script
if args_cli.dataset is None:
    script_dir = Path(__file__).parent
    args_cli.dataset = str(script_dir / "dataset.npz")
    print(f"[INFO] Chemin dataset: {args_cli.dataset}")
else:
    print(f"[INFO] Utilisation dataset: {args_cli.dataset}")

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports après AppLauncher
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from ur10_ros2_cfg import UR10_ROS2_CFG


def main():
    # ------------------ SIM SETUP ------------------
    sim_dt = 0.01  # 100 Hz physics
    sim_cfg = sim_utils.SimulationCfg(
        dt=sim_dt,
        device=args_cli.device,
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.5])

    # Ground plane
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/Ground", ground_cfg)

    # Robot
    print("\n🤖 Chargement UR10 (USD converti URDF)")
    ur10_cfg = UR10_ROS2_CFG.replace(prim_path="/World/UR10")
    ur10 = Articulation(cfg=ur10_cfg)

    sim.reset()
    print("[INFO] Simulation initialized")
    print(f"[INFO] Robot device: {ur10.device}")

    # ------------------ LOAD DATASET ------------------
    print("\n" + "=" * 70)
    print("📁 CHARGEMENT DES TRAJECTOIRES (dataset.npz)")
    print("=" * 70)

    dataset_path = args_cli.dataset
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset non trouvé: {dataset_path}")

    print(f"📂 Chargement: {dataset_path}")
    dataset = np.load(dataset_path, allow_pickle=True)

    trajectories = []

    # ✅ Format MoveIt V2: clé "paths"
    if "paths" in dataset:
        paths = dataset["paths"]

        # Si jamais dtype=object (N,) avec chaque elem (T,6) => densifie
        if isinstance(paths, np.ndarray) and paths.dtype == object:
            paths = np.stack(list(paths), axis=0)

        # Attendu (N,T,6)
        if paths.ndim != 3 or paths.shape[-1] != 6:
            raise ValueError(f"Format paths invalide: shape={paths.shape} (attendu N,T,6)")

        for i in range(paths.shape[0]):
            trajectories.append(torch.tensor(paths[i], dtype=torch.float32, device=ur10.device))
            if i < 10:
                print(f"  ✅ traj_{i:03d}: {paths[i].shape[0]} points")

    else:
        # Fallback legacy: traj_000, traj_001...
        for key in sorted(dataset.keys()):
            if key.startswith("traj_"):
                traj = dataset[key]
                trajectories.append(torch.tensor(traj, dtype=torch.float32, device=ur10.device))
                print(f"  ✅ {key}: {traj.shape[0]} points")

    if len(trajectories) == 0:
        raise ValueError(f"Aucune trajectoire trouvée dans {dataset_path}")

    print(f"\n✅ {len(trajectories)} trajectoires chargées")

    # Workspace info
    if "workspace" in dataset:
        ws = dataset["workspace"]
        print(f"\n📐 Workspace:")
        print(f"   X: [{ws[0]:.3f}, {ws[1]:.3f}]")
        print(f"   Y: [{ws[2]:.3f}, {ws[3]:.3f}]")
        print(f"   Z: {ws[4]:.3f}m (fixe)")

    # Dataset timing info (si présent)
    if "dt" in dataset:
        dt_ds = float(dataset["dt"])
        hz_ds = float(dataset["control_hz"]) if "control_hz" in dataset else 1.0 / dt_ds
        print(f"\n⏱️  Dataset timing: dt={dt_ds:.3f}s (~{hz_ds:.1f} Hz)")
    else:
        dt_ds = 1.0 / 25.0  # défaut 25 Hz

    # On garde chaque waypoint dataset dt_ds pendant plusieurs steps de simu
    steps_per_waypoint = max(1, int(round(dt_ds / sim_dt)))  # 0.04/0.01 = 4
    print(f"⚙️  Sim dt={sim_dt:.3f}s => steps_per_waypoint = {steps_per_waypoint}")

    # ------------------ EXEC STATE ------------------
    current_traj_idx = [0] * args_cli.num_envs
    current_step = [0] * args_cli.num_envs
    steps_on_current_waypoint = [0] * args_cli.num_envs

    def env_ids_tensor(env_id: int):
        # IsaacLab attend souvent un torch.LongTensor
        return torch.tensor([env_id], device=ur10.device, dtype=torch.long)

    def reset_env(env_id: int):
        traj_idx = int(np.random.randint(0, len(trajectories)))
        current_traj_idx[env_id] = traj_idx
        current_step[env_id] = 0
        steps_on_current_waypoint[env_id] = 0

        env_ids_t = env_ids_tensor(env_id)
        first_point = trajectories[traj_idx][0].unsqueeze(0)  # (1,6)
        ur10.set_joint_position_target(first_point, env_ids=env_ids_t)

        print(f"🔄 Env {env_id}: Reset → traj_{traj_idx:03d} ({trajectories[traj_idx].shape[0]} points)")

    print("\n🎲 Initialisation des environnements...")
    for env_id in range(args_cli.num_envs):
        reset_env(env_id)

    # Petit warmup pour stabiliser
    for _ in range(50):
        ur10.write_data_to_sim()
        sim.step()
        ur10.update(dt=sim.get_physics_dt())

    # ------------------ INFO ------------------
    print("\n" + "=" * 70)
    print("✅ SYSTÈME PRÊT - UR10 avec Trajectoires Pré-calculées")
    print("=" * 70)
    print(f"🤖 Environnements: {args_cli.num_envs}")
    print(f"📁 Trajectoires: {len(trajectories)}")
    print(f"🎲 Bruit gaussien: {'OUI' if args_cli.delta_std > 0 else 'NON'}")
    if args_cli.delta_std > 0:
        print(f"   Écart-type: {args_cli.delta_std:.3f} rad (~{np.degrees(args_cli.delta_std):.1f}°)")
    print("=" * 70 + "\n")

    # ------------------ MAIN LOOP ------------------
    count = 0

    try:
        while simulation_app.is_running():
            for env_id in range(args_cli.num_envs):
                traj = trajectories[current_traj_idx[env_id]]  # (T,6)
                step = current_step[env_id]

                # Traj finie -> reset
                if step >= traj.shape[0]:
                    reset_env(env_id)
                    continue

                env_ids_t = env_ids_tensor(env_id)

                # Tenir le même waypoint pendant steps_per_waypoint steps
                if steps_on_current_waypoint[env_id] < steps_per_waypoint:
                    joint_target = traj[step].clone()  # (6,)

                    # Bruit optionnel
                    if args_cli.delta_std > 0:
                        joint_target = joint_target + torch.randn(6, device=ur10.device) * args_cli.delta_std

                    # ✅ Toujours envoyer (1,6) pour env_id unique
                    ur10.set_joint_position_target(joint_target.unsqueeze(0), env_ids=env_ids_t)
                    steps_on_current_waypoint[env_id] += 1
                else:
                    current_step[env_id] += 1
                    steps_on_current_waypoint[env_id] = 0

            # Step simulation
            ur10.write_data_to_sim()
            sim.step()
            ur10.update(dt=sim.get_physics_dt())

            # Log périodique
            if count % 200 == 0:
                jp = ur10.data.joint_pos[0].detach().cpu().numpy()
                jp_deg = np.degrees(jp)

                step0 = current_step[0]
                T0 = trajectories[current_traj_idx[0]].shape[0]
                prog = 100.0 * min(step0, T0) / max(1, T0)

                print(
                    f"[{count:05d}] Env0 traj_{current_traj_idx[0]:03d} {prog:5.1f}% | "
                    f"Joints: [{jp_deg[0]:.1f}, {jp_deg[1]:.1f}, {jp_deg[2]:.1f}, "
                    f"{jp_deg[3]:.1f}, {jp_deg[4]:.1f}, {jp_deg[5]:.1f}]°"
                )

            count += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    print("\n[INFO] Closing...")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()