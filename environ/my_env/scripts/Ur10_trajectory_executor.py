"""
Script IsaacLab simplifié pour exécuter des trajectoires pré-calculées
SANS ROS Bridge - Approche optimale pour multi-environnements

Usage:
    1. Générer trajectoires avec MoveIt:
       cd ~/workspace/sim2real-pnp/environ/ur10
       source install/setup.bash
       ros2 launch ur_coppeliasim ur_isaaclab_moveit.launch.py
       
       # Autre terminal:
       python3 src/ur_coppeliasim/scripts/generate_xy_trajectories.py --num-lines 100
    
    2. Copier trajectories/ vers environ/my_env/scripts/
    
    3. Lancer IsaacLab:
       cd ~/workspace/sim2real-pnp/environ/my_env/scripts
       ./isaaclab.sh -p Ur10_trajectory_executor.py --num_envs 1
"""

import argparse
from pathlib import Path
import os

from isaaclab.app import AppLauncher

# Argparse
parser = argparse.ArgumentParser(description="UR10 avec trajectoires pré-calculées")
parser.add_argument("--num_envs", type=int, default=1, help="Nombre d'environnements")
parser.add_argument("--trajectories_dir", type=str, default=None, 
                    help="Dossier contenant les trajectoires .npy (par défaut: même dossier que ce script)")
parser.add_argument("--apply_delta", action="store_true", 
                    help="Appliquer corrections delta aléatoires")
parser.add_argument("--delta_std", type=float, default=0.01,
                    help="Écart-type perturbation delta (radians)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Si trajectories_dir n'est pas spécifié, utiliser le dossier du script
if args_cli.trajectories_dir is None:
    script_dir = Path(__file__).parent
    args_cli.trajectories_dir = str(script_dir / "trajectories")
    print(f"[INFO] Chemin trajectoires: {args_cli.trajectories_dir}")

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np  # IMPORTER NUMPY ICI, APRÈS AppLauncher
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab_assets import UR10_CFG

# Importer notre executor
from trajectory_executor import TrajectoryExecutor


def main():
    """Main function."""
    
    import omni.usd
    import carb
    
    # ============================================================
    # RÉDUIRE LA QUALITÉ GRAPHIQUE POUR PC PEU PUISSANT
    # ============================================================
    settings = carb.settings.get_settings()
    
    # Limiter FPS drastiquement
    settings.set("/app/runLoops/main/rateLimitEnabled", True)
    settings.set("/app/runLoops/main/rateLimitFrequency", 10)  # 10 FPS au lieu de 60
    
    # Désactiver rendu temps réel (headless partiel)
    settings.set("/app/renderer/enabled", False)  # Pas de rendu 3D
    settings.set("/rtx/rendermode", "rtx")
    settings.set("/rtx/post/aa/op", 0)  # Désactiver antialiasing
    settings.set("/rtx/reflections/enabled", False)  # Pas de réflexions
    settings.set("/rtx/shadows/enabled", False)  # Pas d'ombres
    settings.set("/rtx/ambientOcclusion/enabled", False)  # Pas d'AO
    settings.set("/rtx/gi/enabled", False)  # Pas de GI
    
    print("[INFO] ⚡ Mode performance activé (qualité réduite)")
    
    # Load USD
    usd_path = "/home/ajin/workspace/sim2real-pnp/environ/my_env/source/env_v1.usd"
    print(f"[INFO] Loading USD: {usd_path}")
    omni.usd.get_context().open_stage(usd_path)
    
    # Setup simulation
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.02,  # Augmenter dt = moins de calculs (20ms au lieu de 10ms)
        device=args_cli.device,
        render_interval=4  # Rendre seulement 1 frame sur 4
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([3.0, 3.0, 2.5], [0.0, 0.0, 0.5])
    
    # Setup robot
    robot_prim_path = "/World/Origin2/Table/Robot"
    ur10_cfg = UR10_CFG.replace(prim_path=robot_prim_path)
    ur10 = Articulation(cfg=ur10_cfg)
    
    # Reset simulation
    sim.reset()
    print("[INFO] Simulation initialized")
    
    # ============================================================
    # CHARGER L'EXECUTOR DE TRAJECTOIRES
    # ============================================================
    print("\n" + "=" * 70)
    print("📁 CHARGEMENT DES TRAJECTOIRES PRÉ-CALCULÉES")
    print("=" * 70)
    
    try:
        executor = TrajectoryExecutor(
            robot=ur10,
            num_envs=args_cli.num_envs,
            trajectories_dir=args_cli.trajectories_dir
        )
        
        # Afficher info
        info = executor.get_trajectory_info()
        print(f"✅ {info['num_trajectories']} trajectoires disponibles")
        
        # Initialiser tous les environnements avec trajectoires aléatoires
        print("\n🎲 Initialisation des environnements...")
        executor.reset_env(list(range(args_cli.num_envs)), random_choice=True)
        
    except ValueError as e:
        print(f"\n❌ ERREUR: {e}")
        print("\n💡 Pour générer les trajectoires:")
        print("   1. Lancer MoveIt:")
        print("      ros2 launch ur_coppeliasim ur_isaaclab_moveit.launch.py")
        print("   2. Générer trajectoires:")
        print("      python3 generate_xy_trajectories.py --num-lines 100")
        print("   3. Copier dossier trajectories/ ici")
        print("=" * 70)
        return
    
    # Initialiser robot sur device CUDA
    for _ in range(50):
        ur10.write_data_to_sim()
        sim.step()
        ur10.update(dt=sim.get_physics_dt())
    
    print(f"[INFO] Robot device: {ur10.device}")
    print("=" * 70)
    
    # ============================================================
    # AFFICHAGE INFO
    # ============================================================
    print("\n" + "=" * 70)
    print("✅ SYSTÈME PRÊT - UR10 avec Trajectoires Pré-calculées")
    print("=" * 70)
    print(f"🤖 Environnements: {args_cli.num_envs}")
    print(f"📁 Trajectoires: {info['num_trajectories']}")
    print(f"🎲 Correction delta: {'OUI' if args_cli.apply_delta else 'NON'}")
    if args_cli.apply_delta:
        print(f"   Écart-type: {args_cli.delta_std:.3f} rad (~{np.degrees(args_cli.delta_std):.1f}°)")
    print("")
    print("⚙️  FONCTIONNEMENT:")
    print("   - Chaque env exécute une trajectoire X-Y aléatoire")
    print("   - Z et orientation fixes (stylo vertical)")
    print("   - Auto-reset quand trajectoire terminée")
    print("   - Pas de ROS Bridge - 100% IsaacLab natif")
    print("")
    print("⌨️  CONTRÔLES:")
    print("   - ESC = Quitter")
    print("=" * 70 + "\n")
    
    # ============================================================
    # MAIN LOOP
    # ============================================================
    count = 0
    
    try:
        while simulation_app.is_running():
            # Exécuter un step de trajectoire pour tous les envs
            executor.step(
                apply_delta=args_cli.apply_delta,
                delta_std=args_cli.delta_std
            )
            
            # Write & step simulation
            ur10.write_data_to_sim()
            sim.step()
            ur10.update(dt=sim.get_physics_dt())
            
            # Log périodique
            if count % 500 == 0:
                joint_pos = ur10.data.joint_pos[0].cpu().numpy()
                joint_pos_deg = np.degrees(joint_pos)
                
                # Afficher progrès
                progress = executor.get_progress(0)
                if progress is not None:
                    print(f"[{count:05d}] Env 0 - Progression: {progress*100:.1f}% | "
                          f"Joints: [{joint_pos_deg[0]:.1f}, {joint_pos_deg[1]:.1f}, "
                          f"{joint_pos_deg[2]:.1f}, {joint_pos_deg[3]:.1f}, "
                          f"{joint_pos_deg[4]:.1f}, {joint_pos_deg[5]:.1f}]°")
            
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
