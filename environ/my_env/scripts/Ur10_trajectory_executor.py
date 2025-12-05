"""
Script IsaacLab simplifié pour exécuter des trajectoires pré-calculées
SANS ROS Bridge - Supporte dataset.npz généré par MoveIt

Usage:
    1. Générer trajectoires avec MoveIt V2:
       cd ~/workspace/sim2real-pnp/environ/ur10
       source install/setup.bash
       ros2 launch ur_coppeliasim ur_isaaclab_moveit.launch.py
       
       # Autre terminal:
       python3 src/ur_coppeliasim/scripts/generate_moveit_dataset_v2.py --num-traj 100
    
    2. Copier dataset.npz vers environ/my_env/scripts/
    
    3. Lancer IsaacLab:
       cd ~/workspace/sim2real-pnp/environ/my_env/scripts
       ./isaaclab.sh -p Ur10_trajectory_executor.py --dataset dataset.npz --num_envs 1
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
parser.add_argument("--trajectories_dir", type=str, default=None, 
                    help="[LEGACY] Dossier contenant les trajectoires .npy (ancien format)")
parser.add_argument("--apply_delta", action="store_true", 
                    help="Appliquer corrections delta aléatoires")
parser.add_argument("--delta_std", type=float, default=0.01,
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


    
    print("[INFO] ⚡ Mode performance ULTRA activé (qualité minimale)")

    
    # Load USD
<<<<<<< HEAD
    usd_path = "/home/ajin/work2/sim2real-pnp/environ/my_env/venv/env_v1.usd"
=======
    usd_path = "/home/ajin/workspace/sim2real-pnp/environ/my_env/source/env_v2.usd"
>>>>>>> origin/luca
    print(f"[INFO] Loading USD: {usd_path}")
    omni.usd.get_context().open_stage(usd_path)
    
    # Setup simulation
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.05,  # 50ms = TRÈS LENT mais stable (20 Hz physics)
        device=args_cli.device,
        render_interval=8  # Rendre 1 frame sur 8 au lieu de 4
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
    # CHARGER LES TRAJECTOIRES DEPUIS DATASET.NPZ
    # ============================================================
    print("\n" + "=" * 70)
    print("📁 CHARGEMENT DES TRAJECTOIRES (dataset.npz)")
    print("=" * 70)
    
    try:
        # Charger le fichier npz
        dataset_path = Path(args_cli.dataset)
        
        if not dataset_path.exists():
            raise ValueError(f"Dataset non trouvé: {dataset_path}")
        
        print(f"📂 Chargement: {dataset_path}")
        dataset = np.load(dataset_path)
        
        # Extraire toutes les trajectoires (traj_000, traj_001, ...)
        trajectories = []
        for key in sorted(dataset.keys()):
            if key.startswith('traj_'):
                traj = dataset[key]
                trajectories.append(torch.tensor(traj, dtype=torch.float32, device=ur10.device))
                print(f"  ✅ {key}: {traj.shape[0]} points")
        
        if len(trajectories) == 0:
            raise ValueError(f"Aucune trajectoire trouvée dans {dataset_path}")
        
        print(f"\n✅ {len(trajectories)} trajectoires chargées")
        
        # Afficher info workspace si disponible
        if 'workspace' in dataset:
            ws = dataset['workspace']
            print(f"\n📐 Workspace:")
            print(f"   X: [{ws[0]:.3f}, {ws[1]:.3f}]")
            print(f"   Y: [{ws[2]:.3f}, {ws[3]:.3f}]")
            print(f"   Z: {ws[4]:.3f}m (fixe)")
        
        # Variables pour gérer l'exécution
        current_traj_idx = [0] * args_cli.num_envs  # Index trajectoire actuelle par env
        current_step = [0] * args_cli.num_envs  # Step actuel dans la trajectoire
        
        # Fonction pour reset un env avec une nouvelle trajectoire
        def reset_env(env_id):
            # Choisir une trajectoire aléatoire
            traj_idx = np.random.randint(0, len(trajectories))
            current_traj_idx[env_id] = traj_idx
            current_step[env_id] = 0
            print(f"🔄 Env {env_id}: Nouvelle trajectoire traj_{traj_idx:03d} ({trajectories[traj_idx].shape[0]} points)")
        
        # Initialiser tous les environnements
        print("\n🎲 Initialisation des environnements...")
        for env_id in range(args_cli.num_envs):
            reset_env(env_id)
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        print("\n💡 Pour générer le dataset:")
        print("   1. Lancer MoveIt:")
        print("      ros2 launch ur_coppeliasim ur_isaaclab_moveit.launch.py")
        print("   2. Générer dataset:")
        print("      python3 src/ur_coppeliasim/scripts/generate_moveit_dataset_v2.py --num-traj 100")
        print("   3. Copier dataset.npz ici")
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
    print(f"📁 Trajectoires: {len(trajectories)}")
    print(f"🎲 Bruit gaussien: {'OUI' if args_cli.delta_std > 0 else 'NON'}")
    if args_cli.delta_std > 0:
        print(f"   Écart-type: {args_cli.delta_std:.3f} rad (~{np.degrees(args_cli.delta_std):.1f}°)")
    print("")
    print("⚙️  FONCTIONNEMENT:")
    print("   - Chaque env exécute une trajectoire MoveIt aléatoire")
    print("   - Trajectoires générées par OMPL planner")
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
            # ========================================================
            # SUIVI DES TRAJECTOIRES
            # ========================================================
            for env_id in range(args_cli.num_envs):
                # Récupérer trajectoire et step actuel
                traj = trajectories[current_traj_idx[env_id]]
                step = current_step[env_id]
                
                # Trajectoire terminée → Reset
                if step >= len(traj):
                    reset_env(env_id)
                    continue
                
                # Récupérer les 6 valeurs de joints + deltas aléatoires
                joint_target = traj[step].clone()  # Copie pour ne pas modifier dataset
                
                # Ajouter bruit gaussien (variabilité)
                if args_cli.delta_std > 0:
                    noise = torch.randn(6, device=ur10.device) * args_cli.delta_std
                    joint_target += noise
                
                # Appliquer la commande
                ur10.set_joint_position_target(joint_target, env_ids=[env_id])
                
                # Incrémenter le step
                current_step[env_id] += 1
            
            # Write & step simulation
            ur10.write_data_to_sim()
            sim.step()
            ur10.update(dt=sim.get_physics_dt())
            
            # Log périodique
            if count % 500 == 0:
                joint_pos = ur10.data.joint_pos[0].cpu().numpy()
                joint_pos_deg = np.degrees(joint_pos)
                
                # Afficher progrès
                progress = current_step[0] / len(trajectories[current_traj_idx[0]]) * 100
                print(f"[{count:05d}] Env 0 - traj_{current_traj_idx[0]:03d} {progress:.1f}% | "
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
