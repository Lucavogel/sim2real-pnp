"""
Classe pour charger et exécuter des trajectoires pré-calculées dans IsaacLab
Remplace le système ROS Bridge complexe par un système simple de lecture de fichiers
"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Optional
import random


class TrajectoryExecutor:
    """
    Exécuteur de trajectoires pré-calculées par MoveIt
    
    Usage:
        executor = TrajectoryExecutor(robot, num_envs=16, trajectories_dir="trajectories")
        executor.reset_env([0, 1, 2])  # Charge trajectoires aléatoires pour ces envs
        
        while running:
            executor.step(apply_delta=True)  # Avance dans les trajectoires + correction
            robot.write_data_to_sim()
            sim.step()
    """
    
    def __init__(self, robot, num_envs: int = 1, trajectories_dir: str = "trajectories"):
        """
        Args:
            robot: Articulation Isaac Lab (UR10)
            num_envs: Nombre d'environnements parallèles
            trajectories_dir: Dossier contenant les fichiers .npy de trajectoires
        """
        self.robot = robot
        self.num_envs = num_envs
        self.trajectories_dir = Path(trajectories_dir)
        
        # Charger toutes les trajectoires disponibles
        self.trajectories = self._load_all_trajectories()
        
        if len(self.trajectories) == 0:
            raise ValueError(f"Aucune trajectoire trouvée dans {trajectories_dir}!")
        
        # État de chaque environnement
        self.current_traj = [None] * num_envs  # Trajectoire actuelle [num_envs]
        self.traj_step = [0] * num_envs  # Step actuel dans la trajectoire
        self.traj_indices = [0] * num_envs  # Index de la trajectoire choisie
        self.wait_counter = [0] * num_envs  # Compteur d'attente entre points
        self.steps_per_point = 5  # Nombre de steps à attendre entre chaque point de trajectoire
        
        print(f"✅ TrajectoryExecutor initialisé:")
        print(f"   - {len(self.trajectories)} trajectoires chargées")
        print(f"   - {num_envs} environnements")
        print(f"   - Device: {robot.device}")
        print(f"   - Vitesse: {self.steps_per_point} steps par point")
    
    def _load_all_trajectories(self) -> List[torch.Tensor]:
        """Charge toutes les trajectoires .npy du dossier"""
        trajectories = []
        
        if not self.trajectories_dir.exists():
            print(f"⚠️ Dossier {self.trajectories_dir} inexistant!")
            return trajectories
        
        # Lire tous les fichiers .npy
        traj_files = sorted(self.trajectories_dir.glob("traj_*.npy"))
        
        for traj_file in traj_files:
            try:
                # Charger numpy array [N_points, 6]
                traj_np = np.load(traj_file)
                
                # Convertir en torch tensor
                traj_torch = torch.from_numpy(traj_np).float().to(self.robot.device)
                
                trajectories.append(traj_torch)
                
                print(f"   Chargé: {traj_file.name} → {traj_torch.shape[0]} points")
                
            except Exception as e:
                print(f"   ❌ Erreur chargement {traj_file.name}: {e}")
        
        return trajectories
    
    def reset_env(self, env_ids: List[int], random_choice: bool = True):
        """
        Reset un ou plusieurs environnements avec de nouvelles trajectoires
        
        Args:
            env_ids: Liste des IDs d'environnements à reset
            random_choice: Si True, choisit trajectoire aléatoire, sinon séquentiel
        """
        for env_id in env_ids:
            if env_id >= self.num_envs:
                continue
            
            # Choisir trajectoire
            if random_choice:
                idx = random.randint(0, len(self.trajectories) - 1)
            else:
                idx = self.traj_indices[env_id] % len(self.trajectories)
            
            # Charger trajectoire
            self.current_traj[env_id] = self.trajectories[idx]
            self.traj_step[env_id] = 0
            self.traj_indices[env_id] = idx
            self.wait_counter[env_id] = 0  # Reset compteur d'attente
            
            # Set pose de départ = premier point de la trajectoire
            start_q = self.current_traj[env_id][0]
            self.robot.set_joint_position_target(
                start_q.unsqueeze(0),  # [1, 6]
                env_ids=[env_id]
            )
            
            num_points = len(self.current_traj[env_id])
            print(f"🎲 Env {env_id}: Trajectoire {idx} chargée ({num_points} points)")
    
    def step(self, apply_delta: bool = True, delta_std: float = 0.01):
        """
        Avance d'un step dans les trajectoires de tous les environnements
        Avec délai entre chaque point pour laisser le robot se déplacer
        
        Args:
            apply_delta: Si True, applique correction delta aléatoire
            delta_std: Écart-type de la perturbation (en radians)
        """
        for env_id in range(self.num_envs):
            if self.current_traj[env_id] is None:
                continue
            
            traj = self.current_traj[env_id]
            step = self.traj_step[env_id]
            
            # Si trajectoire terminée, attendre un peu avant de reset
            if step >= len(traj):
                # Attendre 70 steps (~1 seconde) pour voir le mouvement complet
                if self.wait_counter[env_id] < 70:
                    self.wait_counter[env_id] += 1
                    continue
                else:
                    self.reset_env([env_id], random_choice=True)
                    continue
            
            # Attendre entre chaque point de trajectoire
            if self.wait_counter[env_id] < self.steps_per_point:
                self.wait_counter[env_id] += 1
                continue
            
            # Reset compteur et avancer au prochain point
            self.wait_counter[env_id] = 0
            
            # Position nominale (q)
            nominal_q = traj[step]
            
            # Ajouter correction delta (perturbation aléatoire en joint space)
            if apply_delta:
                delta = torch.randn(6, device=self.robot.device) * delta_std
                corrected_q = nominal_q + delta
            else:
                corrected_q = nominal_q
            
            # Appliquer au robot
            self.robot.set_joint_position_target(
                corrected_q.unsqueeze(0),  # [1, 6]
                env_ids=[env_id]
            )
            
            # Incrémenter step
            self.traj_step[env_id] += 1
    
    def get_progress(self, env_id: int) -> Optional[float]:
        """Retourne le pourcentage de complétion de la trajectoire [0-1]"""
        if self.current_traj[env_id] is None:
            return None
        
        total = len(self.current_traj[env_id])
        current = self.traj_step[env_id]
        
        return min(current / total, 1.0)
    
    def is_trajectory_complete(self, env_id: int) -> bool:
        """Vérifie si la trajectoire est terminée"""
        if self.current_traj[env_id] is None:
            return True
        
        return self.traj_step[env_id] >= len(self.current_traj[env_id])
    
    def get_current_target(self, env_id: int) -> Optional[torch.Tensor]:
        """Retourne la position cible actuelle [6] pour un environnement"""
        if self.current_traj[env_id] is None:
            return None
        
        step = self.traj_step[env_id]
        if step >= len(self.current_traj[env_id]):
            return None
        
        return self.current_traj[env_id][step]
    
    def get_trajectory_info(self):
        """Retourne info sur toutes les trajectoires chargées"""
        info = {
            "num_trajectories": len(self.trajectories),
            "num_envs": self.num_envs,
            "trajectories": []
        }
        
        for i, traj in enumerate(self.trajectories):
            info["trajectories"].append({
                "index": i,
                "num_points": len(traj),
                "shape": list(traj.shape)
            })
        
        return info
