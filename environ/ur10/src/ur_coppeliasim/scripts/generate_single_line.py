#!/usr/bin/env python3
"""
Générateur de dataset - lignes droites aléatoires 2D
Génère N trajectoires droites aléatoires dans le workspace

Usage:
   cd ~/workspace/sim2real-pnp/environ/ur10
   source install/setup.bash
   
   # Terminal 1 - MoveIt
   ros2 launch ur_coppeliasim ur_isaaclab_moveit.launch.py
   
   # Terminal 2 - Générer dataset
   python3 src/ur_coppeliasim/scripts/generate_single_line.py --num-traj 100
"""

import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
import numpy as np
import time
import argparse
from pathlib import Path


class RandomLinesGenerator(Node):
    def __init__(self, num_trajectories=100, output_file="random_lines_dataset.npz"):
        super().__init__("random_lines_generator")
        
        self.num_trajectories = num_trajectories
        self.output_file = Path(output_file)
        
        # Workspace fixe
        self.workspace = {
            'x_min': 0.8,
            'x_max': 1.164,
            'y_min': -0.2,
            'y_max': 0.2,
            'z_fixed': 0.15
        }
        
        # Service MoveIt cartésien
        self._cartesian_path_client = self.create_client(
            GetCartesianPath, '/compute_cartesian_path'
        )
        
        self.executor = None
        
        # Orientation fixe (outil vers le bas)
        self.fixed_orientation = [1.0, 0.0, 0.0, 0.0]
        
        # Position initiale (home position UR10)
        self.current_joint_state = [0.0, -1.309, 2.14675, -2.44346, -1.5708, 0.0]
        
        # Nombre de waypoints pour la ligne demandée à MoveIt
        self.num_waypoints = 64
        
        self.get_logger().info("📏 Générateur Dataset - Lignes Droites Aléatoires (PC1)")
        self.get_logger().info("   🎯 Workflow: Génération MoveIt → Dataset Isaac Lab (PC2)")
        self.get_logger().info(f"   Nombre: {self.num_trajectories} trajectoires")
        self.get_logger().info(f"   Workspace: X=[{self.workspace['x_min']:.3f}, {self.workspace['x_max']:.3f}]")
        self.get_logger().info(f"              Y=[{self.workspace['y_min']:.3f}, {self.workspace['y_max']:.3f}]")
        self.get_logger().info(f"              Z={self.workspace['z_fixed']}m (2D planaire strict)")
        self.get_logger().info(f"   Output: {self.output_file}")
        self.get_logger().info("")
    
    def set_executor(self, executor):
        self.executor = executor
    
    def generate_cartesian_waypoints(self, start_pose, end_pose):
        """Génère une ligne droite 2D entre start et end (waypoints cartésiens pour MoveIt)."""
        waypoints = []
        
        for i in range(self.num_waypoints + 1):
            t = i / self.num_waypoints
            
            pose = PoseStamped()
            pose.header.frame_id = "world"
            
            # Interpolation linéaire
            pose.pose.position.x = start_pose[0] + t * (end_pose[0] - start_pose[0])
            pose.pose.position.y = start_pose[1] + t * (end_pose[1] - start_pose[1])
            pose.pose.position.z = self.workspace['z_fixed']
            
            # Orientation fixe
            pose.pose.orientation.x = self.fixed_orientation[0]
            pose.pose.orientation.y = self.fixed_orientation[1]
            pose.pose.orientation.z = self.fixed_orientation[2]
            pose.pose.orientation.w = self.fixed_orientation[3]
            
            waypoints.append(pose.pose)
        
        return waypoints
    
    def plan_cartesian_path(self, start_pose, target_pose):
        """Planifie un chemin cartésien 2D strict."""
        request = GetCartesianPath.Request()
        request.header.frame_id = "world"
        request.group_name = "ur_manipulator"
        request.link_name = "tool0"
        
        # Start state
        start_state = RobotState()
        start_state.joint_state = JointState()
        start_state.joint_state.name = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        start_state.joint_state.position = self.current_joint_state
        request.start_state = start_state
        
        # Waypoints 2D
        request.waypoints = self.generate_cartesian_waypoints(start_pose, target_pose)
        
        # Paramètres
        request.max_step = 0.005
        request.jump_threshold = 0.0
        request.avoid_collisions = True
        
        # Appel asynchrone
        future = self._cartesian_path_client.call_async(request)
        
        timeout = 10.0
        start_time = time.time()
        
        while not future.done():
            if time.time() - start_time > timeout:
                self.get_logger().error(f"   ⏱️ Timeout après {timeout}s")
                return None
            
            if self.executor:
                self.executor.spin_once(timeout_sec=0.1)
            else:
                time.sleep(0.1)
        
        try:
            response = future.result()
            
            # Si succès (fraction = 1.0) : Sauvegarder, sinon jeter
            if response.fraction < 1.0:
                self.get_logger().warn(f"   ⚠️ Trajectoire incomplète ({response.fraction*100:.1f}%) - REJETÉE")
                self.get_logger().warn(f"   💡 Raison: collision ou singularité détectée")
                return None
            
            return response
        except Exception as e:
            self.get_logger().error(f"   ❌ Erreur: {e}")
            return None
    
    def trajectory_to_numpy(self, trajectory):
        """Convertit trajectoire MoveIt en numpy (angles articulaires seulement)."""
        if trajectory is None:
            return None
        
        if hasattr(trajectory, 'joint_trajectory'):
            points = trajectory.joint_trajectory.points
        else:
            return None
        
        if len(points) == 0:
            return None
        
        joint_angles = []
        
        for point in points:
            q = list(point.positions)
            joint_angles.append(q)
        
        return np.array(joint_angles)
    
    def generate_dataset(self):
        """Génère N trajectoires droites aléatoires."""
        self.get_logger().info("🚀 Début génération...")
        
        # Attendre service
        self.get_logger().info("🔍 Recherche service /compute_cartesian_path...")
        if not self._cartesian_path_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("❌ Service indisponible!")
            return
        self.get_logger().info("✅ Service trouvé")
        self.get_logger().info("")
        
        valid_trajectories = []        # liste de q_array (T_i, 6)
        valid_ee_trajectories = []     # liste de ee_array (T_i, 3) : x, y, z linéaires
        
        attempts = 0
        max_attempts = self.num_trajectories * 3  # 3x plus de tentatives possibles
        
        self.get_logger().info(f"🎲 Génération de {self.num_trajectories} trajectoires aléatoires...")
        self.get_logger().info("="*70)
        
        while len(valid_trajectories) < self.num_trajectories and attempts < max_attempts:
            attempts += 1
            
            # Générer points aléatoires dans le workspace
            start_x = np.random.uniform(self.workspace['x_min'], self.workspace['x_max'])
            start_y = np.random.uniform(self.workspace['y_min'], self.workspace['y_max'])
            end_x = np.random.uniform(self.workspace['x_min'], self.workspace['x_max'])
            end_y = np.random.uniform(self.workspace['y_min'], self.workspace['y_max'])
            
            start_pose = (start_x, start_y, self.workspace['z_fixed'])
            end_pose = (end_x, end_y, self.workspace['z_fixed'])
        
            # Log tentative
            self.get_logger().info(f"\n[{len(valid_trajectories)+1}/{self.num_trajectories}] Tentative {attempts}")
            self.get_logger().info(f"   Start: ({start_pose[0]:.3f}, {start_pose[1]:.3f})")
            self.get_logger().info(f"   End:   ({end_pose[0]:.3f}, {end_pose[1]:.3f})")
            
            # Planifier
            response = self.plan_cartesian_path(start_pose, end_pose)
            
            if response is None:
                self.get_logger().warn("   ⚠️ REJETÉE (collision/singularité)")
                continue
        
            # Convertir en numpy
            q_array = self.trajectory_to_numpy(response.solution)
            
            if q_array is None or len(q_array) == 0:
                self.get_logger().warn("   ⚠️ REJETÉE (conversion échouée)")
                continue
            
            num_points = len(q_array)
            self.get_logger().info(f"   ✅ VALIDE: {num_points} points")
        
            # Générer les positions cartésiennes idéales de la ligne (x, y, z) pour ces num_points
            xs = np.linspace(start_pose[0], end_pose[0], num_points)
            ys = np.linspace(start_pose[1], end_pose[1], num_points)
            zs = np.full(num_points, self.workspace['z_fixed'])
            ee_array = np.stack([xs, ys, zs], axis=1)  # shape (num_points, 3)
            
            # Détecter aller-retour en joint space
            start_joints = q_array[0]
            end_joints = q_array[-1]
            distance = np.linalg.norm(end_joints - start_joints)
            
            if distance < 0.5:
                # Trouver point le plus éloigné
                max_dist = 0
                max_idx = 0
                for i in range(len(q_array)):
                    dist = np.linalg.norm(q_array[i] - start_joints)
                    if dist > max_dist:
                        max_dist = dist
                        max_idx = i
                # Couper q_array ET ee_array au même index
                q_array = q_array[:max_idx+1]
                ee_array = ee_array[:max_idx+1]
                self.get_logger().info(f"   ✂️ Coupé au point {max_idx+1}")
            
            # Ajouter à la liste
            valid_trajectories.append(q_array)
            valid_ee_trajectories.append(ee_array)
            
            duration = len(q_array) * 0.02  # 50 Hz
            self.get_logger().info(f"   📊 {len(q_array)} points, ~{duration:.2f}s")
        
        self.get_logger().info("")
        self.get_logger().info("="*70)
        
        if len(valid_trajectories) < self.num_trajectories:
            self.get_logger().error(f"❌ Échec: seulement {len(valid_trajectories)}/{self.num_trajectories} générées")
            self.get_logger().error(f"   Tentatives totales: {attempts}")
            return
        
        self.get_logger().info(f"✅ {len(valid_trajectories)} trajectoires valides générées!")
        self.get_logger().info("")
        
        # Statistiques finales
        total_points = sum(len(traj) for traj in valid_trajectories)
        avg_points = total_points / len(valid_trajectories)
        total_duration = total_points * 0.02
        
        self.get_logger().info("📊 Statistiques du dataset:")
        self.get_logger().info(f"   Trajectoires: {len(valid_trajectories)}")
        self.get_logger().info(f"   Points totaux: {total_points}")
        self.get_logger().info(f"   Points/trajectoire (moy): {avg_points:.1f}")
        self.get_logger().info(f"   Durée totale: {total_duration:.1f}s à 50Hz")
        self.get_logger().info("")
        
        # Sauvegarder format Isaac Lab + XYZ
        self.get_logger().info("💾 Sauvegarde format Isaac Lab...")
        
        # Format requis: np.array(dtype=object) pour longueurs variables
        q_paths_array = np.array(valid_trajectories, dtype=object)       # (T_i, 6)
        ee_paths_array = np.array(valid_ee_trajectories, dtype=object)   # (T_i, 3)
        
        dataset = {
            'paths': q_paths_array,          # joints
            'ee_pos': ee_paths_array,        # x, y, z sur la ligne idéale
            'workspace': np.array([
                self.workspace['x_min'],
                self.workspace['x_max'],
                self.workspace['y_min'],
                self.workspace['y_max'],
                self.workspace['z_fixed']
            ]),
            'num_trajectories': len(q_paths_array)
        }
        
        np.savez_compressed(self.output_file, **dataset)
        self.get_logger().info(f"✅ Sauvegardé: {self.output_file.absolute()}")
        self.get_logger().info("")
        self.get_logger().info("=" * 70)
        self.get_logger().info("✅ TERMINÉ")
        self.get_logger().info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Générateur de dataset - lignes droites aléatoires")
    parser.add_argument('--num-traj', type=int, default=100, help='Nombre de trajectoires à générer')
    parser.add_argument('--output', type=str, default='random_lines_dataset.npz', help='Fichier de sortie')
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("📏 GÉNÉRATEUR DATASET - LIGNES DROITES ALÉATOIRES")
    print("=" * 70)
    print(f"Nombre de trajectoires: {args.num_traj}")
    print(f"Fichier de sortie: {args.output}")
    print("=" * 70 + "\n")
    
    input("Appuyez sur ENTRÉE pour continuer...")
    
    rclpy.init()
    
    node = RandomLinesGenerator(num_trajectories=args.num_traj, output_file=args.output)
    
    from rclpy.executors import SingleThreadedExecutor
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    node.set_executor(executor)
    
    # Init ROS2
    print("\n⏳ Initialisation ROS2 (3 secondes)...")
    start_time = time.time()
    while time.time() - start_time < 3.0:
        executor.spin_once(timeout_sec=0.1)
    print("✅ ROS2 prêt\n")
    
    # Générer dataset
    node.generate_dataset()
    
    executor.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
