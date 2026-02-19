#!/usr/bin/env python3
"""
Générateur de dataset MoveIt STANDALONE - PC1
Génère des trajectoires MoveIt (planification complète, pas cartésienne)

📦 Ce script est AUTONOME:
   - Pas besoin d'Isaac Lab
   - Pas besoin de caméra
   - Workspace fixe défini par 4 coins
   - Utilise la planification MoveIt normale (OMPL)

🎯 Workflow PC1:
   1. Terminal 1 - Lancer MoveIt:
      cd ~/workspace/sim2real-pnp/environ/ur10
      source install/setup.bash
      ros2 launch ur_coppeliasim ur_isaaclab_moveit.launch.py
   
   2. Terminal 2 - Générer dataset:
      cd ~/workspace/sim2real-pnp/environ/ur10
      python3 src/ur_coppeliasim/scripts/generate_moveit_dataset_v2.py --num-traj 100
"""

import rclpy
from rclpy.node import Node
from moveit_msgs.msg import MotionPlanRequest, Constraints, PositionConstraint, OrientationConstraint, JointConstraint
from moveit_msgs.srv import GetMotionPlan
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import PoseStamped
import numpy as np
import argparse
from pathlib import Path
import time


class MoveItDatasetGeneratorV2(Node):
    def __init__(self, num_trajectories=100, output_file="dataset.npz"):
        super().__init__("moveit_dataset_generator_v2")
        
        self.num_trajectories = num_trajectories
        self.output_file = Path(output_file)
        
        # Workspace fixe (inversé en +X pour être devant le robot)
        # Sur le vrai robot: inverser X → multiplier par -1
        self.workspace = {
            'x_min': 0.8,     # +X devant le robot (était -0.8)
            'x_max': 1.164,   # +X devant le robot (était -1.164)
            'y_min': -0.2,
            'y_max': 0.2,
            'z_fixed': 0.15
        }
        
        self.width = self.workspace['x_max'] - self.workspace['x_min']
        self.height = self.workspace['y_max'] - self.workspace['y_min']
        
        # Service MoveIt pour planification CARTÉSIENNE (vrai 2D)
        from moveit_msgs.srv import GetCartesianPath
        self._cartesian_path_client = self.create_client(
            GetCartesianPath, '/compute_cartesian_path'
        )
        
        self.executor = None
        
        # Orientation fixe (outil vers le bas) - CORRIGÉE pour UR10
        # Convention: Z-axis pointing down = rotation 180° autour Y
        self.fixed_orientation = [1.0, 0.0, 0.0, 0.0]  # quaternion (Y-up convention)
        
        # État courant du robot (pour start_state des planifications)
        self.current_joint_state = None
        
        # Nombre de waypoints intermédiaires pour ligne 2D
        self.num_waypoints = 20  # Plus = plus lisse
        
        self.get_logger().info("🏭 Générateur Dataset MoveIt V2 (planification complète)")
        self.get_logger().info(f"   Output: {self.output_file}")
        self.get_logger().info(f"   Nombre: {num_trajectories}")
        self.display_workspace()
    
    def set_executor(self, executor):
        self.executor = executor
    
    def display_workspace(self):
        self.get_logger().info("")
        self.get_logger().info("=" * 70)
        self.get_logger().info("📐 ZONE DE TRAVAIL")
        self.get_logger().info("=" * 70)
        self.get_logger().info(f"  X: [{self.workspace['x_min']:.3f}, {self.workspace['x_max']:.3f}] ({self.width*100:.1f}cm)")
        self.get_logger().info(f"  Y: [{self.workspace['y_min']:.3f}, {self.workspace['y_max']:.3f}] ({self.height*100:.1f}cm)")
        self.get_logger().info(f"  Z: {self.workspace['z_fixed']:.3f}m (fixe)")
        self.get_logger().info("=" * 70)
        self.get_logger().info("")
    
    def generate_random_poses(self):
        """Génère des paires de poses aléatoires dans le workspace"""
        poses = []
        
        for i in range(self.num_trajectories):
            # Pose de départ
            x_start = np.random.uniform(self.workspace['x_min'], self.workspace['x_max'])
            y_start = np.random.uniform(self.workspace['y_min'], self.workspace['y_max'])
            
            # Pose d'arrivée
            x_end = np.random.uniform(self.workspace['x_min'], self.workspace['x_max'])
            y_end = np.random.uniform(self.workspace['y_min'], self.workspace['y_max'])
            
            start_pose = (x_start, y_start, self.workspace['z_fixed'])
            end_pose = (x_end, y_end, self.workspace['z_fixed'])
            
            poses.append((start_pose, end_pose))
        
        return poses
    
    def generate_cartesian_waypoints(self, start_pose, end_pose):
        """Génère une ligne droite 2D entre start et end"""
        waypoints = []
        
        for i in range(self.num_waypoints + 1):
            t = i / self.num_waypoints
            
            pose = PoseStamped()
            pose.header.frame_id = "world"
            
            # Interpolation linéaire en 2D
            pose.pose.position.x = start_pose[0] + t * (end_pose[0] - start_pose[0])
            pose.pose.position.y = start_pose[1] + t * (end_pose[1] - start_pose[1])
            pose.pose.position.z = self.workspace['z_fixed']  # Z strictement fixe
            
            # Orientation fixe partout
            pose.pose.orientation.x = self.fixed_orientation[0]
            pose.pose.orientation.y = self.fixed_orientation[1]
            pose.pose.orientation.z = self.fixed_orientation[2]
            pose.pose.orientation.w = self.fixed_orientation[3]
            
            waypoints.append(pose.pose)
        
        return waypoints
    
    def plan_cartesian_path(self, start_pose, target_pose):
        """Planifie un chemin cartésien 2D strict entre deux poses"""
        from moveit_msgs.srv import GetCartesianPath
        from moveit_msgs.msg import RobotState
        from sensor_msgs.msg import JointState
        
        # Créer le client si nécessaire
        if not hasattr(self, '_cartesian_path_client'):
            self._cartesian_path_client = self.create_client(
                GetCartesianPath, '/compute_cartesian_path'
            )
        
        # Attendre le service
        if not self._cartesian_path_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("❌ Service /compute_cartesian_path indisponible")
            return None
        
        request = GetCartesianPath.Request()
        request.header.frame_id = "world"
        request.group_name = "ur_manipulator"
        request.link_name = "tool0"
        
        # ✅ START STATE - utiliser dernier état connu
        if self.current_joint_state is not None:
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
        
        # ✅ WAYPOINTS 2D - ligne droite garantie
        request.waypoints = self.generate_cartesian_waypoints(start_pose, target_pose)
        
        # Paramètres cartésiens
        request.max_step = 0.01  # 1cm entre chaque point IK
        request.jump_threshold = 0.0  # Pas de sauts de configuration
        request.avoid_collisions = True
        
        # Appel asynchrone
        future = self._cartesian_path_client.call_async(request)
        
        timeout = 10.0
        start_time = time.time()
        
        while not future.done():
            if time.time() - start_time > timeout:
                self.get_logger().error(f"   ⏱️ Timeout planification cartésienne après {timeout}s")
                return None
            
            if self.executor:
                self.executor.spin_once(timeout_sec=0.1)
            else:
                time.sleep(0.1)
        
        try:
            response = future.result()
            
            # Vérifier si le chemin est complet
            if response.fraction < 0.95:  # Au moins 95% du chemin
                self.get_logger().warn(f"   ⚠️ Chemin incomplet ({response.fraction*100:.1f}%)")
                return None
            
            return response
        except Exception as e:
            self.get_logger().error(f"   ❌ Erreur service cartésien: {e}")
            return None
    
    def trajectory_to_numpy(self, trajectory):
        """Convertit une trajectoire MoveIt en numpy"""
        if trajectory is None:
            return None
        
        # CartesianPath retourne RobotTrajectory directement
        if hasattr(trajectory, 'joint_trajectory'):
            points = trajectory.joint_trajectory.points
        else:
            return None
        
        if len(points) == 0:
            return None
        
        joint_angles = []
        for point in points:
            joint_angles.append(list(point.positions))
        
        return np.array(joint_angles)
    
    def generate_dataset(self):
        """Génère le dataset complet"""
        self.get_logger().info("🚀 Début génération dataset...")
        
        # Attendre service cartésien
        self.get_logger().info("🔍 Recherche service MoveIt /compute_cartesian_path...")
        if not self._cartesian_path_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("❌ Service /compute_cartesian_path non disponible!")
            return
        self.get_logger().info("✅ Service MoveIt cartésien trouvé")
        self.get_logger().info("   ℹ️ Mode: Planification 2D stricte (ligne droite)")
        self.get_logger().info("")
        
        # ✅ Initialiser position de départ (home position UR10)
        if self.current_joint_state is None:
            self.current_joint_state = [0.0, -1.309, 2.14675, -2.44346, -1.5708, 0.0]
            self.get_logger().info("🏠 Position initiale: home position UR10")
            self.get_logger().info("")
        
        # Générer poses
        self.get_logger().info(f"📐 Génération de {self.num_trajectories} paires de poses...")
        pose_pairs = self.generate_random_poses()
        self.get_logger().info(f"✅ {len(pose_pairs)} paires générées")
        self.get_logger().info("")
        
        trajectories = {}
        success_count = 0
        failed_count = 0
        
        for i, (start_pose, end_pose) in enumerate(pose_pairs):
            self.get_logger().info(f"🔄 [{i+1}/{len(pose_pairs)}] Planification 2D...")
            self.get_logger().info(f"   Start: ({start_pose[0]:.3f}, {start_pose[1]:.3f})")
            self.get_logger().info(f"   Goal:  ({end_pose[0]:.3f}, {end_pose[1]:.3f})")
            
            try:
                response = self.plan_cartesian_path(start_pose, end_pose)
                
                if response is None:
                    failed_count += 1
                    self.get_logger().warn(f"   ❌ Pas de réponse")
                    continue
                
                # CartesianPath retourne directement la trajectoire
                traj_array = self.trajectory_to_numpy(response.solution)
                
                if traj_array is not None and len(traj_array) > 0:
                    trajectories[f'traj_{success_count:03d}'] = traj_array
                    success_count += 1
                    num_points = len(traj_array)
                    
                    # ✅ Mettre à jour l'état courant avec le dernier point
                    self.current_joint_state = list(traj_array[-1])
                    
                    self.get_logger().info(f"   ✅ Trajectoire 2D valide ({num_points} points, {response.fraction*100:.1f}% complet)")
                else:
                    failed_count += 1
                    self.get_logger().warn(f"   ❌ Trajectoire invalide")
                
            except Exception as e:
                failed_count += 1
                self.get_logger().error(f"   ❌ Erreur: {e}")
        
        # Sauvegarder
        if success_count > 0:
            self.get_logger().info("")
            self.get_logger().info(f"💾 Sauvegarde {success_count} trajectoires...")
            
            trajectories['workspace'] = np.array([
                self.workspace['x_min'],
                self.workspace['x_max'],
                self.workspace['y_min'],
                self.workspace['y_max'],
                self.workspace['z_fixed']
            ])
            trajectories['num_trajectories'] = np.array([success_count])
            
            np.savez(self.output_file, **trajectories)
            self.get_logger().info(f"✅ Dataset sauvegardé: {self.output_file.absolute()}")
        
        # Résumé
        self.get_logger().info("")
        self.get_logger().info("=" * 70)
        self.get_logger().info("✅ GÉNÉRATION TERMINÉE")
        self.get_logger().info("=" * 70)
        self.get_logger().info(f"✅ Succès:  {success_count}/{len(pose_pairs)} ({success_count/len(pose_pairs)*100:.1f}%)")
        self.get_logger().info(f"❌ Échecs:  {failed_count}/{len(pose_pairs)}")
        self.get_logger().info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Génère dataset MoveIt V2 (planification complète)")
    parser.add_argument("--num-traj", type=int, default=100)
    parser.add_argument("--output", type=str, default="dataset.npz")
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("🏭 GÉNÉRATEUR DE DATASET MOVEIT V2")
    print("=" * 70)
    print(f"📊 Trajectoires: {args.num_traj}")
    print(f"📁 Output: {args.output}")
    print("=" * 70 + "\n")
    
    input("Appuyez sur ENTRÉE pour continuer...")
    
    rclpy.init()
    
    node = MoveItDatasetGeneratorV2(
        num_trajectories=args.num_traj,
        output_file=args.output
    )
    
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
