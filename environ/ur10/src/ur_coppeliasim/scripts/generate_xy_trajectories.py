#!/usr/bin/env python3
"""
Script MoveIt pour générer des trajectoires X-Y pures (Z et orientation fixes)
Basé sur apriltag_auto_mover.py mais génère et sauvegarde toutes les trajectoires

⚠️ IMPORTANT: Les AprilTags sont dans IsaacLab!

Usage:
    1. Terminal 1 - Lancer IsaacLab avec AprilTags:
       cd ~/workspace/sim2real-pnp/environ/my_env/scripts
       ./isaaclab.sh -p Ur10_moveit_Apriltag.py
       (Le robot doit être visible et les tags détectés)
    
    2. Terminal 2 - Lancer MoveIt:
       cd ~/workspace/sim2real-pnp/environ/ur10
       source install/setup.bash
       ros2 launch ur_coppeliasim ur_isaaclab_moveit.launch.py
    
    3. Terminal 3 - Générer trajectoires:
       cd ~/workspace/sim2real-pnp/environ/ur10
       python3 src/ur_coppeliasim/scripts/generate_xy_trajectories.py --num-lines 100 --output trajectories/
       
    Les tags doivent être visibles dans TF:
       ros2 run tf2_ros tf2_echo world Tag0
"""

import rclpy
from rclpy.node import Node
import tf2_ros
import numpy as np
import os
import argparse
from pathlib import Path

from geometry_msgs.msg import Pose
from moveit_msgs.srv import GetCartesianPath
from sensor_msgs.msg import JointState


class TrajectoryGenerator(Node):
    def __init__(self, num_trajectories=100, output_dir="trajectories"):
        super().__init__("trajectory_generator")
        
        self.num_trajectories = num_trajectories
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Paramètres
        self.declare_parameter("group_name", "ur_manipulator")
        self.declare_parameter("z_offset", 0.20)  # Hauteur stylo au-dessus du tag (5cm au lieu de 50cm!)
        self.declare_parameter("cartesian_steps", 50)  # Moins de points pour être plus rapide
        
        self.group_name = self.get_parameter("group_name").value
        self.z_offset = self.get_parameter("z_offset").value
        self.cartesian_steps = self.get_parameter("cartesian_steps").value
        
        # TF pour lire positions des AprilTags
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Service Cartesian Path
        self.cartesian_client = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        
        # Executor pour spin pendant les appels asynchrones
        self.executor = None
        
        # Subscriber pour lire la position actuelle du robot
        self.current_joint_state = None
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        # Orientation fixée (stylo vertical vers le bas)
        # Essayer une orientation plus simple (identité = pas de rotation)
        self.fixed_orientation = Pose().orientation
        self.fixed_orientation.x = 1.0
        self.fixed_orientation.y = 0.0
        self.fixed_orientation.z = 0.0
        self.fixed_orientation.w = 0.0  # Quaternion identité
        
        self.get_logger().info("🏭 Générateur de trajectoires X-Y initialisé")
        self.get_logger().info(f"   Output: {self.output_dir}")
        self.get_logger().info(f"   Nombre: {num_trajectories}")
    
    def joint_state_callback(self, msg):
        """Callback pour recevoir l'état actuel des joints"""
        self.current_joint_state = msg
    
    def set_executor(self, executor):
        """Définir l'executor pour les appels asynchrones"""
        self.executor = executor
    
    def get_current_ee_z_position(self):
        """Récupère la position Z actuelle de l'effecteur via TF"""
        # Essayer différents frames possibles pour l'effecteur
        possible_frames = ["tool0", "wrist_3_link", "ee_link", "flange"]
        
        for frame_name in possible_frames:
            try:
                now = rclpy.time.Time()
                tf = self.tf_buffer.lookup_transform(
                    "world",
                    frame_name,
                    now,
                    timeout=rclpy.duration.Duration(seconds=1.0)
                )
                z_pos = tf.transform.translation.z
                self.get_logger().info(f"📍 Position Z actuelle du robot ({frame_name}): {z_pos:.3f}m")
                return z_pos
            except Exception as e:
                self.get_logger().debug(f"Frame '{frame_name}' non trouvé: {e}")
                continue
        
        self.get_logger().error(f"❌ Aucun frame d'effecteur trouvé parmi: {possible_frames}")
        self.get_logger().error("💡 Vérifiez les frames disponibles avec: ros2 run tf2_ros tf2_echo world <frame>")
        return None
    
    def get_tag_positions(self, use_current_z=True):
        """Récupère les positions des AprilTags via TF (depuis IsaacLab!)
        
        Args:
            use_current_z: Si True, utilise la position Z actuelle du robot au lieu de z_offset
        """
        tag_positions = []
        
        self.get_logger().info("📡 Lecture positions AprilTags via TF...")
        
        # Lire position Z actuelle du robot si demandé
        z_to_use = None
        if use_current_z:
            self.get_logger().info("🔍 Tentative de lecture position Z actuelle du robot...")
            z_to_use = self.get_current_ee_z_position()
            if z_to_use is None:
                self.get_logger().warn(f"⚠️  Impossible de lire Z du robot, utilisation de z_offset={self.z_offset}m")
                z_to_use = None
            else:
                self.get_logger().info(f"✅ Utilisation Z du robot: {z_to_use:.3f}m (au lieu de z_offset={self.z_offset}m)")
        
        for i in range(2):  # Tag0 et Tag1
            tag_name = f"Tag{i}"
            try:
                # Utiliser le timestamp actuel
                now = rclpy.time.Time()
                tf = self.tf_buffer.lookup_transform(
                    "world", 
                    tag_name, 
                    now,
                    timeout=rclpy.duration.Duration(seconds=1.0)
                )
                
                # Utiliser Z du robot actuel OU z_offset par défaut
                if z_to_use is not None:
                    z_value = z_to_use
                    self.get_logger().info(f"   → Utilisation Z du robot pour {tag_name}: {z_value:.3f}m")
                else:
                    z_value = tf.transform.translation.z + self.z_offset
                    self.get_logger().info(f"   → Tag {tag_name} Z: {tf.transform.translation.z:.3f}m + offset {self.z_offset:.3f}m = {z_value:.3f}m")
                
                pos = (
                    tf.transform.translation.x,
                    tf.transform.translation.y,
                    z_value  # Z fixe = position actuelle du robot
                )
                tag_positions.append(pos)
                self.get_logger().info(f"✅ {tag_name}: X={pos[0]:.3f}, Y={pos[1]:.3f}, Z={pos[2]:.3f}m")
            except Exception as e:
                self.get_logger().error(f"❌ Impossible de lire {tag_name}: {e}")
                return None
        
        return tag_positions
    
    def generate_xy_lines(self, tag_positions):
        """Génère des lignes de Tag0 vers Tag1 avec variations autour"""
        if len(tag_positions) < 2:
            return []
        
        pos0 = np.array(tag_positions[0])
        pos1 = np.array(tag_positions[1])
        
        # Calculer bounding box ÉLARGIE
        x_min = min(pos0[0], pos1[0])
        x_max = max(pos0[0], pos1[0])
        y_min = min(pos0[1], pos1[1])
        y_max = max(pos0[1], pos1[1])
        z_fixed = pos0[2]  # Z fixe
        
        # Élargir la bounding box de 10cm de chaque côté
        margin = 0.10  # 10 cm
        x_min -= margin
        x_max += margin
        y_min -= margin
        y_max += margin
        
        self.get_logger().info(f"📏 Zone de travail ÉLARGIE:")
        self.get_logger().info(f"   X: [{x_min:.3f}, {x_max:.3f}] (±{margin*100:.0f}cm)")
        self.get_logger().info(f"   Y: [{y_min:.3f}, {y_max:.3f}] (±{margin*100:.0f}cm)")
        self.get_logger().info(f"   Z: {z_fixed:.3f} (fixe)")
        self.get_logger().info("")
        self.get_logger().info(f"📍 Tag0: ({pos0[0]:.3f}, {pos0[1]:.3f})")
        self.get_logger().info(f"📍 Tag1: ({pos1[0]:.3f}, {pos1[1]:.3f})")
        
        lines = []
        
        for i in range(self.num_trajectories):
            # TOUJOURS partir de Tag0 et aller vers Tag1
            # Mais avec petites variations aléatoires autour des tags
            
            # Variation autour de Tag0 (départ)
            offset_start = 0.03  # ±3cm autour de Tag0
            x_start = pos0[0] + np.random.uniform(-offset_start, offset_start)
            y_start = pos0[1] + np.random.uniform(-offset_start, offset_start)
            
            # Variation autour de Tag1 (arrivée)
            offset_end = 0.03  # ±3cm autour de Tag1
            x_end = pos1[0] + np.random.uniform(-offset_end, offset_end)
            y_end = pos1[1] + np.random.uniform(-offset_end, offset_end)
            
            # S'assurer qu'on reste dans la bounding box
            x_start = np.clip(x_start, x_min, x_max)
            y_start = np.clip(y_start, y_min, y_max)
            x_end = np.clip(x_end, x_min, x_max)
            y_end = np.clip(y_end, y_min, y_max)
            
            start = (x_start, y_start, z_fixed)
            end = (x_end, y_end, z_fixed)
            
            lines.append((start, end))
        
        return lines
    
    def compute_cartesian_trajectory(self, start_pos, end_pos):
        """Calcule une trajectoire cartésienne X-Y avec MoveIt"""
        # Générer waypoints intermédiaires
        waypoints = []
        for i in range(self.cartesian_steps + 1):
            t = i / self.cartesian_steps
            waypoint = Pose()
            waypoint.position.x = start_pos[0] + t * (end_pos[0] - start_pos[0])
            waypoint.position.y = start_pos[1] + t * (end_pos[1] - start_pos[1])
            waypoint.position.z = start_pos[2]  # Z fixe
            waypoint.orientation = self.fixed_orientation
            waypoints.append(waypoint)
        
        # Appeler service MoveIt (ASYNCHRONE avec timeout)
        request = GetCartesianPath.Request()
        request.header.frame_id = "world"
        request.group_name = self.group_name
        request.link_name = "tool0"
        request.waypoints = waypoints
        request.max_step = 0.01  # 1cm de résolution
        request.jump_threshold = 0.0
        request.avoid_collisions = True
        
        # Appel ASYNCHRONE
        future = self.cartesian_client.call_async(request)
        
        # Attendre avec timeout ET spin de l'executor
        import time
        timeout = 15.0  # 15 secondes max
        start_time = time.time()
        
        while not future.done():
            if time.time() - start_time > timeout:
                self.get_logger().error(f"   ⏱️ Timeout! MoveIt ne répond pas après {timeout}s")
                return None
            
            # CRITIQUE: Faire tourner l'executor pour recevoir la réponse
            if self.executor:
                self.executor.spin_once(timeout_sec=0.1)
            else:
                time.sleep(0.1)
        
        try:
            response = future.result()
            return response
        except Exception as e:
            self.get_logger().error(f"   ❌ Erreur service: {e}")
            return None
    
    def save_trajectory(self, trajectory, filename):
        """Sauvegarde une trajectoire en format numpy (angles joints)"""
        if trajectory is None or len(trajectory.joint_trajectory.points) == 0:
            return False
        
        # Extraire les angles des joints
        joint_angles = []
        for point in trajectory.joint_trajectory.points:
            joint_angles.append(list(point.positions))
        
        # Sauvegarder en numpy
        joint_array = np.array(joint_angles)
        np.save(self.output_dir / filename, joint_array)
        
        return True
    
    def generate_all_trajectories(self):
        """Génère toutes les trajectoires et les sauvegarde"""
        self.get_logger().info("🚀 Début génération des trajectoires...")
        
        # Attendre service MoveIt
        self.get_logger().info("🔍 Recherche service MoveIt /compute_cartesian_path...")
        if not self.cartesian_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("❌ Service /compute_cartesian_path non disponible!")
            self.get_logger().error("   Lancer: ros2 launch ur_coppeliasim ur_isaaclab_moveit.launch.py")
            return
        self.get_logger().info("✅ Service MoveIt trouvé")
        
        # Lire positions AprilTags (depuis IsaacLab via TF)
        # UTILISER LA POSITION Z ACTUELLE DU ROBOT pour éviter les mouvements en Z
        self.get_logger().info("")
        tag_positions = self.get_tag_positions(use_current_z=True)
        if tag_positions is None:
            self.get_logger().error("")
            self.get_logger().error("❌ ÉCHEC: Impossible de lire les AprilTags")
            return
        
        # Générer lignes X-Y
        self.get_logger().info("📐 Génération lignes X-Y...")
        lines = self.generate_xy_lines(tag_positions)
        
        # Calculer trajectoires MoveIt
        success_count = 0
        failed_count = 0
        
        for i, (start, end) in enumerate(lines):
            self.get_logger().info(f"🔄 [{i+1}/{len(lines)}] Calcul trajectoire...")
            self.get_logger().info(f"   Start: ({start[0]:.3f}, {start[1]:.3f})")
            self.get_logger().info(f"   End:   ({end[0]:.3f}, {end[1]:.3f})")
            
            try:
                response = self.compute_cartesian_trajectory(start, end)
                
                if response is None:
                    failed_count += 1
                    self.get_logger().warn(f"   ❌ Pas de réponse de MoveIt")
                    continue
                
                fraction = response.fraction
                self.get_logger().info(f"   Couverture: {fraction*100:.1f}%")
                
                if fraction > 0.90:  # Réduire à 90% au lieu de 95% pour plus de tolérance
                    filename = f"traj_{i:03d}.npy"
                    if self.save_trajectory(response.solution, filename):
                        success_count += 1
                        num_points = len(response.solution.joint_trajectory.points)
                        self.get_logger().info(f"   ✅ Sauvegardé: {filename} ({num_points} points)")
                    else:
                        failed_count += 1
                        self.get_logger().warn(f"   ⚠️ Échec sauvegarde")
                else:
                    failed_count += 1
                    self.get_logger().warn(f"   ❌ Chemin incomplet ({fraction*100:.1f}%)")
                
            except Exception as e:
                failed_count += 1
                self.get_logger().error(f"   ❌ Erreur: {e}")
        
        # Résumé
        self.get_logger().info("")
        self.get_logger().info("=" * 60)
        self.get_logger().info("✅ GÉNÉRATION TERMINÉE")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"✅ Succès: {success_count}/{len(lines)}")
        self.get_logger().info(f"❌ Échecs: {failed_count}/{len(lines)}")
        self.get_logger().info(f"📁 Dossier: {self.output_dir.absolute()}")
        self.get_logger().info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Génère des trajectoires X-Y avec MoveIt",
        epilog="""
WORKFLOW COMPLET:
  1. Terminal 1 - IsaacLab:
     cd ~/workspace/sim2real-pnp/environ/my_env/scripts
     ./isaaclab.sh -p Ur10_moveit_Apriltag.py
     
  2. Terminal 2 - MoveIt:
     cd ~/workspace/sim2real-pnp/environ/ur10
     source install/setup.bash
     ros2 launch ur_coppeliasim ur_isaaclab_moveit.launch.py
     
  3. Terminal 3 - Ce script:
     python3 src/ur_coppeliasim/scripts/generate_xy_trajectories.py --num-lines 100
        """
    )
    parser.add_argument("--num-lines", type=int, default=100, 
                        help="Nombre de trajectoires à générer")
    parser.add_argument("--output", type=str, default="trajectories",
                        help="Dossier de sortie")
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("🏭 GÉNÉRATEUR DE TRAJECTOIRES X-Y")
    print("=" * 70)
    print(f"📊 Trajectoires à générer: {args.num_lines}")
    print(f"📁 Dossier sortie: {args.output}")
    print("")
    print("⚠️  VÉRIFIEZ AVANT DE CONTINUER:")
    print("   ✓ IsaacLab tourne avec Ur10_moveit_Apriltag.py")
    print("   ✓ AprilTags visibles (Tag0, Tag1)")
    print("   ✓ MoveIt lancé (ur_isaaclab_moveit.launch.py)")
    print("")
    print("Test rapide dans un autre terminal:")
    print("   ros2 topic list | grep joint_states")
    print("   ros2 topic echo /tf | grep -A3 Tag0")
    print("=" * 70 + "\n")
    
    input("Appuyez sur ENTRÉE pour continuer...")
    
    rclpy.init()
    
    node = TrajectoryGenerator(
        num_trajectories=args.num_lines,
        output_dir=args.output
    )
    
    # CRITIQUE: Faire tourner le node pour remplir le TF buffer
    print("\n⏳ Remplissage du TF buffer (spin pendant 5 secondes)...")
    import time
    from rclpy.executors import SingleThreadedExecutor
    
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    
    # Passer l'executor au node pour les appels asynchrones
    node.set_executor(executor)
    
    # Spin pendant 5 secondes pour que le TF buffer se remplisse
    start_time = time.time()
    while time.time() - start_time < 5.0:
        executor.spin_once(timeout_sec=0.1)
    
    print("✅ TF buffer rempli")
    
    # Vérifier que les TF sont disponibles
    print("\n🔍 Test lecture TF...")
    try:
        # Test avec la caméra
        test_tf = node.tf_buffer.lookup_transform(
            "world", "sim_camera", 
            rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=1.0)
        )
        print(f"✅ Frame 'world' existe (world → sim_camera ok)")
    except Exception as e:
        print(f"⚠️  Avertissement: {e}")
    
    try:
        # Test avec Tag0
        test_tf = node.tf_buffer.lookup_transform(
            "world", "Tag0", 
            rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=1.0)
        )
        print(f"✅ Tag0 disponible")
    except Exception as e:
        print(f"❌ Tag0 non disponible: {e}")
        print("\n💡 Vérifiez que IsaacLab publie bien les tags:")
        print("   ros2 topic echo /tf | grep -A5 Tag0")
        executor.shutdown()
        rclpy.shutdown()
        return
    
    # Générer toutes les trajectoires
    print("\n" + "=" * 70)
    node.generate_all_trajectories()
    
    executor.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
