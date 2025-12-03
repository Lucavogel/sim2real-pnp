#!/usr/bin/env python3
"""
Bridge ROS2 <-> CoppeliaSim pour robot UR
Ce script fait le lien entre MoveIt et CoppeliaSim via les topics ROS2
"""

import sys
import os
from pathlib import Path
import numpy as np
import time as time_module

# Ajouter le chemin de CoppeliaSim au PYTHONPATH
coppeliasim_paths = [
    "/home/ajin/Documents/software/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu22_04/programming/legacyRemoteApi/remoteApiBindings/python/python",
    "/home/ajin/Documents/software/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu22_04/programming/legacyRemoteApi/remoteApiBindings/lib/lib/Ubuntu22_04",
    str(Path.home() / "Desktop/UR-10 TP"),  # Chemin alternatif
]

for path in coppeliasim_paths:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from control_msgs.action import FollowJointTrajectory
from std_msgs.msg import Header

try:
    import sim  # Legacy Remote API (port 19999)
    COPPELIASIM_AVAILABLE = True
except ImportError:
    print("=" * 60)
    print("WARNING: sim module (Legacy Remote API) not found")
    print("Vérifiez que CoppeliaSim est installé et que le chemin est correct")
    print("Chemins recherchés:")
    for p in coppeliasim_paths:
        print(f"  - {p}")
    print("=" * 60)
    COPPELIASIM_AVAILABLE = False


class CoppeliaSimBridge(Node):
    """Bridge entre ROS2 et CoppeliaSim pour contrôle du robot UR"""
    
    def __init__(self):
        super().__init__('coppeliasim_bridge')
        
        # Paramètres
        self.declare_parameter('ur_type', 'ur10')
        self.declare_parameter('joint_names', [
            'elbow_joint',
            'shoulder_lift_joint',
            'shoulder_pan_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ])
        # MAPPING ADAPTÉ: CoppeliaSim a joint1 et joint3 inversés physiquement
        # ROS2 joint_names → CoppeliaSim joint names (adapté au modèle CoppeliaSim)
        self.declare_parameter('coppeliasim_joint_names', [
            'UR10_joint3',  # shoulder_pan_joint → joint3 dans CoppeliaSim
            'UR10_joint2',  # shoulder_lift_joint → joint2 dans CoppeliaSim
            'UR10_joint1',  # elbow_joint → joint1 dans CoppeliaSim
            'UR10_joint4',  # wrist_1_joint → joint4 dans CoppeliaSim
            'UR10_joint5',  # wrist_2_joint → joint5 dans CoppeliaSim
            'UR10_joint6'   # wrist_3_joint → joint6 dans CoppeliaSim
        ])
        self.declare_parameter('update_rate', 50.0)  # Hz
        self.declare_parameter('coppeliasim_host', '127.0.0.1')
        self.declare_parameter('coppeliasim_port', 19999)
        
        self.ur_type = self.get_parameter('ur_type').value
        self.joint_names = self.get_parameter('joint_names').value
        self.coppeliasim_joint_names = self.get_parameter('coppeliasim_joint_names').value
        self.update_rate = self.get_parameter('update_rate').value
        self.host = self.get_parameter('coppeliasim_host').value
        self.port = self.get_parameter('coppeliasim_port').value
        
        # IMPORTANT: Offsets ET inversions pour correspondre à la table DH modifiée
        # Votre code Python (UR10 classique) utilise:
        #   - q2 - pi/2 (shoulder_lift avec offset)
        #   - q4 - pi/2 (wrist_1 avec offset)
        #   - a2 et a3 NÉGATIFS → joints 2,3,4 potentiellement inversés
        # ROS2/MoveIt utilise la convention standard
        self.joint_offsets = np.array([
            0.0,           # shoulder_pan_joint: pas d'offset
            -np.pi/2,      # shoulder_lift_joint: offset -pi/2
            0.0,           # elbow_joint: pas d'offset
            -np.pi/2,      # wrist_1_joint: offset -pi/2
            0.0,           # wrist_2_joint: pas d'offset
            0.0            # wrist_3_joint: pas d'offset
        ])
        
        # Directions des moteurs (1 = normal, -1 = inversé)
        # À tester si le robot bouge dans le mauvais sens
        self.joint_directions = np.array([
            1.0,   # shoulder_pan_joint
            1.0,  # shoulder_lift_joint (potentiellement inversé à cause de a2 négatif)
            1.0,  # elbow_joint (potentiellement inversé à cause de a2/a3 négatifs)
            1.0,  # wrist_1_joint (potentiellement inversé à cause de a3 négatif)
            1.0,   # wrist_2_joint
            1.0    # wrist_3_joint
        ])
        
        # Limites de sécurité des joints (DÉSACTIVÉES - mouvement libre)
        self.joint_limit_deg = 360.0  # Limite très large (pas de restriction)
        self.joint_limit_rad = np.radians(self.joint_limit_deg)
        self.get_logger().info(f'Limites joints: ±{self.joint_limit_deg}° (±{self.joint_limit_rad:.2f} rad) - MOUVEMENT LIBRE')
        
        self.get_logger().info(f'Bridge CoppeliaSim pour robot {self.ur_type}')
        self.get_logger().info(f'Offsets DH appliqués: joints 2 et 4 = -π/2')
        self.get_logger().info(f'Inversions moteurs: {self.joint_directions}')
        
        # État des joints
        self.joint_positions = [0.0] * len(self.joint_names)
        self.joint_velocities = [0.0] * len(self.joint_names)
        self.joint_efforts = [0.0] * len(self.joint_names)
        
        # Connection à CoppeliaSim (Legacy API)
        self.clientID = -1
        self.joint_handles = []
        self.connected = False
        
        if COPPELIASIM_AVAILABLE:
            self.connect_coppeliasim()
        
        # Callback group pour exécution parallèle
        self.callback_group = ReentrantCallbackGroup()
        
        # Publisher pour /joint_states
        self.joint_state_pub = self.create_publisher(
            JointState,
            'joint_states',
            10
        )
        
        # Action server pour trajectory controller
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory',
            self.execute_trajectory_callback,
            callback_group=self.callback_group
        )
        
        # Timer pour publier les joint states
        self.create_timer(
            2.0 / self.update_rate,
            self.publish_joint_states,
            callback_group=self.callback_group
        )
        
        self.get_logger().info('Bridge CoppeliaSim prêt!')
    
    def connect_coppeliasim(self):
        """Connecte à CoppeliaSim via Legacy Remote API (port 19999)"""
        try:
            self.get_logger().info(f'Connexion à CoppeliaSim sur {self.host}:{self.port}...')
            
            # Fermer toutes les connexions existantes
            sim.simxFinish(-1)
            
            # Connexion à CoppeliaSim
            self.clientID = sim.simxStart(self.host, self.port, True, True, 5000, 5)
            
            if self.clientID == -1:
                self.get_logger().error('Impossible de se connecter à CoppeliaSim')
                self.connected = False
                return
            
            self.get_logger().info('Connecté au serveur CoppeliaSim!')
            
            # Récupérer les handles des joints (noms CoppeliaSim style UR10_joint1, etc.)
            for i, joint_name in enumerate(self.coppeliasim_joint_names):
                res, handle = sim.simxGetObjectHandle(
                    self.clientID, 
                    joint_name, 
                    sim.simx_opmode_blocking
                )
                
                if res == sim.simx_return_ok:
                    self.joint_handles.append(handle)
                    self.get_logger().info(f'  ✓ {joint_name} (handle: {handle})')
                    
                    # Initialiser le streaming pour lecture de position
                    sim.simxGetJointPosition(
                        self.clientID, 
                        handle, 
                        sim.simx_opmode_streaming
                    )
                else:
                    self.get_logger().error(f'  ✗ {joint_name} NON TROUVÉ (code: {res})')
                    self.joint_handles.append(None)
            
            self.connected = len([h for h in self.joint_handles if h is not None]) > 0
            
            if self.connected:
                self.get_logger().info(f'✓ Bridge prêt avec {len([h for h in self.joint_handles if h is not None])}/6 joints')
            else:
                self.get_logger().error('Aucun joint trouvé dans CoppeliaSim')
                
        except Exception as e:
            self.get_logger().error(f'Erreur connexion CoppeliaSim: {e}')
            self.connected = False
    
    def publish_joint_states(self):
        """Publie l'état actuel des joints depuis CoppeliaSim"""
        if self.connected and self.clientID != -1:
            # Lire les positions depuis CoppeliaSim (mode buffer pour vitesse)
            for i, handle in enumerate(self.joint_handles):
                if handle is not None:
                    try:
                        res, pos = sim.simxGetJointPosition(
                            self.clientID, 
                            handle, 
                            sim.simx_opmode_buffer
                        )
                        if res == sim.simx_return_ok:
                            # APPLIQUER L'OFFSET ET LA DIRECTION: CoppeliaSim -> ROS2
                            # CoppeliaSim utilise votre convention (q2-pi/2, q4-pi/2)
                            # ROS2 attend la convention standard
                            self.joint_positions[i] = (pos * self.joint_directions[i]) + self.joint_offsets[i]
                    except Exception as e:
                        self.get_logger().warn(f'Erreur lecture joint {i}: {e}', throttle_duration_sec=5.0)
        
        # Publier JointState
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.joint_positions
        msg.velocity = self.joint_velocities
        msg.effort = self.joint_efforts
        
        self.joint_state_pub.publish(msg)
    
    def execute_trajectory_callback(self, goal_handle):
        """Exécute une trajectoire reçue de MoveIt"""
        self.get_logger().info('Trajectoire reçue!')
        
        trajectory = goal_handle.request.trajectory
        
        if not self.connected:
            self.get_logger().warn('Non connecté à CoppeliaSim - simulation mode')
            result = FollowJointTrajectory.Result()
            result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
            goal_handle.succeed()
            return result
        
        try:
            # Exécuter chaque point de la trajectoire
            # EXACTEMENT comme dans votre ancien code Python
            start_time = time_module.time()
            previous_time = 0.0
            
            self.get_logger().info(f'Début exécution de {len(trajectory.points)} points')
            
            for idx, point in enumerate(trajectory.points):
                # Obtenir les positions cibles
                target_positions = point.positions
                
                # VÉRIFICATION DES LIMITES DE SÉCURITÉ - DÉSACTIVÉE POUR MOUVEMENT LIBRE
                # positions_safe = True
                # for i, pos in enumerate(target_positions):
                #     if abs(pos) > self.joint_limit_rad:
                #         self.get_logger().error(
                #             f'⚠️ Joint {i+1} ({self.joint_names[i]}) dépasse la limite de sécurité: '
                #             f'{np.degrees(pos):.1f}° (limite: ±{self.joint_limit_deg}°)'
                #         )
                #         positions_safe = False
                
                # if not positions_safe:
                #     self.get_logger().error('❌ Trajectoire rejetée: angles hors limites de sécurité!')
                #     result = FollowJointTrajectory.Result()
                #     result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
                #     goal_handle.abort()
                #     return result
                
                # Calculer le temps à attendre depuis le point précédent
                time_from_start = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
                wait_duration = time_from_start - previous_time
                previous_time = time_from_start
                
                # Log avec angles en degrés
                angles_deg = [np.degrees(p) for p in target_positions]
                self.get_logger().info(f'Point {idx+1}/{len(trajectory.points)}: [{angles_deg[0]:.1f}°, {angles_deg[1]:.1f}°, {angles_deg[2]:.1f}°, {angles_deg[3]:.1f}°, {angles_deg[4]:.1f}°, {angles_deg[5]:.1f}°], attente={wait_duration:.2f}s')
                
                # Envoyer les positions à CoppeliaSim
                # COMME DANS VOTRE CODE: simxSetJointTargetPosition avec streaming
                for i, handle in enumerate(self.joint_handles):
                    if handle is not None and i < len(target_positions):
                        # APPLIQUER L'OFFSET ET LA DIRECTION INVERSE: ROS2 -> CoppeliaSim
                        # ROS2/MoveIt envoie en convention standard
                        # CoppeliaSim attend votre convention (q2-pi/2, q4-pi/2)
                        coppelia_position = (target_positions[i] - self.joint_offsets[i]) * self.joint_directions[i]
                        
                        ret = sim.simxSetJointTargetPosition(
                            self.clientID,
                            handle,
                            coppelia_position,
                            sim.simx_opmode_streaming  # STREAMING comme dans votre code!
                        )
                        if ret != sim.simx_return_ok and ret != sim.simx_return_novalue_flag:
                            self.get_logger().warn(f'Erreur envoi joint {i}: code {ret}')
                
                # IMPORTANT: Attendre comme dans votre code (time.sleep)
                if wait_duration > 0:
                    time_module.sleep(wait_duration)
            
            self.get_logger().info('✅ Trajectoire exécutée avec succès!')
            
            result = FollowJointTrajectory.Result()
            result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
            goal_handle.succeed()
            return result
            
        except Exception as e:
            self.get_logger().error(f'❌ ERREUR exécution trajectoire: {e}')
            import traceback
            self.get_logger().error(f'Traceback: {traceback.format_exc()}')
            result = FollowJointTrajectory.Result()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            goal_handle.abort()
            return result


def main(args=None):
    rclpy.init(args=args)
    
    bridge = CoppeliaSimBridge()
    
    executor = MultiThreadedExecutor()
    executor.add_node(bridge)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
