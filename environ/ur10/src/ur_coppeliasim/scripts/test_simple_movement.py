#!/usr/bin/env python3
"""
Script de test simple pour envoyer une trajectoire au robot UR10 via CoppeliaSim
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import math


class SimpleTrajectoryTest(Node):
    def __init__(self):
        super().__init__('simple_trajectory_test')
        
        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory'
        )
        
        self.get_logger().info('En attente du serveur d\'action...')
        self._action_client.wait_for_server()
        self.get_logger().info('✓ Serveur d\'action connecté!')
    
    def send_test_trajectory(self):
        """Envoie une trajectoire de test simple"""
        
        # Noms des joints (dans l'ordre ROS2)
        joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        # Créer une trajectoire avec plusieurs points
        trajectory = JointTrajectory()
        trajectory.joint_names = joint_names
        
        # Point 1 : Position initiale (home)
        point1 = JointTrajectoryPoint()
        point1.positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        point1.time_from_start = Duration(sec=2, nanosec=0)
        
        # Point 2 : Bouger le joint 3 (elbow) de 20°
        point2 = JointTrajectoryPoint()
        point2.positions = [0.0, 0.0, math.radians(20), 0.0, 0.0, 0.0]
        point2.time_from_start = Duration(sec=5, nanosec=0)
        
        # Point 3 : Bouger aussi le joint 5 (wrist_2) de 20°
        point3 = JointTrajectoryPoint()
        point3.positions = [0.0, 0.0, math.radians(20), 0.0, math.radians(20), 0.0]
        point3.time_from_start = Duration(sec=8, nanosec=0)
        
        # Point 4 : Retour à la position initiale
        point4 = JointTrajectoryPoint()
        point4.positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        point4.time_from_start = Duration(sec=12, nanosec=0)
        
        trajectory.points = [point1, point2, point3, point4]
        
        # Créer le goal
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = trajectory
        
        self.get_logger().info('📤 Envoi de la trajectoire...')
        self.get_logger().info(f'   - {len(trajectory.points)} points')
        self.get_logger().info(f'   - Durée totale: 12 secondes')
        
        # Envoyer le goal
        send_goal_future = self._action_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)
    
    def goal_response_callback(self, future):
        """Callback quand le goal est accepté/rejeté"""
        goal_handle = future.result()
        
        if not goal_handle.accepted:
            self.get_logger().error('❌ Goal rejeté par le serveur')
            return
        
        self.get_logger().info('✓ Goal accepté! Le robot devrait bouger dans CoppeliaSim...')
        
        # Attendre le résultat
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)
    
    def get_result_callback(self, future):
        """Callback quand le mouvement est terminé"""
        result = future.result().result
        
        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().info('✅ Trajectoire exécutée avec succès!')
        else:
            self.get_logger().error(f'❌ Erreur: code {result.error_code}')
        
        # Arrêter le node
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    
    node = SimpleTrajectoryTest()
    
    print("\n" + "="*60)
    print("TEST DE TRAJECTOIRE - UR10 via CoppeliaSim")
    print("="*60)
    print("Le robot va:")
    print("  1. Se mettre en position home (0°)")
    print("  2. Bouger le coude (elbow) à 20°")
    print("  3. Bouger aussi le poignet (wrist_2) à 20°")
    print("  4. Retourner à la position home")
    print("="*60)
    print("\n⏳ Envoi de la trajectoire...\n")
    
    # Envoyer la trajectoire de test
    node.send_test_trajectory()
    
    # Spin pour recevoir les callbacks
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()
