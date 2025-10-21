#!/usr/bin/env python3
"""
Script de test pour vérifier la communication Isaac Lab <-> MoveIt2
Envoie des commandes simples au robot pour tester le bridge
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import numpy as np


class MoveItTestClient(Node):
    """Client de test pour envoyer des trajectoires simples au robot"""
    
    def __init__(self):
        super().__init__('moveit_test_client')
        
        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory'
        )
        
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        self.get_logger().info('Client de test MoveIt2 initialisé')
    
    def send_trajectory(self, positions_list, durations_list):
        """
        Envoie une trajectoire au robot
        
        Args:
            positions_list: Liste de configurations articulaires (chaque élément = [6 angles])
            durations_list: Liste de durées (en secondes) pour chaque point
        """
        # Attendre que l'action server soit disponible
        self.get_logger().info('Attente du serveur d\'action...')
        self._action_client.wait_for_server()
        
        # Créer la trajectoire
        goal_msg = FollowJointTrajectory.Goal()
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names
        
        cumulative_time = 0.0
        for positions, duration in zip(positions_list, durations_list):
            point = JointTrajectoryPoint()
            point.positions = positions
            
            cumulative_time += duration
            point.time_from_start = Duration(sec=int(cumulative_time), 
                                            nanosec=int((cumulative_time % 1) * 1e9))
            
            trajectory.points.append(point)
        
        goal_msg.trajectory = trajectory
        
        # Envoyer le goal
        self.get_logger().info(f'Envoi de {len(positions_list)} points de trajectoire...')
        for i, (pos, dur) in enumerate(zip(positions_list, durations_list)):
            angles_deg = np.degrees(pos)
            self.get_logger().info(
                f'  Point {i+1}: [{angles_deg[0]:.1f}°, {angles_deg[1]:.1f}°, {angles_deg[2]:.1f}°, '
                f'{angles_deg[3]:.1f}°, {angles_deg[4]:.1f}°, {angles_deg[5]:.1f}°], durée={dur:.1f}s'
            )
        
        future = self._action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)
        
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('❌ Goal rejeté!')
            return False
        
        self.get_logger().info('✓ Goal accepté, attente de l\'exécution...')
        
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        
        result = result_future.result().result
        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().info('✅ Trajectoire exécutée avec succès!')
            return True
        else:
            self.get_logger().error(f'❌ Erreur: code {result.error_code}')
            return False


def main(args=None):
    rclpy.init(args=args)
    
    client = MoveItTestClient()
    
    # Test 1: Mouvement simple - Position home
    client.get_logger().info('=== TEST 1: Position Home ===')
    positions = [
        [0.0, -1.57, 1.57, -1.57, -1.57, 0.0],  # Position neutre
    ]
    durations = [3.0]
    
    success = client.send_trajectory(positions, durations)
    
    if success:
        import time
        time.sleep(2.0)
        
        # Test 2: Mouvement en "vague"
        client.get_logger().info('\n=== TEST 2: Mouvement en vague ===')
        positions = [
            [0.5, -1.57, 1.57, -1.57, -1.57, 0.0],   # Pan à droite
            [-0.5, -1.57, 1.57, -1.57, -1.57, 0.0],  # Pan à gauche
            [0.0, -1.57, 1.57, -1.57, -1.57, 0.0],   # Retour centre
        ]
        durations = [2.0, 2.0, 2.0]
        
        success = client.send_trajectory(positions, durations)
    
    if success:
        client.get_logger().info('\n✅ Tous les tests réussis!')
    else:
        client.get_logger().error('\n❌ Tests échoués')
    
    client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
