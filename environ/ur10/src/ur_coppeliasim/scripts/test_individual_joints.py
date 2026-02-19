#!/usr/bin/env python3
"""
Test individuel des joints pour identifier le mapping correct
Bouge chaque joint un par un pour voir lequel bouge réellement
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import numpy as np
import time

class JointTester(Node):
    def __init__(self):
        super().__init__('joint_tester')
        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory'
        )
        
    def test_joint(self, joint_index, angle_deg=20.0):
        """Test un seul joint à la fois"""
        self.get_logger().info(f'\n{"="*60}')
        self.get_logger().info(f'TEST: Bouger joint {joint_index+1} (index {joint_index}) de {angle_deg}°')
        self.get_logger().info(f'{"="*60}')
        
        goal_msg = FollowJointTrajectory.Goal()
        
        # Noms des joints ROS2
        goal_msg.trajectory.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        # Point 1: Position initiale (tous à 0)
        point1 = JointTrajectoryPoint()
        point1.positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        point1.time_from_start = Duration(sec=2, nanosec=0)
        
        # Point 2: Bouger UN SEUL joint
        point2 = JointTrajectoryPoint()
        positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        positions[joint_index] = np.radians(angle_deg)
        point2.positions = positions
        point2.time_from_start = Duration(sec=4, nanosec=0)
        
        # Point 3: Retour à 0
        point3 = JointTrajectoryPoint()
        point3.positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        point3.time_from_start = Duration(sec=6, nanosec=0)
        
        goal_msg.trajectory.points = [point1, point2, point3]
        
        self.get_logger().info(f'Envoi commande: joint {joint_index+1} → {angle_deg}°')
        self.get_logger().info(f'👀 REGARDEZ CoppeliaSim: quel joint bouge réellement?')
        
        self._action_client.wait_for_server()
        send_goal_future = self._action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        
        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error('❌ Goal rejetée!')
            return False
            
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        
        self.get_logger().info(f'✅ Test joint {joint_index+1} terminé\n')
        return True


def main():
    rclpy.init()
    tester = JointTester()
    
    print("\n" + "="*70)
    print("TEST DE MAPPING DES JOINTS")
    print("="*70)
    print("Ce script va bouger chaque joint individuellement.")
    print("REGARDEZ CoppeliaSim et notez quel joint physique bouge!")
    print("="*70 + "\n")
    
    input("Appuyez sur ENTRÉE pour commencer les tests...")
    
    joints_names = [
        "shoulder_pan_joint (rotation base)",
        "shoulder_lift_joint (épaule haut/bas)",
        "elbow_joint (coude)",
        "wrist_1_joint (poignet 1)",
        "wrist_2_joint (poignet 2)",
        "wrist_3_joint (poignet 3)"
    ]
    
    for i in range(6):
        print(f"\n\n🔍 Test {i+1}/6: {joints_names[i]}")
        print(f"   → Commande ROS2: joint index {i}")
        print(f"   → Attendu CoppeliaSim: UR10_joint{i+1}")
        print()
        
        tester.test_joint(i, angle_deg=30.0)
        
        if i < 5:
            response = input(f"\n✍️  Quel joint a RÉELLEMENT bougé dans CoppeliaSim? (1-6, ou 'q' pour quitter): ")
            if response.lower() == 'q':
                break
            print(f"   📝 Vous avez dit: UR10_joint{response}\n")
            time.sleep(1)
    
    print("\n" + "="*70)
    print("Tests terminés!")
    print("="*70)
    
    tester.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
