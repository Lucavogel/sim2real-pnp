#!/usr/bin/env python3
"""
Script de test pour envoyer des commandes au robot via MoveIt
"""

import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetPositionIK
from geometry_msgs.msg import PoseStamped
import sys


class TestMovement(Node):
    """Node de test pour mouvements du robot"""
    
    def __init__(self):
        super().__init__('test_movement')
        self.get_logger().info('Node de test des mouvements')
    
    def test_simple_pose(self):
        """Teste un mouvement vers une pose simple"""
        self.get_logger().info('Test mouvement simple...')
        
        # TODO: Implémenter avec MoveGroupInterface
        # Pour l'instant, juste un placeholder
        
        self.get_logger().info('Test terminé')


def main(args=None):
    rclpy.init(args=args)
    
    node = TestMovement()
    
    try:
        node.test_simple_pose()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
