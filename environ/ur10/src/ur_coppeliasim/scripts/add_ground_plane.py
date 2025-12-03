#!/usr/bin/env python3
"""
Script pour ajouter automatiquement un plan de sol dans la scène MoveIt
pour éviter que le robot planifie des trajectoires passant par le sol.
"""

import rclpy
from rclpy.node import Node
from moveit_msgs.msg import CollisionObject, PlanningScene
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
import time


class GroundPlanePublisher(Node):
    def __init__(self):
        super().__init__('ground_plane_publisher')
        
        # Publisher pour la scène de planification
        self.scene_pub = self.create_publisher(
            PlanningScene,
            '/planning_scene',
            10
        )
        
        # Attendre que le publisher soit prêt
        time.sleep(1.0)
        
        # Publier le plan de sol immédiatement
        self.publish_ground_plane()
        
        # Timer pour republier périodiquement (toutes les 2 secondes)
        self.create_timer(5.0, self.publish_ground_plane)
        
        self.get_logger().info('✅ Plan de sol publié (republication toutes les 2s)')
        
    def publish_ground_plane(self):
        """Crée et publie un objet de collision représentant le sol"""
        
        # Créer l'objet de collision
        collision_object = CollisionObject()
        collision_object.header.frame_id = 'world'
        collision_object.id = 'ground_plane'
        
        # Définir la forme (une boîte plate)
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [5.0, 5.0, 0.01]  # 3m x 5m x 1cm
        
        # Position du sol (juste en dessous de z=0)
        pose = Pose()
        pose.position.x = float(0.0)
        pose.position.y = float(0.0)
        pose.position.z = float(-0.01)  # 5cm en dessous du plan z=0 (float)
        pose.orientation.w = 1.0
        
        # Ajouter à l'objet
        collision_object.primitives.append(box)
        collision_object.primitive_poses.append(pose)
        collision_object.operation = CollisionObject.ADD
        
        # Créer le message de scène
        scene = PlanningScene()
        scene.world.collision_objects.append(collision_object)
        scene.is_diff = True  # C'est une mise à jour incrémentale
        
        # Publier
        self.scene_pub.publish(scene)

def main(args=None):
    rclpy.init(args=args)
    node = GroundPlanePublisher()
    
    # Garder le node actif pour continuer à publier
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Arrêt demandé...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
