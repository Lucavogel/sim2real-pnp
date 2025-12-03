#!/usr/bin/env python3
"""
Script pour ajouter un obstacle carré devant le robot dans la scène MoveIt.
L'obstacle empêchera le robot de passer à travers lors de la planification.
"""

import rclpy
from rclpy.node import Node
from moveit_msgs.msg import CollisionObject, PlanningScene
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
import time


class ObstaclePublisher(Node):
    def __init__(self):
        super().__init__('obstacle_publisher')
        
        # Paramètres de l'obstacle 1
        self.declare_parameter('obstacle_x', 0.8)  # Distance devant le robot (m)
        self.declare_parameter('obstacle_y', 0.55)  # Position latérale (m)
        self.declare_parameter('obstacle_z', 0.3)  # Hauteur du centre (m)
        self.declare_parameter('obstacle_size_x', 1.5)  # Largeur dans l'axe X du robot (m)
        self.declare_parameter('obstacle_size_y', 0.1)  # LONGUEUR dans l'axe Y (perpendiculaire, longue!) (m)
        self.declare_parameter('obstacle_height', 0.8)  # Hauteur de l'obstacle (m)
        
        # Paramètres de l'obstacle 2
        self.declare_parameter('obstacle2_x', 0.6)  # Distance devant le robot (m)
        self.declare_parameter('obstacle2_y', -0.55)  # Position latérale (m)
        self.declare_parameter('obstacle2_z', 0.3)  # Hauteur du centre (m)
        self.declare_parameter('obstacle2_size_x', 1.5)  # Largeur dans l'axe X du robot (m)
        self.declare_parameter('obstacle2_size_y', 0.1)  # LONGUEUR dans l'axe Y (m)
        self.declare_parameter('obstacle2_height', 0.8)  # Hauteur de l'obstacle (m)
        
        # Paramètres de l'obstacle 3 (nouveau)
        self.declare_parameter('obstacle3_x', 0.6)  # Distance devant le robot (m)
        self.declare_parameter('obstacle3_y', 0.0)  # Position latérale (m)
        self.declare_parameter('obstacle3_z', 1.1)  # Hauteur du centre (m)
        self.declare_parameter('obstacle3_size_x', 1.7)  # Largeur dans l'axe X du robot (m)
        self.declare_parameter('obstacle3_size_y', 1.0)  # LONGUEUR dans l'axe Y (m)
        self.declare_parameter('obstacle3_height', 0.1)  # Hauteur de l'obstacle (m)

        # Publisher pour la scène de planification
        self.scene_pub = self.create_publisher(
            PlanningScene,
            '/planning_scene',
            10
        )
        
        # Attendre que le publisher soit prêt
        time.sleep(1.0)
        
        # Publier l'obstacle immédiatement
        self.publish_obstacle()
        
        # Timer pour republier périodiquement (toutes les 2 secondes)
        self.create_timer(5.0, self.publish_obstacle)
        
        
        
    def publish_obstacle(self):
        """Crée et publie des obstacles rectangulaires"""
        
        # Créer le message de scène
        scene = PlanningScene()
        scene.is_diff = True  # C'est une mise à jour incrémentale
        
        # ===== OBSTACLE 1 =====
        x = self.get_parameter('obstacle_x').value
        y = self.get_parameter('obstacle_y').value
        z = self.get_parameter('obstacle_z').value
        size_x = self.get_parameter('obstacle_size_x').value
        size_y = self.get_parameter('obstacle_size_y').value
        height = self.get_parameter('obstacle_height').value

        
        collision_object1 = CollisionObject()
        collision_object1.header.frame_id = 'base_link'
        collision_object1.id = 'obstacle_box_1'
        
        box1 = SolidPrimitive()
        box1.type = SolidPrimitive.BOX
        box1.dimensions = [size_x, size_y, height]
        
        pose1 = Pose()
        pose1.position.x = x
        pose1.position.y = y
        center_z = max(z, height / 2.0)
        pose1.position.z = center_z
        pose1.orientation.w = 1.0
        
        collision_object1.primitives.append(box1)
        collision_object1.primitive_poses.append(pose1)
        collision_object1.operation = CollisionObject.ADD
        
        scene.world.collision_objects.append(collision_object1)
        
        # ===== OBSTACLE 2 =====
        x2 = self.get_parameter('obstacle2_x').value
        y2 = self.get_parameter('obstacle2_y').value
        z2 = self.get_parameter('obstacle2_z').value
        size_x2 = self.get_parameter('obstacle2_size_x').value
        size_y2 = self.get_parameter('obstacle2_size_y').value
        height2 = self.get_parameter('obstacle2_height').value

   
        
        collision_object2 = CollisionObject()
        collision_object2.header.frame_id = 'base_link'
        collision_object2.id = 'obstacle_box_2'
        
        box2 = SolidPrimitive()
        box2.type = SolidPrimitive.BOX
        box2.dimensions = [size_x2, size_y2, height2]
        
        pose2 = Pose()
        pose2.position.x = x2
        pose2.position.y = y2
        center_z2 = max(z2, height2 / 2.0)
        pose2.position.z = center_z2
        pose2.orientation.w = 1.0
        
        collision_object2.primitives.append(box2)
        collision_object2.primitive_poses.append(pose2)
        collision_object2.operation = CollisionObject.ADD
        
        scene.world.collision_objects.append(collision_object2)
        
        # ===== OBSTACLE 3 =====
        x3 = self.get_parameter('obstacle3_x').value
        y3 = self.get_parameter('obstacle3_y').value
        z3 = self.get_parameter('obstacle3_z').value
        size_x3 = self.get_parameter('obstacle3_size_x').value
        size_y3 = self.get_parameter('obstacle3_size_y').value
        height3 = self.get_parameter('obstacle3_height').value

        collision_object3 = CollisionObject()
        collision_object3.header.frame_id = 'base_link'
        collision_object3.id = 'obstacle_box_3'

        box3 = SolidPrimitive()
        box3.type = SolidPrimitive.BOX
        box3.dimensions = [size_x3, size_y3, height3]

        pose3 = Pose()
        pose3.position.x = x3
        pose3.position.y = y3
        center_z3 = max(z3, height3 / 2.0)
        pose3.position.z = center_z3
        pose3.orientation.w = 1.0

        collision_object3.primitives.append(box3)
        collision_object3.primitive_poses.append(pose3)
        collision_object3.operation = CollisionObject.ADD

        scene.world.collision_objects.append(collision_object3)

        # Publier la scène avec les 3 obstacles
        self.scene_pub.publish(scene)



def main(args=None):
    rclpy.init(args=args)
    node = ObstaclePublisher()
    
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
