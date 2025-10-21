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
        
        # Paramètres de l'obstacle
        self.declare_parameter('obstacle_x', 0.7)  # Distance devant le robot (m)
        self.declare_parameter('obstacle_y', 10.0)  # Position latérale (m)
        self.declare_parameter('obstacle_z', 0.3)  # Hauteur du centre (m)
        self.declare_parameter('obstacle_width_x', 0.1)  # LONGUEUR dans le sens X (devant/derrière le robot) (m)
        self.declare_parameter('obstacle_length_y', 0.1)  # Largeur dans le sens Y (perpendiculaire) (m)
        self.declare_parameter('obstacle_height', 0.6)  # Hauteur de l'obstacle (m)
        
        # Publisher pour la scène de planification
        self.scene_pub = self.create_publisher(
            PlanningScene,
            '/planning_scene',
            10
        )
        
        # Attendre que le publisher soit prêt
        time.sleep(1.0)
        
        # Publier l'obstacle
        self.publish_obstacle()
        
        self.get_logger().info('✓ Obstacle rectangulaire ajouté à la scène MoveIt')
        
    def publish_obstacle(self):
        """Crée et publie un obstacle rectangulaire (long dans le sens X, devant/derrière le robot)"""
        
        # Récupérer les paramètres
        x = self.get_parameter('obstacle_x').value
        y = self.get_parameter('obstacle_y').value
        z = self.get_parameter('obstacle_z').value
        width_x = self.get_parameter('obstacle_width_x').value
        length_y = self.get_parameter('obstacle_length_y').value
        height = self.get_parameter('obstacle_height').value
        
        # Créer l'objet de collision
        collision_object = CollisionObject()
        collision_object.header.frame_id = 'base_link'  # Relatif à la base du robot
        collision_object.id = 'obstacle_box'
        
        # Définir la forme (une boîte rectangulaire)
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        # Format: [dimension_X, dimension_Y, hauteur_Z]
        # X = LONG devant/derrière le robot, Y = fin perpendiculaire, Z = hauteur
        box.dimensions = [width_x, length_y, height]
        
        # Position de l'obstacle
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
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
        self.get_logger().info(
            f'Obstacle rectangulaire publié: {width_x:.2f}m (X-long) x {length_y:.2f}m (Y-fin) x {height:.2f}m (Z) '
            f'à position (x={x:.2f}, y={y:.2f}, z={z:.2f})'
        )


def main(args=None):
    rclpy.init(args=args)
    node = ObstaclePublisher()
    
    # Garder le node actif un moment pour s'assurer que le message est reçu
    rclpy.spin_once(node, timeout_sec=2.0)
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
