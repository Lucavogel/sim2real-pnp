#!/usr/bin/env python3
"""
Node ROS2 pour visualiser la zone de travail (workspace) dans RViz
Affiche un rectangle rouge délimitant la zone où le robot peut dessiner

Usage:
    ros2 run ur_coppeliasim visualize_workspace.py
"""

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA


class WorkspaceVisualizer(Node):
    def __init__(self):
        super().__init__('workspace_visualizer')
        
        # Publisher pour les marqueurs RViz
        self.marker_pub = self.create_publisher(MarkerArray, '/workspace_markers', 10)
        self.zfixed = self.declare_parameter('z_fixed', 0.40).value  # Hauteur fixe du plan de travail
        
        # Timer pour publier périodiquement
        self.timer = self.create_timer(1.0, self.publish_markers)
        
        # ===== WORKSPACE FIXE (même que dans generate_moveit_dataset_v2.py) =====
        # Inversé en +X pour être devant le robot (sur vrai robot: inverser X)
        self.workspace = {
            'x_min': 0.7,     # Devant le robot (+X)
            'x_max': 1.000,   # Devant le robot (+X)
            'y_min': -0.2,    # Bas
            'y_max': 0.2,     # Haut
            'z_fixed': self.zfixed   # Hauteur fixe (40cm)
        }
        
        self.get_logger().info("🎨 Visualiseur de workspace démarré")
        self.get_logger().info(f"   Zone: X=[{self.workspace['x_min']:.3f}, {self.workspace['x_max']:.3f}]")
        self.get_logger().info(f"         Y=[{self.workspace['y_min']:.3f}, {self.workspace['y_max']:.3f}]")
        self.get_logger().info(f"         Z={self.workspace['z_fixed']:.3f}m (fixe)")
        self.get_logger().info("   Ouvrez RViz et ajoutez MarkerArray sur topic /workspace_markers")
    
    def publish_markers(self):
        """Publie les marqueurs du workspace"""
        marker_array = MarkerArray()
        
        # ===== 1. Rectangle du workspace (contour rouge) =====
        line_marker = Marker()
        line_marker.header.frame_id = "world"
        line_marker.header.stamp = self.get_clock().now().to_msg()
        line_marker.ns = "workspace"
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        
        # Échelle de la ligne
        line_marker.scale.x = 0.01  # Épaisseur 1cm
        
        # Couleur rouge vif
        line_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        
        # Points du rectangle (dans le sens horaire + fermer le rectangle)
        z = self.workspace['z_fixed']
        points = [
            Point(x=self.workspace['x_min'], y=self.workspace['y_max'], z=z),  # Haut gauche
            Point(x=self.workspace['x_max'], y=self.workspace['y_max'], z=z),  # Haut droit
            Point(x=self.workspace['x_max'], y=self.workspace['y_min'], z=z),  # Bas droit
            Point(x=self.workspace['x_min'], y=self.workspace['y_min'], z=z),  # Bas gauche
            Point(x=self.workspace['x_min'], y=self.workspace['y_max'], z=z),  # Retour au début
        ]
        line_marker.points = points
        
        marker_array.markers.append(line_marker)
        
        # ===== 2. Plan semi-transparent (rectangle rempli) =====
        plane_marker = Marker()
        plane_marker.header.frame_id = "world"
        plane_marker.header.stamp = self.get_clock().now().to_msg()
        plane_marker.ns = "workspace"
        plane_marker.id = 1
        plane_marker.type = Marker.CUBE
        plane_marker.action = Marker.ADD
        
        # Position au centre du rectangle
        x_center = (self.workspace['x_min'] + self.workspace['x_max']) / 2
        y_center = (self.workspace['y_min'] + self.workspace['y_max']) / 2
        plane_marker.pose.position.x = x_center
        plane_marker.pose.position.y = y_center
        plane_marker.pose.position.z = z
        plane_marker.pose.orientation.w = 1.0
        
        # Dimensions du rectangle
        width = self.workspace['x_max'] - self.workspace['x_min']
        height = self.workspace['y_max'] - self.workspace['y_min']
        plane_marker.scale.x = width
        plane_marker.scale.y = height
        plane_marker.scale.z = 0.001  # Très fin (1mm)
        
        # Couleur rouge semi-transparent
        plane_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.2)
        
        marker_array.markers.append(plane_marker)
        
        # ===== 3. Labels texte aux 4 coins =====
        labels = [
            ("P1", self.workspace['x_min'], self.workspace['y_max']),  # Haut gauche
            ("P2", self.workspace['x_min'], self.workspace['y_min']),  # Bas gauche
            ("P3", self.workspace['x_max'], self.workspace['y_max']),  # Haut droit
            ("P4", self.workspace['x_max'], self.workspace['y_min']),  # Bas droit
        ]
        
        for i, (label, x, y) in enumerate(labels):
            text_marker = Marker()
            text_marker.header.frame_id = "world"
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.ns = "workspace"
            text_marker.id = 10 + i
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            text_marker.pose.position.x = x
            text_marker.pose.position.y = y
            text_marker.pose.position.z = z + 0.05  # 5cm au-dessus
            text_marker.pose.orientation.w = 1.0
            
            text_marker.scale.z = 0.05  # Taille du texte
            text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)  # Blanc
            text_marker.text = f"{label}\n({x:.2f}, {y:.2f})"
            
            marker_array.markers.append(text_marker)
        
        # ===== 4. Label central avec dimensions =====
        center_label = Marker()
        center_label.header.frame_id = "world"
        center_label.header.stamp = self.get_clock().now().to_msg()
        center_label.ns = "workspace"
        center_label.id = 20
        center_label.type = Marker.TEXT_VIEW_FACING
        center_label.action = Marker.ADD
        
        center_label.pose.position.x = x_center
        center_label.pose.position.y = y_center
        center_label.pose.position.z = z + 0.1  # 10cm au-dessus
        center_label.pose.orientation.w = 1.0
        
        center_label.scale.z = 0.08  # Texte plus gros
        center_label.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)  # Jaune
        center_label.text = f"WORKSPACE\n{width*100:.1f}cm × {height*100:.1f}cm\nZ={z:.2f}m"
        
        marker_array.markers.append(center_label)
        
        # Publier tous les marqueurs
        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = WorkspaceVisualizer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
