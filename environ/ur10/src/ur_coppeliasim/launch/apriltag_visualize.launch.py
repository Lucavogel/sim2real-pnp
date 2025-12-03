#!/usr/bin/env python3
"""
Launch file pour AprilTag detection + Visualisation RViz2
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import os


def generate_launch_description():
    
    # Chemin vers le fichier de config AprilTag
    config_file = os.path.join(
        os.path.dirname(__file__),
        '..',
        'config',
        'apriltag_config.yaml'
    )
    
    # Node AprilTag detection
    apriltag_node = Node(
        package='apriltag_ros',
        executable='apriltag_node',
        name='apriltag_detector',
        parameters=[config_file],
        remappings=[
            ('image_rect', '/rgb'),
            ('camera_info', '/camera_info'),
        ],
        output='screen'
    )
    
    # Node pour convertir détections en markers visualisables
    visualizer_node = Node(
        package='ur_coppeliasim',
        executable='apriltag_visualizer.py',
        name='apriltag_visualizer',
        output='screen'
    )
    
    # Node RViz2 pour visualisation
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(
            os.path.dirname(__file__),
            '..',
            'config',
            'apriltag_view.rviz'
        )] if os.path.exists(os.path.join(
            os.path.dirname(__file__),
            '..',
            'config',
            'apriltag_view.rviz'
        )) else [],
        output='screen'
    )
    
    return LaunchDescription([
        apriltag_node,
        visualizer_node,
        rviz_node,
    ])
