#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    # chemin vers ton fichier YAML via get_package_share_directory
    pkg_share = get_package_share_directory('ur_coppeliasim')
    config_file = os.path.join(pkg_share, 'config', 'apriltag_config.yaml')

    if not os.path.exists(config_file):
        print(f"[APRILTAG_LAUNCH] ⚠️  Fichier introuvable : {config_file}")
    else:
        print(f"[APRILTAG_LAUNCH] ✅  Chargement config : {config_file}")

    apriltag_node = Node(
        package='apriltag_ros',
        executable='apriltag_node',
        name='apriltag_detector',
        output='screen',
        parameters=[config_file],
        remappings=[
            ('image_rect', '/rgb'),       # topic image Isaac Sim
            ('camera_info', '/camera_info'),  # topic camera info
        ],
    )

    return LaunchDescription([apriltag_node])
