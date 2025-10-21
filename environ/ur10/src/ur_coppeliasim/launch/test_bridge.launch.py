#!/usr/bin/env python3
"""
Launch file simple pour tester le bridge CoppeliaSim seul
Lance uniquement le bridge et vérifie la connexion
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    
    # Arguments
    declared_arguments = []
    
    declared_arguments.append(
        DeclareLaunchArgument(
            "ur_type",
            default_value="ur10",
            description="Type de robot UR",
        )
    )
    
    # Configuration
    ur_type = LaunchConfiguration("ur_type")
    
    # Node bridge CoppeliaSim
    coppeliasim_bridge_node = Node(
        package="ur_coppeliasim",
        executable="coppeliasim_bridge.py",
        name="coppeliasim_bridge",
        output="screen",
        parameters=[
            {"ur_type": ur_type},
        ],
    )
    
    nodes_to_start = [
        coppeliasim_bridge_node,
    ]
    
    return LaunchDescription(declared_arguments + nodes_to_start)
