#!/usr/bin/env python3
"""
Launch file pour MoveIt + CoppeliaSim
Lance le bridge ROS2-CoppeliaSim et MoveIt
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    
    # Arguments
    declared_arguments = []
    
    declared_arguments.append(
        DeclareLaunchArgument(
            "ur_type",
            default_value="ur10",
            description="Type de robot UR (ur3, ur5, ur10, etc.)",
            choices=["ur3", "ur3e", "ur5", "ur5e", "ur7e", "ur10", "ur10e", "ur12e", "ur16e", "ur20", "ur30"],
        )
    )
    
    declared_arguments.append(
        DeclareLaunchArgument(
            "launch_rviz",
            default_value="true",
            description="Lancer RViz",
        )
    )
    
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Utiliser le temps de simulation",
        )
    )
    
    declared_arguments.append(
        DeclareLaunchArgument(
            "add_obstacle",
            default_value="true",
            description="Ajouter un obstacle rectangulaire devant le robot",
        )
    )
    
    declared_arguments.append(
        DeclareLaunchArgument(
            "obstacle_x",
            default_value="0.5",
            description="Distance de l'obstacle devant le robot (m)",
        )
    )
    
    declared_arguments.append(
        DeclareLaunchArgument(
            "obstacle_width_x",
            default_value="0.7",
            description="LONGUEUR de l'obstacle dans le sens X (devant/derrière le robot) (m)",
        )
    )
    
    declared_arguments.append(
        DeclareLaunchArgument(
            "obstacle_length_y",
            default_value="0.1",
            description="Largeur de l'obstacle dans le sens Y (perpendiculaire au robot) (m)",
        )
    )
    
    # Configuration
    ur_type = LaunchConfiguration("ur_type")
    launch_rviz = LaunchConfiguration("launch_rviz")
    use_sim_time = LaunchConfiguration("use_sim_time")
    add_obstacle = LaunchConfiguration("add_obstacle")
    
    # Chemin du fichier de configuration des contrôleurs
    controllers_file = PathJoinSubstitution([
        FindPackageShare("ur_coppeliasim"),
        "config",
        "coppeliasim_controllers.yaml"
    ])
    
    # Node bridge CoppeliaSim
    coppeliasim_bridge_node = Node(
        package="ur_coppeliasim",
        executable="coppeliasim_bridge.py",
        name="coppeliasim_bridge",
        output="screen",
        parameters=[
            {"ur_type": ur_type},
            {"use_sim_time": use_sim_time},
        ],
    )
    
    # Node pour ajouter le plan de sol dans MoveIt
    ground_plane_node = Node(
        package="ur_coppeliasim",
        executable="add_ground_plane.py",
        name="ground_plane_publisher",
        output="screen",
    )
    
    # Node pour ajouter un obstacle rectangulaire
    obstacle_node = Node(
        package="ur_coppeliasim",
        executable="add_obstacle.py",
        name="obstacle_publisher",
        output="screen",
        condition=IfCondition(add_obstacle),
        parameters=[
            {"obstacle_x": LaunchConfiguration("obstacle_x")},
            {"obstacle_width_x": LaunchConfiguration("obstacle_width_x")},
            {"obstacle_length_y": LaunchConfiguration("obstacle_length_y")},
        ],
    )
    
    # MoveIt launch (sans robot_state_publisher car MoveIt le gère déjà)
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("ur_moveit_config"),
                "launch",
                "ur_moveit.launch.py"
            ])
        ]),
        launch_arguments={
            "ur_type": ur_type,
            "launch_rviz": launch_rviz,
            "use_sim_time": use_sim_time,
            "use_fake_hardware": "false",  # On utilise CoppeliaSim, pas fake
            "moveit_controller_manager": "ros_control",
            "controllers_file": controllers_file,
        }.items(),
    )
    
    nodes_to_start = [
        coppeliasim_bridge_node,
        ground_plane_node,
        obstacle_node,
        moveit_launch,
    ]
    
    return LaunchDescription(declared_arguments + nodes_to_start)
