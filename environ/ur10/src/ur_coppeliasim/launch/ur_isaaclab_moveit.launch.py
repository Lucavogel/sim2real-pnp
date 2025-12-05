#!/usr/bin/env python3
"""
Launch file pour MoveIt2 + Isaac Lab
Lance le bridge ROS2-Isaac Lab et MoveIt2
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


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
    
    declared_arguments.append(
        DeclareLaunchArgument(
            "isaaclab_path",
            default_value=os.path.expanduser("~/work2/IsaacLab"),
            description="Chemin vers IsaacLab installation",
        )
    )

    # Nouveau argument pour activer/désactiver AprilTag
    declared_arguments.append(
        DeclareLaunchArgument(
            "enable_apriltag",
            default_value="true",  # Changed from "true"
            description="Activer la détection AprilTag (les nodes sont lancés, pas de second RViz)",
        )
    )
    
    # Argument pour activer/désactiver le mouvement automatique entre tags
    declared_arguments.append(
        DeclareLaunchArgument(
            "auto_move_tags",
            default_value="false",  # Changed from "true"
            description="Déplacer automatiquement le robot entre les AprilTags détectés",
        )
    )
    
    # Paramètres pour le mouvement automatique
    declared_arguments.append(
        DeclareLaunchArgument(
            "move_delay",
            default_value="10.0",
            description="Délai en secondes entre les mouvements automatiques",
        )
    )
    
    declared_arguments.append(
        DeclareLaunchArgument(
            "z_offset",
            default_value="0.10",
            description="Offset vertical au-dessus du tag (en mètres)",
        )
    )
    
    # Configuration
    ur_type = LaunchConfiguration("ur_type")
    launch_rviz = LaunchConfiguration("launch_rviz")
    use_sim_time = LaunchConfiguration("use_sim_time")
    add_obstacle = LaunchConfiguration("add_obstacle")
    isaaclab_path = LaunchConfiguration("isaaclab_path")
    enable_apriltag = LaunchConfiguration("enable_apriltag")
    auto_move_tags = LaunchConfiguration("auto_move_tags")
    move_delay = LaunchConfiguration("move_delay")
    z_offset = LaunchConfiguration("z_offset")
    
    # Chemin du fichier de configuration des contrôleurs
    controllers_file = PathJoinSubstitution([
        FindPackageShare("ur_coppeliasim"),
        "config",
        "coppeliasim_controllers.yaml"
    ])
    
    # Chemin vers le script Isaac Lab bridge
    isaac_bridge_script = PathJoinSubstitution([
        FindPackageShare("ur_coppeliasim"),
        "scripts",
        "isaaclab_bridge.py"
    ])
    
    # IMPORTANT: Isaac Lab nécessite d'être lancé via son wrapper isaaclab.sh
    # On utilise ExecuteProcess au lieu de Node pour lancer via le shell script
    # NOTE: Vous devez lancer Isaac Lab manuellement dans un terminal séparé:
    #   cd ~/work2/IsaacLab
    #   ./isaaclab.sh -p ~/work2/ur10/install/ur_coppeliasim/lib/ur_coppeliasim/isaaclab_bridge.py
    
    # isaaclab_bridge_process = ExecuteProcess(
    #     cmd=[
    #         'bash', '-c',
    #         ['cd ', isaaclab_path, ' && ./isaaclab.sh -p ', isaac_bridge_script]
    #     ],
    #     name='isaaclab_bridge',
    #     output='screen',
    # )
    
    # Alternative: Si vous avez configuré l'environnement Isaac Lab dans votre shell
    # Vous pouvez utiliser directement Python:
    # isaaclab_bridge_node = Node(
    #     package="ur_coppeliasim",
    #     executable="isaaclab_bridge.py",
    #     name="isaaclab_bridge",
    #     output="screen",
    #     parameters=[
    #         {"ur_type": ur_type},
    #         {"use_sim_time": use_sim_time},
    #     ],
    # )
    
    # Node pour ajouter le plan de sol dans MoveIt (planning scene)
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
            "use_fake_hardware": "false",  # On utilise Isaac Lab, pas fake hardware
            "moveit_controller_manager": "ros_control",
            "controllers_file": controllers_file,
        }.items(),
    )

    # Chemin fichier config AprilTag (dans le package ur_coppeliasim/config)
    apriltag_config = PathJoinSubstitution([
        FindPackageShare("ur_coppeliasim"),
        "config",
        "apriltag_config.yaml"
    ])

    # Nodes AprilTag (sans lancer un second RViz)
    apriltag_node = Node(
        package='apriltag_ros',
        executable='apriltag_node',
        name='apriltag_detector',
        parameters=[apriltag_config],
        remappings=[
            ('image_rect', '/rgb'),
            ('camera_info', '/camera_info'),
        ],
        output='screen',
        condition=IfCondition(enable_apriltag),
    )

    visualizer_node = Node(
        package='ur_coppeliasim',
        executable='apriltag_visualizer.py',
        name='apriltag_visualizer',
        output='screen',
        condition=IfCondition(enable_apriltag),
        parameters=[
            {"use_pnp_fallback": True},   # active le fallback solvePnP
            {"tag_size": 0.10},           # taille du tag en m
        ],
    )
    
    # Node pour déplacer automatiquement le robot entre les AprilTags
    auto_mover_node = Node(
        package='ur_coppeliasim',
        executable='apriltag_auto_mover.py',
        name='apriltag_auto_mover',
        output='screen',
        condition=IfCondition(auto_move_tags),
        parameters=[
            {"move_delay": move_delay},
            {"z_offset": z_offset},
            {"planning_time": 5.0},
            {"velocity_scaling": 0.5},
            {"group_name": "ur_manipulator"},
        ],
    )
    
    nodes_to_start = [
        # Isaac Lab bridge doit être lancé manuellement dans un terminal séparé
        # Voir les commandes dans COMMANDS.sh
        ground_plane_node,
        obstacle_node,
        moveit_launch,
        apriltag_node,     # Détection AprilTag
        visualizer_node,   # Convertit detections -> markers + /apriltag_poses
        auto_mover_node,   # Mouvement automatique entre tags
    ]
    
    return LaunchDescription(declared_arguments + nodes_to_start)
