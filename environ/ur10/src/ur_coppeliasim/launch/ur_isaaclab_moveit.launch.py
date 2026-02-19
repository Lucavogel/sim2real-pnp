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
    
    # ============================================================
    # ❌ DÉSACTIVÉ pour génération dataset standalone (pas besoin d'Isaac Lab)
    # ============================================================
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
    # 
    # declared_arguments.append(
    #     DeclareLaunchArgument(
    #         "isaaclab_path",
    #         default_value=os.path.expanduser("~/workspace/IsaacLab"),
    #         description="Chemin vers IsaacLab installation",
    #     )
    # )
    #
    # # Nouveau argument pour activer/désactiver AprilTag
    # declared_arguments.append(
    #     DeclareLaunchArgument(
    #         "enable_apriltag",
    #         default_value="true",
    #         description="Activer la détection AprilTag (les nodes sont lancés, pas de second RViz)",
    #     )
    # )
    # 
    # # Argument pour activer/désactiver le mouvement automatique entre tags
    # declared_arguments.append(
    #     DeclareLaunchArgument(
    #         "auto_move_tags",
    #         default_value="true",
    #         description="Déplacer automatiquement le robot entre les AprilTags détectés",
    #     )
    # )
    # 
    # # Paramètres pour le mouvement automatique
    # declared_arguments.append(
    #     DeclareLaunchArgument(
    #         "move_delay",
    #         default_value="10.0",
    #         description="Délai en secondes entre les mouvements automatiques",
    #     )
    # )
    # 
    # declared_arguments.append(
    #     DeclareLaunchArgument(
    #         "z_offset",
    #         default_value="0.10",
    #         description="Offset vertical au-dessus du tag (en mètres)",
    #     )
    # )
    # ============================================================
    
    # Configuration
    ur_type = LaunchConfiguration("ur_type")
    launch_rviz = LaunchConfiguration("launch_rviz")
    use_sim_time = LaunchConfiguration("use_sim_time")
    add_obstacle = LaunchConfiguration("add_obstacle")  # ❌ Désactivé
    add_obstacle = LaunchConfiguration("add_obstacle")
    # isaaclab_path = LaunchConfiguration("isaaclab_path")  # ❌ Désactivé
    # enable_apriltag = LaunchConfiguration("enable_apriltag")  # ❌ Désactivé
    # auto_move_tags = LaunchConfiguration("auto_move_tags")  # ❌ Désactivé
    # move_delay = LaunchConfiguration("move_delay")  # ❌ Désactivé
    # z_offset = LaunchConfiguration("z_offset")  # ❌ Désactivé
    
    # Chemin du fichier de configuration des contrôleurs
    controllers_file = PathJoinSubstitution([
        FindPackageShare("ur_coppeliasim"),
        "config",
        "coppeliasim_controllers.yaml"
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
    
    # ============================================================
    # ❌ DÉSACTIVÉ - Pas besoin pour génération dataset standalone
    # ============================================================
    # # Node pour ajouter le plan de sol dans MoveIt (planning scene)
    # ground_plane_node = Node(
    #     package="ur_coppeliasim",
    #     executable="add_ground_plane.py",
    #     name="ground_plane_publisher",
    #     output="screen",
    # )
    # 
    # # Node pour ajouter un obstacle rectangulaire
    # obstacle_node = Node(
    #     package="ur_coppeliasim",
    #     executable="add_obstacle.py",
    #     name="obstacle_publisher",
    #     output="screen",
    #     condition=IfCondition(add_obstacle),
    #     parameters=[
    #         {"obstacle_x": LaunchConfiguration("obstacle_x")},
    #         {"obstacle_width_x": LaunchConfiguration("obstacle_width_x")},
    #         {"obstacle_length_y": LaunchConfiguration("obstacle_length_y")},
    #     ],
    # )
        # ============================================================
        # ✅ ACTIF - Ajouter plan de sol et obstacle dans la planning scene
        # ============================================================
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
    
    # ✅ Node pour visualiser le workspace dans RViz
    workspace_visualizer_node = Node(
        package="ur_coppeliasim",
        executable="visualize_workspace.py",
        name="workspace_visualizer",
        output="screen",
    )
    
    # ✅ Node pour publier des joint_states fictifs (pour que MoveIt puisse planifier)
    fake_joint_publisher_node = Node(
        package="ur_coppeliasim",
        executable="fake_joint_publisher.py",
        name="fake_joint_publisher",
        output="screen",
    )
    
    # DÉSACTIVÉ - La transformation TF ne change pas le modèle URDF dans RViz
    # rotate_robot_node = Node(
    #     package="ur_coppeliasim",
    #     executable="rotate_robot_base.py",
    #     name="rotate_robot_base",
    #     output="screen",
    # )
    
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
            "use_fake_hardware": "true",  # ✅ ACTIVER fake hardware pour publier /joint_states
            "moveit_controller_manager": "ros_control",
            "controllers_file": controllers_file,
        }.items(),
    )

    # ============================================================
    # ❌ DÉSACTIVÉ - AprilTag pas nécessaire pour dataset standalone
    # ============================================================
    # # Chemin fichier config AprilTag (dans le package ur_coppeliasim/config)
    # apriltag_config = PathJoinSubstitution([
    #     FindPackageShare("ur_coppeliasim"),
    #     "config",
    #     "apriltag_config.yaml"
    # ])
    #
    # # Node AprilTag SIMPLIFIÉ (1 seul node suffit!)
    # # apriltag_ros détecte + calcule pose 3D + broadcast TF automatiquement
    # apriltag_node = Node(
    #     package='apriltag_ros',
    #     executable='apriltag_node',
    #     name='apriltag_detector',
    #     parameters=[apriltag_config],
    #     remappings=[
    #         ('image_rect', '/rgb'),
    #         ('camera_info', '/camera_info'),
    #     ],
    #     output='screen',
    #     condition=IfCondition(enable_apriltag),
    # )
    # 
    # # ✂️ SUPPRIMÉ: apriltag_visualizer.py (redondant, apriltag_ros fait tout)
    # 
    # # Node pour déplacer automatiquement le robot entre les AprilTags
    # auto_mover_node = Node(
    #     package='ur_coppeliasim',
    #     executable='apriltag_auto_mover.py',
    #     name='apriltag_auto_mover',
    #     output='screen',
    #     condition=IfCondition(auto_move_tags),
    #     parameters=[
    #         {"move_delay": move_delay},
    #         {"z_offset": z_offset},
    #         {"planning_time": 5.0},
    #         {"velocity_scaling": 0.5},
    #         {"group_name": "ur_manipulator"},
    #     ],
    # )
    
    nodes_to_start = [
        # ✅ MoveIt + Visualisation workspace + Fake joints
        moveit_launch,
        workspace_visualizer_node,       # ✅ Affiche le rectangle rouge dans RViz
        fake_joint_publisher_node, 
        ground_plane_node,
        obstacle_node,
        
        # ❌ DÉSACTIVÉ - Pas besoin pour dataset standalone:
        # - Isaac Lab bridge (pas de simulation)
        # - ground_plane_node (pas d'obstacles)
        # - obstacle_node (pas d'obstacles)
        # - apriltag_node (pas de détection)
        # - auto_mover_node (pas de mouvement auto)
        # - rotate_robot_node (TF ne change pas l'URDF)
    ]
    
    return LaunchDescription(declared_arguments + nodes_to_start)
