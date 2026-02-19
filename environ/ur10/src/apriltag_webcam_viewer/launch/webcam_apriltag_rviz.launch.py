from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    video_device = LaunchConfiguration("video_device")
    camera_frame = LaunchConfiguration("camera_frame")
    world_frame = LaunchConfiguration("world_frame")
    tag_config = LaunchConfiguration("tag_config")
    cam_calib = LaunchConfiguration("cam_calib")
    ur_type = LaunchConfiguration("ur_type")

    # NEW: paramètres caméra (base -> cam) pour tag_tf_chain_node
    cam_x = LaunchConfiguration("cam_x")
    cam_y = LaunchConfiguration("cam_y")
    cam_z = LaunchConfiguration("cam_z")
    cam_roll_deg = LaunchConfiguration("cam_roll_deg")
    cam_pitch_deg = LaunchConfiguration("cam_pitch_deg")
    cam_yaw_deg = LaunchConfiguration("cam_yaw_deg")

    pkg_share = FindPackageShare("apriltag_webcam_viewer")
    cam_info_url = [TextSubstitution(text="file://"), cam_calib]

    # Controllers file pour MoveIt
    controllers_file = PathJoinSubstitution([
        FindPackageShare("ur_coppeliasim"),
        "config",
        "coppeliasim_controllers.yaml"
    ])

    return LaunchDescription([
        DeclareLaunchArgument("video_device", default_value="/dev/video3"),
        DeclareLaunchArgument("camera_frame", default_value="camera_link"),
        DeclareLaunchArgument("world_frame", default_value="world"),
        DeclareLaunchArgument("ur_type", default_value="ur10"),

        DeclareLaunchArgument(
            "tag_config",
            default_value=PathJoinSubstitution([pkg_share, "cfg", "tags_36h11.yaml"]),
        ),
        DeclareLaunchArgument(
            "cam_calib",
            default_value=PathJoinSubstitution([pkg_share, "cfg", "c930e_640x480.yaml"]),
        ),

        # NEW: offsets base -> camera (tu changeras plus tard)
        DeclareLaunchArgument("cam_x", default_value="0.0"),
        DeclareLaunchArgument("cam_y", default_value="0.0"),
        DeclareLaunchArgument("cam_z", default_value="2.0"),
        DeclareLaunchArgument("cam_roll_deg", default_value="180.0"),
        DeclareLaunchArgument("cam_pitch_deg", default_value="0.0"),
        DeclareLaunchArgument("cam_yaw_deg", default_value="0.0"),

        # ✅ MoveIt2 (gère robot_state_publisher + joint_states + RViz)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare("ur_moveit_config"),
                    "launch",
                    "ur_moveit.launch.py"
                ])
            ]),
            launch_arguments={
                "ur_type": ur_type,
                "launch_rviz": "true",  # ✅ Lancer RViz avec config MoveIt
                "use_sim_time": "false",
                "use_fake_hardware": "true",
                "moveit_controller_manager": "ros_control",
                "controllers_file": controllers_file,
            }.items(),
        ),

        # TF statique: world -> base_link (origine du robot)
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="world_to_base_link",
            arguments=["0", "0", "0", "0", "0", "0", "1", world_frame, "base_link"],
            output="screen",
        ),

        # Webcam
        Node(
            package="usb_cam",
            executable="usb_cam_node_exe",
            name="usb_cam",
            namespace="camera",
            output="screen",
            parameters=[{
                "video_device": video_device,
                "framerate": 30.0,
                "image_width": 640,
                "image_height": 480,
                "pixel_format": "yuyv",
                "camera_name": "c930e",
                "camera_info_url": cam_info_url,
                "frame_id": camera_frame,
            }],
            remappings=[
                ("image_raw", "image"),
                ("camera_info", "camera_info"),
            ],
        ),

        # AprilTags
        Node(
            package="apriltag_ros",
            executable="apriltag_node",
            name="apriltag",
            namespace="camera",
            output="screen",
            parameters=[
                tag_config,
                {
                    "family": "36h11",
                    "size": 0.1,
                    "max_hamming": 0,
                    "pose_estimation_method": "pnp",
                    "tag.ids": [0, 1, 2, 3],
                    "tag.frames": ["tag36h11:0", "tag36h11:1", "tag36h11:2", "tag36h11:3"],
                    "tag.sizes": [0.1, 0.1, 0.1, 0.1],
                }
            ],
            remappings=[
                ("image_rect", "image"),
                ("camera_info", "camera_info"),
            ],
        ),

        # NEW: Node TF chain (lit /tf et publie PoseArray + logs)
        Node(
            package="apriltag_webcam_viewer",
            executable="tag_tf_chain_node",
            name="tag_tf_chain_node",
            output="screen",
            parameters=[{
                "world_frame": "world",  # ✅ World ancré sur tag1
                "camera_frame": camera_frame,
                "tag_family_prefix": "tag36h11:",
                "anchor_tag_id": 1,
                "tag_ids": [0, 1, 2, 3],

                # 21.5 cm => 0.215 m
                "world_off_x": -0.193,
                "world_off_y": -0.255,
                "world_off_z": -0.01,

                "world_roll_deg": 0.0,
                "world_pitch_deg": 0.0,
                "world_yaw_deg": 0.0,

                "rate_hz": 10.0,
                "log_hz": 2.0,
                "publish_pose_array": True,
            }],
        ),

        # Visualiseur du workspace (rectangle rouge dans RViz)
        Node(
            package="ur_coppeliasim",
            executable="visualize_workspace.py",
            name="workspace_visualizer",
            output="screen",
            parameters=[{
                "z_fixed": 0.01,  # Hauteur fixe du plan de travail
            }],
        ),

        # ❌ Supprimer le RViz manuel (MoveIt lance le sien)
        # Node(
        #     package="rviz2",
        #     executable="rviz2",
        #     name="rviz2",
        #     output="screen",
        # ),
    ])
