from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():
    # ---------- Launch args ----------
    world_frame  = LaunchConfiguration("world_frame")
    camera_frame = LaunchConfiguration("camera_frame")

    # static transform world -> camera
    # x y z qx qy qz qw
    # Ton cas: translation (0.9, 0.0, 1.52164), roll=180°, pitch=0°, yaw=0°
    # quaternion (x,y,z,w) = (1,0,0,0) pour roll=pi
    static_xyz_q = LaunchConfiguration("static_xyz_q")

    # topics
    image_topic = LaunchConfiguration("image_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")

    # tag config
    tag_family = LaunchConfiguration("tag_family")
    tag_size   = LaunchConfiguration("tag_size")

    # relative TF (A -> B)
    tag_a = LaunchConfiguration("tag_a")
    tag_b = LaunchConfiguration("tag_b")

    declare_args = [
        DeclareLaunchArgument("world_frame", default_value="world"),
        DeclareLaunchArgument("camera_frame", default_value="camera"),
        DeclareLaunchArgument(
            "static_xyz_q",
            default_value="0.9 0.0 1.52164 1.0 0.0 0.0 0.0",
            description="Static TF world->camera: x y z qx qy qz qw",
        ),
        DeclareLaunchArgument("image_topic", default_value="/camera/image_raw"),
        DeclareLaunchArgument("camera_info_topic", default_value="/camera/camera_info"),
        DeclareLaunchArgument("tag_family", default_value="tag36h11"),
        DeclareLaunchArgument("tag_size", default_value="0.08"),  # meters
        DeclareLaunchArgument("tag_a", default_value="tag36h11:0"),
        DeclareLaunchArgument("tag_b", default_value="tag36h11:5"),
    ]

    # ---------- Nodes ----------

    static_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_camera_static_tf",
        # args: x y z qx qy qz qw frame_id child_frame_id
        arguments=[
            *static_xyz_q.perform(None).split(),  # note: launch will substitute at runtime
            world_frame,
            camera_frame,
        ],
        output="screen",
    )

    # apriltag_ros (les noms peuvent varier selon ta distro/installation)
    # Si ça ne démarre pas, remplace executable="apriltag_node" par "apriltag_ros_node"
    apriltag_node = Node(
        package="apriltag_ros",
        executable="apriltag_node",
        name="apriltag",
        output="screen",
        parameters=[{
            "family": tag_family,
            "size": tag_size,
            "camera_frame": camera_frame,
            "image_transport": "raw",
        }],
        remappings=[
            ("image_rect", image_topic),
            ("camera_info", camera_info_topic),
        ],
    )

    # ton node tag->tag (dans ton package)
    tag_to_tag_node = Node(
        package="your_pkg",
        executable="tag_to_tag_tracker",  # ton entrypoint/install
        name="tag_to_tag_tracker",
        output="screen",
        parameters=[{
            "target_frame": tag_a,
            "source_frame": tag_b,
        }],
    )

    return LaunchDescription(
        declare_args + [
            static_tf_node,
            apriltag_node,
            tag_to_tag_node,
        ]
    )
