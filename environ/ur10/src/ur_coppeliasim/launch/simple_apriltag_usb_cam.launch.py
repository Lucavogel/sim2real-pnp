#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    pkg_dir = os.path.dirname(__file__)
    cfg = os.path.join(pkg_dir, "..", "config", "apriltag_simple.yaml")
    script = os.path.join(pkg_dir, "..", "scripts", "print_detections.py")

    usb_cam = Node(
        package="usb_cam",
        executable="usb_cam_node_exe",
        name="usb_cam",
        parameters=[{
            "video_device": "/dev/video2",
            "image_width": 640,
            "image_height": 480,
            "frame_id": "camera_optical_frame",

            # IMPORTANT: doit matcher camera_name dans ton YAML camera_info
            "camera_name": "default_cam",
            "camera_info_url": "file:///home/ajin/workspace/sim2real-pnp/environ/ur10/src/ur_coppeliasim/config/camera_info_webcam.yaml",
        }],
        output="screen",
    )

    apriltag = Node(
        package="apriltag_ros",
        executable="apriltag_node",
        name="apriltag_detector",
        parameters=[cfg],
        remappings=[
            ("image_rect", "/image_raw"),
            ("camera_info", "/camera_info"),
        ],
        output="screen",
    )

    printer = ExecuteProcess(
        cmd=["python3", script],
        output="screen",
    )

    return LaunchDescription([usb_cam, apriltag, printer])
