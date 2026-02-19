from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="ur_coppeliasim",          # <- remplace par le nom de ton package
            executable="tag_to_tag_tracker", # <- nom de l’exécutable/entrypoint
            name="tag_to_tag_tracker",
            output="screen",
            respawn=True,
            respawn_delay=2.0,
        )
    ])
