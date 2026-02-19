import rclpy
from rclpy.node import Node
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import time

class TagToTagTracker(Node):
    def __init__(self):
        super().__init__('tag_to_tag_tracker')

        # 1. Initialize the TF2 buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 2. Define your source and target frames
        # "target_frame" is the reference frame (Coordinate System Origin)
        # "source_frame" is the object you want to locate
        self.target_frame = 'tag36h11:0' 
        self.source_frame = 'tag36h11:5'

        # Create a timer to query the transform periodically
        self.timer = self.create_timer(1.0, self.on_timer)

    def on_timer(self):
        try:
            # 3. Look up the transform
            # We use rclpy.time.Time() to get the latest available transform
            t = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.source_frame,
                rclpy.time.Time()
            )

            # 4. Process the data
            translation = t.transform.translation
            rotation = t.transform.rotation

            self.get_logger().info(
                f'\nTag {self.source_frame} relative to {self.target_frame}:'
                f'\n  Translation: x={translation.x:.3f}, y={translation.y:.3f}, z={translation.z:.3f}'
                f'\n  Rotation: x={rotation.x:.3f}, y={rotation.y:.3f}, z={rotation.z:.3f}, w={rotation.w:.3f}'
            )

        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {self.source_frame} to {self.target_frame}: {ex}'
            )

def main():
    rclpy.init()
    node = TagToTagTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()