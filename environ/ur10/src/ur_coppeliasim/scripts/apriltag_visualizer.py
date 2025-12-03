#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
import tf2_ros
from apriltag_msgs.msg import AprilTagDetectionArray
from geometry_msgs.msg import PoseArray, PoseStamped
from tf2_geometry_msgs import do_transform_pose
from tf_transformations import quaternion_from_euler
import math
from sensor_msgs.msg import CameraInfo

class TFAndAprilTagNode(Node):
    def __init__(self):
        super().__init__('tf_and_apriltag_node')

        # Parameters
        self.declare_parameter('tag_size', 0.10)
        self.tag_size = float(self.get_parameter('tag_size').value)

        # Frames
        self.world_frame = 'world'
        self.camera_frame = 'sim_camera'

        # TF broadcaster
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.publish_static_transforms()

        # TF buffer pour lire les transformations depuis /tf
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribers / publishers
        self.sub = self.create_subscription(AprilTagDetectionArray, 'detections', self.detections_callback, 10)
        self.pose_pub = self.create_publisher(PoseArray, '/apriltag_poses', 10)
        self.latest_cam_info = None
        self.create_subscription(CameraInfo, '/camera_info', self.camera_info_callback, 10)

        self.get_logger().info(f"TF & AprilTag Node démarré | world_frame={self.world_frame} camera_frame={self.camera_frame} tag_size={self.tag_size}")

    def camera_info_callback(self, msg: CameraInfo):
        self.latest_cam_info = msg
        K = list(msg.k)
        D = list(msg.d)
        self.get_logger().debug(f"CameraInfo reçu: width={msg.width} height={msg.height} K={K} D={D}")

    def publish_static_transforms(self):
        t_camera = TransformStamped()
        t_camera.header.stamp = self.get_clock().now().to_msg()
        t_camera.header.frame_id = self.world_frame
        t_camera.child_frame_id = self.camera_frame
        t_camera.transform.translation.x = 0.9
        t_camera.transform.translation.y = 0.0
        t_camera.transform.translation.z = 1.52164
        roll  = math.radians(180.0)
        pitch = 0.0
        yaw   = 0.0
        qx, qy, qz, qw = quaternion_from_euler(roll, pitch, yaw)
        t_camera.transform.rotation.x = qx
        t_camera.transform.rotation.y = qy
        t_camera.transform.rotation.z = qz
        t_camera.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform([t_camera])
        self.get_logger().info(f"Transform statique publié: world -> sim_camera | trans=({t_camera.transform.translation.x:.3f}, {t_camera.transform.translation.y:.3f}, {t_camera.transform.translation.z:.3f}) quat=({qx:.3f},{qy:.3f},{qz:.3f},{qw:.3f})")

    def detections_callback(self, msg: AprilTagDetectionArray):
        if not msg.detections:
            self.get_logger().debug("Aucune détection reçue")
            return

        pose_array = PoseArray()
        pose_array.header.stamp = msg.header.stamp
        pose_array.header.frame_id = self.world_frame

        for det in msg.detections:
            det_ids = getattr(det, 'id', getattr(det, 'ids', "unknown"))
            tag_frame = f"Tag{det_ids}" if isinstance(det_ids, int) else str(det_ids)

            try:
                # Lookup TF world <- tag
                transform = self.tf_buffer.lookup_transform(
                    self.world_frame,
                    tag_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )

                # Créer une pose directement depuis la transformation TF
                from geometry_msgs.msg import Pose
                world_pose = Pose()
                world_pose.position.x = transform.transform.translation.x
                world_pose.position.y = transform.transform.translation.y
                world_pose.position.z = transform.transform.translation.z
                world_pose.orientation = transform.transform.rotation

                pose_array.poses.append(world_pose)

            except Exception as e:
                self.get_logger().warn(f"Impossible d'obtenir la transformation TF pour tag ids={det_ids}: {e}")

        if pose_array.poses:
            self.pose_pub.publish(pose_array)

def main(args=None):
    rclpy.init(args=args)
    node = TFAndAprilTagNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
