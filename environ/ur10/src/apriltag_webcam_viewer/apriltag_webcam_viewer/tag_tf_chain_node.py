import math
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from geometry_msgs.msg import TransformStamped, PoseArray, Pose
import tf2_ros
from tf_transformations import quaternion_from_euler


class TagWorldChainNode(Node):
    """
    Publie une TF statique tag1 -> world (world ancré sur tag1 avec offset).
    Ensuite publie les poses des tags dans 'world' et log world->tag / cam->tag.
    """

    def __init__(self):
        super().__init__("tag_world_chain_node")

        # ---- Params
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("camera_frame", "camera_link")
        self.declare_parameter("tag_family_prefix", "tag36h11:")
        self.declare_parameter("anchor_tag_id", 1)       # tag qui définit le monde
        self.declare_parameter("tag_ids", [0, 1,2,3])         # tags à suivre

        # Offset du monde par rapport à tag0 (en mètres)
        self.declare_parameter("world_off_x", 0.0)      # 21.5 cm = 0.215 m
        self.declare_parameter("world_off_y", 0.0)
        self.declare_parameter("world_off_z", -0.215)

        # Orientation world par rapport à camera_link (deg)
        self.declare_parameter("world_roll_deg", 0.0)
        self.declare_parameter("world_pitch_deg", 180.0)
        self.declare_parameter("world_yaw_deg", 0.0)

        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("lookup_timeout_s", 0.05)
        self.declare_parameter("publish_pose_array", True)
        self.declare_parameter("log_hz", 2.0)

        self.world_frame = str(self.get_parameter("world_frame").value)
        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.tag_prefix = str(self.get_parameter("tag_family_prefix").value)
        self.anchor_tag_id = int(self.get_parameter("anchor_tag_id").value)
        self.tag_ids = list(self.get_parameter("tag_ids").value)

        self.lookup_timeout = Duration(seconds=float(self.get_parameter("lookup_timeout_s").value))
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.publish_pose_array = bool(self.get_parameter("publish_pose_array").value)
        self.log_period = 1.0 / max(0.1, float(self.get_parameter("log_hz").value))
        self._last_log_t = 0.0

        # TF
        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Pub
        if self.publish_pose_array:
            self.pose_pub = self.create_publisher(PoseArray, "tag_poses_in_world", 10)

        # Publie le TF statique tag1 -> world (offset fixe par rapport au tag)
        self.publish_tag_to_world_static()

        # Timer
        self.timer = self.create_timer(1.0 / max(1e-3, self.rate_hz), self.on_timer)


    def publish_tag_to_world_static(self):
        """Publie TF statique: tag1 -> world"""
        tag1_frame = f"{self.tag_prefix}{self.anchor_tag_id}"

        x = float(self.get_parameter("world_off_x").value)
        y = float(self.get_parameter("world_off_y").value)
        z = float(self.get_parameter("world_off_z").value)

        roll = math.radians(float(self.get_parameter("world_roll_deg").value))
        pitch = math.radians(float(self.get_parameter("world_pitch_deg").value))
        yaw = math.radians(float(self.get_parameter("world_yaw_deg").value))

        qx, qy, qz, qw = quaternion_from_euler(roll, pitch, yaw)

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = tag1_frame           # PARENT = tag1
        t.child_frame_id = self.world_frame      # CHILD  = world

        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        self.static_broadcaster.sendTransform(t)


    def _should_log(self) -> bool:
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._last_log_t >= self.log_period:
            self._last_log_t = now
            return True
        return False

    def on_timer(self):
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = self.world_frame

        do_log = self._should_log()

        for tag_id in self.tag_ids:
            tag_frame = f"{self.tag_prefix}{int(tag_id)}"

            # world -> tag
            try:
                tf_world_tag = self.tf_buffer.lookup_transform(
                    self.world_frame, tag_frame, rclpy.time.Time(), timeout=self.lookup_timeout
                )
            except Exception:
                continue

            # camera -> tag (optionnel pour debug)
            tf_cam_tag = None
            try:
                tf_cam_tag = self.tf_buffer.lookup_transform(
                    self.camera_frame, tag_frame, rclpy.time.Time(), timeout=self.lookup_timeout
                )
            except Exception:
                pass

            wt = tf_world_tag.transform.translation
            if do_log:

                if tf_cam_tag is not None:
                    ct = tf_cam_tag.transform.translation

                    

            if self.publish_pose_array:
                p = Pose()
                p.position.x = wt.x
                p.position.y = wt.y
                p.position.z = wt.z
                p.orientation = tf_world_tag.transform.rotation
                pose_array.poses.append(p)

        if self.publish_pose_array and pose_array.poses:
            self.pose_pub.publish(pose_array)


def main(args=None):
    rclpy.init(args=args)
    node = TagWorldChainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
