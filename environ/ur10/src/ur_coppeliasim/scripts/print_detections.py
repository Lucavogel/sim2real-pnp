import rclpy
from rclpy.node import Node
from apriltag_msgs.msg import AprilTagDetectionArray

class PrintDetections(Node):
    def __init__(self):
        super().__init__("print_detections")
        self.sub = self.create_subscription(AprilTagDetectionArray, "/detections", self.cb, 10)
        self.get_logger().info("✅ print_detections prêt. Montre un tag à la caméra.")

    def cb(self, msg: AprilTagDetectionArray):
        if not msg.detections:
            return
        ids = []
        for det in msg.detections:
            det_id = det.id if isinstance(det.id, int) else (det.id[0] if det.id else -1)
            ids.append(det_id)
        self.get_logger().info(f"🎯 tags détectés: {ids}")

def main():
    rclpy.init()
    node = PrintDetections()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
