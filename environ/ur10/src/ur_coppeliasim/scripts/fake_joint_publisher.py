#!/usr/bin/env python3
"""
Publie un état fictif des joints pour que MoveIt puisse planifier
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header


class FakeJointPublisher(Node):
    def __init__(self):
        super().__init__('fake_joint_publisher')
        
        self.publisher = self.create_publisher(JointState, '/joint_states', 10)
        self.timer = self.create_timer(0.02, self.publish_joint_states)  # 50Hz
        
        # Position home du UR10 (safe position)
        self.joint_positions = [0.0, -1.309, 2.14675, -2.44346, -1.5708, 0.0]  # En radians
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        self.get_logger().info('🤖 Publication joint_states fictifs pour MoveIt')
        self.get_logger().info(f'   Position: {[f"{p:.2f}" for p in self.joint_positions]}')
    
    def publish_joint_states(self):
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.joint_positions
        msg.velocity = [0.0] * 6
        msg.effort = [0.0] * 6
        
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FakeJointPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
