#!/usr/bin/env python3
"""
Version simplifiée du bridge Isaac Lab pour test rapide
Lance uniquement Isaac Lab et publie joint_states, sans MoveIt2
"""

import sys
import os
import numpy as np
import time as time_module
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

# Isaac Lab imports
import torch
import argparse
from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Isaac Lab Simple Bridge Test")
parser.add_argument("--device", type=str, default="cuda:0", help="Device")
AppLauncher.add_app_launcher_args(parser)
args_cli, unknown = parser.parse_known_args()

# Launch Isaac
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import Isaac Lab after app launch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab_assets import UR10_CFG


class SimpleIsaacBridge(Node):
    """Bridge minimal Isaac Lab → ROS2 (joint_states seulement)"""
    
    def __init__(self):
        super().__init__('simple_isaac_bridge')
        
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        self.joint_positions = [0.0] * 6
        self.joint_velocities = [0.0] * 6
        
        # Isaac Lab
        self.sim = None
        self.robot = None
        self.isaac_ready = False
        
        # Publisher
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        
        # Timer 20Hz
        self.create_timer(0.05, self.publish_joints)
        
        self.get_logger().info('🚀 Simple Isaac Bridge démarré')
    
    def setup_isaac(self):
        """Configure Isaac Lab"""
        try:
            # Simulation
            sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
            self.sim = sim_utils.SimulationContext(sim_cfg)
            self.sim.set_camera_view([3.0, 3.0, 2.0], [0.0, 0.0, 1.0])
            
            # Ground + Light
            cfg_ground = sim_utils.GroundPlaneCfg()
            cfg_ground.func("/World/Ground", cfg_ground)
            
            cfg_light = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
            cfg_light.func("/World/Light", cfg_light, translation=(1, 0, 10))
            
            # Robot UR10
            ur10_cfg = UR10_CFG.replace(prim_path="/World/UR10")
            ur10_cfg.init_state.pos = (0.0, 0.0, 0.0)
            ur10_cfg.init_state.joint_pos = {
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -1.57,
                "elbow_joint": 1.57,
                "wrist_1_joint": -1.57,
                "wrist_2_joint": -1.57,
                "wrist_3_joint": 0.0,
            }
            
            self.robot = Articulation(cfg=ur10_cfg)
            self.sim.reset()
            
            self.get_logger().info('✅ Isaac Lab prêt!')
            self.isaac_ready = True
            
        except Exception as e:
            self.get_logger().error(f'❌ Erreur Isaac Lab: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def publish_joints(self):
        """Publie joint_states depuis Isaac Lab"""
        if self.isaac_ready:
            try:
                pos = self.robot.data.joint_pos[0].cpu().numpy()
                vel = self.robot.data.joint_vel[0].cpu().numpy()
                
                msg = JointState()
                msg.header = Header()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.name = self.joint_names
                msg.position = pos[:6].tolist()
                msg.velocity = vel[:6].tolist()
                msg.effort = [0.0] * 6
                
                self.joint_pub.publish(msg)
                
            except Exception as e:
                self.get_logger().warn(f'Erreur lecture: {e}', throttle_duration_sec=5.0)
        else:
            # Publier des valeurs par défaut
            msg = JointState()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = self.joint_names
            msg.position = self.joint_positions
            msg.velocity = self.joint_velocities
            msg.effort = [0.0] * 6
            self.joint_pub.publish(msg)
    
    def run_isaac(self):
        """Boucle Isaac Lab"""
        while rclpy.ok() and simulation_app.is_running():
            if self.isaac_ready:
                try:
                    dt = self.sim.get_physics_dt()
                    self.robot.write_data_to_sim()
                    self.sim.step()
                    self.robot.update(dt)
                    time_module.sleep(dt)
                except Exception as e:
                    self.get_logger().error(f'Erreur sim: {e}', throttle_duration_sec=5.0)
            else:
                time_module.sleep(0.1)


def main(args=None):
    rclpy.init(args=args)
    
    bridge = SimpleIsaacBridge()
    bridge.setup_isaac()
    
    if not bridge.isaac_ready:
        bridge.get_logger().error('❌ Isaac Lab non démarré')
        simulation_app.close()
        return
    
    # Thread Isaac
    isaac_thread = threading.Thread(target=bridge.run_isaac, daemon=True)
    isaac_thread.start()
    
    bridge.get_logger().info('✅ Bridge actif! Topics:')
    bridge.get_logger().info('  - /joint_states (publié)')
    bridge.get_logger().info('')
    bridge.get_logger().info('Test avec: ros2 topic echo /joint_states')
    
    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy_node()
        rclpy.shutdown()
        simulation_app.close()


if __name__ == '__main__':
    main()
