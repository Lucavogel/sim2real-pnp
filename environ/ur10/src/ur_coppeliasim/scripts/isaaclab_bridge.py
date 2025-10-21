#!/usr/bin/env python3
"""
Bridge ROS2 <-> Isaac Lab pour robot UR10
Ce script fait le lien entre MoveIt2 et Isaac Lab via les topics ROS2
Inspiré du bridge CoppeliaSim mais adapté pour Isaac Lab
"""

import sys
import os
import numpy as np
import time as time_module
import threading

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from control_msgs.action import FollowJointTrajectory
from std_msgs.msg import Header

# Isaac Lab imports
import torch
import argparse
from isaaclab.app import AppLauncher

# Create argparser for Isaac Lab
parser = argparse.ArgumentParser(description="Isaac Lab Bridge for ROS2/MoveIt2")
parser.add_argument("--device", type=str, default="cuda:0", help="Device to run Isaac Lab on")
AppLauncher.add_app_launcher_args(parser)

# Parse arguments (will be set by launch file or defaults)
args_cli, unknown = parser.parse_known_args()

# Launch Isaac Sim/Lab application
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import Isaac Lab modules AFTER launching app
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets import UR10_CFG


class IsaacLabBridge(Node):
    """Bridge entre ROS2/MoveIt2 et Isaac Lab pour contrôle du robot UR10"""
    
    def __init__(self):
        super().__init__('isaaclab_bridge')
        
        # Paramètres ROS2
        self.declare_parameter('ur_type', 'ur10')
        self.declare_parameter('joint_names', [
            'elbow_joint',
            'shoulder_lift_joint',
            'shoulder_pan_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ])
        self.declare_parameter('update_rate', 50.0)  # Hz
        
        self.ur_type = self.get_parameter('ur_type').value
        self.joint_names = self.get_parameter('joint_names').value
        self.update_rate = self.get_parameter('update_rate').value
        
        # État des joints (sera synchronisé avec Isaac Lab)
        self.joint_positions = [0.0] * len(self.joint_names)
        self.joint_velocities = [0.0] * len(self.joint_names)
        self.joint_efforts = [0.0] * len(self.joint_names)
        
        # Lock pour accès thread-safe à l'état des joints
        self.state_lock = threading.Lock()
        
        # Isaac Lab components (seront initialisés dans setup_isaac_lab)
        self.sim = None
        self.robot = None
        self.isaac_ready = False
        
        self.get_logger().info(f'Isaac Lab Bridge pour robot {self.ur_type}')
        
        # Callback group pour exécution parallèle
        self.callback_group = ReentrantCallbackGroup()
        
        # Publisher pour /joint_states
        self.joint_state_pub = self.create_publisher(
            JointState,
            'joint_states',
            10
        )
        
        # Action server pour trajectory controller
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory',
            self.execute_trajectory_callback,
            callback_group=self.callback_group
        )
        
        # Timer pour publier les joint states
        self.create_timer(
            1.0 / self.update_rate,
            self.publish_joint_states,
            callback_group=self.callback_group
        )
        
        self.get_logger().info('Bridge Isaac Lab prêt! Initialisation de la scène...')
    
    def setup_isaac_lab(self):
        """Configure la simulation Isaac Lab avec le robot UR10"""
        try:
            # Créer le contexte de simulation
            sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
            self.sim = sim_utils.SimulationContext(sim_cfg)
            
            # Configurer la vue de la caméra
            self.sim.set_camera_view([3.0, 3.0, 2.0], [0.0, 0.0, 1.0])
            
            # Créer le sol
            cfg_ground = sim_utils.GroundPlaneCfg()
            cfg_ground.func("/World/defaultGroundPlane", cfg_ground)
            
            # Lumière
            cfg_light = sim_utils.DistantLightCfg(
                intensity=3000.0,
                color=(0.75, 0.75, 0.75),
            )
            cfg_light.func("/World/lightDistant", cfg_light, translation=(1, 0, 10))
            
            # Créer le robot UR10
            ur10_cfg = UR10_CFG.replace(prim_path="/World/UR10_Robot")
            ur10_cfg.init_state.pos = (0.0, 0.0, 0.0)
            
            # Appliquer la configuration initiale des joints (position neutre)
            ur10_cfg.init_state.joint_pos = {
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -1.57,  # -90°
                "elbow_joint": 1.57,            # +90°
                "wrist_1_joint": -1.57,         # -90°
                "wrist_2_joint": -1.57,         # -90°
                "wrist_3_joint": 0.0,
            }
            
            self.robot = Articulation(cfg=ur10_cfg)
            
            # Reset de la simulation
            self.sim.reset()
            
            self.get_logger().info('✓ Isaac Lab initialisé avec succès!')
            self.isaac_ready = True
            
            # Lire l'état initial des joints
            self.update_joint_state_from_isaac()
            
        except Exception as e:
            self.get_logger().error(f'❌ Erreur initialisation Isaac Lab: {e}')
            import traceback
            self.get_logger().error(f'Traceback: {traceback.format_exc()}')
            self.isaac_ready = False
    
    def update_joint_state_from_isaac(self):
        """Lit l'état actuel des joints depuis Isaac Lab"""
        if not self.isaac_ready or self.robot is None:
            return
        
        try:
            # Récupérer les positions des joints depuis Isaac Lab
            # robot.data.joint_pos est un tensor [num_instances, num_joints]
            joint_pos = self.robot.data.joint_pos[0].cpu().numpy()  # Premier robot (instance 0)
            joint_vel = self.robot.data.joint_vel[0].cpu().numpy()
            
            with self.state_lock:
                # Isaac Lab donne les joints dans l'ordre de la configuration
                # Vérifier que l'ordre correspond à self.joint_names
                for i in range(min(len(self.joint_names), len(joint_pos))):
                    self.joint_positions[i] = float(joint_pos[i])
                    self.joint_velocities[i] = float(joint_vel[i])
                    # Efforts non disponibles directement, on peut les laisser à 0
                    self.joint_efforts[i] = 0.0
                    
        except Exception as e:
            self.get_logger().warn(f'Erreur lecture état Isaac Lab: {e}', throttle_duration_sec=5.0)
    
    def publish_joint_states(self):
        """Publie l'état actuel des joints sur ROS2"""
        # Mettre à jour depuis Isaac Lab
        self.update_joint_state_from_isaac()
        
        # Publier JointState
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        with self.state_lock:
            msg.name = self.joint_names
            msg.position = self.joint_positions.copy()
            msg.velocity = self.joint_velocities.copy()
            msg.effort = self.joint_efforts.copy()
        
        self.joint_state_pub.publish(msg)
    
    def execute_trajectory_callback(self, goal_handle):
        """Exécute une trajectoire reçue de MoveIt2"""
        self.get_logger().info('🎯 Trajectoire MoveIt2 reçue!')
        
        trajectory = goal_handle.request.trajectory
        
        if not self.isaac_ready:
            self.get_logger().error('❌ Isaac Lab non initialisé!')
            result = FollowJointTrajectory.Result()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            goal_handle.abort()
            return result
        
        try:
            self.get_logger().info(f'📊 Exécution de {len(trajectory.points)} points de trajectoire')
            
            # Exécuter chaque point de la trajectoire
            previous_time = 0.0
            
            for idx, point in enumerate(trajectory.points):
                # Positions cibles
                target_positions = np.array(point.positions)
                
                # Calculer le temps à attendre
                time_from_start = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
                wait_duration = time_from_start - previous_time
                previous_time = time_from_start
                
                # Log
                angles_deg = np.degrees(target_positions)
                self.get_logger().info(
                    f'Point {idx+1}/{len(trajectory.points)}: '
                    f'[{angles_deg[0]:.1f}°, {angles_deg[1]:.1f}°, {angles_deg[2]:.1f}°, '
                    f'{angles_deg[3]:.1f}°, {angles_deg[4]:.1f}°, {angles_deg[5]:.1f}°], '
                    f'attente={wait_duration:.2f}s'
                )
                
                # Envoyer les positions à Isaac Lab
                self.set_joint_positions_isaac(target_positions)
                
                # Simuler pendant la durée nécessaire
                if wait_duration > 0:
                    self.step_simulation(wait_duration)
            
            self.get_logger().info('✅ Trajectoire exécutée avec succès!')
            
            result = FollowJointTrajectory.Result()
            result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
            goal_handle.succeed()
            return result
            
        except Exception as e:
            self.get_logger().error(f'❌ ERREUR exécution trajectoire: {e}')
            import traceback
            self.get_logger().error(f'Traceback: {traceback.format_exc()}')
            result = FollowJointTrajectory.Result()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            goal_handle.abort()
            return result
    
    def set_joint_positions_isaac(self, positions):
        """Envoie des positions cibles au robot dans Isaac Lab"""
        if not self.isaac_ready or self.robot is None:
            return
        
        try:
            # Convertir en tensor Isaac Lab [num_instances, num_joints]
            target_tensor = torch.tensor([positions], device=args_cli.device, dtype=torch.float32)
            
            # Définir les positions cibles pour le robot
            # Isaac Lab utilise set_joint_position_target pour le contrôle
            self.robot.set_joint_position_target(target_tensor)
            
        except Exception as e:
            self.get_logger().error(f'Erreur envoi positions Isaac Lab: {e}')
    
    def step_simulation(self, duration):
        """Fait avancer la simulation Isaac Lab pendant une durée donnée"""
        if not self.isaac_ready or self.sim is None:
            return
        
        dt = self.sim.get_physics_dt()
        num_steps = int(duration / dt)
        
        for _ in range(num_steps):
            # Write robot actions (les targets ont déjà été définis)
            self.robot.write_data_to_sim()
            
            # Step simulation
            self.sim.step()
            
            # Update robot state
            self.robot.update(dt)
    
    def run_isaac_loop(self):
        """Boucle principale Isaac Lab (dans un thread séparé)"""
        while rclpy.ok() and simulation_app.is_running():
            if self.isaac_ready:
                try:
                    # Step simulation à la fréquence Isaac Lab
                    dt = self.sim.get_physics_dt()
                    
                    # Write current targets
                    self.robot.write_data_to_sim()
                    
                    # Step
                    self.sim.step()
                    
                    # Update robot
                    self.robot.update(dt)
                    
                    # Petite pause pour ne pas surcharger
                    time_module.sleep(dt)
                    
                except Exception as e:
                    self.get_logger().error(f'Erreur boucle Isaac: {e}', throttle_duration_sec=5.0)
            else:
                time_module.sleep(0.1)


def main(args=None):
    """Point d'entrée principal"""
    rclpy.init(args=args)
    
    # Créer le bridge
    bridge = IsaacLabBridge()
    
    # Initialiser Isaac Lab dans le thread principal
    bridge.setup_isaac_lab()
    
    if not bridge.isaac_ready:
        bridge.get_logger().error('❌ Impossible de démarrer Isaac Lab')
        simulation_app.close()
        return
    
    # Créer un executor multi-thread pour ROS2
    executor = MultiThreadedExecutor()
    executor.add_node(bridge)
    
    # Lancer la boucle Isaac Lab dans un thread séparé
    isaac_thread = threading.Thread(target=bridge.run_isaac_loop, daemon=True)
    isaac_thread.start()
    
    bridge.get_logger().info('🚀 Bridge Isaac Lab <-> ROS2 actif!')
    
    try:
        # Spin ROS2 dans le thread principal
        executor.spin()
    except KeyboardInterrupt:
        bridge.get_logger().info('Arrêt demandé...')
    finally:
        bridge.destroy_node()
        rclpy.shutdown()
        simulation_app.close()


if __name__ == '__main__':
    main()
