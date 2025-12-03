"""
Script pour connecter le robot UR10 dans Isaac Lab à MoveIt2 via ROS2.
Utilise un bridge Python pour publier /joint_states et écouter /joint_command.

Usage:
    cd ~/work2/IsaacLab
    ./isaaclab.sh -p /home/ajin/work2/my_env/scripts/zero_agent.py
    
    Puis dans un autre terminal:
    cd ~/work2/ur10 && source install/setup.bash
    ros2 launch ur_coppeliasim ur_isaaclab_moveit.launch.py
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import threading

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="UR10 robot control script with ROS2 bridge.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app with ROS2 enabled
app_launcher = AppLauncher(args_cli, enable_ros2_bridge=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils 
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab_assets import UR10_CFG
# ROS2 imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header


class JointStateBridge(Node):
    """Bridge minimal pour publier /joint_states et recevoir /joint_command"""
    
    def __init__(self, robot: Articulation):
        super().__init__('ur10_joint_state_bridge')
        self.robot = robot
        
        # Noms des joints dans l'ordre MoveIt2/ROS2 standard
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        # Publisher /joint_states
        self.joint_state_pub = self.create_publisher(JointState, 'joint_states', 10)
        
        # Timer pour publier à 50Hz
        self.create_timer(0.02, self.publish_joint_states)
        
        # Cible de position (sera mise à jour par MoveIt2)
        self.target_positions = None
        
        # Subscriber /joint_command (optionnel, pour compatibilité)
        self.create_subscription(JointState, 'joint_command', self.joint_command_callback, 10)
        
        self.get_logger().info('✅ Bridge ROS2 actif!')
    
    def publish_joint_states(self):
        """Publie l'état actuel des joints"""
        try:
            # Vérifier que le robot est initialisé
            if self.robot.data.joint_pos is None:
                return
            
            # Lire positions depuis Isaac Lab
            joint_pos = self.robot.data.joint_pos[0].cpu().numpy()
            joint_vel = self.robot.data.joint_vel[0].cpu().numpy()
            
            # Créer message ROS2
            msg = JointState()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = self.joint_names
            msg.position = joint_pos.tolist()
            msg.velocity = joint_vel.tolist()
            msg.effort = [0.0] * 6
            
            self.joint_state_pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f'Error publishing joint states: {e}', throttle_duration_sec=5.0)
    
    def joint_command_callback(self, msg: JointState):
        """Reçoit des commandes de position (optionnel)"""
        if len(msg.position) == 6:
            self.target_positions = torch.tensor(
                [msg.position], 
                device=self.robot.device, 
                dtype=torch.float32
            )


def main():
    """Main function."""
    # Load USD stage FIRST (before creating SimulationContext)
    import omni.usd
    import omni.graph.core as og

    usd_path = "/home/ajin/work2/my_env/env_v3.usd"
    print(f"[INFO] Loading USD file: {usd_path}")
    omni.usd.get_context().open_stage(usd_path)
    
    # Désactiver les Action Graphs ROS2 du USD s'ils existent (pour éviter les conflits)
    try:
        stage = omni.usd.get_context().get_stage()
        # Lister tous les graphs pour les désactiver
        for prim in stage.Traverse():
            if prim.GetTypeName() == "OmniGraph":
                prim_path = str(prim.GetPath())
                if "ROS" in prim_path or "Joint" in prim_path:
                    print(f"[INFO] Désactivation du graph: {prim_path}")
                    # Pas de méthode directe, on laisse faire
    except Exception as e:
        print(f"[WARN] Could not disable USD graphs: {e}")
    
    # Setup simulation context AFTER loading USD
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

  


    # Set camera view
    sim.set_camera_view([0.5, 0.0, 2.5], 
                        [0.0, -1.0, 0.5])
    
    # Reference the UR10 robot that already exists in the USD file
    robot_prim_path = "/World/Origin2/Table/Robot"
    ur10_cfg = UR10_CFG.replace(prim_path=robot_prim_path)
    ur10 = Articulation(cfg=ur10_cfg)
    
    # Play simulation (this initializes physics)
    sim.reset()
    print("[INFO] Setup complete...")
    
    # Initialize targets to current position
    # Option 1: Définir une position personnalisée en radians
    current_joint_pos = torch.tensor([[
        0.0,           # shoulder_pan_joint (rotation base)
        -0.79,         # shoulder_lift_joint (-45°)
        1.39,          # elbow_joint (90°)
        -2.36,         # wrist_1_joint (-90°)
        -1.57,         # wrist_2_joint (-90°)
        0.0            # wrist_3_joint
    ]], device=ur10.device, dtype=torch.float32)

    # Option 2: Ou en degrés (plus lisible)
    # import math
    # current_joint_pos = torch.tensor([[
    #     math.radians(0),      # shoulder_pan
    #     math.radians(-90),    # shoulder_lift
    #     math.radians(90),     # elbow
    #     math.radians(-90),    # wrist_1
    #     math.radians(-90),    # wrist_2
    #     math.radians(0)       # wrist_3
    # ]], device=ur10.device, dtype=torch.float32)

    ur10.set_joint_position_target(current_joint_pos)
    ur10.write_data_to_sim()
    
    # IMPORTANT: Step simulation plusieurs fois pour initialiser complètement le robot
    print("[INFO] Initializing robot...")
    for _ in range(10):
        ur10.write_data_to_sim()
        sim.step()
        ur10.update(dt=sim.get_physics_dt())
    print("[INFO] Robot initialized!")
    
    # Initialiser ROS2
    rclpy.init()
    
    # Créer le bridge ROS2
    ros_bridge = JointStateBridge(ur10)
    
    # Thread ROS2 pour spin
    def spin_ros():
        try:
            rclpy.spin(ros_bridge)
        except Exception as e:
            print(f"ROS2 error: {e}")
    
    ros_thread = threading.Thread(target=spin_ros, daemon=True)
    ros_thread.start()
    
    print("\n" + "=" * 60)
    print("✅ SYSTÈME PRÊT - UR10 + MoveIt2 via ROS2")
    print("=" * 60)
    print("Isaac Lab: Robot UR10 simulé")
    print("ROS2: Bridge Python actif")
    print("  - /joint_states publié (50Hz)")
    print("  - /joint_command écouté")
    print("")
    print("Lancez MoveIt2:")
    print("  cd ~/work2/ur10 && source install/setup.bash")
    print("  ros2 launch ur_coppeliasim ur_isaaclab_moveit.launch.py")
    print("=" * 60 + "\n")
    
    # Simulation loop
    count = 0

    try:
        while simulation_app.is_running() and rclpy.ok():
            # Appliquer la cible si elle existe
            if ros_bridge.target_positions is not None:
                ur10.set_joint_position_target(ros_bridge.target_positions)

            # Écrire les commandes dans la simulation
            ur10.write_data_to_sim()

            # Faire avancer la physique
            sim.step()

            # Mettre à jour l'état du robot
            ur10.update(dt=sim.get_physics_dt())

            # Log périodique de la position des joints
            if count % 500 == 0:
                joint_pos = ur10.data.joint_pos[0].cpu().numpy()
                joint_pos_deg = np.degrees(joint_pos)
                print(f"[{count:05d}] Joints (deg): "
                    f"[{joint_pos_deg[0]:.1f}, {joint_pos_deg[1]:.1f}, {joint_pos_deg[2]:.1f}, "
                    f"{joint_pos_deg[3]:.1f}, {joint_pos_deg[4]:.1f}, {joint_pos_deg[5]:.1f}]")


            count += 1

    finally:
        # Cleanup ROS2
        ros_bridge.destroy_node()
        rclpy.shutdown()
    
    print("\n[INFO] Closing...")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        raise e
    finally:
        simulation_app.close()

