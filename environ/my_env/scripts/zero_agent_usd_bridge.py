"""
Script simplifié pour UR10 avec ROS2 bridge natif Isaac Sim.
Le Action Graph ROS2 est déjà configuré dans le USD (env_v1.usd).

Usage:
    ./environ/my_env/scripts/launch_ur10_ros_native.sh
    
    Puis dans un autre terminal:
    source /opt/ros/humble/setup.bash
    ros2 topic list
    ros2 topic echo /joint_states
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="UR10 robot with native ROS2 bridge.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app WITH ROS2 bridge enabled
app_launcher = AppLauncher(args_cli, enable_ros2_bridge=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab_assets import UR10_CFG


def main():
    """Main function."""
    
    # Import Omniverse modules AFTER app is launched
    import omni.usd
    
    # Load USD stage FIRST (le Action Graph ROS2 est déjà dedans!)
    usd_path = "/home/ajin/workspace/sim2real-pnp/environ/my_env/source/env_v1.usd"
    print(f"[INFO] Loading USD file: {usd_path}")
    omni.usd.get_context().open_stage(usd_path)
    
    # Setup simulation context AFTER loading USD
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    
    # Set camera view
    sim.set_camera_view([3.0, 3.0, 2.5], [0.0, 0.0, 0.5])
    
    # Reference the UR10 robot that already exists in the USD file
    robot_prim_path = "/World/Origin2/Table/Robot"
    ur10_cfg = UR10_CFG.replace(prim_path=robot_prim_path)
    ur10 = Articulation(cfg=ur10_cfg)
    
    # Play simulation (this initializes physics)
    sim.reset()
    print("[INFO] Setup complete...")
    
    # Initialize targets to current position
    current_joint_pos = ur10.data.default_joint_pos.clone()
    ur10.set_joint_position_target(current_joint_pos)
    ur10.write_data_to_sim()
    
    # IMPORTANT: Step simulation plusieurs fois pour initialiser complètement le robot
    print("[INFO] Initializing robot...")
    for _ in range(10):
        ur10.write_data_to_sim()
        sim.step()
        ur10.update(dt=sim.get_physics_dt())
    print("[INFO] Robot initialized!")
    
    print("\n" + "=" * 70)
    print("✅ SYSTÈME PRÊT - UR10 avec ROS2 Bridge Natif Isaac Sim")
    print("=" * 70)
    print("Isaac Lab: Robot UR10 simulé")
    print("ROS2: Bridge natif Isaac Sim via Action Graph (dans USD)")
    print("  📡 Publisher: /joint_states (automatique)")
    print("  📥 Subscriber: /joint_command (avec Move Robot activé)")
    print("")
    print("Testez dans un autre terminal:")
    print("  source /opt/ros/humble/setup.bash")
    print("  ros2 topic list")
    print("  ros2 topic echo /joint_states")
    print("")
    print("Pour commander le robot:")
    print("  ros2 topic pub /joint_command sensor_msgs/msg/JointState ...")
    print("")
    print("Pour MoveIt2:")
    print("  cd ~/workspace/ur10 && source install/setup.bash")
    print("  ros2 launch ur_coppeliasim ur_isaaclab_moveit.launch.py")
    print("=" * 70 + "\n")
    
    # Simulation loop
    count = 0
    try:
        while simulation_app.is_running():
            # Write robot commands to simulation
            ur10.write_data_to_sim()
            
            # Step simulation physics
            sim.step()
            
            # Update robot state
            ur10.update(dt=sim.get_physics_dt())
            
            # Log périodique
            if count % 500 == 0:
                joint_pos = ur10.data.joint_pos[0].cpu().numpy()
                joint_pos_deg = np.degrees(joint_pos)
                print(f"[{count:05d}] Joints (deg): " + 
                      f"[{joint_pos_deg[0]:.1f}, {joint_pos_deg[1]:.1f}, {joint_pos_deg[2]:.1f}, " +
                      f"{joint_pos_deg[3]:.1f}, {joint_pos_deg[4]:.1f}, {joint_pos_deg[5]:.1f}]")
            
            count += 1
            
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    print("\n[INFO] Closing...")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
