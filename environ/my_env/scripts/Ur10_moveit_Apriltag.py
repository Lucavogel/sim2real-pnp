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
import threading
import traceback

from isaaclab.app import AppLauncher


# add argparse arguments
parser = argparse.ArgumentParser(description="UR10 robot with native ROS2 bridge.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app WITH ROS2 bridge enabled
app_launcher = AppLauncher(args_cli, enable_ros2_bridge=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab_assets import UR10_CFG

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import PoseStamped, PoseArray
from std_msgs.msg import Bool
import numpy as np

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from std_msgs.msg import Header
from rclpy.action import ActionClient


class MoveItBridge(Node):
    """Bridge ROS2 pour recevoir des trajectoires de MoveIt2 et contrôler le robot Isaac Lab"""
    
    def __init__(self, robot: Articulation, sim: SimulationContext):
        super().__init__('moveit_bridge')
        
        self.robot = robot
        self.sim = sim
        
        # Noms des joints UR10 (ordre ROS2/MoveIt2 standard)
        self.joint_names = [
            'elbow_joint',
            'shoulder_lift_joint', 
            'shoulder_pan_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        # MAPPING: Isaac Lab index → ROS2 index (du script qui marche!)
        self.isaac_to_ros2 = [2, 1, 0, 3, 4, 5]  # Swap moteurs 0 et 2
        self.ros2_to_isaac = [2, 1, 0, 3, 4, 5]  # Inverse
        
        # INVERSIONS de direction des moteurs
        self.joint_directions = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        
        self.get_logger().info(f'🔄 Mapping Isaac→ROS2: {self.isaac_to_ros2}')
        self.get_logger().info(f'🔄 Directions: {self.joint_directions.tolist()}')
        
        # Trajectoire en cours d'exécution
        self.current_trajectory = None
        self.trajectory_index = 0
        self.trajectory_start_time = None
        self.executing_trajectory = False
        self.trajectory_success = False
        self.current_goal_handle = None
        
        # Lock pour thread safety
        self.trajectory_lock = threading.Lock()
        
        # Callback group pour exécution parallèle
        callback_group = ReentrantCallbackGroup()
        
        # PUBLISHER /joint_states (on le fait en Python, pas via USD!)
        self.joint_state_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.create_timer(0.02, self.publish_joint_states, callback_group=callback_group)  # 50Hz
        
        # PUBLISHER /start_motion (flag pour déclencher les mouvements depuis Isaac Lab)
        self.motion_flag_pub = self.create_publisher(Bool, '/start_motion', 10)
        self.motion_trigger_sent = False  # Pour envoyer un pulse au lieu d'un état
        
        # Action server pour MoveIt2 (interface standard pour trajectory execution)
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory',
            self.execute_trajectory_callback,
            callback_group=callback_group
        )
        
        # Subscriber simple pour /joint_command (backup, si pas d'action server)
        self.joint_command_sub = self.create_subscription(
            JointState,
            'joint_command',
            self.joint_command_callback,
            10,
            callback_group=callback_group
        )
        
        self.get_logger().info('✅ Isaac Lab Bridge initialisé!')
        self.get_logger().info('  - Publisher: /joint_states (50Hz avec mapping)')
        self.get_logger().info('  - Publisher: /start_motion (trigger par ESPACE)')
        self.get_logger().info('  - Action Server: /joint_trajectory_controller/follow_joint_trajectory')
        self.get_logger().info('  - Subscriber: /joint_command (backup)')
        self.get_logger().info('')
        self.get_logger().info('🤖 Ce bridge reçoit les trajectoires de MoveIt2')
        self.get_logger().info('⌨️  Appuyez sur ESPACE pour déclencher un mouvement vers le prochain tag')
        self.get_logger().info('📡 Lancez le launch file pour MoveIt + AprilTag + Auto Mover')
    
    def trigger_motion(self):
        """Trigger un mouvement (appelé par la touche ESPACE)"""
        # Envoyer un pulse True puis False
        msg = Bool()
        msg.data = True
        self.motion_flag_pub.publish(msg)
        self.get_logger().info('🎮 TRIGGER → Mouvement vers prochain tag!')
    
    def publish_joint_states(self):
        """Publie l'état actuel des joints (appelé à 50Hz) AVEC MAPPING"""
        try:
            # Lire positions depuis Isaac Lab (ordre physique Isaac)
            joint_pos_isaac = self.robot.data.joint_pos[0].cpu().numpy()
            joint_vel_isaac = self.robot.data.joint_vel[0].cpu().numpy()
            
            # Appliquer le mapping Isaac → ROS2 (comme dans Ur10_ros.py)
            joint_pos_ros2 = [0.0] * 6
            joint_vel_ros2 = [0.0] * 6
            for isaac_idx in range(6):
                ros2_idx = self.isaac_to_ros2[isaac_idx]
                # Appliquer direction
                joint_pos_ros2[ros2_idx] = float(joint_pos_isaac[isaac_idx] * self.joint_directions[isaac_idx])
                joint_vel_ros2[ros2_idx] = float(joint_vel_isaac[isaac_idx] * self.joint_directions[isaac_idx])
            
            # Créer message ROS2
            msg = JointState()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = self.joint_names
            msg.position = joint_pos_ros2
            msg.velocity = joint_vel_ros2
            msg.effort = [0.0] * 6
            
            self.joint_state_pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f'Erreur publication joint_states: {e}', throttle_duration_sec=5.0)
    
    def joint_command_callback(self, msg: JointState):
        """Callback simple pour recevoir des positions cibles (sans trajectoire temporelle)"""
        if len(msg.position) == 6:
            self.get_logger().info(f'📥 Commande reçue via /joint_command')
            try:
                # Convertir en tensor et appliquer au robot
                target_positions = torch.tensor(
                    [msg.position], 
                    device=self.robot.device, 
                    dtype=torch.float32
                )
                self.robot.set_joint_position_target(target_positions)
                self.get_logger().info(f'✅ Position cible appliquée: {np.degrees(msg.position)}°')
            except Exception as e:
                self.get_logger().error(f'❌ Erreur application commande: {e}')
    
    def execute_trajectory_callback(self, goal_handle):
        """
        Callback pour l'action server - stocke la trajectoire pour exécution dans le main loop
        CRITICAL: Ne PAS appeler sim.step() ici! Cela cause des erreurs PhysX.
        """
        self.get_logger().info('🎯 Trajectoire MoveIt2 reçue via Action Server!')
        
        trajectory = goal_handle.request.trajectory
        
        try:
            self.get_logger().info(f'📊 Stockage de {len(trajectory.points)} points de trajectoire')
            
            # Stocker la trajectoire pour exécution dans le main loop
            with self.trajectory_lock:
                self.current_trajectory = trajectory
                self.trajectory_index = 0
                self.trajectory_start_time = None
                self.executing_trajectory = True
                self.current_goal_handle = goal_handle
            
            # Attendre que la trajectoire soit exécutée (par le main loop)
            while self.executing_trajectory and rclpy.ok():
                import time
                time.sleep(0.01)  # Check toutes les 10ms
            
            # Trajectoire terminée
            if hasattr(self, 'trajectory_success') and self.trajectory_success:
                self.get_logger().info('✅ Trajectoire exécutée avec succès!')
                result = FollowJointTrajectory.Result()
                result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                goal_handle.succeed()
                return result
            else:
                raise Exception("Trajectory execution failed or was interrupted")
            
        except Exception as e:
            self.get_logger().error(f'❌ Erreur exécution trajectoire: {e}')
            import traceback
            self.get_logger().error(f'Traceback: {traceback.format_exc()}')
            
            with self.trajectory_lock:
                self.executing_trajectory = False
            
            result = FollowJointTrajectory.Result()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            goal_handle.abort()
            return result
    
    def update_trajectory(self, current_time: float):
        """
        Appelée depuis le main loop - met à jour la position du robot selon la trajectoire
        Cette méthode est safe car elle est appelée AVANT sim.step()
        """
        with self.trajectory_lock:
            if not self.executing_trajectory or self.current_trajectory is None:
                return
            
            # Initialiser le temps de départ
            if self.trajectory_start_time is None:
                self.trajectory_start_time = current_time
                self.get_logger().info(f'🚀 Début exécution trajectoire à t={current_time:.2f}s')
            
            elapsed = current_time - self.trajectory_start_time
            
            # Trouver le waypoint actuel
            trajectory = self.current_trajectory
            points = trajectory.points
            
            # Si on a terminé tous les points
            if self.trajectory_index >= len(points):
                self.get_logger().info(f'✅ Trajectoire terminée! ({len(points)} points)')
                self.executing_trajectory = False
                self.trajectory_success = True
                self.current_trajectory = None
                return
            
            # Récupérer le point actuel
            point = points[self.trajectory_index]
            point_time = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
            
            # Si on a atteint ou dépassé ce point
            if elapsed >= point_time:
                # Appliquer les positions avec mapping ROS2 → Isaac
                if len(point.positions) == 6:
                    # Convertir positions ROS2 vers ordre Isaac
                    isaac_positions = [0.0] * 6
                    for ros2_idx in range(6):
                        isaac_idx = self.ros2_to_isaac[ros2_idx]
                        # Appliquer direction inverse
                        isaac_positions[isaac_idx] = point.positions[ros2_idx] * self.joint_directions[isaac_idx]
                    
                    target_positions = torch.tensor(
                        [isaac_positions], 
                        device=self.robot.device, 
                        dtype=torch.float32
                    )
                    self.robot.set_joint_position_target(target_positions)
                    
                    # Log plus détaillé
                    isaac_deg = np.degrees(isaac_positions)
                    ros2_deg = np.degrees(point.positions)
                    self.get_logger().info(
                        f'  ▶️ Point {self.trajectory_index+1}/{len(points)} appliqué: '
                        f't={point_time:.2f}s (elapsed={elapsed:.2f}s)'
                    )
                    self.get_logger().info(
                        f'     ROS2 (deg): [{ros2_deg[0]:.1f}, {ros2_deg[1]:.1f}, {ros2_deg[2]:.1f}, {ros2_deg[3]:.1f}, {ros2_deg[4]:.1f}, {ros2_deg[5]:.1f}]'
                    )
                    self.get_logger().info(
                        f'     Isaac (deg): [{isaac_deg[0]:.1f}, {isaac_deg[1]:.1f}, {isaac_deg[2]:.1f}, {isaac_deg[3]:.1f}, {isaac_deg[4]:.1f}, {isaac_deg[5]:.1f}]'
                    )
                
                # Passer au point suivant
                self.trajectory_index += 1


def main():
    """Main function."""
    
    # Import Omniverse modules AFTER app is launched
    import omni.usd
    import omni.graph.core as og
    import carb
    
    # Limiter les FPS pour réduire la charge GPU
    settings = carb.settings.get_settings()
    settings.set("/app/runLoops/main/rateLimitEnabled", True)
    settings.set("/app/runLoops/main/rateLimitFrequency", 20)  # 20 FPS au lieu de 60
    
    # Load USD stage FIRST (le Action Graph ROS2 est déjà dedans!)
<<<<<<< HEAD
    usd_path = "/home/ajin/work2/sim2real-pnp/environ/my_env/venv/env_v1.usd"
=======
    usd_path = "/home/ajin/workspace/sim2real-pnp/environ/my_env/source/env_v2.usd"
>>>>>>> origin/luca
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
    
    # CRITICAL: Force le device CUDA pour éviter les erreurs "expected device 0, received device -1"
    # On doit recréer l'articulation APRÈS le reset pour qu'elle soit sur le bon device
    
    ur10 = Articulation(cfg=ur10_cfg)
    
    print("[INFO] Native Isaac Sim ROS2 bridge actif")
    print("[INFO] /joint_states publisher Python (pas USD)")
    
    # Play simulation (this initializes physics)
    sim.reset()
    print("[INFO] Setup complete...")
    
    terget_joint_pose = torch.tensor([[0.0, -0.9, 1.6, -2.3, -1.57, 0.0]], 
                                  device=ur10.device, dtype=torch.float32)
    
   
    # ============================================================
    
    ur10.set_joint_position_target(terget_joint_pose)
    ur10.write_data_to_sim()
    
    # IMPORTANT: Step simulation plusieurs fois pour initialiser complètement le robot
    print("[INFO] Initializing robot...")
    for _ in range(50):  # Plus d'itérations pour bien initialiser
        ur10.write_data_to_sim()
        sim.step()
        ur10.update(dt=sim.get_physics_dt())
    print("[INFO] Robot initialized!")
    
    # Vérifier que le robot est bien sur CUDA
    print(f"[INFO] Robot device: {ur10.device}")
    print(f"[INFO] Joint positions shape: {ur10.data.joint_pos.shape}")
    print(f"[INFO] Joint positions device: {ur10.data.joint_pos.device}")
    
    # Initialiser ROS2
    rclpy.init()
    
    # Créer le bridge MoveIt2 (pour recevoir les trajectoires)
    moveit_bridge = MoveItBridge(ur10, sim)
    
    # Thread ROS2 pour spin
    def spin_ros():
        try:
            rclpy.spin(moveit_bridge)
        except Exception as e:
            print(f"[ERROR] ROS2 error: {e}")
    
    ros_thread = threading.Thread(target=spin_ros, daemon=True)
    ros_thread.start()
    
    print("\n" + "=" * 70)
    print("✅ SYSTÈME PRÊT - UR10 avec ROS2 Bridge Python + MoveIt2")
    print("=" * 70)
    print("Isaac Lab: Robot UR10 simulé")
    print("ROS2: Bridge Python natif (rclpy)")
    print("  📡 Publisher: /joint_states (50Hz, depuis Python)")
    print("  🎮 Publisher: /start_motion (contrôlé par ESPACE)")
    print("   Action Server: /joint_trajectory_controller/follow_joint_trajectory")
    print("  📥 Subscriber: /joint_command (backup)")
    print("")
    print("⌨️  CONTRÔLES CLAVIER:")
    print("  ESPACE = Déclencher un mouvement vers le prochain tag")
    print("")
    print("Testez dans un autre terminal:")
    print("  source /opt/ros/humble/setup.bash")
    print("  ros2 topic list")
    print("  ros2 topic echo /joint_states")
    print("")
    print("Pour MoveIt2:")
    print("  cd ~/workspace/sim2real-pnp/environ/ur10")
    print("  source install/setup.bash")
    print("  ros2 launch ur_coppeliasim ur_isaaclab_moveit.launch.py")
    print("")
    print("Test manuel de commande:")
    print("  ros2 topic pub --once /joint_command sensor_msgs/msg/JointState \\")
    print("    '{name: [shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint], \\")
    print("     position: [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]}'")
    print("=" * 70 + "\n")
    
    # Simulation loop
    count = 0
    simulation_time = 0.0
    
    # Variable pour éviter les double-presses
    space_was_pressed = False
    
    # Import des modules carb après le lancement de l'app
    import carb
    from pxr import Sdf
    
    try:
        while simulation_app.is_running() and rclpy.ok():
            # Vérifier si ESPACE est appuyé via carb
            try:
                input_provider = carb.input.acquire_input_provider()
                keyboard = input_provider.get_keyboard()
                space_pressed = bool(keyboard.get_key_state(carb.input.KeyboardInput.SPACE))
                
                # Détecter le front montant (passage de non-pressé à pressé)
                if space_pressed and not space_was_pressed:
                    moveit_bridge.trigger_motion()
                
                space_was_pressed = space_pressed
            except Exception as e:
                # Si l'API clavier ne fonctionne pas, ignorer silencieusement
                pass
            
            # Update trajectory execution (AVANT write_data_to_sim!)
            moveit_bridge.update_trajectory(simulation_time)
            
            # Write robot commands to simulation
            ur10.write_data_to_sim()
            
            # Step simulation physics
            sim.step()
            
            # Update robot state
            dt = sim.get_physics_dt()
            ur10.update(dt=dt)
            simulation_time += dt
            
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
    finally:
        # Cleanup ROS2
        moveit_bridge.destroy_node()
        rclpy.shutdown()
    
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
