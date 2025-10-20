# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn prims into the scene.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/00_sim/spawn_prims.py

"""

"""Launch Isaac Sim Simulator first."""

import numpy as np
import torch
import time

import argparse
from isaaclab.app import AppLauncher
import os

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on spawning prims into the scene.")
parser.add_argument("--save", action="store_true", help="Save ray-caster outputs to disk")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app with ROS2 bridge enabled
app_launcher = AppLauncher(args_cli, enable_ros2_bridge=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.camera import Camera

from isaaclab_assets import UR10_CFG

import imageio.v3 as iio

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from control_msgs.action import FollowJointTrajectory
from std_msgs.msg import Header
import threading
import time as time_module

# Small helper to convert tensors to numpy (used by the writer)
def convert_dict_to_backend(d: dict, backend: str = "numpy"):
    if backend != "numpy":
        return d
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().numpy()
        else:
            out[k] = v
    return out

# Simple pinhole helpers (used for validation)
def unproject_depth(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    # depth: [N,H,W] or [N,H,W,1]; K: [N,3,3]
    if depth.ndim == 4:
        depth = depth[..., 0]
    N, H, W = depth.shape
    device = depth.device
    ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    ones = torch.ones_like(xs, dtype=depth.dtype)
    uv1 = torch.stack([xs, ys, ones], dim=-1).view(1, H, W, 3).repeat(N, 1, 1, 1)  # [N,H,W,3]
    K_inv = torch.inverse(K)[:, None, None, :, :]  # [N,1,1,3,3]
    xyznorm = torch.matmul(uv1.unsqueeze(-2), K_inv.transpose(-1, -2)).squeeze(-2)  # [N,H,W,3]
    xyz = xyznorm * depth[..., None]  # z = depth
    return xyz  # [N,H,W,3]

def project_points(points_3d: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    # points_3d: [N,H,W,3]; K: [N,3,3]
    N, H, W, _ = points_3d.shape
    fx = K[:, 0, 0].view(N, 1, 1)
    fy = K[:, 1, 1].view(N, 1, 1)
    cx = K[:, 0, 2].view(N, 1, 1)
    cy = K[:, 1, 2].view(N, 1, 1)
    X = points_3d[..., 0]
    Y = points_3d[..., 1]
    Z = points_3d[..., 2].clamp(min=1e-6)
    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy
    uvz = torch.stack([u, v, Z], dim=-1).reshape(N, H * W, 3)  # [N,HW,3]
    return uvz

def define_rgb_camera(parent_prim_path: str) -> Camera:
    """Defines an RGB pinhole camera under the given parent prim (e.g. robot wrist)."""
    cam_prim = f"{parent_prim_path}/wrist_cam"
    # Spawn the USD camera via CameraCfg
    cam_cfg = CameraCfg(
        prim_path=cam_prim,
        update_period=0.1,  # 10 Hz
        height=480,
        width=640,
        data_types=["rgb"],  # RGB only
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        # Orientation pour regarder dans l'axe du poignet (ajustez au besoin)
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, -4.0),
            rot=(0.7071068, 0.0, 0.0, 0.7071068),  # regarde droit vers le bas
            convention="ros",
        ),
    )
    return Camera(cfg=cam_cfg)




def design_scene():
    """Designs the scene by spawning ground plane, light, objects and meshes from usd files."""
    # Ground-plane (plan plat par défaut)
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # spawn distant light
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))

    # create a new xform prim for all objects to be spawned under
    prim_utils.create_prim("/World/Objects", "Xform")
    def spawn_red_cone():
        # cfg_cone = sim_utils.ConeCfg(
        #     radius=0.15,
        #     height=0.5,
        #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        # )
            # cfg_cone.func("/World/Objects/Cone1", cfg_cone, translation=(-1.0, 1.0, 1.0))
            # cfg_cone.func("/World/Objects/Cone2", cfg_cone, translation=(-1.0, -1.0, 1.0))

            # spawn a green cone with colliders and rigid body
            # cfg_cone_rigid = sim_utils.ConeCfg(
            #     radius=0.15,
            #     height=0.5,
            #     rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            #     mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            #     collision_props=sim_utils.CollisionPropertiesCfg(),
            #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            # )
            # cfg_cone_rigid.func(
            #     "/World/Objects/ConeRigid", cfg_cone_rigid, translation=(-0.2, 0.0, 2.0), orientation=(0.5, 0.0, 0.5, 0.0)
            # )

            # spawn a blue cuboid with deformable body
            # cfg_cuboid_deformable = sim_utils.MeshCuboidCfg(
            #     size=(0.2, 0.5, 0.2),
            #     deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
            #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            #     physics_material=sim_utils.DeformableBodyMaterialCfg(),
            # )
            # cfg_cuboid_deformable.func("/World/Objects/CuboidDeformable", cfg_cuboid_deformable, translation=(0.15, 0.0, 2.0))
            pass

    # spawn a usd file of a table into the scene
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
    cfg.func(
        "/World/Objects/Table",
        cfg,
        translation=(1.0, 0.08, 1.05),
        orientation=(0.0, 0.0, 0.0, 0.7071068)
    )

    
    # Origin 2 with UR10
    origin = torch.tensor([[0.0, 0.0, 0.0]], device=args_cli.device)
    prim_utils.create_prim("/World/Origin2", "Xform", translation=origin[0].tolist())
    # -- Table
    cfg = sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
    )
    cfg.func("/World/Origin2/Table", cfg, translation=(0.0, 0.0, 1.03))
    # -- Robot
    ur10_cfg = UR10_CFG.replace(prim_path="/World/Origin2/Robot")
    ur10_cfg.init_state.pos = (0.0, 0.0, 1.03)
    ur10 = Articulation(cfg=ur10_cfg)
    # add to sensor 

    # RETIRER le terrain accidenté:
    # cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd")
    # cfg.func("/World/ground", cfg)

    # -- Lights (optionnel second éclairage)
    cfg = sim_utils.DistantLightCfg(intensity=600.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # -- Sensors
    camera = define_rgb_camera("/World/Origin2/Robot/wrist_3_link")
    scene_entities = {"camera": camera}
    return ur10, origin, scene_entities

class UR10MoveItBridge(Node):
    """Bridge ROS2 pour connecter UR10 Isaac Lab à MoveIt2"""
    
    def __init__(self, robot: Articulation, sim: sim_utils.SimulationContext):
        super().__init__('ur10_moveit_bridge')
        
        self.robot = robot
        self.sim = sim
        
        # Noms des joints UR10 (ordre ROS2/MoveIt2 standard)
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'

             



        ]
        
        # MAPPING: Isaac Lab index → ROS2 index (swap moteurs 0 et 2)
        self.isaac_to_ros2 = [2, 1, 0, 3, 4, 5]  # Moteur Isaac 0→ROS2 elbow(2), Isaac 2→ROS2 shoulder_pan(0)
        self.ros2_to_isaac = [2, 1, 0, 3, 4, 5]  # Inverse (même car c'est un swap)
        
        # INVERSIONS de direction des moteurs
        self.joint_directions = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # Tous positifs pour l'instant
        
        self.get_logger().info(f'🔄 Mapping Isaac→ROS2: {self.isaac_to_ros2}')
        self.get_logger().info(f'🔄 Directions: {self.joint_directions.tolist()}')
        
        # Callback group pour exécution parallèle
        self.callback_group = ReentrantCallbackGroup()
        
        # Publisher pour /joint_states (pour MoveIt2)
        self.joint_state_pub = self.create_publisher(JointState, 'joint_states', 10)
        
        # Action server pour recevoir trajectoires de MoveIt2
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory',
            execute_callback=self.execute_trajectory_callback,
            callback_group=self.callback_group,
            goal_callback=self.goal_callback
        )
        
        # Variables pour stocker la trajectoire en cours
        self.current_trajectory = None
        self.trajectory_start_time = None
        self.current_goal_handle = None
        self.trajectory_index = 0
        self.trajectory_complete = False
        self.trajectory_error = None
        
        # Timer pour publier joint_states à 50Hz
        self.create_timer(0.02, self.publish_joint_states, callback_group=self.callback_group)
        
        self.get_logger().info('✅ UR10 connecté à MoveIt2 via ROS2!')
    
    def goal_callback(self, goal_request):
        """Callback appelé quand un nouveau goal arrive - accepte toujours"""
        from rclpy.action.server import GoalResponse
        self.get_logger().info('📩 Nouveau goal reçu, acceptation...')
        return GoalResponse.ACCEPT
    
    def publish_joint_states(self):
        """Publie l'état actuel des joints vers MoveIt2"""
        try:
            # Lire positions depuis Isaac Lab (ordre physique Isaac)
            joint_pos_isaac = self.robot.data.joint_pos[0].cpu().numpy()
            joint_vel_isaac = self.robot.data.joint_vel[0].cpu().numpy()
            
            # Appliquer le mapping Isaac → ROS2
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
    
    def execute_trajectory_callback(self, goal_handle):
        """Reçoit une trajectoire de MoveIt2 et la stocke pour exécution"""
        self.get_logger().info('🎯 Trajectoire MoveIt2 reçue - exécution dans loop principale!')
        
        # Réinitialiser les flags
        self.current_trajectory = goal_handle.request.trajectory
        self.trajectory_start_time = time.time()
        self.current_goal_handle = goal_handle
        self.trajectory_index = 0
        self.trajectory_complete = False
        self.trajectory_error = None
        
        self.get_logger().info(f'   → {len(self.current_trajectory.points)} points à exécuter')
        
        # Attendre que la trajectoire soit exécutée dans la boucle principale
        while not self.trajectory_complete and self.trajectory_error is None:
            time.sleep(0.01)  # Vérifier toutes les 10ms
        
        # Retourner le résultat
        result = FollowJointTrajectory.Result()
        if self.trajectory_error:
            self.get_logger().error(f'❌ Erreur: {self.trajectory_error}')
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            return result
        else:
            self.get_logger().info('✅ Trajectoire terminée!')
            result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
            return result
    
    def update_trajectory(self):
        """Met à jour l'exécution de la trajectoire (appelé depuis la boucle principale)"""
        if self.current_trajectory is None:
            return
        
        try:
            elapsed_time = time.time() - self.trajectory_start_time
            
            # Trouver le point de trajectoire actuel basé sur le temps
            for idx in range(self.trajectory_index, len(self.current_trajectory.points)):
                point = self.current_trajectory.points[idx]
                point_time = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
                
                if elapsed_time >= point_time:
                    # Positions reçues de MoveIt2 (ordre ROS2)
                    target_positions_ros2 = list(point.positions)
                    
                    # Appliquer le mapping inverse: ROS2 → Isaac Lab avec inversions
                    target_positions_isaac = [0.0] * 6
                    for ros2_idx in range(6):
                        isaac_idx = self.ros2_to_isaac[ros2_idx]
                        # Appliquer direction inverse
                        target_positions_isaac[isaac_idx] = target_positions_ros2[ros2_idx] * self.joint_directions[isaac_idx]
                    
                    # Convertir en tensor Isaac Lab
                    target_tensor = torch.tensor(
                        [target_positions_isaac], 
                        device=args_cli.device, 
                        dtype=torch.float32
                    )
                    
                    # Envoyer au robot Isaac Lab
                    self.robot.set_joint_position_target(target_tensor)
                    
                    # Log uniquement tous les 5 points pour éviter le spam
                    if idx % 5 == 0:
                        angles_deg = np.degrees(target_positions_ros2)
                        self.get_logger().info(
                            f'Point {idx+1}/{len(self.current_trajectory.points)}: '
                            f'[{angles_deg[0]:.1f}°, {angles_deg[1]:.1f}°, {angles_deg[2]:.1f}°]'
                        )
                    
                    self.trajectory_index = idx + 1
                else:
                    break
            
            # Vérifier si la trajectoire est terminée
            if self.trajectory_index >= len(self.current_trajectory.points):
                # Marquer comme complète
                self.trajectory_complete = True
                
                # Réinitialiser
                self.current_trajectory = None
                self.current_goal_handle = None
                self.trajectory_index = 0
                
        except Exception as e:
            # Marquer l'erreur
            self.trajectory_error = str(e)
            import traceback
            self.get_logger().error(f'❌ ERREUR: {traceback.format_exc()}')
            
            # Réinitialiser
            self.current_trajectory = None
            self.current_goal_handle = None
            self.trajectory_index = 0


def run_simulator(sim: sim_utils.SimulationContext, robot: Articulation, scene_entities: dict, ros_bridge: UR10MoveItBridge):
    """Run the simulator avec ROS2."""
    camera: Camera = scene_entities["camera"]

    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "rgb_camera")
    os.makedirs(output_dir, exist_ok=True)

    print("[INFO]: Boucle principale Isaac Lab + ROS2 active!")
    print("  - Isaac Lab: Simulation UR10 + Caméra")
    print("  - ROS2: /joint_states publié, trajectoires acceptées")
    print("  - Lancez MoveIt2 dans un autre terminal!")

    frame_count = 0
    while simulation_app.is_running() and rclpy.ok():
        # Update trajectoire ROS2 (avant simulation step)
        if ros_bridge:
            ros_bridge.update_trajectory()
        
        # Update Isaac Lab
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt=sim.get_physics_dt())
        camera.update(dt=sim.get_physics_dt())

        # Vision (moins verbose)
        if frame_count % 50 == 0:  # Tous les 50 frames
            if "rgb" in camera.data.output:
                print(f"Frame {frame_count}: RGB shape = {camera.data.output['rgb'].shape}")

        if args_cli.save and frame_count % 10 == 0:  # Sauver moins souvent
            frame = int(camera.frame[0])
            rgb = camera.data.output["rgb"][0, ...]  # [H,W,3 or 4], float
            rgb = rgb[..., :3].detach().cpu().numpy()
            # Normalise si nécessaire
            if rgb.max() <= 1.0:
                rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
            else:
                rgb = rgb.clip(0, 255).astype(np.uint8)
            iio.imwrite(os.path.join(output_dir, f"rgb_{frame:04d}.png"), rgb)
        
        frame_count += 1

def main():
    """Main function avec ROS2."""
    
    # Initialiser ROS2
    rclpy.init()
    
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # Set main camera
    sim.set_camera_view([6.0, 0.0, 5.5], [-0.5, 0.0, 0.5])
    
    # Design scene
    ur10, origin, scene_entities = design_scene()
    
    # Play the simulator
    sim.reset()
    print("[INFO]: Isaac Lab setup complete...")
    
    # Créer le bridge ROS2 pour MoveIt2
    ros_bridge = UR10MoveItBridge(ur10, sim)
    
    # Thread ROS2 pour spin en parallèle
    def spin_ros():
        try:
            rclpy.spin(ros_bridge)
        except Exception as e:
            print(f"ROS2 spin error: {e}")
    
    ros_thread = threading.Thread(target=spin_ros, daemon=True)
    ros_thread.start()
    
    print("[INFO]: ✅ UR10 + ROS2 Bridge prêt!")
    print("")
    print("=" * 60)
    print("🚀 SYSTÈME PRÊT")
    print("=" * 60)
    print("Isaac Lab: Robot UR10 + Caméra simulés")
    print("ROS2: /joint_states publié, trajectoires acceptées")
    print("")
    print("Prochaine étape: Lancez MoveIt2 dans un autre terminal:")
    print("  cd ~/work2/ur10 && source install/setup.bash")
    print("  ros2 launch ur_coppeliasim ur_isaaclab_moveit.launch.py")
    print("=" * 60)
    print("")
    
    # Delegate to simulator loop with sensors
    try:
        run_simulator(sim, ur10, scene_entities, ros_bridge)
    except KeyboardInterrupt:
        print("\n[INFO]: Arrêt demandé...")
    finally:
        # Cleanup ROS2
        ros_bridge.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()