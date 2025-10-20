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

import argparse
from isaaclab.app import AppLauncher



# create argparser
parser = argparse.ArgumentParser(description="Tutorial on spawning prims into the scene.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets import UR10_CFG

def design_scene():
    """Designs the scene by spawning ground plane, light, objects and meshes from usd files."""
    # Ground-plane
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
    # spawn a red cone
    cfg_cone = sim_utils.ConeCfg(
        radius=0.15,
        height=0.5,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    )
    cfg_cone.func("/World/Objects/Cone1", cfg_cone, translation=(-1.0, 1.0, 1.0))
    cfg_cone.func("/World/Objects/Cone2", cfg_cone, translation=(-1.0, -1.0, 1.0))

    # spawn a green cone with colliders and rigid body
    cfg_cone_rigid = sim_utils.ConeCfg(
        radius=0.15,
        height=0.5,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    )
    cfg_cone_rigid.func(
        "/World/Objects/ConeRigid", cfg_cone_rigid, translation=(-0.2, 0.0, 2.0), orientation=(0.5, 0.0, 0.5, 0.0)
    )

    # commzment mettre la caméra 
    # camera = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
    #     update_period=0.1,
    #     height=480,
    #     width=640,
    #     data_types=["rgb", "distance_to_image_plane"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    # )
    # cfg = sim_utils.UsdFileCfg(
    #     usd_path=f"{ISAAC_NUCLEUS_DIR}/Assets/Isaac/5.0/Isaac/Sensors/Sensing/SG2/H60YA/Camera_SG2_OX03CC_5200_GMSL2_H60YA.usd",
    #     scale=(0.5, 0.5, 0.5),
    # )
    # cam_prim = "/World/Camera"
    # cfg.func(cam_prim, cfg, translation=(3.0, 0.0, 2.0), orientation=(0.707, 0.0, 0.707, 0.0))
    # Spawn the USD camera via CameraCfg

    # spawn a blue cuboid with deformable body
    cfg_cuboid_deformable = sim_utils.MeshCuboidCfg(
        size=(0.2, 0.5, 0.2),
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        physics_material=sim_utils.DeformableBodyMaterialCfg(),
    )
    cfg_cuboid_deformable.func("/World/Objects/CuboidDeformable", cfg_cuboid_deformable, translation=(0.15, 0.0, 2.0))

    # spawn a usd file of a table into the scene
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
    cfg.func("/World/Objects/Table", cfg, translation=(0.0, 0.0, 1.05))

    
    # Origin 2 with UR10
    origin = torch.tensor([[2.5, 0.0, 0.0]], device=args_cli.device)
    prim_utils.create_prim("/World/Origin2", "Xform", translation=origin[0].tolist())
    # -- Table
    cfg = sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
    )
    cfg.func("/World/Origin2/Table", cfg, translation=(0.0, 0.0, 0.725))
    # -- Robot
    ur10_cfg = UR10_CFG.replace(prim_path="/World/Origin2/Robot")
    ur10_cfg.init_state.pos = (0.0, 0.0, 0.725)
    ur10 = Articulation(cfg=ur10_cfg)



    #--------------
    # camera
    # camera = sim_utils.CameraCfg(
    #     prim_path="/World/Camera",
    #     update_period=0.1,
    #     height=480,
    #     width=640,
    #     data_types=["rgb", "distance_to_image_plane"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
    #     ),
    #     offset=sim_utils.CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    # )
    
    return ur10, origin


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([6.0, 0.0, 5.5], 
                        [-0.5, 0.0, 0.5])
    # Design scene
    ur10, origin = design_scene()
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Choix du joint et du mouvement (autour de la position par défaut)
    joint_idx = 1   # 0=base, 1=shoulder lift, 2=elbow, 3-5=poignet
    amp = 0.4       # amplitude en radians
    freq = 0.25     # fréquence en Hz
    
    # Simulate physics
    while simulation_app.is_running():
        # reset every 200 steps
        if count % 200 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset the robot
            # root state
            root_state = ur10.data.default_root_state.clone()
            root_state[:, :3] += origin
            ur10.write_root_pose_to_sim(root_state[:, :7])
            ur10.write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions
            joint_pos, joint_vel = ur10.data.default_joint_pos.clone(), ur10.data.default_joint_vel.clone()
            ur10.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            ur10.reset()
            print("[INFO]: Resetting robot state...")
        
        # apply random actions to the robot
        # generate random joint positions (small perturbations around default)
        # joint_pos_target = ur10.data.default_joint_pos + torch.randn_like(ur10.data.joint_pos) * 0.1

        # Mouvement sinusoïdal d’un seul joint autour de la position par défaut
        joint_pos_target = ur10.data.default_joint_pos.clone()
        phase = torch.tensor(2 * np.pi * freq * sim_time, device=args_cli.device, dtype=joint_pos_target.dtype)
        offset = amp * torch.sin(phase)
        joint_pos_target[:, joint_idx] = ur10.data.default_joint_pos[:, joint_idx] + offset

        # clamp to joint limits
        joint_pos_target = joint_pos_target.clamp_(
            ur10.data.soft_joint_pos_limits[..., 0], ur10.data.soft_joint_pos_limits[..., 1]
        )
        # apply action to the robot
        ur10.set_joint_position_target(joint_pos_target)
        # write data to sim
        ur10.write_data_to_sim()
        
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        ur10.update(sim_dt)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()