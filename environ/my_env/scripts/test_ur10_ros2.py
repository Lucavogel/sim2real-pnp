#!/usr/bin/env python3
"""
Test UR10 ROS2 USD - mouvement en position control
Compatible MoveIt / RL résiduel
"""

import argparse
import torch

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------- #
# App launcher
# ---------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description="Test UR10 ROS2 USD")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------- #
# Imports Isaac Lab
# ---------------------------------------------------------------------------- #
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from ur10_ros2_cfg import UR10_ROS2_CFG

# ---------------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------------- #
def main():
    # Simulation
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.01,
        device=args_cli.device,
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.5])

    # Ground
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/Ground", ground_cfg)

    # Robot
    print("\n" + "=" * 70)
    print("🤖 CHARGEMENT UR10 (ROS2 URDF → USD)")
    print("=" * 70)

    ur10_cfg = UR10_ROS2_CFG.replace(prim_path="/World/UR10")
    ur10 = Articulation(cfg=ur10_cfg)

    # Reset
    sim.reset()

    print(f"✅ Robot chargé: {ur10.num_instances}")
    print(f"✅ Nombre de joints: {ur10.num_joints}")
    print(f"✅ Joints: {ur10.joint_names}")
    print(f"✅ Device: {ur10.device}")
    print("=" * 70)

    # ---------------------------------------------------------------------- #
    # ✅ TARGET DE TEST (VISIBLE)
    # ---------------------------------------------------------------------- #
    import torch

    # Position initiale (définie dans ur10_ros2_cfg.py)
    init_q = torch.tensor([[0.0, -0.9, 1.6, -2.3, -1.57, 0.0]], device=ur10.device)
    
    # Target finale
    target_q = torch.tensor([[0.3, -1.2, 1.3, -1.5, -1.2, 0.5]], device=ur10.device)

    print("\n▶️ Simulation (10 secondes)")
    for step in range(1000):
        # Garder init pendant 2 secondes (200 steps), puis aller vers target
        if step < 200:
            ur10.set_joint_position_target(init_q)
        else:
            ur10.set_joint_position_target(target_q)

        # Mettre à jour la simulation
        ur10.write_data_to_sim()
        sim.step()
        ur10.update(sim.get_physics_dt())

        if step % 100 == 0:
            q = ur10.data.joint_pos[0].cpu().numpy()
            print(f"Step {step:04d} | q = {q}")


    print("\n✅ TEST TERMINÉ AVEC SUCCÈS")

# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
