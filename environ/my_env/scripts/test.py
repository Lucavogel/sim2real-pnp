from isaaclab.app import AppLauncher
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation
from ur10_ros2_cfg import UR10_ROS2_CFG
import numpy as np

import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app

def main():
    # config simulation légère
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.02,
        device=args.device,
        render_interval=2,
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.0, 2.0, 1.5], [0.0, 0.0, 0.5])

    # charger le robot
    ur10_cfg = UR10_ROS2_CFG.replace(prim_path="/World/UR10")
    ur10 = Articulation(cfg=ur10_cfg)

    sim.reset()
    print("[INFO] UR10 chargé, joints :", ur10.joint_names)

    # petite boucle de test
    count = 0
    try:
        while app.is_running() and count < 500:
            # juste pour tester : léger mouvement sinusoïdal sur le premier joint
            q = ur10.data.joint_pos.clone()
            q[:, 0] = 0.2 * np.sin(0.01 * count)
            ur10.set_joint_position_target(q)

            ur10.write_data_to_sim()
            sim.step()
            ur10.update(dt=sim.get_physics_dt())

            if count % 50 == 0:
                print(f"[{count:04d}] joint 0 = {float(q[0,0]):.3f} rad")
            count += 1
    finally:
        print("[INFO] Fin du test.")

if __name__ == "__main__":
    main()
    app.close()
