from isaaclab.app import AppLauncher
from pathlib import Path

# Lancer Isaac en mode headless
app_launcher = AppLauncher()
simulation_app = app_launcher.app

import omni
from omni.isaac.urdf import _urdf

def main():
    urdf_path = "/home/ajin/workspace/sim2real-pnp/environ/ur10/ur10_isaac.urdf"
    output_dir = "/home/ajin/workspace/sim2real-pnp/environ/ur10/urdf_converted"
    root_prim = "/World/UR10"

    print(f"📂 URDF : {urdf_path}")
    print(f"📁 Output USD : {output_dir}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    urdf_interface = _urdf.acquire_urdf_interface()

    cfg = _urdf.ImportConfig()
    cfg.merge_fixed_joints = True
    cfg.fix_base = True
    cfg.convex_decomp = True
    cfg.self_collision = False

    print("🚀 Import URDF → USD...")
    success = urdf_interface.parse_urdf(
        file_path=urdf_path,
        import_config=cfg,
        prim_path=root_prim,
        destination_path=output_dir,
    )

    if not success:
        print("❌ Import échoué")
    else:
        print("✅ Import réussi")
        print(f"   USD générés dans : {output_dir}")

if __name__ == "__main__":
    main()
    simulation_app.close()
