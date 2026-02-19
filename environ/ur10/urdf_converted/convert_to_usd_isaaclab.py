#!/usr/bin/env python3
"""
Script pour convertir l'URDF UR10 en USD avec Isaac Lab
Compatible avec IsaacLab (pas besoin d'Isaac Sim complet)

Usage:
    cd IsaacLab
    ./isaaclab.sh -p /path/to/convert_to_usd_isaaclab.py --urdf ur10.urdf --output ur10.usd
"""

import argparse
from pathlib import Path

# Isaac Lab imports
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convertir URDF UR10 en USD")
parser.add_argument("--urdf", type=str, required=True, help="Chemin vers le fichier URDF")
parser.add_argument("--output", type=str, required=True, help="Chemin de sortie pour le USD")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Lancer Isaac Sim en mode headless
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports après lancement
import omni.isaac.core.utils.extensions as ext_utils
import omni.kit.commands
from pxr import Usd

def convert_urdf_to_usd(urdf_path: str, usd_path: str):
    """Convertit URDF en USD"""
    
    print(f"\n{'='*70}")
    print("CONVERSION URDF → USD avec Isaac Lab")
    print('='*70)
    print(f"📂 Input:  {urdf_path}")
    print(f"💾 Output: {usd_path}")
    print()
    
    # Activer l'extension URDF
    print("🔧 Activation extension URDF...")
    ext_utils.enable_extension("omni.isaac.urdf")
    
    # Vérifier que le fichier URDF existe
    if not Path(urdf_path).exists():
        raise FileNotFoundError(f"URDF non trouvé: {urdf_path}")
    
    print(f"✅ URDF trouvé: {Path(urdf_path).stat().st_size} bytes")
    print()
    
    # Créer une nouvelle scène
    print("🌍 Création de la scène...")
    stage = omni.usd.get_context().get_stage()
    
    # Importer URDF
    print("📥 Import URDF...")
    print("   (Cela peut prendre quelques secondes...)")
    
    success, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=urdf_path,
        import_config=omni.isaac.urdf.ImportConfig(
            set_default_prim=True,
            create_physics_scene=False,
            import_inertia_tensor=True,
            fix_base=False,
            merge_fixed_joints=False,
            self_collision=False,
            default_drive_type=omni.isaac.urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION,
            default_position_drive_damping=1000.0,
            default_position_drive_stiffness=10000.0,
        ),
        dest_path="/World/ur10"
    )
    
    if not success:
        raise RuntimeError("Import URDF échoué!")
    
    print(f"✅ URDF importé: {prim_path}")
    print()
    
    # Sauvegarder USD
    print(f"💾 Sauvegarde USD...")
    
    # Créer le dossier parent si nécessaire
    output_path = Path(usd_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Exporter
    stage.Export(str(output_path))
    
    print(f"✅ USD sauvegardé: {output_path}")
    print(f"   Taille: {output_path.stat().st_size} bytes")
    print()
    print('='*70)
    print("✅ CONVERSION RÉUSSIE!")
    print('='*70)
    print()
    print("📝 UTILISATION dans Isaac Lab:")
    print()
    print("from isaaclab.assets import Articulation, ArticulationCfg")
    print("import isaaclab.sim as sim_utils")
    print()
    print("ur10_cfg = ArticulationCfg(")
    print("    prim_path='/World/UR10',")
    print("    spawn=sim_utils.UsdFileCfg(")
    print(f"        usd_path='{output_path}'")
    print("    ),")
    print("    actuators={")
    print("        'ur10_joints': ImplicitActuatorCfg(")
    print("            joint_names_expr=['.*'],")
    print("            stiffness=1000.0,")
    print("            damping=100.0,")
    print("        )")
    print("    }")
    print(")")
    print()
    
    return True


if __name__ == "__main__":
    try:
        convert_urdf_to_usd(args_cli.urdf, args_cli.output)
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    finally:
        simulation_app.close()
