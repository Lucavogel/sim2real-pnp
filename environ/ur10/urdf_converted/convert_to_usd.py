#!/usr/bin/env python3
"""Script pour convertir URDF en USD avec Isaac Sim"""

import sys
import os
from pathlib import Path

# Importer Isaac Sim
from omni.isaac.kit import SimulationApp

# Lancer Isaac Sim en mode headless
simulation_app = SimulationApp({"headless": True})

# Importer les modules Isaac après lancement
from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.isaac.urdf")

import omni.kit.commands
from pxr import Usd, UsdGeom

def convert_urdf_to_usd(urdf_path: str, usd_path: str):
    """Convertit URDF en USD"""
    print(f"📂 Chargement URDF: {urdf_path}")
    
    # Importer URDF
    omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=urdf_path,
        import_config=omni.isaac.urdf.ImportConfig(
            set_default_prim=True,
            create_physics_scene=False,
            import_inertia_tensor=True,
            fix_base=False,
        ),
        dest_path="/World/ur10"
    )
    
    print(f"✅ URDF importé dans la scène")
    
    # Sauvegarder USD
    stage = omni.usd.get_context().get_stage()
    stage.Export(usd_path)
    
    print(f"💾 USD sauvegardé: {usd_path}")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_to_usd.py <urdf_path> <usd_path>")
        sys.exit(1)
    
    urdf_path = sys.argv[1]
    usd_path = sys.argv[2]
    
    try:
        convert_urdf_to_usd(urdf_path, usd_path)
        print("\n✅ CONVERSION RÉUSSIE!")
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        simulation_app.close()
