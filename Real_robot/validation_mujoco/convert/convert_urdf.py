#!/usr/bin/env python3
"""
Convertit l'URDF UR10 en XML MuJoCo (MJCF) avec les meshes visuels.

Structure requise:
    validation_mujoco/
    ├── convert_urdf.py  (ce script)
    ├── ur10.urdf
    └── meshes/
        └── ur10/
            ├── visual/
            └── collision/

Utilisation:
    python convert_urdf.py
"""
import mujoco
import os
import sys

# --- FIX AUTOMATIQUE DU CHEMIN ---
# On récupère le dossier où se trouve CE script (convert_urdf.py)
script_dir = os.path.dirname(os.path.abspath(__file__))
# On force Python à travailler dans ce dossier
os.chdir(script_dir)
print(f"📂 Répertoire de travail: {script_dir}")

# Noms des fichiers
INPUT_URDF = "ur10_stl.urdf"  # URDF avec meshes STL uniquement (compatible MuJoCo)
OUTPUT_XML = "ur10.xml"

def convert():
    """Convertit l'URDF en XML MuJoCo natif."""
    print(f"🔄 Conversion: {INPUT_URDF} → {OUTPUT_XML}")

    # 1. Vérifications préalables
    if not os.path.exists(INPUT_URDF):
        print(f"❌ ERREUR: Fichier '{INPUT_URDF}' introuvable!")
        sys.exit(1)
    
    if not os.path.exists("meshes"):
        print(f"⚠️  ATTENTION: Dossier 'meshes' introuvable dans {os.getcwd()}!")
        print("   Le XML sera généré sans géométries visuelles.")
    
    meshes_ur10 = os.path.join("meshes", "ur10", "collision")
    if os.path.exists(meshes_ur10):
        mesh_count = len([f for f in os.listdir(meshes_ur10) if f.endswith('.stl')])
        print(f"✅ {mesh_count} fichiers mesh détectés")

    try:
        # 2. Chargement de l'URDF
        print("⏳ Chargement URDF...")
        model = mujoco.MjModel.from_xml_path(INPUT_URDF)
        
        # 3. Sauvegarde en XML natif (MJCF)
        print("💾 Sauvegarde XML...")
        mujoco.mj_saveLastXML(OUTPUT_XML, model)
        
        # 4. Vérification
        print(f"\n✅ SUCCÈS: '{OUTPUT_XML}' généré!")
        print(f"   Bodies: {model.nbody}")
        print(f"   Joints: {model.njnt}")
        print(f"   DOF: {model.nv}")
        print(f"   Meshes: {model.nmesh}")
        print(f"\n👉 ETAPE SUIVANTE: Ajouter les <actuator> dans {OUTPUT_XML}")
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    convert()