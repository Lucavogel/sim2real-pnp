#!/usr/bin/env python3
"""
Test du modèle UR10 MuJoCo avec visualisation.
"""
import mujoco
import mujoco.viewer
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

XML_PATH = "ur10_complete.xml"

print(f"🤖 Chargement du modèle: {XML_PATH}")

# Charger le modèle
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

print(f"✅ Modèle chargé:")
print(f"   Bodies: {model.nbody}")
print(f"   Joints: {model.njnt}")
print(f"   DOF: {model.nv}")
print(f"   Actuateurs: {model.nu}")
print(f"   Meshes: {model.nmesh}")

# Définir la position initiale (Isaac Lab home position)
home_position = [0.01420452, -0.8636804, 1.4162203, -2.1233363, -1.5707963, -1.5565917]
data.qpos[:6] = home_position

# Fixer les commandes des actuateurs à la position initiale (pour stabiliser)
data.ctrl[:] = home_position

print(f"\n🎮 Lancement du viewer interactif...")
print("   Utilisez la souris pour tourner la caméra")
print("   Appuyez sur ESC pour quitter")

# Lancer le viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Maintenir la position home (contrôle simple)
        data.ctrl[:] = home_position
        
        # Simulation step
        mujoco.mj_step(model, data)
        viewer.sync()

print("\n✅ Terminé!")
