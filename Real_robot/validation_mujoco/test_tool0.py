#!/usr/bin/env python3
"""
Test rapide : vérifie que tool0 est bien placé en chargeant le modèle MuJoCo.
"""
import mujoco
import numpy as np

XML_PATH = "ur10_complete.xml"

print("\n" + "="*70)
print("TEST TOOL0 POSITION")
print("="*70)

# Charger modèle
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# Trouver tool0
try:
    tool0_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'tool0')
    print(f"\n✅ tool0 trouvé (ID: {tool0_id})")
except:
    print(f"\n❌ tool0 introuvable !")
    exit(1)

# Position home
home_q = [0.01420452, -0.8636804, 1.4162203, -2.1233363, -1.5707963, -1.5565917]

# Trouver les joints
joint_ids = []
for name in ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
             'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    joint_ids.append(jid)

# Mettre en position home
data.qpos[joint_ids] = home_q
mujoco.mj_forward(model, data)

# Lire position tool0
tool0_pos = data.xpos[tool0_id]
print(f"\n📍 Position tool0 (home):")
print(f"   X: {tool0_pos[0]:.4f} m")
print(f"   Y: {tool0_pos[1]:.4f} m")
print(f"   Z: {tool0_pos[2]:.4f} m")

# Comparer avec le dataset
npz_path = "/home/ajin/workspace/sim2real-pnp/environ/ur10/tag_to_tag.npz"
data_npz = np.load(npz_path, allow_pickle=True)
ee_ref_home = data_npz["ee_ref_pos"][0][0]  # Premier point

print(f"\n📍 Position dataset (référence):")
print(f"   X: {ee_ref_home[0]:.4f} m")
print(f"   Y: {ee_ref_home[1]:.4f} m")
print(f"   Z: {ee_ref_home[2]:.4f} m")

diff = np.linalg.norm(tool0_pos - ee_ref_home) * 1000
print(f"\n📏 Différence: {diff:.1f} mm")

if diff < 50:
    print("   ✅ EXCELLENT ! tool0 est bien placé.")
elif diff < 100:
    print("   ✅ BON ! Proche de la référence.")
else:
    print("   ⚠️  ATTENTION ! Écart significatif - vérifier la cinématique.")

print("="*70)
