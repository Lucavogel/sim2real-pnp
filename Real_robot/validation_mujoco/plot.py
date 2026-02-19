import numpy as np
import matplotlib.pyplot as plt

# Charger les données complètes
data = np.load("/home/ajin/workspace/sim2real-pnp/Real_robot/validation_mujoco/rapport_mujoco.npz")
times = data["time"]

# Données Joints
q_actual = data["q_actual"]
q_ref = data["q_ref"]
velocities = data["velocities"]
actions = data["actions"]

# Données Cartésiennes (converties en mm)
ee_actual = data["ee_actual"] * 1000
ee_ref = data["ee_ref"] * 1000

# === PAGE 1 : ANALYSE MOTEURS (Joints - Position, Vitesse, Action) ===
joint_names = ["Shoulder Pan", "Shoulder Lift", "Elbow", "Wrist 1", "Wrist 2", "Wrist 3"]

# 1.1 POSITION TRACKING (6 Subplots)
plt.figure(figsize=(20, 12))
plt.suptitle("Suivi Articulaire (Position Rad) - Réel vs Référence", fontsize=16)
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.title(joint_names[i])
    plt.plot(times, q_ref[:, i], 'k--', label="Ref")
    plt.plot(times, q_actual[:, i], 'b', label="Real")
    plt.grid(True)
    if i == 0: plt.legend()
plt.tight_layout()
plt.savefig("rapport_1a_joints_position.png")
print("✅ Image 1a générée : rapport_1a_joints_position.png")

# 1.2 VELOCITY (6 Subplots)
plt.figure(figsize=(20, 12))
plt.suptitle("Vitesses Articulaires (Rad/s)", fontsize=16)
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.title(joint_names[i])
    plt.plot(times, velocities[:, i], 'g')
    plt.grid(True)
plt.tight_layout()
plt.savefig("rapport_1b_joints_velocity.png")
print("✅ Image 1b générée : rapport_1b_joints_velocity.png")

# 1.3 ACTIONS (6 Subplots)
plt.figure(figsize=(20, 12))
plt.suptitle("Actions IA (Normalisées)", fontsize=16)
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.title(joint_names[i])
    plt.plot(times, actions[:, i], 'r', alpha=0.7)
    plt.grid(True)
    plt.ylim(-1.1, 1.1)
plt.tight_layout()
plt.savefig("rapport_1c_joints_action.png")
print("✅ Image 1c générée : rapport_1c_joints_action.png")

# === PAGE 2 : ANALYSE SPATIALE (Cartésien) ===
plt.figure(figsize=(15, 10))

axes = ["X", "Y", "Z"]
for i in range(3):
    plt.subplot(4, 1, i+1)
    plt.title(f"Axe {axes[i]} (mm)")
    plt.plot(times, ee_ref[:, i], 'k--', label="Cible")
    plt.plot(times, ee_actual[:, i], 'b', label="Réel")
    plt.grid(True)
    if i==0: plt.legend()

plt.subplot(4, 1, 4)
error = np.linalg.norm(ee_ref - ee_actual, axis=1)
plt.title("Erreur de Position Totale (mm)")
plt.plot(times, error, 'r')
plt.xlabel("Temps (s)")
plt.ylabel("Erreur (mm)")
plt.grid(True)

plt.tight_layout()
plt.savefig("rapport_2_cartesien.png")
print("✅ Image 2 générée : rapport_2_cartesien.png")

# === PAGE 3 : VUE 3D ===
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Trajectoire 3D Sim-to-Real")
ax.plot(ee_ref[:, 0], ee_ref[:, 1], ee_ref[:, 2], 'k--', label="Cible")
ax.plot(ee_actual[:, 0], ee_actual[:, 1], ee_actual[:, 2], 'b', label="Robot", linewidth=2)
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.legend()
plt.savefig("rapport_3_trajectoire3D.png")
print("✅ Image 3 générée : rapport_3_trajectoire3D.png")

plt.show()