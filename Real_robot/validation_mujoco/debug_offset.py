
import mujoco
import numpy as np
import os

# Paths
xml_path = "/home/ajin/workspace/sim2real-pnp/Real_robot/validation_mujoco/ur10_complete.xml"
npz_path = "/home/ajin/workspace/sim2real-pnp/Real_robot/validation_mujoco/tag2_to_tag3.npz"

# Load Model
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Load Dataset
dataset = np.load(npz_path, allow_pickle=True)
q_ref_0 = dataset["paths"][0, 0] # First path, first point
ee_ref_0 = dataset["ee_ref_pos"][0, 0]

# Set Robot to q_ref
joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
for i, name in enumerate(joint_names):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[jid] = q_ref_0[i]

mujoco.mj_kinematics(model, data)

# Get Wrist 3 Pos
wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'wrist_3_link')
wrist_pos = data.xpos[wrist_id]

print(f"Dataset EE Ref: {ee_ref_0}")
print(f"Model Wrist Pos: {wrist_pos}")
print(f"Diff: {np.linalg.norm(ee_ref_0 - wrist_pos)}")

# Check Tool Offset assumptions
# Assume tool is offset in Z?
diff_vec = ee_ref_0 - wrist_pos
print(f"Vector Diff: {diff_vec}")
