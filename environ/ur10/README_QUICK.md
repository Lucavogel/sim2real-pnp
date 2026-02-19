# 🚀 Isaac Lab + MoveIt2 - COMMANDES ESSENTIELLES

## ⚡ Installation (1 fois)

### 1️⃣ [CONDA] Installer ROS2 dans Isaac Lab
```bash
conda activate env_isaaclab
conda install -c conda-forge -c robostack-staging ros-humble-rclpy ros-humble-control-msgs ros-humble-trajectory-msgs ros-humble-sensor-msgs
```

### 2️⃣ [ROS2] Compiler le workspace
```bash
conda deactivate
cd ~/work2/ur10
colcon build
source install/setup.bash
```

---

## 🎯 Test Rapide (2 terminaux)

### Terminal 1 [CONDA]
```bash
cd ~/work2/IsaacLab
./isaaclab.sh -p ~/work2/ur10/install/ur_coppeliasim/lib/ur_coppeliasim/simple_isaaclab_test.py
```

### Terminal 2 [ROS2]
```bash
ros2 topic echo /joint_states
```

---

## 🤖 Test Complet (3 terminaux)

### Terminal 1 [CONDA] - Isaac Lab
```bash
cd ~/work2/IsaacLab
./isaaclab.sh -p ~/work2/ur10/install/ur_coppeliasim/lib/ur_coppeliasim/isaaclab_bridge.py
```
⏳ Attendez: `✅ Bridge Isaac Lab prêt!`

### Terminal 2 [ROS2] - MoveIt2
```bash
conda deactivate
cd ~/work2/ur10
source install/setup.bash
ros2 launch ur_coppeliasim ur_isaaclab_moveit.launch.py
```
⏳ Attendez: RViz s'ouvre

### Terminal 3 [ROS2] - Test
```bash
cd ~/work2/ur10
source install/setup.bash
ros2 run ur_coppeliasim test_isaaclab_moveit.py
```

---

## 🐛 Problèmes ?

**"ModuleNotFoundError: rclpy"**
```bash
conda activate env_isaaclab
conda install -c conda-forge ros-humble-rclpy
```

**"package not found"**
```bash
cd ~/work2/ur10
colcon build
source install/setup.bash
```

**Isaac Lab ne démarre pas**
```bash
cd ~/work2/IsaacLab
./isaaclab.sh -p scripts/tutorials/00_sim/spawn_prims.py
```

---

## 📝 Règles

- **[CONDA]** = avec `conda activate env_isaaclab`
- **[ROS2]** = SANS Conda (`conda deactivate`)
- **Ne jamais mélanger !**

---

Plus de détails → voir `COMMANDS.sh`
