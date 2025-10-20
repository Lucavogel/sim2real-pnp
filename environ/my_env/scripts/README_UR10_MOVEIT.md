# 🤖 Ur10.py avec MoveIt2 - Guide d'utilisation

## ✨ Qu'est-ce qui a changé ?

Votre `Ur10.py` maintenant :
- ✅ Lance Isaac Lab avec le robot UR10
- ✅ Active votre caméra sur le poignet (comme avant)
- ✅ **NOUVEAU** : Communique avec MoveIt2 via ROS2
- ✅ Publie `/joint_states` pour MoveIt2
- ✅ Reçoit des trajectoires de MoveIt2

**Avantage** : Un seul processus Isaac Lab au lieu de 2 !

---

## 🚀 Utilisation (2 terminaux au lieu de 3)

### Terminal 1 [CONDA] : Isaac Lab + UR10
```bash
cd ~/work2/IsaacLab
./isaaclab.sh -p ../my_env/scripts/Ur10.py
```

⏳ **Attendez de voir** :
```
✅ UR10 + ROS2 Bridge prêt!
🚀 SYSTÈME PRÊT
Isaac Lab: Robot UR10 + Caméra simulés
ROS2: /joint_states publié, trajectoires acceptées
```

---

### Terminal 2 [ROS2] : MoveIt2 + RViz
```bash
conda deactivate
cd ~/work2/ur10
source install/setup.bash
ros2 launch ur_coppeliasim ur_isaaclab_moveit.launch.py
```

⏳ **Attendez** : RViz s'ouvre

---

### Terminal 2 [ROS2] : Test Automatique (optionnel)
```bash
cd ~/work2/ur10
source install/setup.bash
ros2 run ur_coppeliasim test_isaaclab_moveit.py
```

🎉 **Le robot bouge dans Isaac Lab !**

---

## 🔍 Vérification

### Vérifier les topics ROS2
```bash
ros2 topic list
# Devrait montrer:
#   /joint_states
#   /joint_trajectory_controller/follow_joint_trajectory
```

### Voir les joint states
```bash
ros2 topic echo /joint_states
```

---

## 📸 Votre Caméra fonctionne toujours !

La caméra continue de publier les images (comme avant) :
- Position : Sur le poignet (`wrist_3_link`)
- Orientation : Regarde vers le bas
- Fréquence : 20 Hz

Pour sauvegarder les images :
```bash
cd ~/work2/IsaacLab
./isaaclab.sh -p ../my_env/scripts/Ur10.py --save
```

Images sauvées dans : `my_env/scripts/output/rgb_camera/`

---

## 🆚 Avant vs Maintenant

### Avant (3 terminaux)
```
Terminal 1 [CONDA]: isaaclab_bridge.py
Terminal 2 [ROS2]: MoveIt2
Terminal 3 [ROS2]: Test
```

### Maintenant (2 terminaux)
```
Terminal 1 [CONDA]: Ur10.py (Isaac Lab + ROS2 Bridge intégré)
Terminal 2 [ROS2]: MoveIt2 + Test
```

---

## 🎯 Ce que vous pouvez faire

1. **Contrôler le robot avec MoveIt2** :
   - Ouvrez RViz
   - Déplacez le marker interactif
   - Cliquez "Plan" puis "Execute"
   - Le robot bouge dans Isaac Lab !

2. **Utiliser votre caméra** :
   - Les images RGB sont disponibles en temps réel
   - Compatible avec votre code vision/RL

3. **Développer votre logique** :
   - Ajoutez votre code dans `run_simulator()`
   - Le bridge ROS2 fonctionne en parallèle

---

## 🐛 Dépannage

**"ModuleNotFoundError: rclpy"**
```bash
conda activate env_isaaclab
conda install -c conda-forge ros-humble-rclpy ros-humble-control-msgs
```

**"No joint_states published"**
- Vérifiez que Isaac Lab est bien lancé
- Regardez les logs dans Terminal 1

**"Robot doesn't move"**
- Vérifiez que MoveIt2 envoie bien des trajectoires
- Regardez les logs : `🎯 Trajectoire MoveIt2 reçue!`

---

## 📝 Architecture

```
┌────────────────────────────────────────┐
│         Ur10.py (Isaac Lab)            │
│  ┌──────────────┐  ┌────────────────┐  │
│  │ Simulation   │  │  ROS2 Bridge   │  │
│  │ + Caméra     │  │  (intégré)     │  │
│  └──────────────┘  └────────────────┘  │
│         [CONDA - Python 3.11]          │
└────────────────────────────────────────┘
              │
              │ ROS2 Topics
              │ /joint_states
              │ /joint_trajectory
              ▼
┌────────────────────────────────────────┐
│         MoveIt2 + RViz                 │
│       [ROS2 - Python 3.10]             │
└────────────────────────────────────────┘
```

---

**Bon développement ! 🎉**
