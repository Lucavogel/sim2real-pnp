# 🎯 Guide rapide - CoppeliaSim + ROS2 pour UR10

## ✅ STATUS : Bridge fonctionnel !

Le bridge ROS2 ↔ CoppeliaSim est **connecté et opérationnel** avec les 6 joints UR10.

---

## 🚀 Utilisation

### 1. Lancer CoppeliaSim avec votre scène UR10

Assurez-vous que CoppeliaSim tourne avec votre scène contenant les joints :
- `UR10_joint1` à `UR10_joint6`

### 2. Lancer le bridge ROS2

```bash
cd /home/luca/Documents/UR_WS
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch ur_coppeliasim test_bridge.launch.py
```

**Vous devriez voir** :
```
✓ UR10_joint1 (handle: 18)
✓ UR10_joint2 (handle: 21)
...
✓ Bridge prêt avec 6/6 joints
```

### 3. Vérifier les topics ROS2

Dans un **nouveau terminal** :

```bash
# Lister les topics
ros2 topic list

# Voir les positions des joints
ros2 topic echo /joint_states

# Voir l'action server disponible
ros2 action list
```

Vous devriez voir :
- `/joint_states` → positions actuelles du robot
- `/joint_trajectory_controller/follow_joint_trajectory` → pour envoyer des trajectoires

### 4. Tester un mouvement simple

**Option A - Avec votre script Python existant**

Votre script `ik_move_cpp` ou votre code de cinématique inverse devrait fonctionner directement en publiant sur `/joint_trajectory_controller/follow_joint_trajectory`.

**Option B - Test manuel rapide**

```bash
# Envoyer une commande de test (nécessite ros2_control_test_assets ou similaire)
ros2 topic pub --once /joint_trajectory_controller/follow_joint_trajectory/goal \
  control_msgs/action/FollowJointTrajectory \
  "{'trajectory': ...}"
```

---

## 📂 Fichiers créés

```
ur_coppeliasim/
├── launch/
│   ├── test_bridge.launch.py          # Test simple du bridge
│   └── ur_coppeliasim_moveit.launch.py # (À venir) Bridge + MoveIt complet
├── scripts/
│   ├── coppeliasim_bridge.py          # Bridge ROS2 ↔ CoppeliaSim
│   └── test_movement.py               # Script de test
├── config/
│   └── bridge_config.yaml
└── README.md
```

---

## 🔧 Architecture

```
CoppeliaSim (port 19999)
    ↕ Legacy Remote API
coppeliasim_bridge.py
    ↕ ROS2 Topics/Actions
Votre code Python / MoveIt / ik_move_cpp
```

**Le bridge fait** :
1. **Lit** les positions des joints depuis CoppeliaSim (50 Hz)
2. **Publie** `/joint_states` pour ROS2
3. **Écoute** les trajectoires sur l'action server
4. **Envoie** les commandes à CoppeliaSim

---

## ✨ Prochaines étapes

### Pour utiliser avec MoveIt

1. **Modifier le contrôleur MoveIt** pour qu'il utilise `joint_trajectory_controller` au lieu de `scaled_joint_trajectory_controller`

2. **Lancer MoveIt + Bridge** :
   ```bash
   # Terminal 1 : Bridge
   ros2 launch ur_coppeliasim test_bridge.launch.py
   
   # Terminal 2 : MoveIt
   ros2 launch ur_moveit_config ur_moveit.launch.py \
       ur_type:=ur10 \
       use_fake_hardware:=false \
       launch_rviz:=true
   ```

### Pour utiliser avec votre code de cinématique inverse

Votre code Python actuel doit juste publier sur l'action `/joint_trajectory_controller/follow_joint_trajectory`.

**Exemple d'adaptation** de votre code existant :

```python
import rclpy
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

# ... votre code de cinématique inverse ...

# Au lieu de sim.simxSetJointTargetPosition directement :
# Créer une trajectoire ROS2
trajectory = JointTrajectory()
trajectory.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 
                          'elbow_joint', 'wrist_1_joint', 
                          'wrist_2_joint', 'wrist_3_joint']

point = JointTrajectoryPoint()
point.positions = joint_pos.tolist()  # Vos positions calculées
point.time_from_start = Duration(sec=2, nanosec=0)
trajectory.points = [point]

# Envoyer via l'action client
action_client.send_goal_async(FollowJointTrajectory.Goal(trajectory=trajectory))
```

---

## 🐛 Dépannage

### "Impossible de se connecter à CoppeliaSim"
- Vérifiez que CoppeliaSim est lancé
- Vérifiez que le port 19999 est utilisé (Legacy API)
- Dans CoppeliaSim : Tools → User Settings → vérifier que Legacy Remote API est activé

### "Joint UR10_jointX NON TROUVÉ"
- Vérifiez les noms des joints dans votre scène CoppeliaSim (clic droit → Properties)
- Ils doivent être exactement : `UR10_joint1`, `UR10_joint2`, etc.

### Le robot ne bouge pas
- Vérifiez que les joints sont en mode "Torque/Force" ou "Hybrid" dans CoppeliaSim
- Vérifiez que la simulation est démarrée (bouton Play)

---

## 📝 Notes

- Le bridge utilise l'**ancienne API CoppeliaSim** (Legacy Remote API, port 19999)
- Fréquence de mise à jour : **50 Hz**
- Compatible avec votre code Python existant (même structure que `ur10_etudiant.py`)

---

**Besoin d'aide ?** Vérifiez les logs avec :
```bash
ros2 topic echo /rosout
```
#

ros2 launch ur_coppeliasim test_bridge.launch.py

ros2 launch ur_simulation_gazebo ur_sim_moveit.launch.py ur_type:=ur10