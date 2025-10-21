# ur_coppeliasim - Intégration CoppeliaSim pour Universal Robots

Package ROS2 pour connecter MoveIt à CoppeliaSim avec les robots UR.

## Prérequis

1. **CoppeliaSim** installé (https://www.coppeliarobotics.com/)
2. **Python ZMQ Remote API** pour CoppeliaSim :
   ```bash
   pip3 install coppeliasim-zmqremoteapi-client
   ```

3. **MoveIt2** et packages UR :
   ```bash
   sudo apt install ros-humble-moveit ros-humble-ur-moveit-config
   ```

## Installation

```bash
cd ~/Documents/UR_WS
colcon build --packages-select ur_coppeliasim
source install/setup.bash
```

## Utilisation

### 1. Lancer CoppeliaSim avec votre scène UR

Ouvrez CoppeliaSim et chargez votre scène avec le robot UR.

**Important** : Vérifiez que les noms des joints dans CoppeliaSim correspondent à :
- `shoulder_pan_joint`
- `shoulder_lift_joint`
- `elbow_joint`
- `wrist_1_joint`
- `wrist_2_joint`
- `wrist_3_joint`

### 2. Lancer le bridge ROS2 + MoveIt

Dans un terminal :
```bash
source ~/Documents/UR_WS/install/setup.bash
ros2 launch ur_coppeliasim ur_coppeliasim_moveit.launch.py ur_type:=ur10
```

Options disponibles :
- `ur_type` : Type de robot (ur3, ur5, ur10, etc.) - default: ur10
- `launch_rviz` : Lancer RViz (true/false) - default: true
- `use_sim_time` : Utiliser le temps de simulation (true/false) - default: false

### 3. Tester les mouvements

#### Via RViz (interface graphique)
1. Dans RViz, onglet "MotionPlanning"
2. Sélectionnez "Planning Group" = `ur_manipulator`
3. Déplacez le robot avec les marqueurs interactifs
4. Cliquez "Plan" puis "Execute"

#### Via code Python
Utilisez votre script existant `ik_move_cpp` ou créez un script Python :

```python
import rclpy
from rclpy.node import Node
from moveit.planning import MoveGroupInterface

rclpy.init()
node = Node("test_movement")
move_group = MoveGroupInterface(node, "ur_manipulator")

# Définir une pose cible
move_group.set_pose_target([0.3, 0.2, 0.4, 0, 0, 0])  # x, y, z, roll, pitch, yaw
move_group.go(wait=True)
```

## Architecture

```
CoppeliaSim <--ZMQ--> coppeliasim_bridge <--ROS2--> MoveIt <--> RViz
                            |
                            +---> /joint_states (publie)
                            +---> /joint_trajectory_controller/follow_joint_trajectory (action server)
```

## Dépannage

### CoppeliaSim ne se connecte pas
- Vérifiez que CoppeliaSim est lancé
- Vérifiez le port ZMQ (par défaut 23000)
- Dans CoppeliaSim, allez dans Tools > Settings > ZMQ Remote API

### Les noms de joints ne correspondent pas
Modifiez les noms dans votre scène CoppeliaSim ou adaptez le fichier `config/bridge_config.yaml`

### MoveIt planifie mais le robot ne bouge pas
- Vérifiez les logs du bridge : `ros2 topic echo /joint_states`
- Vérifiez que les joints sont en mode "torque/force" dans CoppeliaSim

## Exemple de scène CoppeliaSim

Les joints doivent être configurés ainsi dans CoppeliaSim :
1. Mode : "Torque/force mode" ou "Hybrid operation"
2. Nom du joint : exactement comme dans la liste ci-dessus
3. Control loop enabled : coché

## Alternative sans CoppeliaSim

Si CoppeliaSim n'est pas disponible, le bridge fonctionne en mode "simulation" :
- Les trajectoires sont acceptées mais pas exécutées physiquement
- Utile pour tester MoveIt sans simulateur

## Voir aussi

- Documentation MoveIt2 : https://moveit.picknik.ai/
- CoppeliaSim tutorials : https://www.coppeliarobotics.com/helpFiles/
- UR ROS2 driver : https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver


 source /home/luca/ros_projetcs/ur_ws/install/setup.bash && ros2 launch ur_coppeliasim ur_coppeliasim_moveit.launch.py ur_type:=ur10 launch_rviz:=true