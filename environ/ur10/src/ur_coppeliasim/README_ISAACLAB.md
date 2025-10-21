# Bridge Isaac Lab ↔ ROS2/MoveIt2

Ce package contient un bridge pour contrôler le robot UR10 dans Isaac Lab via ROS2 et MoveIt2.

## Architecture

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│                 │         │                  │         │                 │
│   MoveIt2       │ ◄─────► │  Isaac Lab       │ ◄─────► │   Isaac Sim     │
│   (Planning)    │  ROS2   │  Bridge          │  API    │   (Simulation)  │
│                 │         │                  │         │                 │
└─────────────────┘         └──────────────────┘         └─────────────────┘
        │                            │
        │                            │
        ▼                            ▼
  /joint_states              /follow_joint_trajectory
  (publié)                   (action server)
```

## Fichiers principaux

1. **`isaaclab_bridge.py`** : Bridge principal qui :
   - Lance Isaac Lab avec le robot UR10
   - Publie l'état des joints sur `/joint_states`
   - Écoute les commandes de trajectoire via action server `/joint_trajectory_controller/follow_joint_trajectory`
   - Synchronise Isaac Lab et ROS2

2. **`ur_isaaclab_moveit.launch.py`** : Fichier de lancement qui démarre :
   - Le bridge Isaac Lab
   - MoveIt2 avec la configuration UR10
   - Les obstacles dans la scène de planning
   - RViz pour visualisation

3. **`test_isaaclab_moveit.py`** : Script de test pour valider la connexion

## Installation

### Prérequis

1. **Isaac Lab** installé et configuré (testez avec `./isaaclab.sh -p scripts/tutorials/00_sim/spawn_prims.py`)
2. **ROS2 Humble** installé
3. **MoveIt2** installé : `sudo apt install ros-humble-moveit`
4. **UR packages** installés (déjà dans votre workspace)

### Configuration

1. Compiler votre workspace ROS2 :
```bash
cd ~/work2/ur10
colcon build
source install/setup.bash
```

2. Vérifier le chemin Isaac Lab dans le launch file :
```python
# Dans ur_isaaclab_moveit.launch.py, ligne ~75
isaaclab_path = LaunchConfiguration("isaaclab_path")
# Par défaut: ~/work2/IsaacLab
```

## Utilisation

### 1. Lancer le système complet

```bash
# Terminal 1: Lancer Isaac Lab + MoveIt2
cd ~/work2/ur10
source install/setup.bash
ros2 launch ur_coppeliasim ur_isaaclab_moveit.launch.py
```

Vous devriez voir :
- ✅ Isaac Lab se lancer avec le robot UR10
- ✅ MoveIt2 démarrer avec RViz
- ✅ Le robot visible dans Isaac Lab ET dans RViz

### 2. Tester avec des commandes simples

```bash
# Terminal 2: Lancer le script de test
cd ~/work2/ur10
source install/setup.bash
ros2 run ur_coppeliasim test_isaaclab_moveit.py
```

Le robot devrait :
1. Aller à une position "home"
2. Faire un mouvement de "vague" (pan gauche-droite)

### 3. Utiliser MoveIt2 dans RViz

Dans RViz (lancé automatiquement) :
1. Dans le panneau "Planning" → "Planning Group" : sélectionnez `ur_manipulator`
2. Déplacez le marker interactif (flèches 3D bleues/vertes/rouges)
3. Cliquez sur "Plan" pour calculer une trajectoire
4. Cliquez sur "Execute" pour l'exécuter dans Isaac Lab

### 4. Utiliser depuis Python (MoveIt Python API)

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetPositionFK
from geometry_msgs.msg import PoseStamped
# ... (exemple complet à venir)
```

## Paramètres du launch file

```bash
ros2 launch ur_coppeliasim ur_isaaclab_moveit.launch.py \
    ur_type:=ur10 \
    launch_rviz:=true \
    add_obstacle:=true \
    obstacle_x:=0.5 \
    isaaclab_path:=/chemin/vers/IsaacLab
```

- `ur_type` : Type de robot UR (ur10, ur5, etc.)
- `launch_rviz` : Lancer RViz (true/false)
- `add_obstacle` : Ajouter un obstacle devant le robot (true/false)
- `obstacle_x` : Distance de l'obstacle (en mètres)
- `isaaclab_path` : Chemin vers votre installation Isaac Lab

## Debugging

### Le bridge ne démarre pas

```bash
# Vérifier que Isaac Lab fonctionne seul
cd ~/work2/IsaacLab
./isaaclab.sh -p scripts/tutorials/00_sim/spawn_prims.py
```

### MoveIt ne voit pas le robot

```bash
# Vérifier que les joint_states sont publiés
ros2 topic echo /joint_states

# Vérifier que l'action server est actif
ros2 action list
# Devrait montrer: /joint_trajectory_controller/follow_joint_trajectory
```

### Le robot ne bouge pas dans Isaac Lab

- Vérifier les logs du bridge : cherchez "✅ Trajectoire exécutée"
- Vérifier que Isaac Lab est bien lancé : vous devriez voir une fenêtre de simulation
- Vérifier les positions cibles : elles sont loggées en degrés

### Problème de performance

Si la simulation est trop lente :
1. Réduire la fréquence de publication dans `isaaclab_bridge.py` (ligne ~60) : `update_rate = 20.0` au lieu de 50.0
2. Ajuster le `dt` de simulation (ligne ~111) : `dt=0.02` au lieu de 0.01

## Architecture technique détaillée

### Flux de données : MoveIt2 → Isaac Lab

1. **MoveIt2** calcule une trajectoire (liste de positions articulaires)
2. **Action Client** envoie via `/follow_joint_trajectory` (action)
3. **Isaac Lab Bridge** reçoit la trajectoire :
   - Convertit les positions en tensors PyTorch
   - Envoie à `robot.set_joint_position_target()`
   - Step la simulation Isaac Lab
4. **Isaac Sim** exécute le mouvement physique du robot

### Flux de données : Isaac Lab → MoveIt2

1. **Isaac Lab** simule le robot à chaque step
2. **Bridge** lit `robot.data.joint_pos` et `robot.data.joint_vel`
3. **Bridge** publie sur `/joint_states` (sensor_msgs/JointState)
4. **MoveIt2** et **RViz** reçoivent l'état en temps réel

### Différences avec CoppeliaSim Bridge

| Aspect | CoppeliaSim Bridge | Isaac Lab Bridge |
|--------|-------------------|------------------|
| API | Legacy Remote API (TCP) | Python API native |
| Threading | ROS2 + Async API | ROS2 + Thread séparé pour Isaac |
| Performance | ~20-50 Hz | ~50-100 Hz (GPU) |
| Visualisation | CoppeliaSim + RViz | Isaac Sim + RViz |
| Physique | CoppeliaSim (Bullet/ODE) | PhysX (NVIDIA) |

## Prochaines étapes

- [ ] Ajouter le support des grippers (Robotiq)
- [ ] Ajouter des capteurs (caméra, force/torque)
- [ ] Intégration avec Isaac Lab environments pour RL
- [ ] Support multi-robots

## Troubleshooting commun

**Q: "Import isaaclab could not be resolved"**  
R: C'est normal, Isaac Lab doit être lancé via `isaaclab.sh`. Les imports sont résolus à l'exécution.

**Q: "Connection refused to Isaac Lab"**  
R: Le bridge lance Isaac Lab directement, pas besoin de connexion externe.

**Q: "Robot ne bouge pas après 'Execute' dans RViz"**  
R: Vérifiez les logs du bridge. Si vous voyez "❌ Isaac Lab non initialisé", redémarrez le système.

**Q: "Trajectoire rejetée: angles hors limites"**  
R: Les limites de sécurité sont désactivées par défaut. Vérifiez votre configuration MoveIt (SRDF).

## Références

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [MoveIt2 Tutorials](https://moveit.picknik.ai/main/index.html)
- [Universal Robots ROS2](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver)
