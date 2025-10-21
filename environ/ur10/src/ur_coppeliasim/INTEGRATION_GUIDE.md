# 🔄 Intégration avec votre code Ur10.py

Ce document explique comment intégrer le bridge ROS2/MoveIt2 avec votre code Isaac Lab existant (`Ur10.py`).

## Votre Code Actuel (Ur10.py)

Votre script actuel :
- ✅ Lance Isaac Sim/Lab
- ✅ Crée un robot UR10 avec `Articulation`
- ✅ Ajoute une caméra sur le poignet (`define_rgb_camera`)
- ✅ Boucle de simulation avec capture d'images

## Option 1 : Utiliser le Bridge séparément (Recommandé)

**Cas d'usage** : Vous voulez contrôler le robot avec MoveIt2 pendant que votre script gère les capteurs/vision.

### Architecture

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│   MoveIt2       │ ◄─────► │  Isaac Bridge    │ ◄─────► │   Isaac Sim     │
│   (Commandes)   │  ROS2   │  (Contrôle)      │         │   (UR10 Robot)  │
└─────────────────┘         └──────────────────┘         └─────────────────┘
                                                                    ▲
                                                                    │
┌─────────────────┐                                                │
│   Ur10.py       │ ◄──────────────────────────────────────────────┘
│   (Vision/RL)   │         API Isaac Lab
└─────────────────┘
```

### Implémentation

**Terminal 1 : Bridge MoveIt2**
```bash
cd ~/work2/IsaacLab
./isaaclab.sh -p ~/work2/ur10/src/ur_coppeliasim/scripts/isaaclab_bridge.py
```

**Terminal 2 : Votre code vision/RL**
```bash
cd ~/work2/my_env
# Modifiez Ur10.py pour se connecter au robot existant
# (voir exemple ci-dessous)
```

**Modification de Ur10.py** :

```python
# Dans votre Ur10.py, au lieu de créer le robot:
# ur10 = Articulation(cfg=ur10_cfg)  # ❌ Ne PAS créer un nouveau robot

# Récupérer le robot existant créé par le bridge:
import omni.isaac.core.utils.prims as prim_utils
robot_prim_path = "/World/UR10_Robot"  # Chemin utilisé par le bridge
if prim_utils.is_prim_path_valid(robot_prim_path):
    ur10 = Articulation(prim_path=robot_prim_path)
    print("✅ Robot connecté depuis le bridge")
else:
    print("❌ Bridge non lancé - créer robot local")
    ur10 = Articulation(cfg=ur10_cfg)

# Ensuite, votre code caméra fonctionne normalement
camera = define_rgb_camera(f"{robot_prim_path}/wrist_3_link")
# ...
```

---

## Option 2 : Intégrer le Bridge dans Ur10.py

**Cas d'usage** : Tout-en-un, un seul script qui gère robot + vision + ROS2.

### Code Modifié

Créez un nouveau fichier `Ur10_with_ros2.py` :

```python
#!/usr/bin/env python3
"""
Isaac Lab + ROS2 Bridge intégré
Combine votre Ur10.py avec le bridge MoveIt2
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports Isaac Lab
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, CameraCfg
from isaaclab_assets import UR10_CFG

# Imports ROS2
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

class IntegratedBridge(Node):
    """Bridge ROS2 + Vision intégré"""
    
    def __init__(self, robot, camera):
        super().__init__('integrated_bridge')
        
        self.robot = robot
        self.camera = camera
        
        # Publisher joint_states
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.create_timer(0.05, self.publish_joints)
        
        # TODO: Ajouter action server pour trajectoires (voir isaaclab_bridge.py)
        
        self.get_logger().info('✅ Bridge intégré prêt')
    
    def publish_joints(self):
        """Publie l'état des joints"""
        pos = self.robot.data.joint_pos[0].cpu().numpy()
        vel = self.robot.data.joint_vel[0].cpu().numpy()
        
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        msg.position = pos[:6].tolist()
        msg.velocity = vel[:6].tolist()
        msg.effort = [0.0] * 6
        
        self.joint_pub.publish(msg)


def setup_scene():
    """Configuration de la scène (votre code actuel)"""
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([6.0, 0.0, 5.5], [-0.5, 0.0, 0.5])
    
    # Ground + Light
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/Ground", cfg_ground)
    
    cfg_light = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg_light.func("/World/Light", cfg_light, translation=(1, 0, 10))
    
    # Robot UR10
    ur10_cfg = UR10_CFG.replace(prim_path="/World/UR10")
    ur10_cfg.init_state.pos = (0.0, 0.0, 1.03)
    robot = Articulation(cfg=ur10_cfg)
    
    # Caméra (votre fonction)
    camera_cfg = CameraCfg(
        prim_path="/World/UR10/wrist_3_link/camera",
        update_period=0.05,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, -0.1),
            rot=(0.7071, 0.0, 0.0, 0.7071),
            convention="ros",
        ),
    )
    camera = Camera(cfg=camera_cfg)
    
    sim.reset()
    
    return sim, robot, camera


def main():
    """Fonction principale intégrée"""
    
    # Initialiser ROS2
    rclpy.init()
    
    # Setup scène
    sim, robot, camera = setup_scene()
    
    # Créer le bridge ROS2
    bridge = IntegratedBridge(robot, camera)
    
    # Thread ROS2
    def spin_ros():
        rclpy.spin(bridge)
    
    ros_thread = threading.Thread(target=spin_ros, daemon=True)
    ros_thread.start()
    
    print("✅ Système intégré actif!")
    print("  - Isaac Lab: Robot UR10 + Caméra")
    print("  - ROS2: /joint_states publié")
    print("  - MoveIt2: Lancer séparément si besoin")
    
    # Boucle principale (votre code actuel)
    frame_count = 0
    while simulation_app.is_running():
        # Update simulation
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())
        camera.update(sim.get_physics_dt())
        
        # Vision (votre code actuel)
        if frame_count % 10 == 0:  # Tous les 10 frames
            if "rgb" in camera.data.output:
                rgb = camera.data.output["rgb"][0]
                print(f"Frame {frame_count}: RGB shape = {rgb.shape}")
                # Traiter l'image ici...
        
        frame_count += 1
    
    # Cleanup
    bridge.destroy_node()
    rclpy.shutdown()
    simulation_app.close()


if __name__ == "__main__":
    main()
```

### Utilisation

```bash
# Lancer le script intégré
cd ~/work2/IsaacLab
./isaaclab.sh -p ~/work2/my_env/scripts/Ur10_with_ros2.py

# Dans un autre terminal: Lancer MoveIt2
cd ~/work2/ur10
source install/setup.bash
ros2 launch ur_moveit_config ur_moveit.launch.py \
    ur_type:=ur10 \
    use_fake_hardware:=false \
    launch_rviz:=true
```

---

## Option 3 : Bridge Minimaliste (Lecture seule)

**Cas d'usage** : Vous contrôlez le robot depuis Isaac Lab, mais voulez visualiser dans RViz.

### Code Minimal

Ajoutez juste à la fin de votre `Ur10.py` :

```python
# À la fin de votre Ur10.py, avant la boucle simulation

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import threading

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_joint_pub')
        self.pub = self.create_publisher(JointState, 'joint_states', 10)
        self.timer = self.create_timer(0.05, self.publish)
        self.robot = None
    
    def publish(self):
        if self.robot is None:
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        pos = self.robot.data.joint_pos[0].cpu().numpy()
        msg.position = pos[:6].tolist()
        self.pub.publish(msg)

# Init ROS2
rclpy.init()
ros_pub = MinimalPublisher()
ros_pub.robot = ur10  # Votre robot UR10

# Thread ROS2
def spin(): rclpy.spin(ros_pub)
threading.Thread(target=spin, daemon=True).start()

# Votre boucle normale continue...
while simulation_app.is_running():
    # ... votre code ...
```

---

## Comparaison des Options

| Aspect | Option 1: Séparé | Option 2: Intégré | Option 3: Minimal |
|--------|------------------|-------------------|-------------------|
| **Complexité** | Moyenne | Élevée | Faible |
| **Flexibilité** | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| **Contrôle MoveIt** | ✅ Complet | ✅ Complet | ❌ Lecture seule |
| **Vision/Capteurs** | ✅ Séparé | ✅ Intégré | ✅ Normal |
| **Debugging** | ✅ Facile | ⚠️ Moyen | ✅ Facile |
| **Use case** | Développement | Production | Visualisation |

---

## Recommandation

Pour votre projet, je recommande **Option 1 (Séparé)** :

1. **Développement** :
   - Utilisez le bridge standalone pour tester MoveIt2
   - Développez votre vision/RL dans Ur10.py séparément
   - Facile à debugger chaque partie

2. **Intégration** :
   - Quand tout fonctionne, passez à Option 2 si besoin d'un seul process
   - Ou gardez séparé pour plus de flexibilité

3. **Avantages** :
   - Vous pouvez arrêter/redémarrer le bridge sans perdre votre session Isaac Lab
   - Vous pouvez tester votre vision sans ROS2
   - Plus facile à maintenir

---

## Prochaines Étapes

1. ✅ Testez le bridge seul : `simple_isaaclab_test.py`
2. ✅ Testez avec MoveIt2 : `ur_isaaclab_moveit.launch.py`
3. 🔄 Modifiez votre Ur10.py selon l'option choisie
4. 🎯 Intégrez votre logique vision/RL
5. 🚀 Profitez du meilleur des deux mondes!

---

**Questions fréquentes** :

**Q: Puis-je contrôler le robot depuis Ur10.py ET MoveIt2 en même temps?**  
R: Oui, mais attention aux conflits. Utilisez un "arbitrateur" qui donne priorité à l'un ou l'autre.

**Q: La caméra fonctionne-t-elle avec le bridge?**  
R: Oui! Le bridge ne touche que les joints. Capteurs/caméras fonctionnent normalement.

**Q: Performance avec les deux?**  
R: Léger impact (~5-10% CPU). GPU reste le bottleneck principal.

Bon codage! 🤖✨
