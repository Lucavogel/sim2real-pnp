# GUIDE: Convertir URDF UR10 → USD pour Isaac Lab

## ✅ ÉTAPE 1: URDF généré

Le fichier URDF a été généré avec succès:
```
📂 /home/ajin/workspace/sim2real-pnp/environ/ur10/urdf_converted/ur10.urdf
```

Ce fichier contient le modèle exact de votre robot UR10 depuis `Universal_Robots_ROS2_Description`.

---

## 🔄 ÉTAPE 2: Conversion URDF → USD

### Option A: Via Isaac Sim GUI (RECOMMANDÉ - Le plus simple)

1. **Lancer Isaac Sim:**
   ```bash
   # Si installé via Omniverse Launcher:
   ~/.local/share/ov/pkg/isaac-sim-*/isaac-sim.sh
   
   # Ou depuis le menu Omniverse Launcher
   ```

2. **Importer l'URDF:**
   - File → Import → URDF
   - Ou: Isaac Utils → Workflows → URDF Importer
   
3. **Configuration de l'import:**
   ```
   Input File: /home/ajin/workspace/sim2real-pnp/environ/ur10/urdf_converted/ur10.urdf
   
   Options:
   ✅ Import Inertia Tensor
   ✅ Create Physics Scene: NON (on ajoutera ça dans Isaac Lab)
   ✅ Fix Base Link: NON (le robot doit pouvoir bouger)
   ✅ Merge Fixed Joints: NON
   ✅ Self Collision: OUI
   
   Joint Drive Type: Position
   Joint Drive Stiffness: 10000.0
   Joint Drive Damping: 1000.0
   ```

4. **Sauvegarder en USD:**
   - File → Save As
   - Chemin: `/home/ajin/workspace/sim2real-pnp/environ/ur10/urdf_converted/ur10.usd`

---

### Option B: Via script Python Isaac Sim (si GUI ne marche pas)

Créer un script `convert_manual.py`:

```python
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.isaac.core.utils.extensions as ext_utils
ext_utils.enable_extension("omni.isaac.urdf")

import omni.kit.commands
from pxr import Usd

urdf_path = "/home/ajin/workspace/sim2real-pnp/environ/ur10/urdf_converted/ur10.urdf"
usd_path = "/home/ajin/workspace/sim2real-pnp/environ/ur10/urdf_converted/ur10.usd"

print(f"Importing URDF: {urdf_path}")

success, prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=urdf_path,
    import_config=omni.isaac.urdf.ImportConfig(
        set_default_prim=True,
        create_physics_scene=False,
        import_inertia_tensor=True,
        fix_base=False,
        merge_fixed_joints=False,
        self_collision=True,
        default_drive_type=omni.isaac.urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION,
        default_position_drive_damping=1000.0,
        default_position_drive_stiffness=10000.0,
    ),
    dest_path="/World/ur10"
)

if success:
    stage = omni.usd.get_context().get_stage()
    stage.Export(usd_path)
    print(f"✅ USD saved: {usd_path}")
else:
    print("❌ Import failed!")

simulation_app.close()
```

Lancer avec:
```bash
~/.local/share/ov/pkg/isaac-sim-*/python.sh convert_manual.py
```

---

## 📝 ÉTAPE 3: Utiliser le USD dans Isaac Lab

Une fois le fichier `ur10.usd` créé, modifiez vos scripts Isaac Lab:

### Dans `Ur10_trajectory_executor.py`:

```python
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg

# Remplacer UR10_CFG par votre USD converti
ur10_cfg = ArticulationCfg(
    prim_path="/World/Origin2/Table/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/ajin/workspace/sim2real-pnp/environ/ur10/urdf_converted/ur10.usd",
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.57,
            "elbow_joint": 1.57,
            "wrist_1_joint": -1.57,
            "wrist_2_joint": -1.57,
            "wrist_3_joint": 0.0,
        },
    ),
    actuators={
        "ur10_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=10000.0,
            damping=1000.0,
        ),
    },
)

ur10 = Articulation(cfg=ur10_cfg)
```

---

## ✅ RÉSULTAT ATTENDU

Une fois cette conversion faite:

✔️ MoveIt et Isaac Lab utilisent **exactement** le même modèle de robot
✔️ Les trajectoires MoveIt fonctionnent **sans offset** dans Isaac Lab
✔️ Les frames (`tool0`, `base_link`, etc.) sont **identiques**
✔️ Les longueurs de liens sont **identiques**
✔️ Les inerties sont **identiques**

➡️ **Plus besoin de mapping ou d'offset !**

---

## 🐛 Dépannage

### "Cannot find package ur_description"
- Assurez-vous que le workspace ROS2 est sourcé:
  ```bash
  source /home/ajin/workspace/sim2real-pnp/environ/ur10/install/setup.bash
  ```

### "URDF import failed in Isaac Sim"
- Vérifiez que le fichier URDF est valide:
  ```bash
  check_urdf ur10.urdf
  ```

### "Le robot ne bouge pas correctement"
- Vérifiez les valeurs de `stiffness` et `damping` des actuators
- Augmentez si le robot est mou: `stiffness=50000.0, damping=5000.0`

---

## 📞 Prochaines étapes

1. Faites la conversion URDF → USD (Option A ou B)
2. Testez avec un script simple Isaac Lab
3. Vérifiez que les poses correspondent entre MoveIt et Isaac Lab
4. Remplacez `UR10_CFG` dans tous vos scripts

**Status actuel:**
- ✅ URDF généré
- ⏳ USD à créer (vous devez faire Option A ou B)
- ⏳ Tester l'alignement MoveIt ↔ Isaac Lab
