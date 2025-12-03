# 🎯 Workflow de Génération des Trajectoires

## Vue d'ensemble

Les AprilTags sont dans **IsaacLab**, donc il faut lancer IsaacLab AVANT MoveIt pour que les TF soient disponibles.

```
IsaacLab (Tags) → TF → MoveIt → Trajectoires .npy
```

---

## 📋 Étapes détaillées

### 1️⃣ Terminal 1 : Lancer IsaacLab avec AprilTags

```bash
cd ~/workspace/sim2real-pnp/environ/my_env/scripts
./isaaclab.sh -p Ur10_moveit_Apriltag.py
```

**Ce qui doit fonctionner:**
- ✅ Robot UR10 visible et initialisé
- ✅ AprilTags Tag0 et Tag1 visibles dans la scène
- ✅ ROS2 Bridge actif (publication /joint_states, TF)
- ✅ Détection des tags par caméra

**Vérification:**
```bash
# Dans un autre terminal
source /opt/ros/humble/setup.bash
ros2 topic list | grep joint_states  # Doit afficher /joint_states
ros2 topic echo /joint_states         # Doit afficher positions robot
ros2 run tf2_ros tf2_echo world Tag0  # Doit afficher position Tag0
```

---

### 2️⃣ Terminal 2 : Lancer MoveIt

```bash
cd ~/workspace/sim2real-pnp/environ/ur10
source install/setup.bash
ros2 launch ur_coppeliasim ur_isaaclab_moveit.launch.py
```

**Ce qui doit fonctionner:**
- ✅ MoveIt démarre sans erreur
- ✅ Robot affiché dans RViz (synchronisé avec IsaacLab)
- ✅ Service `/compute_cartesian_path` disponible
- ✅ TF de IsaacLab visible dans RViz

**Vérification:**
```bash
ros2 service list | grep compute_cartesian_path  # Doit exister
ros2 run tf2_ros tf2_monitor                     # Vérifier TF actifs
```

---

### 3️⃣ Terminal 3 : Générer les trajectoires

```bash
cd ~/workspace/sim2real-pnp/environ/ur10
python3 src/ur_coppeliasim/scripts/generate_xy_trajectories.py \
    --num-lines 100 \
    --output trajectories
```

**Ce qui se passe:**
1. Script détecte service MoveIt
2. Lit positions Tag0 et Tag1 via TF (depuis IsaacLab)
3. Génère 100 lignes X-Y aléatoires entre les tags
4. Pour chaque ligne:
   - Génère waypoints cartésiens
   - Appelle MoveIt `/compute_cartesian_path`
   - Si succès (>95%), sauvegarde en `.npy`
5. Résultat: `trajectories/traj_000.npy` à `traj_099.npy`

**Sortie attendue:**
```
✅ Tag0: (1.000, 0.200, 0.900)
✅ Tag1: (1.000, -0.200, 0.900)
📏 Zone de travail X: [0.950, 1.050]
📏 Zone de travail Y: [-0.250, 0.250]
🔄 [1/100] Calcul trajectoire...
   ✅ Sauvegardé: traj_000.npy (100 points)
...
✅ Succès: 95/100
```

---

### 4️⃣ Copier trajectoires vers IsaacLab

```bash
cp -r ~/workspace/sim2real-pnp/environ/ur10/trajectories \
      ~/workspace/sim2real-pnp/environ/my_env/scripts/
```

---

### 5️⃣ Tester dans IsaacLab (nouveau script)

```bash
cd ~/workspace/sim2real-pnp/environ/my_env/scripts
./isaaclab.sh -p Ur10_trajectory_executor.py \
    --num_envs 1 \
    --apply_delta
```

---

## 🔍 Débogage

### Problème: Pas de TF pour Tag0/Tag1

**Cause:** IsaacLab ne publie pas les TF des tags

**Solution:**
1. Vérifier que les tags sont détectés dans IsaacLab (caméra doit voir les tags)
2. Vérifier ROS Bridge actif: `ros2 topic list`
3. Vérifier TF: `ros2 run tf2_ros tf2_monitor`

### Problème: Service MoveIt introuvable

**Cause:** MoveIt pas lancé ou erreur au démarrage

**Solution:**
```bash
ros2 service list  # Vérifier /compute_cartesian_path existe
ros2 launch ur_coppeliasim ur_isaaclab_moveit.launch.py  # Relancer
```

### Problème: Trajectoires fraction < 95%

**Cause:** Positions impossibles à atteindre (IK fail)

**Solution:**
- Vérifier que z_offset est correct (pas trop haut/bas)
- Augmenter zone de travail (moins de trajectoires extrêmes)
- Vérifier orientation fixée est atteignable

---

## 📊 Structure finale

```
sim2real-pnp/
├── environ/
│   ├── ur10/                          # Côté ROS2/MoveIt
│   │   ├── trajectories/              # Généré par MoveIt
│   │   │   ├── traj_000.npy
│   │   │   ├── traj_001.npy
│   │   │   └── ...
│   │   └── src/ur_coppeliasim/scripts/
│   │       └── generate_xy_trajectories.py  # Générateur
│   │
│   └── my_env/scripts/                # Côté IsaacLab
│       ├── trajectories/              # Copié depuis ur10/
│       ├── Ur10_moveit_Apriltag.py    # Ancien (avec ROS Bridge)
│       ├── Ur10_trajectory_executor.py  # Nouveau (sans ROS)
│       └── trajectory_executor.py     # Classe utilitaire
```

---

## 🎯 Résumé commandes

```bash
# Terminal 1
cd ~/workspace/sim2real-pnp/environ/my_env/scripts
./isaaclab.sh -p Ur10_moveit_Apriltag.py

# Terminal 2
cd ~/workspace/sim2real-pnp/environ/ur10
source install/setup.bash
ros2 launch ur_coppeliasim ur_isaaclab_moveit.launch.py

# Terminal 3
cd ~/workspace/sim2real-pnp/environ/ur10
python3 src/ur_coppeliasim/scripts/generate_xy_trajectories.py --num-lines 100

# Copier
cp -r ~/workspace/sim2real-pnp/environ/ur10/trajectories \
      ~/workspace/sim2real-pnp/environ/my_env/scripts/

# Tester nouveau système
cd ~/workspace/sim2real-pnp/environ/my_env/scripts
./isaaclab.sh -p Ur10_trajectory_executor.py --num_envs 1 --apply_delta
```
