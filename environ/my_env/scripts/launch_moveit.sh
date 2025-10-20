#!/bin/bash
# Script pour lancer MoveIt2 + RViz (ROS2 système Python 3.10)

echo "📦 Lancement de MoveIt2 + RViz..."
echo "   (ROS2 système Python 3.10)"

# ⚠️ DÉSACTIVER CONDA si activé
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠️  Désactivation de Conda ($CONDA_DEFAULT_ENV)..."
    conda deactivate 2>/dev/null || true
fi

# Nettoyer les variables Conda restantes
unset CONDA_DEFAULT_ENV
unset CONDA_PREFIX
unset CONDA_PROMPT_MODIFIER
unset CONDA_SHLVL
unset CONDA_PYTHON_EXE
unset CONDA_EXE
unset _CE_CONDA
unset _CE_M

# Nettoyer PYTHONPATH
unset PYTHONPATH

# Réinitialiser PATH (enlever les chemins Conda)
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

echo "🔍 Vérification environnement:"
echo "   Python: $(which python3)"
echo "   ROS2: $(which ros2)"

# Aller dans le workspace ROS2
cd ~/work2/ur10

colcon build
# Source ROS2 système
source /opt/ros/humble/setup.bash

# Source le workspace local
source install/setup.bash

# Lancer MoveIt2
echo ""
echo "🚀 ros2 launch ur_coppeliasim ur_isaaclab_moveit.launch.py"
ros2 launch ur_coppeliasim ur_isaaclab_moveit.launch.py
