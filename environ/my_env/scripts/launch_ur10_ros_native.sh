#!/bin/bash
echo "🤖 Lancement de Isaac Lab avec UR10 + ROS2 Native Bridge..."
echo ""

# ✅ ACTIVER ROS2 système (OBLIGATOIRE pour le bridge)
echo "🔧 Sourcing ROS2 Humble..."
source /opt/ros/humble/setup.bash

# ✅ Configurer ROS2
export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH=/opt/ros/humble/lib:$LD_LIBRARY_PATH

echo "✅ ROS2 Configuration:"
echo "   ROS_DISTRO: $ROS_DISTRO"
echo "   RMW_IMPLEMENTATION: $RMW_IMPLEMENTATION"
echo ""

#Activer Conda Isaac Lab
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

#Lancer Isaac Lab avec le Action Graph déjà dans le USD
cd ~/workspace/IsaacLab
echo "🚀 Lancement avec Action Graph ROS2 (configuré dans USD)..."
echo "   ✅ Pas de conflit Python - bridge natif Isaac Sim en C++"
echo "   📡 /joint_states sera publié automatiquement"
echo "   📥 /joint_command sera écouté automatiquement"
echo ""
./isaaclab.sh -p ../sim2real-pnp/environ/my_env/scripts/zero_agent_usd_bridge.py "$@"
