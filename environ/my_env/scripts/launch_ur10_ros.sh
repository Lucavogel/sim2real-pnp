echo "🤖 Lancement de Isaac Lab avec UR10..."

# Désactiver ROS2 système
unset ROS_DISTRO
unset AMENT_PREFIX_PATH
unset CMAKE_PREFIX_PATH
unset ROS_VERSION
unset ROS_PYTHON_VERSION
unset PYTHONPATH

#Activer Conda Isaac Lab
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

#Lancer Isaac Lab
cd ~/workspace/IsaacLab
echo "   (Conda Python 3.11 avec ROS2 Humble)"
./isaaclab.sh -p ../sim2real-pnp/environ/my_env/scripts/zero_agent.py "$@"