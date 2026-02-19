source /opt/ros/humble/setup.bash

# Aller dans workspace UR10
cd ~/workspace/sim2real-pnp/environ/ur10

# Build (commentez après premier build)
echo "🔨 Build du workspace..."
colcon build

# Source workspace
source install/setup.bash

python3 src/ur_coppeliasim/scripts/generate_tag_to_tag_trajectory.py --output tag_to_tag_12HZ_Final.npz
