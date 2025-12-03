# sim2real-pnp

/ur10e/joints/ee_joint

commende reduit "./isaaclab.sh --python scripts/reinforcement_learning/skrl/train.py --task=Isaac-Velocity-Rough-H1-v0 --num_envs=20 --headless

lance un train avec environ " ./isaaclab.sh --python scripts/reinforcement_learning/skrl/train.py --task=Isaac-Velocity-Rough-H1-v0 --num_envs=20
##cree un envireonement

#cmd 1
nouveau projet " ./isaaclab.sh --new
#2
python -m pip install -e source/<given-project-name>
#3
./isaaclab.sh --python scripts/reinforcement_learning/skrl/train.py --task=Isaac-Reach-UR10-v0 --num_envs=20






ros2 launch ur_coppeliasim apriltag_detection.launch.py

source /opt/ros/humble/setup.bash && timeout 5 ros2 topic echo /detections --once


rm -rf ~/.ros/log/*

ros2 topic pub --once /start_motion std_msgs/msg/Bool "data: true"

