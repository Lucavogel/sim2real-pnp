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



🧮 Calibration...

✅ Résultats
RMS (OpenCV ret): 3.6761 px
Erreur moyenne (mean): 2.7165 px  (cible: < 0.5)
Pires vues (index, erreur_moy_px): [(8, 6.688069820404053), (29, 6.199604034423828), (31, 5.719046115875244), (9, 5.64873743057251), (55, 5.550762176513672)]

--- MATRICE CAMÉRA ---
fx=367.42, fy=366.75, cx=321.64, cy=242.03

--- DIST ---
[-0.16213709  0.32844888 -0.01198295 -0.00561599 -0.21310417]



21,5