"""
Configuration UR10 basée sur Universal_Robots_ROS2_Description
Source unique URDF → compatible MoveIt2 + vrai robot
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


# ---------------------------------------------------------------------------- #
# UR10 ROS2 CONFIG
# ---------------------------------------------------------------------------- #
UR10_ROS2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/ajin/workspace/sim2real-pnp/environ/ur10/urdf_converted/ur10.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            fix_root_link=True,  # ✅ FIXER LE BASE_LINK!
        ),
        activate_contact_sensors=False,
    ),

    # ⚠️ 0.0, -1.309, 2.14675, -2.44346, -1.5708, 0.0
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "shoulder_pan_joint": 0.0,    # -75°
            "shoulder_lift_joint": -1.309,    # 123°
            "elbow_joint": 2.147,           # -140°
            "wrist_1_joint": -2.443,         # -90°
            "wrist_2_joint": -1.571,            # 0°
            "wrist_3_joint": 0.0,            # 0°
        },
    ),

    # ✅ ACTUATORS - GAINS RÉALISTES UR10 (specs constructeur)
    actuators={
        "ur10_joints": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            effort_limit=330.0,     # UR10 nominal torque (Nm)
            stiffness=10000.0,      # Gains typiques robots industriels
            damping=1000.0,         # Ratio critique ~0.1 * stiffness
        ),
    },
)
