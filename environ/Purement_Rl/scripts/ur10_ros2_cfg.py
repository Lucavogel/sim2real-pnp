import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg



# Configuration de la simulation
usd_config = sim_utils.UsdFileCfg(
    usd_path="/home/ajin/work2/my_env/ur10/ur10.usd",
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        max_depenetration_velocity=5.0,
    ),
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        fix_root_link=True,  # Fixe le base_link
    ),
    activate_contact_sensors=False,
)

# Initialisation de l'articulation sans `sim_config`
UR10_ROS2_CFG = ArticulationCfg(
    spawn=usd_config,  # Passez la configuration de l'USD dans le champ `spawn`
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "shoulder_pan_joint": 0.0,  # Exemple pour un robot UR10
            "shoulder_lift_joint": -1.309,
            "elbow_joint": 2.147,
            "wrist_1_joint": -2.443,
            "wrist_2_joint": -1.571,
            "wrist_3_joint": 0.0,
        },
    ),
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
            effort_limit=330.0,  # UR10 nominal torque (Nm)
            stiffness=10000.0,   # Gains typiques robots industriels
            damping=1000.0,      # Ratio critique ~0.1 * stiffness
        ),
    },
)
