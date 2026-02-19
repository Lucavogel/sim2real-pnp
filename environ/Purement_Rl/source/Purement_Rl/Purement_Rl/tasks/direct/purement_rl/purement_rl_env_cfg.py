from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
import math


@configclass
class PurementRlEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 20.0

    action_space = 6

    # obs = q(6)+qd(6)+q_ref(6)+err_q(6)+ee_meas(3)+ee_des(3)+e_meas(3)+||e_meas||(1)+actions(6) = 40
    observation_space = 40
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation)

    traj_path: str = "/home/ajin/work2/sim2real-pnp/environ/Purement_Rl/scripts/single_line_12HZ_Final.npz"
    traj_dt: float = 1.0 / 30.0

    # How many dataset waypoints to advance per RL step.
    # If 0, it is auto-computed as round(step_dt / traj_dt) where step_dt = sim.dt * decimation.
    traj_phase_inc: int = 0

    # ============================================================
    # End-of-trajectory behavior
    # ============================================================
    # What to do when the phase reaches the end of a trajectory:
    # - "hold": keep using the last waypoint (current behavior)
    # - "reset": truncate episode when the trajectory ends (no "stuck" hold)
    # - "loop": wrap phase around to the start (cyclic reference)
    # NOTE: For sim2real tracking, "reset" is recommended.
    # It avoids a discontinuity jump from the last waypoint back to the first ("loop"),
    # and triggers an episode reset where the env teleports the robot to the next
    # trajectory start (joint_pos overwritten, joint_vel zeroed).
    traj_end_mode: str = "reset"

    # When an episode resets, whether to sample a new trajectory id.
    # - True (training default): each episode may start from a different trajectory.
    # - False (deployment-friendly): keep the current traj_id and restart from phase=0.
    resample_traj_id_on_reset: bool = True

    # décor
    env_usd_path: str = "/home/ajin/work2/my_env/Origin2_asset1.usd"
    env_asset_cfg: sim_utils.UsdFileCfg = sim_utils.UsdFileCfg(
        usd_path="/home/ajin/work2/my_env/Origin2_asset1.usd",
    )

    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/ajin/work2/my_env/ur10bras.usd",
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "shoulder_pan_joint": 0.0,
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
                effort_limit=330.0,
                stiffness=10000.0,
                damping=1000.0,
            ),
        },
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=5.0,
        replicate_physics=True,
    )

    joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    # ----------------------------
    # Lookahead + Action scaling
    # ----------------------------
    lookahead_steps: int = 1
    # Residual authority in radians (50 mrad ≈ 2.8°)
    action_scale: float = 0.05
    reward_warmup_steps: int = 0

    # ----------------------------
    # Reward 3.4
    # ----------------------------
    # NOTE: si la reward est très négative, c'est souvent que les pénalités dominent
    # et/ou que le noyau exp() est trop "étroit" (sigma trop petit => reward ~0 loin de la cible).
    # User-spec reward uses: exp(-e^2 / sigma)
    sigma_pos: float = 0.02
    sigma_joint: float = 0.05

    # Legacy (kept for backwards compatibility; no longer used by default reward)
    rew_scale_track: float = 5.0
    rew_scale_guide: float = 1.0

    # Shaping additionnel
    # - success: bonus par step quand l'EE est dans le seuil
    # - progress: encourage la réduction de l'erreur (delta error)
    rew_scale_success: float = 0.5
    rew_scale_progress: float = 0.1
    lambda_action: float = 0.01
    lambda_smooth: float = 0.1

    # ----------------------------
    # Stabilité OT / régularisation vitesses
    # ----------------------------
    # pénalité vitesse EE (anti-tremblement): w * ||v_ee||^2
    lambda_ee_speed: float = 0.03
    # pénalité accélération EE (stabilité): w * ||a_ee||^2
    lambda_ee_accel: float = 0.01

    # Réduction mouvement vertical (axe Z)
    # Désactivé par défaut: l'agent peut bouger en Z sans pénalités/contraintes.
    lambda_ee_speed_z: float = 0.0
    lambda_ee_accel_z: float = 0.0

    # pénalité dérive verticale: w * (z_des - z_true)^2
    lambda_ee_z_pos: float = 0.0
    ee_z_pos_sq_clip: float = 0.05  # (m^2) ~ 22cm RMS si 0.05, mettre plus petit si tu veux strict

    # Soft speed limits (met <= 0 pour désactiver)
    max_ee_speed_m_s: float = 0.0
    lambda_ee_speed_over: float = 0.0

    # Cap vitesse Z (met <= 0 pour désactiver)
    max_ee_speed_z_m_s: float = 0.0
    lambda_ee_speed_z_over: float = 0.0

    # Terminations "hard" sur Z (désactivé par défaut)
    terminate_on_ee_speed_z_over: bool = False
    max_ee_z_error_m: float = 0.0
    terminate_on_ee_z_error_over: bool = False
    # régularisation vitesse joints: w * ||qd||^2
    lambda_joint_vel: float = 0.05

    # Clipping des pénalités (évite les épisodes avec reward énorme négative)
    # Mettre <= 0 pour désactiver.
    ee_speed_sq_clip: float = 4.0
    ee_accel_sq_clip: float = 100.0

    # Clipping optionnel spécifique à l'axe Z (met <= 0 pour désactiver)
    ee_speed_z_sq_clip: float = 4.0
    ee_accel_z_sq_clip: float = 100.0

    # ----------------------------
    # Success metric (success_rate)
    # ----------------------------
    success_ee_thresh_m: float = 0.2     # 2 cm
    success_hold_steps: int = 2           # doit être bon 1 step d’affilée pour compter "success"

    # ----------------------------
    # Latence système (délai d’action)
    # ----------------------------
    action_delay_min: int = 1
    action_delay_max: int = 3

    # ============================================================
    # CAMERA PINHOLE (TES INTRINSICS + REPROJECTION ERROR)
    # ============================================================
    cam_fx: float = 491.6
    cam_fy: float = 488.0
    cam_cx: float = 381.4
    cam_cy: float = 221.9

    # Bruit en pixels (reprojection error moyen)
    cam_pixel_noise_std: float = 0.355

    # Optionnel : en plus du bruit pixel, tu peux ajouter un bruit direct XYZ.
    # IMPORTANT: si `use_camera_error_for_reward=True`, un jitter trop grand peut écraser la reward.
    cam_xyz_jitter_std_m: float = 0.002  # 2 mm

    # Bruit profondeur optionnel
    cam_depth_noise_std: float = 0.0

    # Décalage de calibration extrinsèque (CONSTANT par épisode)
    cam_extrinsic_bias_max_m: float = 0.02  # +/- 2 cm

    # Si False: le biais extrinsèque est tiré une seule fois au démarrage (erreur de calibration fixe)
    # Si True: le biais est re-tiré à chaque reset d'épisode (erreur variable)
    resample_cam_extrinsic_bias_each_episode: bool = True

    # Pose caméra dans repère env local (exemple)
    cam_pos_local: tuple[float, float, float] = (0.725, 0.0057, 1.565)
    cam_rpy_local: tuple[float, float, float] = (-math.pi, 0.0, 0.0)

    # Reward sur erreur caméra ?
    # Recommandé: False (reward sur erreur vraie), tout en gardant l'observation bruitée pour sim2real.
    use_camera_error_for_reward: bool = False

    # ============================================================
    # Bruit sur la cible (ee_des) - par timestep
    # ============================================================
    # Ajoute un bruit Gaussien N(0, std) sur la cible ee_des du dataset.
    # Par défaut on ne bruit que X/Y (pas Z) pour simuler une erreur 2D.
    ee_des_noise_std_m: float = 0.002  # 2 mm
    ee_des_noise_xy_only: bool = True

    # Où appliquer le bruit de ee_des:
    # - observation=True: l'agent voit une cible bruitée (robustesse)
    # - reward=False: la reward reste basée sur la trajectoire propre (objectif inchangé)
    ee_des_noise_in_observation: bool = True
    ee_des_noise_in_reward: bool = False

    # Si True: on force ee_des.z à une valeur fixe (celle de l'EE au reset)
    # => tracking 2D (XY) sans consigne de mouvement vertical.
    # Désactivé par défaut: la politique peut bouger en Z.
    freeze_ee_des_z: bool = False

    # ============================================================
    # Denoising caméra (correction bruit gaussien)
    # ============================================================
    # Filtre exponentiel (EMA) appliqué à la mesure XYZ locale.
    # - alpha proche de 1.0 => très peu de filtrage (presque brut)
    # - alpha petit (ex: 0.1..0.3) => lisse davantage
    use_camera_filter: bool = True
    cam_filter_alpha: float = 0.2

    # ============================================================
    # Domain randomization physique (damping + payload)
    # ============================================================
    randomize_damping: bool = True
    damping_scale_range: tuple[float, float] = (0.8, 1.2)

    # Si False: mêmes paramètres pour chaque env tout du long (pas de re-échantillonnage au reset)
    # Si True: re-échantillonne à chaque reset d'épisode
    resample_damping_each_episode: bool = True

    randomize_payload: bool = True
    payload_mass_range_kg: tuple[float, float] = (0.0, 0.5)  # +0..0.5 kg
    payload_body_name: str = "wrist_3_link"  # on essaye de modifier la masse de ce body

    # Si True: applique la masse additive sur TOUS les links (tous les rigid bodies du robot).
    # Par défaut, on DISTRIBUE la masse totale uniformément sur tous les links
    # (i.e. +payload_mass/num_links sur chaque link) pour éviter d'exploser la masse totale.
    payload_apply_to_all_links: bool = True
    # Mode d'application quand payload_apply_to_all_links=True:
    # - "distribute": masse totale distribuée uniformément sur tous les links
    # - "per_link": ajoute payload_mass à chaque link (masse totale augmente de num_links*payload_mass)
    payload_all_links_mode: str = "distribute"

    # Si False: même payload (par env) tout du long
    # Si True: re-échantillonne à chaque reset d'épisode
    resample_payload_each_episode: bool = True
    # ============================================================
    # Limits vitesse joints
    # ============================================================
    # une valeur unique pour tous les joints:
    max_joint_vel_rad_s: float = 1.5
    # ou bien par joint:
    # max_joint_vel_rad_s: tuple[float, ...] = (1.0, 1.0, 1.0, 1.5, 1.5, 2.0)

    # ============================================================
    # Domain randomization physique (friction)
    # ============================================================
    # NOTE: Les modifications PhysX doivent être faites côté env (runtime), pas dans la cfg.
    randomize_friction: bool = True
    friction_static_range: tuple[float, float] = (0.6, 1.2)
    friction_dynamic_range: tuple[float, float] = (0.5, 1.0)

    # Si False: friction FIXE par env pendant toute l'exécution (même valeur au reset)
    # Si True: re-échantillonne à chaque reset d'épisode
    resample_friction_each_episode: bool = False

    def __post_init__(self):
        super().__post_init__()

