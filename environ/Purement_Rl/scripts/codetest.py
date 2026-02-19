from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher

# Launch Isaac Sim / Kit first (required for carb/omni)
parser = argparse.ArgumentParser(description="Policy inference in Isaac Lab")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane  # noqa: E402

# ============================================================
# CONFIG
# ============================================================

USD_ROBOT_PATH = "/home/ajin/work2/my_env/ur10bras.usd"   # 🔴 adapte
POLICY_PATH = "/home/ajin/work2/IsaacLab/logs/rsl_rl/cartpole_direct/2026-01-04_04-26-52/model_115000.pt"
DT = 0.01
DOF = 6
ACTION_SCALE = 0.5
MAX_JOINT_VEL = torch.tensor([1.5, 1.5, 1.5, 2.0, 2.0, 2.0])

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

EE_LINK_NAME = "wrist_3_link"

# ============================================================
# SIMULATION
# ============================================================

sim_cfg = sim_utils.SimulationCfg(
    dt=DT,
    device="cuda",
    gravity=(0.0, 0.0, -9.81),
)

sim = SimulationContext(sim_cfg)
sim.reset()

# Ground (robust IsaacLab pattern)
ground_cfg = GroundPlaneCfg()
ground_cfg.func("/World/Ground", ground_cfg)

# ============================================================
# ROBOT
# ============================================================

robot_cfg = ArticulationCfg(
    prim_path="/World/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_ROBOT_PATH,
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=JOINT_NAMES,
            stiffness=10000.0,
            damping=1000.0,
        )
    },
)

robot = Articulation(robot_cfg)

# Initialize buffers/state (API differs across IsaacLab versions)
sim.reset()
try:
    robot.reset()
except Exception:
    pass

joint_ids, _ = robot.find_joints(JOINT_NAMES)
ee_body_id, _ = robot.find_bodies([EE_LINK_NAME])
ee_body_id = int(ee_body_id[0])

# ============================================================
# POLICY
# ============================================================

def load_policy(path: str, device: str | torch.device):
    """Load a TorchScript policy.

    Note: RSL-RL checkpoints are usually *not* TorchScript.
    If you point this to a training checkpoint (dict with model_state_dict),
    export a JIT policy first (see message below).
    """
    exported_path = Path(path).parent / "exported" / "policy.pt"

    # 1) Normal case: a real TorchScript module
    try:
        policy_module = torch.jit.load(path, map_location=device)
        policy_module.eval()
        return policy_module
    except Exception as e:
        # 2) If user pointed at an RSL-RL checkpoint, auto-fallback to exported policy if present.
        try:
            ckpt = torch.load(path, map_location="cpu")
        except Exception:
            ckpt = None

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            if exported_path.is_file():
                policy_module = torch.jit.load(str(exported_path), map_location=device)
                policy_module.eval()
                print(f"[INFO] POLICY_PATH is a checkpoint; using exported policy: {exported_path}")
                return policy_module

            raise RuntimeError(
                "POLICY_PATH points to an RSL-RL training checkpoint (state_dict), not a TorchScript policy.\n"
                f"- Current: {path}\n"
                "Export the policy once using the repo play script, then rerun this script.\n"
                "Example command (exports to exported/policy.pt):\n"
                "  cd /home/ajin/work2/IsaacLab\n"
                "  ./isaaclab.sh -p /home/ajin/work2/sim2real-pnp/environ/Purement_Rl/scripts/rsl_rl/play.py \\\n+    --task Template-Purement-Rl-Direct-v0 \\\n+    --checkpoint "
                + path
                + " \\\n+    --headless --video --video_length 1\n"
                f"Then set POLICY_PATH to: {exported_path}\n"
            ) from e

        # Not a known checkpoint format -> rethrow original error
        raise


policy = load_policy(POLICY_PATH, sim.device)

last_action = torch.zeros(DOF, device=sim.device)

# ============================================================
# TRAJECTOIRE (dataset Isaac)
# ============================================================

data_path = Path(__file__).parent / "tag2_to_tag3.npz"
data = np.load(str(data_path), allow_pickle=True)
q_traj = torch.tensor(data["paths"][0], device=sim.device, dtype=torch.float32)

# Dataset key differs depending on generator version
_ee_key = None
for candidate in ["ee_pos", "ee_ref_pos", "ee_des", "ee_traj", "eef_pos"]:
    if candidate in data:
        _ee_key = candidate
        break
if _ee_key is None:
    raise KeyError(
        f"No end-effector trajectory found in {data_path}. "
        f"Available keys: {list(data.keys())}. "
        "Expected one of: ee_ref_pos / ee_pos."
    )

ee_traj = torch.tensor(data[_ee_key][0], device=sim.device, dtype=torch.float32)
traj_len = q_traj.shape[0]
traj_phase = 0

# ============================================================
# UTILS
# ============================================================

def get_ee_pos_local():
    d = robot.data
    ee_state = d.body_state_w[:, ee_body_id, :]
    ee_pos_w = ee_state[:, 0:3]
    origin = torch.zeros_like(ee_pos_w)
    return ee_pos_w - origin

# ============================================================
# MAIN LOOP
# ============================================================

print("[INFO] Démarrage inference Isaac Lab")

try:
    while sim.is_running() and traj_phase < traj_len - 1:

        # --------------------------------------------------------
        # READ STATE
        # --------------------------------------------------------
        q = robot.data.joint_pos[:, joint_ids][0]
        qd = robot.data.joint_vel[:, joint_ids][0]
        ee_meas = get_ee_pos_local()[0]

        # --------------------------------------------------------
        # REFERENCES
        # --------------------------------------------------------
        q_ref = q_traj[traj_phase]
        ee_des = ee_traj[traj_phase]

        err_q = q_ref - q
        e_meas = ee_des - ee_meas
        e_meas_norm = torch.norm(e_meas).unsqueeze(0)

        # --------------------------------------------------------
        # OBSERVATION (STRICTEMENT IDENTIQUE)
        # --------------------------------------------------------
        obs = torch.cat(
            [
                q,
                qd,
                q_ref,
                err_q,
                ee_meas,
                ee_des,
                e_meas,
                e_meas_norm,
                last_action,
            ],
            dim=0,
        ).unsqueeze(0)

        # --------------------------------------------------------
        # POLICY
        # --------------------------------------------------------
        with torch.no_grad():
            action = policy(obs)[0]

        action = torch.clamp(action, -1.0, 1.0)

        # --------------------------------------------------------
        # ACTION → COMMANDE (IDENTIQUE TRAINING)
        # --------------------------------------------------------
        q_des = q_ref + action * ACTION_SCALE
        qd_cmd = (q_des - q) / DT
        qd_cmd = torch.clamp(qd_cmd, -MAX_JOINT_VEL, MAX_JOINT_VEL)

        robot.set_joint_velocity_target(qd_cmd.unsqueeze(0), joint_ids)
        robot.set_joint_position_target((q + qd_cmd * DT).unsqueeze(0), joint_ids)

        last_action = action.clone()
        traj_phase += 1

        # Push targets to simulator (if required by this IsaacLab version)
        if hasattr(robot, "write_data_to_sim"):
            robot.write_data_to_sim()

        sim.step()

        # Refresh internal buffers
        if hasattr(robot, "update"):
            robot.update(sim.dt)

finally:
    simulation_app.close()

print("[INFO] Fin de trajectoire")
