# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--save_dir",
    type=str,
    default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "outputmodel")),
    help="Directory where the trained model will be saved (in addition to logs).",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import torch
import numpy as np  # <--- Ajout de l'import numpy
from datetime import datetime

import omni
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import Purement_Rl.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


# <--- Ajout de la classe CustomOnPolicyRunner
class CustomOnPolicyRunner(OnPolicyRunner):
    """
    Custom runner with automatic stopping based on reward convergence.
    Adapted for UR10 Reach task (dense reward).
    """

    def __init__(
        self,
        *args,
        save_dir: str | None = None,
        auto_stop_enabled: bool = True,
        auto_stop_window: int = 50,
        auto_stop_min_logs: int = 50,
        auto_stop_var: float = 1e-5,
        auto_stop_std: float = 2.0,
        auto_stop_span: float = 5.0,
        auto_stop_slope: float = 0.005,
        auto_stop_delta_mean: float = 1.0,
        auto_stop_debug: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.save_dir = save_dir

        self.auto_stop_enabled = bool(auto_stop_enabled)
        self.auto_stop_debug = bool(auto_stop_debug)

        # Historique des rewards moyens (log points)
        self.reward_mean_history: list[float] = []

        # Fenêtre glissante / seuils (valeurs en unité de reward)
        self.history_len = int(auto_stop_window)
        self.min_logs_before_stop = int(auto_stop_min_logs)
        # Variance-only auto-stop threshold.
        self.var_threshold = float(auto_stop_var)
        # (Unused now; kept in signature for compatibility with existing scripts)
        self.std_threshold = float(auto_stop_std)
        self.span_threshold = float(auto_stop_span)
        self.slope_threshold = float(auto_stop_slope)
        self.delta_mean_threshold = float(auto_stop_delta_mean)

        self.log_counter = 0

        self._printed_locs_keys = False
        self._warned_no_reward = False

    def log(self, locs, width=80, pad=35):
        # Logger standard RSL-RL
        super().log(locs, width, pad)
        self.log_counter += 1

        if not self.auto_stop_enabled:
            return

        # --------- One-shot debug: quelles clés existent vraiment ? ---------
        if self.auto_stop_debug and not getattr(self, "_printed_locs_keys", False):
            try:
                keys = sorted(list(locs.keys()))
            except Exception:
                keys = []

            print(f"[AUTO-STOP] locs keys ({len(keys)}): {keys}")

            # Affiche quelques clés probables
            for k in ["mean_reward", "Train/mean_reward", "rewbuffer", "ep_infos"]:
                if isinstance(locs, dict) and k in locs:
                    v = locs.get(k)
                    try:
                        v_len = len(v)
                    except Exception:
                        v_len = None
                    extra = f" len={v_len}" if v_len is not None else ""
                    print(f"[AUTO-STOP] locs[{k!r}] type={type(v).__name__}{extra}")

            self._printed_locs_keys = True

        # --------- Helpers ---------
        def _to_float(x):
            if x is None:
                return None
            try:
                # torch scalar / numpy scalar
                if hasattr(x, "item"):
                    return float(x.item())
            except Exception:
                pass
            try:
                return float(x)
            except Exception:
                return None

        def _is_finite(x):
            try:
                return bool(np.isfinite(x))
            except Exception:
                return False

        # --------- Extraction robuste du mean reward ---------
        mean_rew = None

        # 0) Meilleur signal si présent (souvent celui qui alimente Train/mean_reward)
        for key in ("mean_reward", "Train/mean_reward"):
            if isinstance(locs, dict) and key in locs:
                v = _to_float(locs.get(key))
                if v is not None and _is_finite(v):
                    mean_rew = v
                    break

        # 1) RSL-RL standard: rewbuffer (liste/deque des returns)
        if mean_rew is None and isinstance(locs, dict) and "rewbuffer" in locs:
            rewbuffer = locs.get("rewbuffer")
            values = []
            if rewbuffer is not None:
                try:
                    for x in list(rewbuffer):
                        fx = _to_float(x)
                        if fx is not None and _is_finite(fx):
                            values.append(fx)
                except Exception:
                    values = []
            if values:
                mean_rew = float(np.mean(values))

        # 2) Fallback: ep_infos (formats variés)
        if mean_rew is None and isinstance(locs, dict) and "ep_infos" in locs:
            ep_infos = locs.get("ep_infos") or []
            values = []
            try:
                for info in ep_infos:
                    if not isinstance(info, dict):
                        continue
                    if "return" in info:
                        fx = _to_float(info.get("return"))
                    elif "episode" in info and isinstance(info.get("episode"), dict) and "r" in info["episode"]:
                        fx = _to_float(info["episode"].get("r"))
                    elif "r" in info:
                        fx = _to_float(info.get("r"))
                    else:
                        fx = None

                    if fx is not None and _is_finite(fx):
                        values.append(fx)
            except Exception:
                values = []
            if values:
                mean_rew = float(np.mean(values))

        # Si on n’a aucun signal: on sort (et on warn une fois)
        if mean_rew is None:
            if not getattr(self, "_warned_no_reward", False):
                print(
                    "[AUTO-STOP] WARNING: impossible d'extraire un signal reward depuis `locs` "
                    "(pas de mean_reward/rewbuffer/ep_infos exploitable)."
                )
                self._warned_no_reward = True
            return

        # --------- Fenêtre glissante ---------
        self.reward_mean_history.append(float(mean_rew))
        if len(self.reward_mean_history) > self.history_len:
            self.reward_mean_history.pop(0)

        # Pas assez d’historique / pas assez de logs
        if (
            self.history_len < 10
            or len(self.reward_mean_history) < self.history_len
            or self.log_counter < self.min_logs_before_stop
        ):
            return

        # --------- Check plateau (robuste) ---------
        window = np.asarray(self.reward_mean_history, dtype=np.float32)

        reward_mean = float(window.mean())
        reward_var = float(window.var())

        if self.auto_stop_debug:
            print(
                f"[AUTO-STOP CHECK] MeanReward={reward_mean:.6f} | Var={reward_var:.8f} (th={self.var_threshold})"
            )

        # Stop condition (variance-only)
        should_stop = reward_var < self.var_threshold

        if should_stop:
            print(
                "\n[AUTO-STOP] Condition variance satisfaite → arrêt training\n"
                f"[AUTO-STOP] log_counter={self.log_counter} | window={len(window)} | "
                f"Var={reward_var:.8f} < th={self.var_threshold}"
            )
            # Save in log dir
            save_path = os.path.join(self.log_dir, "model_converged.pt")
            self.save(save_path)
            print(f"[AUTO-STOP] Modèle sauvegardé: {save_path}")

            # Also save in user requested directory (if provided)
            if self.save_dir:
                try:
                    os.makedirs(self.save_dir, exist_ok=True)
                    save_path_user = os.path.join(self.save_dir, "model_converged.pt")
                    self.save(save_path_user)
                    print(f"[AUTO-STOP] Modèle aussi sauvegardé: {save_path_user}")
                except Exception as e:
                    print(f"[AUTO-STOP] WARNING: échec sauvegarde dans save_dir: {e}")
            raise KeyboardInterrupt
# <--- Fin de la classe CustomOnPolicyRunner


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    
    # <--- Ajout: Force un grand nombre d'itérations si non spécifié pour permettre la convergence auto
    if args_cli.max_iterations is None:
        agent_cfg.max_iterations = 1000000

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        # <--- Modification: Utiliser CustomOnPolicyRunner
        auto_stop_enabled = True
        if getattr(args_cli, "disable_auto_stop", False):
            auto_stop_enabled = False
        # --auto_stop can explicitly re-enable (useful for scripts that always pass disable)
        if getattr(args_cli, "auto_stop", False):
            auto_stop_enabled = True
        runner = CustomOnPolicyRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=log_dir,
            device=agent_cfg.device,
            save_dir=args_cli.save_dir,
            auto_stop_enabled=auto_stop_enabled,
            auto_stop_window=getattr(args_cli, "auto_stop_window", 50),
            auto_stop_min_logs=getattr(args_cli, "auto_stop_min_logs", 50),
            auto_stop_var=getattr(args_cli, "auto_stop_var", 1e-5),
            auto_stop_std=getattr(args_cli, "auto_stop_std", 2.0),
            auto_stop_span=getattr(args_cli, "auto_stop_span", 5.0),
            auto_stop_slope=getattr(args_cli, "auto_stop_slope", 0.005),
            auto_stop_delta_mean=getattr(args_cli, "auto_stop_delta_mean", 1.0),
            auto_stop_debug=getattr(args_cli, "auto_stop_debug", False),
        )
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    except KeyboardInterrupt:
        print("[INFO] Training interrupted (KeyboardInterrupt or Convergence). Closing...")

    # Always try to save a copy of the final model in the user-defined directory
    try:
        if args_cli.save_dir:
            os.makedirs(args_cli.save_dir, exist_ok=True)
            run_tag = os.path.basename(log_dir.rstrip(os.sep))
            save_path_user = os.path.join(args_cli.save_dir, f"model_{run_tag}.pt")
            runner.save(save_path_user)
            print(f"[INFO] Model saved to: {save_path_user}")
    except Exception as e:
        print(f"[WARNING] Failed to save model to save_dir: {e}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
