# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # Longer rollouts reduce gradient variance and stabilize training.
    num_steps_per_env = 32
    max_iterations = 150
    save_interval = 50
    experiment_name = "cartpole_direct"
    policy = RslRlPpoActorCriticCfg(
        # If noise grows too much during training, start lower.
        init_noise_std=0.25,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[32, 32],
        critic_hidden_dims=[32, 32],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        # Smaller clip makes PPO updates more conservative/stable.
        clip_param=0.1,
        # Entropy bonus encourages exploration. If you see mean_noise_std/entropy trending up,
        # it usually means this term dominates and the policy is getting more stochastic.
        # Lower it to favor stability/convergence.
        entropy_coef=1.0e-4,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )