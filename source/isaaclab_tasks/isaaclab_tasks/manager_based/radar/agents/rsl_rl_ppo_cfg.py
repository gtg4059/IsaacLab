# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlRNNModelCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48
    max_iterations = 9000000
    save_interval = 100
    clip_actions = 50.0
    empirical_normalization = False
    experiment_name = "g1_radar_oracle"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlRNNModelCfg(
        hidden_dims=[64],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.8),
        rnn_hidden_dim=128,
        rnn_num_layers=1,
        rnn_type="lstm",
    )
    critic = RslRlRNNModelCfg(
        hidden_dims=[64],
        activation="elu",
        obs_normalization=False,
        rnn_hidden_dim=128,
        rnn_num_layers=1,
        rnn_type="lstm",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )