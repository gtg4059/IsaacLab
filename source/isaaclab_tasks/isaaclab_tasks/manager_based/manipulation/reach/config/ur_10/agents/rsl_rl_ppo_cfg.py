# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticRecurrentCfg, RslRlPpoAlgorithmCfg


@configclass
class UR10ReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 100000
    save_interval = 100
    experiment_name = "reach_ur10"
    run_name = ""
    # resume = True
    empirical_normalization = False
    clip_actions = 10.0
    max_iterations = 9000000
    # LSTM + MLP 아키텍처 사용
    # notion의 "[정리] 휴머노이드 걷기 학습" 참조
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=0.8,
        actor_hidden_dims=[32],
        critic_hidden_dims=[32],
        activation="elu",
        rnn_hidden_dim=64,
        rnn_num_layers=1,
        rnn_type="lstm",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
