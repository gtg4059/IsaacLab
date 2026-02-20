# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlDistillationStudentTeacherCfg,
    RslRlDistillationAlgorithmCfg,
)


@configclass
class G1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 500
    experiment_name = "g1_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002, #0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1FlatPPORunnerCfg(G1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.clip_actions = 50
        self.max_iterations = 25000
        self.experiment_name = "g1_flat"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]        

############################################################

# @configclass
# class G1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
#     num_steps_per_env = 8
#     max_iterations = 3000
#     save_interval = 100
#     experiment_name = "g1_rough"
#     empirical_normalization = False
#     clip_actions = 50.0
#     max_iterations = 9000
#     # policy = RslRlPpoActorCriticCfg(
#     #     init_noise_std=1.0,
#     #     actor_hidden_dims=[512, 256, 128],
#     #     critic_hidden_dims=[512, 256, 128],
#     #     activation="elu",
#     # )
#     policy = RslRlPpoActorCriticRecurrentCfg(
#         init_noise_std=0.8,
#         actor_hidden_dims=[32],
#         critic_hidden_dims=[32],
#         activation="elu",
#         rnn_hidden_dim=64,
#         rnn_num_layers=1,
#         rnn_type="lstm",
#     )
#     algorithm = RslRlPpoAlgorithmCfg(
#         value_loss_coef=1.0,
#         use_clipped_value_loss=True,
#         clip_param=0.2,
#         entropy_coef=0.008,
#         num_learning_epochs=5,
#         num_mini_batches=4,
#         learning_rate=1.0e-3,
#         schedule="adaptive",
#         gamma=0.99,
#         lam=0.95,
#         desired_kl=0.01,
#         max_grad_norm=1.0,
#     )


# @configclass
# class G1FlatPPORunnerCfg(G1RoughPPORunnerCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         # self.clip_actions = 50.0
#         self.max_iterations = 25000
#         self.experiment_name = "g1_flat"
#         # self.policy.actor_hidden_dims = [512, 256, 128]
#         # self.policy.critic_hidden_dims = [512, 256, 128]
###############################################################3
# # Teacher-Student Distillation:
# # - Teacher: 기존에 학습한 .pt (load_run + load_checkpoint으로 로드)
# # - Student: 다른 obs 개수/종류, 다른 reward로 새로 학습 → 저장되는 .pt는 student
# # - Obs/Reward 차이는 환경 설정(env_cfg)에서 policy observation 그룹과 rewards 항목으로 조정
# @configclass
# class G1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
#     num_steps_per_env = 24
#     max_iterations = 3000
#     save_interval = 500
#     experiment_name = "g1_rough"
#     empirical_normalization = False
#     # Teacher .pt 로드: 학습 시 --load_run <teacher_run폴더> --checkpoint <model_xxx.pt> 지정
#     policy = RslRlDistillationStudentTeacherCfg(
#         init_noise_std=1.0,
#         student_hidden_dims=[512, 256, 128],
#         teacher_hidden_dims=[512, 256, 128],
#         num_obs_teacher=278,  # teacher 체크포인트가 278차원 obs로 학습됐을 때 설정 (현재 env 269와 다르면 필수)
#         activation="elu",
#     )
#     # Distillation 전용 알고리즘 (PPO와 필드가 다름: gradient_length 등)
#     algorithm = RslRlDistillationAlgorithmCfg(
#         num_learning_epochs=5,
#         learning_rate=1.0e-3,
#         gradient_length=24,  # 환경 step 기준 gradient 흐름 길이 (num_steps_per_env와 맞추거나 조정)
#         max_grad_norm=1.0,
#     )


# @configclass
# class G1FlatPPORunnerCfg(G1RoughPPORunnerCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         self.clip_actions = 50
#         self.max_iterations = 100000
#         self.experiment_name = "g1_flat"
#         self.policy.teacher_hidden_dims = [512, 256, 128]
#         self.policy.student_hidden_dims = [512, 256, 128]