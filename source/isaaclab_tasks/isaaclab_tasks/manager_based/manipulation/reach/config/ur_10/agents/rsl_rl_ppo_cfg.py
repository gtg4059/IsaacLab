# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class UR10ReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48
    max_iterations = 10000000
    save_interval = 500
    experiment_name = "reach_ur10"
    empirical_normalization = True

    # None keeps algorithm.learning_rate and algorithm.schedule for the whole run.
    # Set an iteration to switch once, at that iteration and after, to the values below.
    # These stay on the runner config. rsl_rl passes ``algorithm`` into PPO, so a field
    # placed there is rejected. P2P and CRI-F inherit this; set ``lr_switch_iter = None``
    # on those classes when this one is enabled.
    lr_switch_iter: int | None = None
    lr_switch_learning_rate: float = 1.0e-5
    lr_switch_schedule: str = "fixed"
    
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_hidden_dims=[1024, 512, 256, 128],
        critic_hidden_dims=[1024, 512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=8,
        num_mini_batches=1,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.995,
        lam=0.97,
        desired_kl=0.016,
        max_grad_norm=1.0,
    )


@configclass
class UR10ReachP2PPPORunnerCfg(UR10ReachPPORunnerCfg):
    experiment_name = "reach_ur10_p2p"


@configclass
class UR10ReachCRIFPPORunnerCfg(UR10ReachPPORunnerCfg):
    experiment_name = "reach_ur10_cri_f"
