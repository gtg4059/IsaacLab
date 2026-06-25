# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from isaaclab_tasks.manager_based.pickandplace import agents

##
# Register Gym environments.
#
# Pipeline:
#   1) Train lift + place-down – Isaac-Pickandplace-Cube-Franka-v0
#   2) Play                     – auto gripper open at floor target
##

gym.register(
    id="Isaac-Pickandplace-Cube-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaCubePickandplaceLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickandplaceLiftPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Pickandplace-Cube-Franka-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaCubePickandplaceLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickandplaceLiftPPORunnerCfg",
    },
)
