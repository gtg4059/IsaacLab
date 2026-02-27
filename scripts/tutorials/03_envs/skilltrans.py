# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates policy inference in a prebuilt USD environment.

In this example, we use a locomotion policy to control the H1 robot. The robot was trained
using Isaac-Velocity-Rough-H1-v0. The robot is commanded to move forward at a constant velocity.

.. code-block:: bash

    # Run the script
    ./isaaclab.sh -p scripts/tutorials/03_envs/policy_inference_in_usd.py --checkpoint /path/to/jit/checkpoint.pt

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on inferencing a policy on an H1 robot in a warehouse.")
# parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint exported as jit.", required=True)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import io
import os
import torch

import omni

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg_PLAY
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg_PLAY
import torch
CLIP_ACTIONS = 50.0

def main():
    """Main function."""
    # load the trained jit policy
    policy_path = "./logs/rsl_rl/g1_rough/2026-02-23_10-41-29/exported/policy.pt"
    policy_run = torch.jit.load(policy_path, map_location="cpu")
    # env
    env_cfg = G1RoughEnvCfg_PLAY()
    
    env_cfg.scene.num_envs = 1
    env_cfg.curriculum = None
    env_cfg.sim.device = "cpu"

    env = ManagerBasedRLEnv(cfg=env_cfg)
    obs, _ = env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            action = policy_run(obs["Run"])
            action = torch.clamp(action, -CLIP_ACTIONS, CLIP_ACTIONS)
            obs, _, _, _, _ = env.step(action)



if __name__ == "__main__":
    main()
    simulation_app.update()
    simulation_app.close()
    
