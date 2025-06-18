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
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg_PLAY




def main():
    """Main function."""
    # load the trained jit policy
    # policy_path = os.path.abspath(args_cli.checkpoint)
    # runner
    policy_path1 = "/home/robotics/IsaacLab/logs/rsl_rl/g1_flat/run_for_scenario/exported/policy.pt"
    file_content1 = omni.client.read_file(policy_path1)[2]
    file1 = io.BytesIO(memoryview(file_content1).tobytes())
    policy1 = torch.jit.load(file1)
    # stop
    policy_path2 = "/home/robotics/IsaacLab/logs/rsl_rl/g1_flat/2025-06-17_20-31-34/exported/policy.pt"
    file_content2 = omni.client.read_file(policy_path2)[2]
    file2 = io.BytesIO(memoryview(file_content2).tobytes())
    policy2 = torch.jit.load(file2)
    # pickup
    policy_path2 = "/home/robotics/IsaacLab/logs/rsl_rl/g1_flat/2025-06-17_20-31-34/exported/policy.pt"
    file_content2 = omni.client.read_file(policy_path2)[2]
    file2 = io.BytesIO(memoryview(file_content2).tobytes())
    policy2 = torch.jit.load(file2)
    # env
    env_cfg = G1FlatEnvCfg_PLAY()
    env_cfg.scene.num_envs = 1
    env_cfg.curriculum = None
    # env_cfg.scene.terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="usd",
    #     usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
    # )
    env_cfg.sim.device = "cpu"

    flag=False
    def print_cb():
        print("pressed")
        nonlocal flag
        flag = not flag
    # env_cfg.sim.use_fabric = False
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.keyboard.add_callback("A", print_cb)
    # env.keyboard.add_callback("a", print_cb))
    obs, _ = env.reset()
    while simulation_app.is_running():
        # print(env.keyboard.is_pressed("a"))
        if flag:
            action = policy2(obs["policy"])
        elif obs["policy"][:, :-3]>1:

        else:
            action = policy1(obs["policy"][:, :-3])
        # run inference
        obs, _, _, _, _ = env.step(action)


if __name__ == "__main__":
    main()
    simulation_app.close()
