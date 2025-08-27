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
import torch
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.devices import Se2Keyboard
from isaaclab.devices.keyboard.se2_keyboard import Se2KeyboardCfg

keyboard_cfg = Se2KeyboardCfg(
    v_x_sensitivity=0.8,
    v_y_sensitivity=0.4,
    omega_z_sensitivity=1.0,
)

def main():
    """Main function."""
    # load the trained jit policy
    # policy_path = os.path.abspath(args_cli.checkpoint)
    # runner
    policy_path1 = "./scripts/tutorials/03_envs/policy_run.pt"
    file_content1 = omni.client.read_file(policy_path1)[2]
    file1 = io.BytesIO(memoryview(file_content1).tobytes())
    policy_run = torch.jit.load(file1)
    # stop
    policy_path2 = "./scripts/tutorials/03_envs/policy_stop.pt"
    file_content2 = omni.client.read_file(policy_path2)[2]
    file2 = io.BytesIO(memoryview(file_content2).tobytes())
    policy_stop = torch.jit.load(file2)
    # pickup
    policy_path3 = "./scripts/tutorials/03_envs/policy_pickup.pt"
    file_content3 = omni.client.read_file(policy_path3)[2]
    file3 = io.BytesIO(memoryview(file_content3).tobytes())
    policy_pickup = torch.jit.load(file3)
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
    
    # commands = keyboard.advance()
    
    # env.keyboard.add_callback("a", print_cb))
    env = ManagerBasedRLEnv(cfg=env_cfg)
    # command = env_cfg.keyboard.advance()
    env_cfg.keyboard.add_callback("A", print_cb)
    obs, _ = env.reset()
    while simulation_app.is_running():
        command = env_cfg.keyboard.advance()
        print("command",command)
        # action = policy_run(obs["Run"])
        # # print(env.keyboard.is_pressed("a"))
        # # print(obs["policy"][:, 93:96])
        if not flag and torch.norm(command)>0.02: #run
            action = policy_run(obs["Run"])
        elif not flag and torch.norm(command)<=0.1: #stop
            action = policy_stop(obs["Run"])
        elif flag:
            robot = env.scene["robot"]
            joint_indices, joint_names = robot.find_joints(['.*_proximal_joint'])
            joint_idx = robot.set_joint_effort_target(torch.zeros_like(robot.data.default_joint_pos[:,joint_indices]),joint_indices)
            num_envs = env.num_envs
            num_joints = robot.num_joints
            efforts = 0.02*torch.ones((num_envs, num_joints), device=env.device)
            efforts[:, joint_idx] = 0.02
            robot.set_joint_effort_target(efforts)
            robot.write_data_to_sim()
            # action = policy3(obs["policy"])
        # run inference
        obs, _, _, _, _ = env.step(action)


if __name__ == "__main__":
    main()
    simulation_app.close()
