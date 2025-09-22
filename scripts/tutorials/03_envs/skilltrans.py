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
from isaaclab.devices.gamepad import Se2Gamepad, Se2GamepadCfg
import carb.input

import pandas as pd
import datetime
# keyboard_cfg = Se2KeyboardCfg(
#     v_x_sensitivity=0.8,
#     v_y_sensitivity=0.4,
#     omega_z_sensitivity=1.0,
# )

robot_data = []

def save_data_to_csv(filename=None):
    """
    수집된 로봇 데이터를 CSV 파일로 저장
    """
    if not robot_data:
        print("No data to save.")
        return
    
    if filename is None:
        # 현재 시간을 포함한 파일명 생성
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"robot_data_{timestamp}.csv"
    
    # DataFrame 생성
    df = pd.DataFrame(robot_data)
    # CSV 파일로 저장
    df.to_csv(filename, index=False)
    
    # action 통계 (처음 5개 관절)
    print("Action (first 5 joints):")
    for j in range(min(5, 23)):
        col = f'action_{j}'
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            min_val = df[col].min()
            max_val = df[col].max()
            print(f"  joint_{j}: mean={mean_val:.6f}, std={std_val:.6f}, range=[{min_val:.6f}, {max_val:.6f}]")
    
    return filename

def main():
    """Main function."""
    # load the trained jit policy
    # policy_path = os.path.abspath(args_cli.checkpoint)
    # runner
    policy_path1 = "./scripts/tutorials/03_envs/policy_run27.pt"
    file_content1 = omni.client.read_file(policy_path1)[2]
    file1 = io.BytesIO(memoryview(file_content1).tobytes())
    policy_run = torch.jit.load(file1)
    # stop
    policy_path2 = "./scripts/tutorials/03_envs/policy_stop27.pt"
    file_content2 = omni.client.read_file(policy_path2)[2]
    file2 = io.BytesIO(memoryview(file_content2).tobytes())
    policy_stop = torch.jit.load(file2)
    # pickup
    policy_path3 = "./scripts/tutorials/03_envs/policy_pickup.pt"
    file_content3 = omni.client.read_file(policy_path3)[2]
    file3 = io.BytesIO(memoryview(file_content3).tobytes())
    policy_pickup = torch.jit.load(file3)
    # pick_walk
    policy_path4 = "./scripts/tutorials/03_envs/policy_pick_walk.pt"
    file_content4 = omni.client.read_file(policy_path4)[2]
    file4 = io.BytesIO(memoryview(file_content4).tobytes())
    policy_pick_walk = torch.jit.load(file4)
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

    # flag=False
    # def print_cb():
    #     print("pressed")
    #     nonlocal flag
    #     flag = not flag
    # env_cfg.sim.use_fabric = False
    
    # commands = keyboard.advance()
    
    # env.keyboard.add_callback("a", print_cb))
    env = ManagerBasedRLEnv(cfg=env_cfg)
    # command = env_cfg.keyboard.advance()
    # env_cfg.gamepad.add_callback(carb.input.GamepadInput.X, print_cb)
    obs, _ = env.reset()
    
    k = 0
    while simulation_app.is_running():
        command = obs["policy"][:, -3:]
        # action = policy_run(obs["Run"])
        # # print(env.keyboard.is_pressed("a"))
        # # print(obs["policy"][:, 93:96])
        # print("commands",env.gamepad.advance()[3],env.gamepad.advance()[5])
        # X, Y, A, B button: 3,4,5,6
        # print(env.gamepad.advance()[3],env.gamepad.advance()[5],env.gamepad.advance()[6])
        if env.gamepad.advance()[3]>0.0 and env.gamepad.advance()[5]>0 and env.gamepad.advance()[6]>0.0:# (0, 0, 0)
            # if abs(env.gamepad.advance()[0])<0.1 and abs(env.gamepad.advance()[1])<0.1:
            action = policy_stop(torch.cat((obs["policy"][:,:-3],command*0),dim=1))
            # action = policy_run(obs["Run"])
        elif env.gamepad.advance()[3]<0 and env.gamepad.advance()[5]>0.0 and env.gamepad.advance()[6]>0.0:# (1, 0, 0)
            action = policy_run(obs["policy"])
        # elif env.gamepad.advance()[3]>0.0 and env.gamepad.advance()[5]<0 and env.gamepad.advance()[6]>0.0: # (0, 1, 0) up
        #     print(obs["Pickup"][:,60:87])
        #     action = policy_pickup(obs["Pickup"])
        # elif env.gamepad.advance()[3]>0.0 and env.gamepad.advance()[5]<0.0 and env.gamepad.advance()[6]<0.0: # (0, 1, 1) down
        #     x = obs["Pickup"].clone()
        #     x[:,89] = 0.06
        #     x[:,96] = 0.06
        #     # print("down")
        #     # print(obs["Pickup"][:,89],obs["Pickup"][:,96])
        #     # obs["Pickup"][:,89] = 0.06
        #     # obs["Pickup"][:,:96] = 0.06
        #     action = policy_pickup(x)
        # elif env.gamepad.advance()[3]<0.0 and env.gamepad.advance()[5]<0.0 and env.gamepad.advance()[6]>0.0: # (1, 1, 0)
        #     # print(obs["PickWalk"][:,60:87])
        #     x = obs["PickWalk"].clone()
        #     x[:,92] = 0.06
        #     x[:,99] = 0.06
        #     action = policy_pick_walk(x)

        # data_row = {}
        # # for i in range(len(obs["Run"][0])):
        # #     data_row[f'obs_{i}'] = float(obs["Run"][0,i])
        # for i in range(len(action[0])):
        #     data_row[f'action_{i}'] = float(action[0,i])
        # robot_data.append(data_row)

        obs, _, _, _, _ = env.step(action)

        # k += 1
        # if k >= 250:
        #     break


if __name__ == "__main__":
    main()
    # print("Saving robot data...")
    # # print(robot_data)
    # #  print(robot_data[0,2])
    # csv_filename = save_data_to_csv()
    # print(f"Data saved to: {csv_filename}")
    # simulation_app.update()
    # simulation_app.update()
    # simulation_app.update()
    simulation_app.close()
    
