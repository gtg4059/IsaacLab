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
from isaaclab.sensors import ContactSensor

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
    policy_path1 = "./logs/rsl_rl/ptcontainer/policy.pt"
    file_content1 = omni.client.read_file(policy_path1)[2]
    file1 = io.BytesIO(memoryview(file_content1).tobytes())
    policy_run = torch.jit.load(file1)
    # # stop
    # policy_path2 = "./logs/rsl_rl/ptcontainer/policy_stop.pt"
    # file_content2 = omni.client.read_file(policy_path2)[2]
    # file2 = io.BytesIO(memoryview(file_content2).tobytes())
    # policy_stop = torch.jit.load(file2)
    # # pickup
    # policy_path3 = "./logs/rsl_rl/ptcontainer/policy_pickup.pt"
    # file_content3 = omni.client.read_file(policy_path3)[2]
    # file3 = io.BytesIO(memoryview(file_content3).tobytes())
    # policy_pickup = torch.jit.load(file3)
    # # pick_walk
    # policy_path4 = "./logs/rsl_rl/ptcontainer/policy_pick_walk.pt"
    # file_content4 = omni.client.read_file(policy_path4)[2]
    # file4 = io.BytesIO(memoryview(file_content4).tobytes())
    # policy_pick_walk = torch.jit.load(file4)
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
    command = env_cfg.keyboard.advance()
    # env_cfg.gamepad.add_callback(carb.input.GamepadInput.X, print_cb)
    obs, _ = env.reset()
    
        # Contact sensor 접근
    contact_sensor = env.scene.sensors["contact_forces"]
    # 타입 체크: ContactSensor인지 확인
    assert isinstance(contact_sensor, ContactSensor), "contact_forces sensor must be a ContactSensor"
    # ankle_roll_link의 body ID 찾기
    ankle_roll_body_ids, ankle_roll_body_names = contact_sensor.find_bodies(".*_ankle_roll_link")
    print(f"Found ankle_roll_link bodies: {ankle_roll_body_names} with IDs: {ankle_roll_body_ids}")
    
    k = 0
    while simulation_app.is_running():

        # walk command
        # if k > 50 and k <= 170:
        #     command[0] = 2.0
        # else:
        #     command[0] = 0.0
        # run command
        if k > 50 and k <= 220:
            command[0] = 4.0
        else:
            command[0] = 0.0
        # # sim test
        # if k > 20 and k <= 120:
        #     command[0] = 2.0
        # elif k > 120 and k <= 240:
        #     command[0] = 4.0
        # elif k > 240 and k <= 340:
        #     command[0] = 2.0
        # else:
        #     command[0] = 0.0
        action = policy_run(torch.cat((obs["Run"][:,:-3],command.unsqueeze(0)),dim=1))
        # ankle_roll_link의 contact sensor 데이터 가져오기
        # net_forces_w: (num_envs, num_bodies, 3) - 현재 접촉 힘
        contact_data = contact_sensor.data
        if contact_data.net_forces_w is not None:
            ankle_roll_forces = contact_data.net_forces_w[:, ankle_roll_body_ids, :]
        # net_forces_w_history: (num_envs, history_length, num_bodies, 3) - 히스토리
        if contact_data.net_forces_w_history is not None:
            ankle_roll_forces_history = contact_data.net_forces_w_history[:, :, ankle_roll_body_ids, :]
        # current_contact_time: (num_envs, num_bodies) - 현재 접촉 시간
        if contact_data.current_contact_time is not None:
            ankle_roll_contact_time = contact_data.current_contact_time[:, ankle_roll_body_ids]
        # current_air_time: (num_envs, num_bodies) - 현재 공중에 있는 시간
        if contact_data.current_air_time is not None:
            ankle_roll_air_time = contact_data.current_air_time[:, ankle_roll_body_ids]
        
        # # 예시: 첫 번째 환경의 ankle_roll_link contact force 출력
        # if k % 100 == 0:  # 100 스텝마다 출력
        #     print(f"Step {k}:")
        #     if contact_data.net_forces_w is not None:
        #         print(f"  Left ankle roll force: {ankle_roll_forces[0, 0, :]}")
        #         print(f"  Right ankle roll force: {ankle_roll_forces[0, 1, :]}")
        #     if contact_data.current_contact_time is not None:
        #         print(f"  Contact time: {ankle_roll_contact_time[0, :]}")
        #     if contact_data.current_air_time is not None:
        #         print(f"  Air time: {ankle_roll_air_time[0, :]}")

        # print(obs["Run"][0,6:35])

        # action = policy_run(obs["Run"])
        # # print(env.keyboard.is_pressed("a"))
        # # print(obs["policy"][:, 93:96])
        # print("commands",env.gamepad.advance()[3],env.gamepad.advance()[5])
        # X, Y, A, B button: 3,4,5,6
        # print(env.gamepad.advance()[3],env.gamepad.advance()[5],env.gamepad.advance()[6])
        # if env.gamepad.advance()[3]>0.0 and env.gamepad.advance()[5]>0 and env.gamepad.advance()[6]>0.0:# (0, 0, 0)
        #     # if abs(env.gamepad.advance()[0])<0.1 and abs(env.gamepad.advance()[1])<0.1:
        #     # action = policy_stop(torch.cat((obs["Run"][:,:-3],command*0),dim=1))
        #     action = policy_stop(obs["Run"])
        # elif env.gamepad.advance()[3]<0 and env.gamepad.advance()[5]>0.0 and env.gamepad.advance()[6]>0.0:# (1, 0, 0)
        #     action = policy_run(obs["Run"])
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

        target_dof_pos = action * 0.25
        # 데이터 수집 (매 스텝마다)
        data_row = {}
        # 액션과 목표 위치 추가
        for i in range(len(action[0])):
            
            data_row[f'action_{i}'] = float(action[0,i])
            data_row[f'target_dof_pos_{i}'] = float(target_dof_pos[0,i])
            data_row[f'qj{i}'] = float(obs["Run"][0,6+i])
            data_row[f'dqj{i}'] = float(obs["Run"][0,35+i])
        # 힘의 크기(norm)가 0보다 큰지 확인하여 접촉 여부 판단
        left_force_magnitude = torch.norm(ankle_roll_forces[0, 0, :]).item()
        right_force_magnitude = torch.norm(ankle_roll_forces[0, 1, :]).item()
        data_row[f'left'] = int(left_force_magnitude > 0 and right_force_magnitude <= 0)
        data_row[f'right'] = int(right_force_magnitude > 0 and left_force_magnitude <= 0)
        data_row[f'double'] = int(left_force_magnitude > 0 and right_force_magnitude > 0)
        # 로봇의 전방 속도 (world 좌표계 x 방향 속도)
        robot = env.scene["robot"]
        forward_velocity = robot.data.root_lin_vel_w[0, 0].item()  # x 방향 속도
        data_row[f'command'] = command[0].item()/2
        data_row[f'forward_velocity'] = forward_velocity
        # # obs 위치 추가
        # for i in range(len(self.obs)):
        #     data_row[f'obs_{i}'] = float(self.obs[i])
        
        robot_data.append(data_row)
        # print(data_row)

        k += 1
        if k >= 300:
            break


if __name__ == "__main__":
    main()
    print("Saving robot data...")
    # # print(robot_data)
    # #  print(robot_data[0,2])
    csv_filename = save_data_to_csv()
    print(f"Data saved to: {csv_filename}")
    simulation_app.update()
    simulation_app.update()
    simulation_app.update()
    simulation_app.close()
    
