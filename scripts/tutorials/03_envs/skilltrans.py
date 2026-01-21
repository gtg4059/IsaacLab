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
import tensorrt as trt
from pathlib import Path
import pycuda.driver as cuda
import pycuda.autoinit  # 또는 직접 context 관리
import numpy as np

# keyboard_cfg = Se2KeyboardCfg(
#     v_x_sensitivity=0.8,
#     v_y_sensitivity=0.4,
#     omega_z_sensitivity=1.0,
# )



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

def build_engine_from_onnx(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    workspace_size_bytes: int = 1 << 30,  # 1GB
):
    logger = trt.Logger(trt.Logger.WARNING)
    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)

    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")

    # EXPLICIT_BATCH 네트워크 생성
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(logger) as builder, \
         builder.create_network(flags=flags) as network, \
         trt.OnnxParser(network, logger) as parser:

        # 1) ONNX 파싱
        with open(onnx_path, "rb") as f:
            model_bytes = f.read()
        if not parser.parse(model_bytes):
            print("[TensorRT] ONNX parsing failed")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

        # 2) BuilderConfig 생성
        config = builder.create_builder_config()
        # 워크스페이스 메모리 한도 설정 (TRT10 방식)
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, workspace_size_bytes
        )

        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        # (동적 shape가 있으면 여기서 optimization profile 추가)

        # 3) build_serialized_network 사용 (TRT10)
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build serialized TensorRT engine")

        # 4) 바로 파일로 저장
        engine_path.write_bytes(bytes(serialized_engine))
        print(f"[TensorRT] Saved engine to: {engine_path}")

class TrtPolicyRunner:
    def __init__(self, engine_path: str, num_layers: int, hidden_size: int):
        self.logger = trt.Logger(trt.Logger.WARNING)

        # 1) 엔진 로드
        with open(engine_path, "rb") as f:
            engine_bytes = f.read()
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")

        # 2) 실행 컨텍스트
        self.context = self.engine.create_execution_context()

        # 3) I/O 텐서 이름 모으기
        self.input_names = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        # 기대: ['obs','h_in','c_in'], ['actions','h_out','c_out']
        required_inputs = {"obs", "h_in", "c_in"}
        required_outputs = {"actions", "h_out", "c_out"}
        if set(self.input_names) != required_inputs or set(self.output_names) != required_outputs:
            raise RuntimeError(
                f"Unexpected IO tensors: inputs={self.input_names}, outputs={self.output_names}"
            )

        # 이름을 고정 순서로 보관
        self.obs_name = "obs"
        self.h_in_name = "h_in"
        self.c_in_name = "c_in"
        self.actions_name = "actions"
        self.h_out_name = "h_out"
        self.c_out_name = "c_out"

        # 4) 입력 shape 설정
        #    obs: (1, 96) 고정, h/c: (num_layers, 1, hidden_size) 고정이라고 가정
        self.obs_shape = (1, 96)
        self.h_shape = (num_layers, 1, hidden_size)
        self.c_shape = (num_layers, 1, hidden_size)

        self.context.set_input_shape(self.obs_name, self.obs_shape)
        self.context.set_input_shape(self.h_in_name, self.h_shape)
        self.context.set_input_shape(self.c_in_name, self.c_shape)

        # 실제 런타임 shape 확인
        self.obs_shape = tuple(self.context.get_tensor_shape(self.obs_name))
        self.h_shape = tuple(self.context.get_tensor_shape(self.h_in_name))
        self.c_shape = tuple(self.context.get_tensor_shape(self.c_in_name))

        self.actions_shape = tuple(self.context.get_tensor_shape(self.actions_name))
        self.h_out_shape = tuple(self.context.get_tensor_shape(self.h_out_name))
        self.c_out_shape = tuple(self.context.get_tensor_shape(self.c_out_name))

        # 5) 크기 계산
        self.obs_size = int(np.prod(self.obs_shape))
        self.h_size = int(np.prod(self.h_shape))
        self.c_size = int(np.prod(self.c_shape))
        self.actions_size = int(np.prod(self.actions_shape))
        self.h_out_size = int(np.prod(self.h_out_shape))
        self.c_out_size = int(np.prod(self.c_out_shape))

        # 6) GPU 메모리 할당
        self.d_obs = cuda.mem_alloc(self.obs_size * np.float32().nbytes)
        self.d_h_in = cuda.mem_alloc(self.h_size * np.float32().nbytes)
        self.d_c_in = cuda.mem_alloc(self.c_size * np.float32().nbytes)

        self.d_actions = cuda.mem_alloc(self.actions_size * np.float32().nbytes)
        self.d_h_out = cuda.mem_alloc(self.h_out_size * np.float32().nbytes)
        self.d_c_out = cuda.mem_alloc(self.c_out_size * np.float32().nbytes)

        # 7) 호스트 버퍼
        self.h_actions = np.empty(self.actions_shape, dtype=np.float32)
        self.h_h = np.zeros(self.h_shape, dtype=np.float32)
        self.h_c = np.zeros(self.c_shape, dtype=np.float32)

        # 8) 텐서 주소 바인딩 (TensorRT 10: 이름 기준)
        self.context.set_tensor_address(self.obs_name, int(self.d_obs))
        self.context.set_tensor_address(self.h_in_name, int(self.d_h_in))
        self.context.set_tensor_address(self.c_in_name, int(self.d_c_in))

        self.context.set_tensor_address(self.actions_name, int(self.d_actions))
        self.context.set_tensor_address(self.h_out_name, int(self.d_h_out))
        self.context.set_tensor_address(self.c_out_name, int(self.d_c_out))

        # CUDA 스트림
        self.stream = cuda.Stream()

    def reset(self):
        """에피소드 시작 시 hidden/cell state 리셋."""
        self.h_h.fill(0.0)
        self.h_c.fill(0.0)

    def infer(self, obs_np: np.ndarray) -> np.ndarray:
        """
        obs_np: shape (96,) 또는 (1,96) float32.
        내부적으로 h_h, h_c를 유지하면서 매 스텝 업데이트.
        """
        # 1) obs shape 맞추기
        if obs_np.ndim == 1:
            obs_np = obs_np.reshape(self.obs_shape)
        elif obs_np.shape != self.obs_shape:
            raise ValueError(f"Expected obs shape {self.obs_shape}, got {obs_np.shape}")
        obs_np = np.ascontiguousarray(obs_np, dtype=np.float32)

        # 2) Host -> Device 복사 (obs, h_in, c_in)
        cuda.memcpy_htod_async(self.d_obs, obs_np, self.stream)
        cuda.memcpy_htod_async(self.d_h_in, self.h_h, self.stream)
        cuda.memcpy_htod_async(self.d_c_in, self.h_c, self.stream)

        # 3) 실행
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # 4) Device -> Host 복사 (actions, h_out, c_out)
        cuda.memcpy_dtoh_async(self.h_actions, self.d_actions, self.stream)
        cuda.memcpy_dtoh_async(self.h_h, self.d_h_out, self.stream)
        cuda.memcpy_dtoh_async(self.h_c, self.d_c_out, self.stream)

        self.stream.synchronize()

        # actions만 리턴 (1D)
        return self.h_actions.squeeze()



robot_data = []

def main():
    """Main function."""

    onnx_path = "/home/safetics/IsaacLab/logs/rsl_rl/ptcontainer/policy.onnx"
    engine_path = "/home/safetics/IsaacLab/logs/rsl_rl/ptcontainer/policy_fp16.engine"
    build_engine_from_onnx(onnx_path, engine_path, fp16=True)

    # # runner
    # policy_path1 = "./logs/rsl_rl/ptcontainer/policy.pt"
    # file_content1 = omni.client.read_file(policy_path1)[2]
    # file1 = io.BytesIO(memoryview(file_content1).tobytes())
    # policy_run = torch.jit.load(file1)

    engine_path = "/home/safetics/IsaacLab/logs/rsl_rl/ptcontainer/policy_fp16.engine"
    runner = TrtPolicyRunner(engine_path, num_layers=1, hidden_size=32)
    # runner = TrtPolicyRunner(engine_path)

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
        # if k > 50 and k <= 220:
        #     command[0] = 4.0
        # else:
        #     command[0] = 0.0
        # # sim test
        # if k > 20 and k <= 120:
        #     command[0] = 2.0
        # elif k > 120 and k <= 240:
        #     command[0] = 4.0
        # elif k > 240 and k <= 340:
        #     command[0] = 2.0
        # else:
        #     command[0] = 0.0
        # action = policy_run(obs["Run"])
        action = runner.infer(np.array(obs["Run"]))
        # action = policy_run(torch.cat((obs["Run"][:,:-3],command.unsqueeze(0)),dim=1))
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

        obs, _, _, _, _ = env.step(torch.tensor(action).unsqueeze_(0))

        # target_dof_pos = action * 0.25
        # # 데이터 수집 (매 스텝마다)
        # data_row = {}
        # # 액션과 목표 위치 추가
        # for i in range(len(action[0])):
            
        #     data_row[f'action_{i}'] = float(action[0,i])
        #     data_row[f'target_dof_pos_{i}'] = float(target_dof_pos[0,i])
        #     data_row[f'qj{i}'] = float(obs["Run"][0,6+i])
        #     data_row[f'dqj{i}'] = float(obs["Run"][0,35+i])
        # # 힘의 크기(norm)가 0보다 큰지 확인하여 접촉 여부 판단
        # left_force_magnitude = torch.norm(ankle_roll_forces[0, 0, :]).item()
        # right_force_magnitude = torch.norm(ankle_roll_forces[0, 1, :]).item()
        # data_row[f'left'] = int(left_force_magnitude > 0 and right_force_magnitude <= 0)
        # data_row[f'right'] = int(right_force_magnitude > 0 and left_force_magnitude <= 0)
        # data_row[f'double'] = int(left_force_magnitude > 0 and right_force_magnitude > 0)
        # # 로봇의 전방 속도 (world 좌표계 x 방향 속도)
        # robot = env.scene["robot"]
        # forward_velocity = robot.data.root_lin_vel_w[0, 0].item()  # x 방향 속도
        # data_row[f'command'] = command[0].item()/2
        # data_row[f'forward_velocity'] = forward_velocity
        # # # obs 위치 추가
        # # for i in range(len(self.obs)):
        # #     data_row[f'obs_{i}'] = float(self.obs[i])
        
        # robot_data.append(data_row)
        # # print(data_row)

        # k += 1
        # if k >= 300:
        #     break


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
    
