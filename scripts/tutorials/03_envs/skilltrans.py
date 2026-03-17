# # Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# """
# This script demonstrates policy inference in a prebuilt USD environment.

# In this example, we use a locomotion policy to control the g1 robot. The robot was trained
# using Isaac-Velocity-Rough-G1-v0. The robot is commanded to move forward at a constant velocity.

# .. code-block:: bash

#     # Run the script
#     ./isaaclab.sh -p scripts/tutorials/03_envs/skilltrans.py --checkpoint /path/to/jit/checkpoint.pt

# """

# """Launch Isaac Sim Simulator first."""


# import argparse

# from isaaclab.app import AppLauncher

# # add argparse arguments
# parser = argparse.ArgumentParser(description="Tutorial on inferencing a policy on an g1 robot in a warehouse.")
# parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint exported as jit.", required=True)

# # append AppLauncher cli args
# AppLauncher.add_app_launcher_args(parser)
# # parse the arguments
# args_cli = parser.parse_args()

# # launch omniverse app
# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app

# """Rest everything follows."""
# import io
# import os
# import torch

# import omni

# from isaaclab.envs import ManagerBasedRLEnv

# from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg_PLAY
# from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg_PLAY
# import torch
# CLIP_ACTIONS = 50.0

# def main():
#     """Main function."""
#     # load the trained jit policy
#     policy_path = "./logs/rsl_rl/g1_flat/2026-02-24_10-29-55/exported/policy.pt"
#     policy_run = torch.jit.load(policy_path, map_location="cpu")
#     # env
#     env_cfg = G1RoughEnvCfg_PLAY()
    
#     env_cfg.scene.num_envs = 1
#     env_cfg.curriculum = None
#     env_cfg.sim.device = "cpu"

#     env = ManagerBasedRLEnv(cfg=env_cfg)
#     obs, _ = env.reset()
#     while simulation_app.is_running():
#         with torch.inference_mode():
#             action = policy_run(obs["policy"])
#             action = torch.clamp(action, -CLIP_ACTIONS, CLIP_ACTIONS)
#             obs, _, _, _, _ = env.step(action)



# if __name__ == "__main__":
#     main()
#     simulation_app.update()
#     simulation_app.close()
    

#########################################################################################
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
보행(속도)만 키보드로 제어하는 G1 인터랙티브 데모. 학습된 .pt 체크포인트를 불러와 사용.
외력은 관측에서 0으로 고정하여 속도 추종만 확인.

Usage:
    ./isaaclab.sh -p scripts/tutorials/03_envs/skilltrans.py --policy_checkpoint /path/to/model_XXXXX.pt

Controls:
    - Click a robot in the viewport to select it
    - UP:    move forward  (+x)
    - DOWN:  move backward (-x)
    - LEFT:  move left     (+y)
    - RIGHT: move right    (-y)
    - C: toggle between third-person and perspective cameras
    - ESC: exit current third-person view (deselect)
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

# 프로젝트 루트(IsaacLab)를 path 맨 앞에 넣어 scripts 패키지 인식
_project_root = os.environ.get("ISAACLAB_PATH")
if not _project_root or not os.path.isdir(_project_root):
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.normpath(os.path.join(_script_dir, "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
# cli_args: scripts 패키지에서 로드, 실패 시 rsl_rl 디렉터리에서 직접 로드
try:
    import scripts.reinforcement_learning.rsl_rl.cli_args as cli_args  # noqa: E402
except ModuleNotFoundError:
    _rsl_dir = os.path.join(_project_root, "scripts", "reinforcement_learning", "rsl_rl")
    if _rsl_dir not in sys.path:
        sys.path.insert(0, _rsl_dir)
    import cli_args  # noqa: E402

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description=(
        "Interactive demo for the Unitree G1 velocity environment using a custom trained RSL-RL checkpoint."
    )
)
# path to a trained checkpoint (from IsaacLab/logs/rsl_rl/.../*.pt)
parser.add_argument(
    "--policy_checkpoint",
    type=str,
    required=True,
    help=(
        "Path to a trained RSL-RL checkpoint (.pt), "
        "e.g. C:/Users/USER/IsaacLab/logs/rsl_rl/g1_flat/.../model_76000.pt"
    ),
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of robot instances in the scene (default: 1).",
)

# append RSL-RL cli arguments (device, etc.); this also defines its own --checkpoint
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import carb
import omni
from isaacsim.core.utils.stage import get_current_stage
from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, Sdf
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg_PLAY

TASK = "Isaac-Velocity-Flat-G1-v0"

# policy 관측에서 velocity_commands에 적용되는 scale (velocity_env_cfg와 동일하게)
VELOCITY_CMD_SCALE = (2.0, 2.0, 0.25)


def _get_policy_obs_term_slice(env, group_name: str, term_name: str):
    """policy observation에서 term_name에 해당하는 concat 구간 (start, end) 반환."""
    om = env.unwrapped.observation_manager
    names = om._group_obs_term_names.get(group_name, [])
    dims = om._group_obs_term_dim.get(group_name, [])
    if term_name not in names:
        return None
    idx = names.index(term_name)
    if idx >= len(dims):
        return None
    # term shape (num_envs, feature_dim) → feature_dim 사용; (num_envs,) → 1
    def _feat_size(d):
        if not isinstance(d, (list, tuple)) or len(d) == 0:
            return 0
        return d[1] if len(d) >= 2 else 1
    start = sum(_feat_size(d) for d in dims[:idx])
    end = start + _feat_size(dims[idx])
    return start, end


class G1RoughDemo:
    """키보드로 속도 명령을 넣어 보행만 확인하는 인터랙티브 데모. 외력은 관측에서 0으로 고정."""

    def __init__(self):
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(TASK, args_cli)
        checkpoint = args_cli.policy_checkpoint

        # Flat env 사용 (보행/속도만 볼 때 적합, 외력 평가 안 함)
        env_cfg = G1FlatEnvCfg_PLAY()
        env_cfg.scene.num_envs = getattr(args_cli, "num_envs", 1)
        env_cfg.episode_length_s = 1_000_000
        env_cfg.curriculum = None
        # 플레이에서는 키보드로만 command를 넣을 것이므로, command resampling을 사실상 비활성화
        env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # 외력(base_force) 관련 command / event는 데모에서 완전히 비활성화 (관측도 0으로 고정)
        if hasattr(env_cfg.commands, "base_force") and env_cfg.commands.base_force is not None:
            env_cfg.commands.base_force.resampling_time_range = (1.0e9, 1.0e9)
            # force command 자체가 0만 생성하도록 설정
            env_cfg.commands.base_force.ranges.force_range_fx = (0.0, 0.0)
            env_cfg.commands.base_force.ranges.force_range_fy = (0.0, 0.0)
            env_cfg.commands.base_force.ranges.force_range_fz = (0.0, 0.0)
            env_cfg.commands.base_force.ranges.duration_range_s = (1.0e9, 1.0e9)
            env_cfg.commands.base_force.ranges.interval_range_s = (1.0e9, 1.0e9)
        if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "apply_base_force_from_command"):
            env_cfg.events.apply_base_force_from_command = None

        self.env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
        self.device = self.env.unwrapped.device

        ppo_runner = OnPolicyRunner(self.env, agent_cfg.to_dict(), log_dir=None, device=self.device)
        ppo_runner.load(checkpoint)
        self.policy = ppo_runner.get_inference_policy(device=self.device)

        self.create_camera()
        # base_velocity는 (lin_vel_x, lin_vel_y, ang_vel_z) 3차원
        # 시작 시에는 반드시 0으로 (키 입력 전 자동 이동 방지)
        self.commands = torch.zeros(env_cfg.scene.num_envs, 3, device=self.device)
        self.set_up_keyboard()
        self._prim_selection = omni.usd.get_context().get_selection()
        self._selected_id = None
        self._previous_selected_id = None
        self._camera_local_transform = torch.tensor([-2.5, 0.0, 0.8], device=self.device)
        # 관측 내 velocity_commands / base_force_commands 구간 (reset 후 한 번 계산)
        self._vel_slice = None
        self._force_slice = None

    def create_camera(self):
        stage = get_current_stage()
        self.viewport = get_viewport_from_window_name("Viewport")
        self.camera_path = "/World/Camera"
        self.perspective_path = "/OmniverseKit_Persp"
        camera_prim = stage.DefinePrim(self.camera_path, "Camera")
        camera_prim.GetAttribute("focalLength").Set(8.5)
        coi_prop = camera_prim.GetProperty("omni:kit:centerOfInterest")
        if not coi_prop or not coi_prop.IsValid():
            camera_prim.CreateAttribute(
                "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d, True, Sdf.VariabilityUniform
            ).Set(Gf.Vec3d(0, 0, -10))
        self.viewport.set_active_camera(self.perspective_path)

    def set_up_keyboard(self):
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        # [lin_vel_x, lin_vel_y, ang_vel_z] (m/s, m/s, rad/s)
        T = 0.8
        S = 0.6
        self._key_to_control = {
            "UP": torch.tensor([T, 0.0, 0.0], device=self.device),
            "DOWN": torch.tensor([-T, 0.0, 0.0], device=self.device),
            "LEFT": torch.tensor([0.0, S, 0.0], device=self.device),
            "RIGHT": torch.tensor([0.0, -S, 0.0], device=self.device),
            "ZEROS": torch.tensor([0.0, 0.0, 0.0], device=self.device),
        }

    def _on_keyboard_event(self, event):
        # num_envs==1이면 선택 없이도 env 0에 키보드 적용
        env_id = self._selected_id if self._selected_id is not None else (0 if self.env.unwrapped.num_envs == 1 else None)
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._key_to_control:
                if env_id is not None:
                    self.commands[env_id] = self._key_to_control[event.input.name]
            elif event.input.name == "ESCAPE":
                self._prim_selection.clear_selected_prim_paths()
            elif event.input.name == "C":
                if self._selected_id is not None:
                    if self.viewport.get_active_camera() == self.camera_path:
                        self.viewport.set_active_camera(self.perspective_path)
                    else:
                        self.viewport.set_active_camera(self.camera_path)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if env_id is not None:
                self.commands[env_id] = self._key_to_control["ZEROS"]

    def update_selected_object(self):
        self._previous_selected_id = self._selected_id
        selected_prim_paths = self._prim_selection.get_selected_prim_paths()
        if len(selected_prim_paths) == 0:
            self._selected_id = None
            self.viewport.set_active_camera(self.perspective_path)
        elif len(selected_prim_paths) > 1:
            print("Multiple prims are selected. Please only select one!")
        else:
            prim_splitted_path = selected_prim_paths[0].split("/")
            if len(prim_splitted_path) >= 4 and prim_splitted_path[3][0:4] == "env_":
                self._selected_id = int(prim_splitted_path[3][4:])
                if self._previous_selected_id != self._selected_id:
                    self.viewport.set_active_camera(self.camera_path)
                self._update_camera()
            else:
                print("The selected prim was not a G1 robot")

        if self._previous_selected_id is not None and self._previous_selected_id != self._selected_id:
            self.env.unwrapped.command_manager.reset([self._previous_selected_id])
            # 선택이 바뀌면 키보드 명령은 0으로 리셋
            self.commands[:] = 0.0

    def set_velocity_command_from_keyboard(self):
        """매 스텝 키보드 값을 base_velocity에 직접 넣어, 로봇이 키보드 명령만 따르도록 함."""
        vel_term = self.env.unwrapped.command_manager.get_term("base_velocity")
        if hasattr(vel_term, "vel_command_b"):
            vel_term.vel_command_b.copy_(self.commands)

    def _update_camera(self):
        base_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[self._selected_id, :]
        base_quat = self.env.unwrapped.scene["robot"].data.root_quat_w[self._selected_id, :]
        camera_pos = quat_apply(base_quat, self._camera_local_transform) + base_pos
        camera_state = ViewportCameraState(self.camera_path, self.viewport)
        eye = Gf.Vec3d(camera_pos[0].item(), camera_pos[1].item(), camera_pos[2].item())
        target = Gf.Vec3d(base_pos[0].item(), base_pos[1].item(), base_pos[2].item() + 0.6)
        camera_state.set_position_world(eye, True)
        camera_state.set_target_world(target, True)

    def _ensure_obs_slices(self):
        if self._vel_slice is None:
            self._vel_slice = _get_policy_obs_term_slice(self.env, "policy", "velocity_commands")
            self._force_slice = _get_policy_obs_term_slice(self.env, "policy", "base_force_commands")

    def inject_commands_into_obs(self, obs):
        """키보드 속도 명령을 관측에 반영하고, 외력 관측은 0으로 고정(보행만 평가). inference tensor는 in-place 수정 불가하므로 clone 후 수정해 반환."""
        self._ensure_obs_slices()
        if self._vel_slice is None or self._force_slice is None:
            return obs
        start_vel, end_vel = self._vel_slice
        start_f, end_f = self._force_slice
        n_vel = end_vel - start_vel
        if n_vel <= 0:
            return obs
        obs = obs.clone()
        e = self._selected_id if self._selected_id is not None else 0
        scale_t = torch.tensor(VELOCITY_CMD_SCALE, device=self.device, dtype=obs.dtype)
        vals = (self.commands[e].to(obs.dtype) * scale_t)[:n_vel]
        obs[e, start_vel:end_vel] = vals
        obs[:, start_f:end_f] = 0.0
        return obs


def main():
    demo = G1RoughDemo()
    obs, _ = demo.env.reset()
    while simulation_app.is_running():
        demo.update_selected_object()
        # 키보드 값을 velocity command에 직접 설정 (resample에 덮어씌워지기 전에 다음 스텝에서 다시 설정)
        demo.set_velocity_command_from_keyboard()
        obs = demo.inject_commands_into_obs(obs)
        with torch.inference_mode():
            action = demo.policy(obs)
            obs, _, _, _ = demo.env.step(action)


if __name__ == "__main__":
    main()
    simulation_app.close()

