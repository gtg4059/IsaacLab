# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .rough_env_cfg import G1RoughEnvCfg
# from isaaclab_assets.robots import G1_DEX_FIX, G1_DEX_EASY
from isaaclab_assets import G1_DEX_FIX
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import math

@configclass
class G1FlatEnvCfg(G1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.robot = G1_DEX_FIX.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None
        # no height scan
        self.scene.height_scanner = None


        # # Rewards
        # self.rewards.track_ang_vel_z_exp.weight = 1.0
        # self.rewards.lin_vel_z_l2.weight = -0.2
        # self.rewards.action_rate_l2.weight = -0.001
        # self.rewards.dof_acc_l2.weight = -1.0e-7
        # self.rewards.feet_air_time.weight = 3.0 # 0.75
        # self.rewards.feet_air_time.params["threshold"] = 0.4
        # self.rewards.dof_torques_l2.weight = -2.0e-4
        # self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
        #     "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        # )

        # # Commands
        # self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # # 전역 외력 (모든 링크 동일 범위)
        # self.commands.base_force.ranges.force_range_fx = (-40.0, -40.0)
        # self.commands.base_force.ranges.force_range_fy = (0.0, 0.0)
        # self.commands.base_force.ranges.force_range_fz = (0.0, 0.0)
        
        # # 링크별 외력: body_names 순서대로 [left_wrist_yaw_link, right_wrist_yaw_link]
        # # 각 원소 = ((fx_min, fx_max), (fy_min, fy_max), (fz_min, fz_max)) (base frame, N)
        # self.commands.base_force.ranges.force_ranges_per_link = [
        #     ((-40.0, -40.0), (0.0, 0.0), (0.0, 0.0)),   # link0 왼쪽 wrist: -x 방향 40N
        #     ((20.0, 20.0), (0.0, 0.0), (0.0, 0.0)),    # link1 오른쪽 wrist: +x 방향 20N
        # ]


class G1FlatEnvCfg_PLAY(G1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        self.scene.robot = G1_DEX_FIX.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        # self.observations.policy.enable_corruption = False
        # remove random pushing
        # self.events.randomize_friction = None
        self.events.push_robot = None

        # self.events.randomize_base_mass = None
        # self.events.randomize_base_com = None
        # self.events.randomize_pd_gains = None
        # self.events.randomize_link_mass = None
        # self.events.randomize_motor_zero_offset = None
        # self.events.randomize_joint_param = None