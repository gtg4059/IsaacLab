# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets import UR10e_ROBOTIQ_2F_85_CFG, UR10e_JOINT_ORDER  # isort: skip


@configclass
class UR10eReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UR10e_ROBOTIQ_2F_85_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.events.reset_robot_joints.params["position_range"] = (0.75, 1.25)
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=UR10e_JOINT_ORDER)
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=UR10e_JOINT_ORDER)
        self.observations.policy.CRI.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=UR10e_JOINT_ORDER)
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["base_link_0"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["base_link_0"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["base_link_0"]
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=UR10e_JOINT_ORDER, scale=0.5, use_default_offset=True
        )
        self.commands.ee_pose.body_name = "base_link_0"
        self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)


@configclass
class UR10eReachEnvCfg_PLAY(UR10eReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
