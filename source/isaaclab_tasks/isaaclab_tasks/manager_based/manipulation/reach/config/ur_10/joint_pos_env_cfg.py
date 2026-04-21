# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets import UR10_CFG  # isort: skip


##
# Environment configuration
##


@configclass
class UR10ReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to ur10
        self.scene.robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["ee_link"]
        # # self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["ee_link"]
        # # self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["ee_link"]
        # self.rewards.end_effector_pos_orientation_tracking.params["asset_cfg"].body_names = ["ee_link"]
        # self.rewards.end_effector_pos_orientation_tracking_fine_grained.params["asset_cfg"].body_names = ["ee_link"]
        # # self.rewards.command_error_tanh.params["asset_cfg"].body_names = ["ee_link"]
        # # self.rewards.position_orientation_command_error.params["asset_cfg"].body_names = ["ee_link"]
        # self.observations.policy.ee_pose_error.params["asset_cfg"].body_names = ["ee_link"]
        self.commands.ee_pose.body_name = "ee_link"
        self.actions.arm_action = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=[".*"],scale=0.2)
        # self.actions.arm_action = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"],scale=0.01)
        # shoulder_lift_joint 와 나머지 관절에 서로 다른 위치/속도 reset 오프셋 범위 (uniform 샘플)
        self.events.reset_robot_joints.params["primary_position_range"] = (-math.pi/2, math.pi/2)
        self.events.reset_robot_joints.params["primary_velocity_range"] = (0.0, 0.0)
        self.events.reset_robot_joints.params["secondary_position_range"] = (-math.pi*0.8, math.pi*0.8)
        self.events.reset_robot_joints.params["secondary_velocity_range"] = (0.0, 0.0)

        self.commands.ee_pose.ranges.pos_th = (-math.pi, math.pi)
        self.commands.ee_pose.ranges.roll = (-math.pi, math.pi)
        self.commands.ee_pose.ranges.pitch = (-math.pi, math.pi)
        self.commands.ee_pose.ranges.yaw = (-math.pi, math.pi)

        # self.rewards.end_effector_position_tracking = None


@configclass
class UR10ReachEnvCfg_PLAY(UR10ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.viewer.eye = (4.5, 4.5, 4.5)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
