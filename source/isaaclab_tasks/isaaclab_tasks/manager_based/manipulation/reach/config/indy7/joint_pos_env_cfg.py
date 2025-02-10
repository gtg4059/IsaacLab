# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg
from isaaclab.managers import SceneEntityCfg

##
# Pre-defined configs
##
from isaaclab_assets import UR10_CFG  # isort: skip


##
# Environment configuration
##


@configclass
class Indy7ReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to Indy7
        self.scene.robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.events.reset_robot_joints.params["position_range"] = (-math.pi,math.pi)#(0.0, 1.57)
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["ee_link"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["ee_link"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["ee_link"]
        self.rewards.end_effector_tracking_fine_grained.params["asset_cfg"].body_names = ["ee_link"]

        self.actions.arm_action = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=[".*"],scale=0.2)
        
        # self.actions.arm_action = mdp.JointVelocityActionScaleCfg(asset_name="robot", joint_names=[".*"],scale=0.21,
        #                                                           body="ee_link",asset_cfg=SceneEntityCfg("robot"),
        #                                                           command_name="ee_pose")


        self.commands.ee_pose.body_name = "ee_link"
        self.commands.ee_pose.resampling_time_range=(80,80)
        # self.commands.ee_pose.resampling_trigger=resample_trig
        self.commands.ee_pose.ranges.pos_th = (-math.pi, math.pi)
        self.commands.ee_pose.ranges.roll = (-math.pi, math.pi)
        self.commands.ee_pose.ranges.pitch = (-math.pi,math.pi)#(-math.pi/2,-math.pi/2)
        self.commands.ee_pose.ranges.yaw = (-math.pi,math.pi)


@configclass
class Indy7ReachEnvCfg_PLAY(Indy7ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
