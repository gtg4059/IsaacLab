# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.cri_F_reach_env_cfg import (
    CRIFReachEnvCfg,
    CRIFReachEnvCfg_PLAY,
)
from isaaclab_tasks.manager_based.manipulation.reach.cri_reach_env_cfg import (
    CRIReachEnvCfg,
    _spawn_ur10_base_collider_only,
)

##
# Pre-defined configs
##
from isaaclab_assets import UR10_CFG  # isort: skip


##
# Environment configuration
##


@configclass
class UR10ReachEnvCfg(CRIReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UR10_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=UR10_CFG.spawn.replace(func=_spawn_ur10_base_collider_only),
        )
        self.commands.ee_pose.body_name = "ee_link"
        self.actions.arm_action = mdp.JointVelocitySplineOffsetActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        )

        # Every reset draws a new arm. A horizon timeout keeps the target.
        self.events.reset_robot_joints.params["position_range"] = {
            "shoulder_pan_joint": (-math.pi, math.pi),
            "shoulder_lift_joint": (-math.pi, 0.0),
            "elbow_joint": (-math.pi * 2 / 3, math.pi * 2 / 3),
            "wrist_1_joint": (-math.pi, math.pi),
            "wrist_2_joint": (-math.pi, math.pi),
            "wrist_3_joint": (-math.pi, math.pi),
        }
        self.events.reset_robot_joints.params["velocity_range"] = {
            "shoulder_pan_joint": (0.0, 0.0),
            "shoulder_lift_joint": (0.0, 0.0),
            "elbow_joint": (0.0, 0.0),
            "wrist_1_joint": (0.0, 0.0),
            "wrist_2_joint": (0.0, 0.0),
            "wrist_3_joint": (0.0, 0.0),
        }

        self.commands.ee_pose.ranges.pos_th = (-math.pi, math.pi)
        self.commands.ee_pose.ranges.roll = (-math.pi, math.pi)
        self.commands.ee_pose.ranges.pitch = (-math.pi, math.pi)
        self.commands.ee_pose.ranges.yaw = (-math.pi, math.pi)


@configclass
class UR10ReachEnvCfg_PLAY(UR10ReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.viewer.eye = (4.5, 4.5, 4.5)
        self.observations.policy.enable_corruption = False
        self.terminations.OVF.params["threshold"] = 0.96
        # Match training: a timeout keeps the target and draws a new arm.
        self.commands.ee_pose.keep_command_until_success = True
        self.events.reset_robot_joints.params["keep_failed_spawn"] = False
        if self.curriculum is not None and hasattr(self.curriculum, "resample_on_reset"):
            self.curriculum.resample_on_reset.params["switch_step"] = 0
        if hasattr(self.events, "early_spline_velocity_offset"):
            self.events.early_spline_velocity_offset.params["std"] = 0.0
        if self.curriculum is not None and hasattr(self.curriculum, "early_spline_velocity_offset"):
            spline = self.curriculum.early_spline_velocity_offset.params["modify_params"]
            spline["amplitude"] = 0.0
        self.commands.ee_pose.debug_vis = True
        if hasattr(self.rewards, "CRI_OVF"):
            self.rewards.CRI_OVF.params["limit"] = 0.96


# CRI-F 본체는 cri_F_reach_env_cfg.py. 여기 이름은 Gym/기존 import 호환용.
UR10CRIFReachEnvCfg = CRIFReachEnvCfg
UR10CRIFReachEnvCfg_PLAY = CRIFReachEnvCfg_PLAY