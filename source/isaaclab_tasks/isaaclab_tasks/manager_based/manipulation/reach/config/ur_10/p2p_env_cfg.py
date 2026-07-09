# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .joint_pos_env_cfg import UR10ReachEnvCfg

##
# Environment configuration
##

# --task Isaac-Reach-UR10-P2P-v0

@configclass
class UR10ReachP2PEnvCfg(UR10ReachEnvCfg):
    """UR10 reach with a single point-to-point goal per episode (no in-episode resample)."""

    def __post_init__(self):
        super().__post_init__()

        # Single P2P: disable reach-triggered command resampling.
        self.events.resample_ee_pose_on_reach = None
        # Curriculum syncs thresholds to that event; disable together.
        self.curriculum.reach_success_criteria = None
        # P2P accumulates success bonus every step at goal; use a lower weight than recurr (600).
        self.rewards.reach_success_bonus.weight = 10.0

        self.events.reset_robot_joints.params["primary_position_range"] = (0.0, 0.0)
        self.events.reset_robot_joints.params["secondary_position_range"] = (0.0, 0.0)
        self.events.reset_robot_joints.params["primary_velocity_range"] = (0.0, 0.0)
        self.events.reset_robot_joints.params["secondary_velocity_range"] = (0.0, 0.0)

        self.commands.ee_pose.ranges.pos_th = (0.0, 0.0)
        self.commands.ee_pose.ranges.roll = (0.0, 0.0)
        self.commands.ee_pose.ranges.pitch = (0.0, 0.0)
        self.commands.ee_pose.ranges.yaw = (0.0, 0.0)


@configclass
class UR10ReachP2PEnvCfg_PLAY(UR10ReachP2PEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.viewer.eye = (4.5, 4.5, 4.5)
        self.observations.policy.enable_corruption = False
