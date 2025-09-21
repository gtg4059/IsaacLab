# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .rough_env_cfg import G1RoughEnvCfg
from isaaclab_assets import G1_DEX_FIX


@configclass
class G1FlatEnvCfg(G1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # New Rewards
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.dof_acc_l2.weight = -1.0e-7

        # Rewards
        # self.rewards.lin_vel_z_l2.weight = -0.2
        # self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        # self.rewards.feet_air_time.weight = 5.0
        # self.rewards.base_height_l2.weight = -20.0
        # self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )
        # # disable randomization for play
        self.commands.dual_ee_pose.ranges.pos_x = (0.31, 0.31)
        self.commands.dual_ee_pose.ranges.pos_y = (0.12, 0.12)
        self.commands.dual_ee_pose.ranges.pos_z = (0.10, 0.10)
        # self.observations.policy.enable_corruption = False
        # self.events.randomize_friction_hand.params["static_friction_range"] = (0.6, 1.1)
        # self.events.randomize_friction_hand.params["dynamic_friction_range"] = (0.6, 1.1)
        # self.events.randomize_friction_hand.params["make_consistent"] = True
        # self.events.physics_material_obj.params["static_friction_range"] = (0.6, 1.1)
        # self.events.physics_material_obj.params["dynamic_friction_range"] = (0.6, 1.1)
        # self.events.physics_material_obj.params["make_consistent"] = True
        # # self.commands.dual_ee_pose.resampling_time_range = (5.0, 10.0)
        # self.commands.dual_ee_pose.ranges.pos_z = (0.20, 0.20)


class G1FlatEnvCfg_PLAY(G1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        self.scene.robot = G1_DEX_FIX.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # remove random pushing
        # delattr(self.events, 'push_robot')
        # delattr(self.events, 'push_robot_interval')
