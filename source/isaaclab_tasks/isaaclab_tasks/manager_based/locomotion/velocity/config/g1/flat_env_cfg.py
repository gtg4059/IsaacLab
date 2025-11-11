# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .rough_env_cfg import G1RoughEnvCfg
from isaaclab_assets import G1_DEX_FIX
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import math

@configclass
class G1FlatEnvCfg(G1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        # self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # New Rewards
        # self.rewards.joint_deviation_arms.weight = -0.2
        # # self.rewards.joint_deviation_fingers.weight = -0.1
        # self.rewards.joint_deviation_torso.weight = -0.2
        # self.rewards.joint_deviation_hip.weight = -0.2
        # Main Rewards
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        # Rewards
        self.rewards.lin_vel_z_l2.weight = -0.2
        
        
        # self.rewards.feet_air_time.weight = 0.75
        # self.rewards.feet_air_time.params["threshold"] = 0.4
        
        # G1_29_no_hand
        self.rewards.joint_deviation_arms.weight = -1.0
        self.rewards.joint_deviation_torso.weight = -1.0

        # self.events.push_robot = None
        # curriculum
        # self.rewards.dof_acc_l2.weight = -1.0e-7
        # self.rewards.track_lin_vel_xy_exp.weight = 2.0
        # self.rewards.track_ang_vel_z_exp.weight = 2.0
        # self.rewards.foot_clearance.weight = 0.75
        # self.rewards.feet_air_time.weight = 0.0
        # self.rewards.contact_forces.weight = -0.0
        # self.rewards.dof_torques_l2.weight = -2.0e-6
        # self.rewards.action_rate_l2.weight = -0.002
        # self.commands.base_velocity = mdp.UniformVelocityCommandCfg(
        #     asset_name="robot",
        #     resampling_time_range=(5.0, 10.0),
        #     rel_standing_envs=0.1,
        #     rel_heading_envs=0.8,
        #     heading_command=True,
        #     heading_control_stiffness=2.0,
        #     debug_vis=True,
        #     ranges=mdp.UniformVelocityCommandCfg.Ranges(
        #         lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        #     ),
        # )




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
        
        # self.events.randomize_base_mass = None
        # self.events.randomize_base_com = None
        # self.events.randomize_pd_gains = None
        # self.events.randomize_link_mass = None
        # self.events.randomize_motor_zero_offset = None
        # self.events.randomize_joint_param = None
