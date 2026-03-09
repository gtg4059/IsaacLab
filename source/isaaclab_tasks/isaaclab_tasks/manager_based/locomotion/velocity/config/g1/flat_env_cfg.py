# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .rough_env_cfg import G1RoughEnvCfg
from isaaclab_assets import G1_DEX_FIX, G1_DEX_EASY
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg, CurriculumCfg

@configclass
class G1FlatCurriculumCfg(CurriculumCfg):
    """Curriculum configuration for G1 flat environment."""
    foot_clearance_weight = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "foot_clearance", "weight": 0.0, "num_steps": 0*8}
    )
    feet_land_time_weight = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "feet_land_time", "weight": 0.6, "num_steps": 0*8}
    )
    contact_forces_weight = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "contact_forces", "weight": -0.0000005, "num_steps": 0*8}
    )
    action_rate_l2_weight = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate_l2", "weight": -0.05, "num_steps": 0*8}
    )
    dof_acc_l2_weight = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "dof_acc_l2", "weight": -1.0e-6, "num_steps": 0*8}
    )
    dof_torques_l2_weight = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "dof_torques_l2", "weight": -1.0e-6, "num_steps": 0*8}
    )
    joint_deviation_hip_yaw_weight = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_deviation_hip_yaw", "weight": -0.1, "num_steps": 0*8}
    )

    # 상체 arm joint target position_range를 0 → ±0.6 rad로 점진 증가
    arm_joint_targets_position_range = CurrTerm(
        func=mdp.modify_arm_joint_targets_position_range,
        params={
            "event_term_name": "set_arm_joint_targets_interval",
            "start_step": 0*8,
            "end_step": 0*8,
            "max_range": 1.6,
        },
    )
    
@configclass
class G1FlatEnvCfg(G1RoughEnvCfg):
    curriculum: G1FlatCurriculumCfg = G1FlatCurriculumCfg()
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
        # reward for init model file
        self.rewards.foot_clearance.weight = 0.75
        self.rewards.feet_land_time.weight = 0.0
        self.rewards.contact_forces.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.001
        self.rewards.dof_acc_l2.weight = 0.0
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.joint_deviation_hip_yaw.weight = -1.0

class G1FlatEnvCfg_PLAY(G1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        self.scene.robot = G1_DEX_FIX.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.curriculum.arm_joint_targets_position_range = None
        self.events.set_arm_joint_targets_interval.params["position_range"] = (-0.8, 0.8)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.events.push_robot = None
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
