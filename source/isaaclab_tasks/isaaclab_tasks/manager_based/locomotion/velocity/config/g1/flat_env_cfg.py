# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .rough_env_cfg import G1RoughEnvCfg, G1RoughRewards
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab_assets import G1_DEX_FIX, G1_SHOE
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import CurriculumCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg
from isaaclab.managers import RewardTermCfg as RewTerm
import math

STEP = 32000*8
RESUME = 0*8

@configclass
class G1FlatRewards(RewardsCfg):
    """Reward terms for the MDP."""
    base_height = RewTerm(func=mdp.base_height_l2, weight=-100.0, params={
        "target_height": 0.78,
    })
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 1.0},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"command_name": "base_velocity", "std": 1.0}
    )

    foot_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=0.75,
        params={
            "std": 0.05,
            "target_height": 0.2,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )

    feet_land_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
            ".*_ankle_pitch_joint", 
            ".*_ankle_roll_joint",
            ".*_knee_joint"
            ])},
    )
    
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    # ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    # ".*_shoulder_yaw_joint",
                    # ".*_elbow_joint",
                    ".*_wrist_roll_joint",
                    ".*_wrist_pitch_joint",
                    ".*_wrist_yaw_joint",
                ],
            )
        },
    )
    
    joint_deviation_shoulders = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    # ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    # ".*_wrist_roll_joint",
                    # ".*_wrist_pitch_joint",
                    # ".*_wrist_yaw_joint",
                ],
            )
        },
    )

    # G1_29_no_hand
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
            "waist_roll_joint",
            "waist_pitch_joint",
            "waist_yaw_joint",
        ])},
    )

    contact_forces = RewTerm(
        func=mdp.contact_forces_minimize,
        weight=-0.0,
        params={
            "threshold": 0.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )

@configclass
class G1FlatCurriculumCfg(CurriculumCfg):
    """Curriculum configuration for G1 flat environment."""
    foot_clearance_weight = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "foot_clearance", "weight": 0.0, "num_steps": STEP-RESUME}
    )
    # feet_land_time_weight = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "feet_land_time", "weight": 1.6, "num_steps": STEP-RESUME}
    # )
    feet_land_time_weight = CurrTerm(
        func=mdp.modify_reward_weight_linear_decay,
        params={
            "term_name": "feet_land_time",
            "initial_weight": 10.0,
            "final_weight": 1.2,
            "num_steps": 1.25*STEP-RESUME,
            "min_steps": 1.2*STEP-RESUME,  # 이 스텝 이후부터 감쇠 시작 (이전에는 weight=0)
        },
    )
    contact_forces_weight = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "contact_forces", "weight": -0.0000005, "num_steps": STEP-RESUME}
    )
    action_rate_l2_weight = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate_l2", "weight": -0.05, "num_steps": STEP-RESUME}
    )

    # joint_deviation_hip_yaw_weight = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "joint_deviation_hip_yaw", "weight": -0.1, "num_steps": STEP}
    # )

@configclass
class G1FlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    curriculum: G1FlatCurriculumCfg = G1FlatCurriculumCfg()
    rewards: G1FlatRewards = G1FlatRewards()
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = G1_SHOE.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.81)
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        # Randomization
        self.events.randomize_friction.params["asset_cfg"].body_names = [".*_ankle_roll_link"]
        # self.events.randomize_friction.params["static_friction_range"] = (0.1, 1.0)
        # self.events.randomize_friction.params["dynamic_friction_range"] = (0.1, 1.0)
        # Rewards
        self.rewards.undesired_contacts = None
        self.rewards.action_rate_l2.weight = -0.001
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None
        # no height scan
        self.scene.height_scanner = None


class G1FlatEnvCfg_PLAY(G1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()
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
