# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg, CurriculumCfg

##
# Pre-defined configs
##
from isaaclab_assets import G1_MINIMAL_CFG, G1_DEX_FIX  # isort: skip


def _curriculum_base_force_ranges(
    env,
    env_ids,
    old_value,
    num_steps_start=0,
    num_steps_end=5000,
    max_force_xy=30.0,
    max_force_z=10.0,
):
    """커리큘럼: 학습 스텝에 따라 base_force 범위를 0에서 목표(합 약 50)까지 선형 증가.

    max_force_xy=30, max_force_z=10 이면 sqrt(40^2+40^2+20^2) ≈ 60 N.
    """
    if env.common_step_counter < num_steps_start:
        return mdp.modify_env_param.NO_CHANGE
    progress = (env.common_step_counter - num_steps_start) / max(1, num_steps_end - num_steps_start)
    scale = min(1.0, progress)
    RangesClass = type(old_value)
    return RangesClass(
        force_range_fx=(-max_force_xy * scale, max_force_xy * scale),
        force_range_fy=(-max_force_xy * scale, max_force_xy * scale),
        force_range_fz=(-max_force_z * scale, max_force_z * scale),
        duration_range_s=old_value.duration_range_s,
        interval_range_s=old_value.interval_range_s,
    )


@configclass
class G1Rewards(RewardsCfg):
    """Reward terms for the MDP."""

    base_height = RewTerm(func=mdp.base_height_l2, weight=-100.0, params={"target_height": 0.78}) #weight=-100.0
    
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=6.0, #4.0,
        params={"command_name": "base_velocity", "std": 0.25},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=5.0,#2.0, 
        params={"command_name": "base_velocity", "std": 0.25}
    )

    foot_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=0.0, #0.75,0.0
        params={
            "std": 0.05,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.5, #0.0,1.2
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-1.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    # balance_air_time = RewTerm(
    #     func=mdp.balance_air_time_reward,
    #     weight=-0.3,
    #     params={
    #         #"command_name": "base_velocity",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    #     },
    # )

    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,#-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
            ".*_ankle_pitch_joint", 
            ".*_ankle_roll_joint",
            ".*_knee_joint"
            ])},
    )
    
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint"])},
    )

    joint_deviation_hip_yaw = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1, #-1.5,-0.1
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint"])},
    )

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.3,#-0.3
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    # ".*_shoulder_pitch_joint",
                    # ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
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
        weight=-0.1,#-0.1?
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    # ".*_shoulder_yaw_joint",
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
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
            "waist_roll_joint",
            "waist_pitch_joint",
            "waist_yaw_joint",
        ])},
    )
    # shoulder_roll_limit = RewTerm(
    #     func=mdp.shoulder_roll_limit,
    #     weight=0.1,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 "left_shoulder_roll_joint",
    #                 "right_shoulder_roll_joint",
    #             ],
    #         )
    #     },
    # )
    contact_forces = RewTerm(
        func=mdp.contact_forces_minimize,
        weight=-0.0000005,# 0.0,-0.0000005
        params={
            "threshold": 0.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )


##################### force reward terms #####################

    tracking_lin_vel_force = RewTerm(
        func=mdp.tracking_lin_vel_force_reward,
        weight=0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "force_command_name": "base_force", 
            "vel_command_name": "base_velocity",
            "damping": 2.0,
            "sigma": 0.25,       
        },
    )

    force_compliance_reward = RewTerm(
        func=mdp.compliance_with_external_force_reward,
        weight=10.0,
        params={"sigma": 1.0,
                "force_threshold": 20.0, # 링크마다 같은 임계값
                # base_force.body_names 순서대로
                # 링크별 임계값을 다르게 적용하고 싶으면 아래 리스트를 수정
                "force_thresholds_per_link": [15.0, 15.0, 20.0, 30.0, 30.0],
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_wrist_yaw_link"),
                "force_command_name": "base_force", 
                "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    standing_arm_compliance = RewTerm(
        func=mdp.standing_arm_compliance,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "force_command_name": "base_force",
            "trigger_body_cfg": SceneEntityCfg("robot", body_names=".*_wrist_yaw_link"),
            "neighbor_body_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    ".*_shoulder_.*_link",
                    ".*_elbow_link",
                    ".*_wrist_.*_link",
                ],
            ),
            "force_threshold": 10.0,
            "standing_lin_vel_threshold": 0.02,
            "standing_ang_vel_threshold": 0.02,
        },
    )

    # 다리(발/무릎) 링크가 외력을 받으면, 외력 방향으로 다리 주변 링크들이 같이 움직이도록 유도
    # (단, 지면 접촉 상태(contact sensor)를 만족할 때만 reward를 주어 공중 발/한쪽만 튀는 현상을 완화)
    left_standing_leg_complianced = RewTerm(
        func=mdp.standing_leg_compliance,
        weight=0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "force_command_name": "base_force",
            # 외력을 받는 '트리거' 다리 링크(기본: 발목)
            "trigger_body_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "left_ankle_roll_link",
                    "left_ankle_pitch_link",
                ],
            ),
            # 트리거 주변(기본: 발목/무릎)
            "neighbor_body_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "left_ankle_roll_link",
                    "left_ankle_pitch_link",
                    "left_knee_link",
                ],
            ),
            "force_threshold": 15.0,
            "standing_lin_vel_threshold": 0.02,
            "standing_ang_vel_threshold": 0.02,
            # 공중 발 방지용 접촉 게이팅
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names= ["left_ankle_roll_link", "left_ankle_pitch_link"]),
            "contact_force_threshold": 1.0,
        },
    )

    right_standing_leg_complianced = RewTerm(
        func=mdp.standing_leg_compliance,
        weight=0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "force_command_name": "base_force",
            # 외력을 받는 '트리거' 다리 링크(기본: 발목)
            "trigger_body_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "right_ankle_roll_link",
                    "right_ankle_pitch_link",
                ],
            ),
            # 트리거 주변(기본: 발목/무릎)
            "neighbor_body_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "right_ankle_roll_link",
                    "right_ankle_pitch_link",
                    "right_knee_link",
                ],
            ),
            "force_threshold": 15.0,
            "standing_lin_vel_threshold": 0.02,
            "standing_ang_vel_threshold": 0.02,
            # 공중 발 방지용 접촉 게이팅
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["right_ankle_roll_link", "right_ankle_pitch_link"]),
            "contact_force_threshold": 1.0,
        },
    )

########## log data ###########
    # pos_data = RewTerm(
    #     func=mdp.joint_pos_data,
    #     weight=1.0e-11,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=['left_hip_pitch_joint'])},
    # )
    # vel_data = RewTerm(
    #     func=mdp.joint_vel_data,
    #     weight=1.0e-11,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=['left_hip_pitch_joint'])},
    # )
    # acc_data = RewTerm(
    #     func=mdp.joint_acc_data,
    #     weight=1.0e-11,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=['left_hip_pitch_joint'])},
    # )
    # applied_torque_data = RewTerm(
    #     func=mdp.joint_applied_torque_data,
    #     weight=1.0e-11,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=['left_hip_pitch_joint'])},
    # )

@configclass
class G1RoughCurriculumCfg(CurriculumCfg):
    """Curriculum configuration for G1 flat environment."""
    
    foot_clearance_weight = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "foot_clearance", "weight": 0.0, "num_steps": 50000*24}
    )
    feet_air_time_weight = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "feet_air_time", "weight": 1.2, "num_steps": 50000*24}
    )
    contact_forces_weight = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "contact_forces", "weight": -0.0000005, "num_steps": 50000*24}
    )
    action_rate_l2_weight = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate_l2", "weight": -0.1, "num_steps": 50000*24}
    )
    dof_acc_l2_weight = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "dof_acc_l2", "weight": -1.0e-6, "num_steps": 50000*24}
    )
    dof_torques_l2_weight = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "dof_torques_l2", "weight": -1.0e-6, "num_steps": 50000*24}
    )
    joint_deviation_hip_yaw_weight = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_deviation_hip_yaw", "weight": -0.1, "num_steps": 50000*24}
    )

    # # base_force 커리큘럼: 외력 크기를 0 → 약 60 N(합)으로 단계적 증가
    # base_force_ranges = CurrTerm(
    #     func=mdp.modify_env_param,
    #     params={
    #         "address": "command_manager.cfg.base_force.ranges",
    #         "modify_fn": _curriculum_base_force_ranges,
    #         "modify_params": {
    #             "num_steps_start": 10000*24,
    #             "num_steps_end": 40000*24,
    #             "max_force_xy": 40.0,
    #             "max_force_z": 10.0,
    #         },
    #     },
    # )


@configclass
class G1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: G1Rewards = G1Rewards()
    curriculum: G1RoughCurriculumCfg = G1RoughCurriculumCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = G1_DEX_FIX.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        self.curriculum.terrain_levels = None
        # Randomization
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        # Rewards
        # self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.undesired_contacts = None
        # self.rewards.flat_orientation_l2.weight = -1.0
        # self.rewards.action_rate_l2.weight = -0.005
        # self.rewards.dof_acc_l2.weight = -1.25e-7
        # self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
        #     "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        # )
        # self.rewards.dof_torques_l2.weight = -1.5e-7
        # self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
        #     "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        # )
        # no height scan
        self.scene.height_scanner = None
        # randomization select
        # self.events.randomize_friction = None
        # self.events.randomize_joint_param = None
        # self.events.randomize_link_mass = None
        # self.events.randomize_base_mass = None
        # self.events.randomize_base_com = None
        # self.events.randomize_pd_gains = None
        # self.events.randomize_motor_zero_offset = None

        # commands
        # self.commands.base_force.resampling_time_range = (5.0, 5.0)
        # self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)



@configclass
class G1RoughEnvCfg_PLAY(G1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
