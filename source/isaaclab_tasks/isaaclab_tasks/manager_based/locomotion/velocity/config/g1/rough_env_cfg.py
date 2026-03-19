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
from isaaclab_assets import G1_DEX_FIX, G1_DEX_EASY  # isort: skip


@configclass
class G1Rewards(RewardsCfg):
    """Reward terms for the MDP."""

    base_height = RewTerm(func=mdp.base_height_l2, weight=-100.0, params={
        "target_height": 0.76,
        "sensor_cfg": SceneEntityCfg("height_scanner"),
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
        weight=0.0,
        params={
            "std": 0.05,
            "target_height": 0.2,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )

    feet_land_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.2,
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
    joint_deviation_hip_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint"])},
    )

    joint_deviation_hip_yaw = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint"])},
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
        weight=-0.0000005,
        params={
            "threshold": 0.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )

@configclass
class G1RoughCurriculumCfg(CurriculumCfg):
    """Curriculum configuration for G1 flat environment."""
    # 10000 step마다 최대 난이도의 10%씩 증가 (거리 기반 아님)
    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_step_schedule,
        params={
            "step_interval": 1*8,
            "percent_per_interval": 0.5,
            "min_steps": 1*8,
        },
    ) 
    foot_clearance_weight = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "foot_clearance", "weight": 0.0, "num_steps": 1*8}
    )
    # 초기 50.0에서 num_steps 동안 선형 감쇠하여 0으로 수렴
    feet_land_time_weight = CurrTerm(
        func=mdp.modify_reward_weight_linear_decay,
        params={
            "term_name": "feet_land_time",
            "initial_weight": 1.2,
            "final_weight": 1.2,
            "num_steps": 16000*8,
        },
    )
    contact_forces_weight = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "contact_forces", "weight": -0.0000002, "num_steps": 1*8}
    )
    action_rate_l2_weight = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate_l2", "weight": -0.02, "num_steps": 1*8}
    )
    dof_acc_l2_weight = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "dof_acc_l2", "weight": -1.0e-7, "num_steps": 1*8}
    )
    dof_torques_l2_weight = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "dof_torques_l2", "weight": -1.0e-6, "num_steps": 1*8}
    )
    joint_deviation_hip_yaw_weight = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_deviation_hip_yaw", "weight": -0.1, "num_steps": 1*8}
    )

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
        # start at minimum terrain difficulty (curriculum increases after min_steps)
        self.scene.terrain.max_init_terrain_level = 0
        # self.curriculum.terrain_levels = None
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
        self.rewards.undesired_contacts = None
        # no height scan
        # self.scene.height_scanner = None
        # reward for init model file
        # self.commands.base_velocity.ranges.lin_vel_x = (0.0, 2.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.01, 0.01)
        self.rewards.base_height.weight = -10.0
        self.rewards.base_height.params["target_height"] = 0.72
        self.rewards.foot_clearance.weight = 0.75
        self.rewards.feet_land_time.weight = 0.0
        self.rewards.contact_forces.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.001
        self.rewards.dof_acc_l2.weight = 0.0
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.joint_deviation_hip_yaw.weight = -1.0

        self.events.randomize_friction.params["asset_cfg"].body_names = [".*_ankle_roll_link"]
        # self.events.randomize_joint_param = None
        self.events.randomize_link_mass = None
        # self.events.randomize_base_mass = None
        # self.events.randomize_base_com = None
        self.events.randomize_pd_gains = None
        self.events.randomize_motor_zero_offset = None




@configclass
class G1RoughEnvCfg_PLAY(G1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.curriculum.terrain_levels = None
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn only on easiest terrain (minimum difficulty)
        self.scene.terrain.terrain_generator.difficulty_range = (1.0, 1.0)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        # if self.scene.terrain.terrain_generator is not None:
        #     self.scene.terrain.terrain_generator.num_rows = 5
        #     self.scene.terrain.terrain_generator.num_cols = 5
        #     self.scene.terrain.terrain_generator.curriculum = False

        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None
        # self.curriculum.terrain_levels = None

        # self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        # self.events.push_robot = None

        # disable randomization for play
        self.observations.policy.enable_corruption = False
