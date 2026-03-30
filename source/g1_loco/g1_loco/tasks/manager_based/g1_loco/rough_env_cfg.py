# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rough(험지) 전용 환경 설정.

`g1_loco_env_cfg.py`의 공통 베이스를 상속하여,
거친 지형 학습에 필요한 보상/커리큘럼/랜덤화 강도를 조정한다.
"""

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from . import mdp
from .g1_loco_env_cfg import G1LocoEnvCfg, RewardsCfg

STEP = 32000*8
RESUME = 0*8

@configclass
class G1RoughRewards(RewardsCfg):
    """Rough 환경용 보상 항목 정의.

    베이스 `RewardsCfg`를 확장/재정의하여 험지 보행 안정성에 맞게 조정한다.
    """

    # 높이 스캐너를 활용해 지형 높이를 반영한 베이스 높이 패널티
    base_height = RewTerm(func=mdp.base_height_log, weight=-100.0, params={
        "target_height": 0.78,
        "sensor_cfg": SceneEntityCfg("height_scanner"),
    })
    # 종료 시 큰 패널티
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    # 목표 선속도/각속도 추종 보상
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

    # 발목/무릎 관절 제한 위반 패널티
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
            ".*_ankle_pitch_joint", 
            ".*_ankle_roll_joint",
            ".*_knee_joint"
            ])},
    )
    
    # 보행 핵심이 아닌 관절의 과도한 편차 억제
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
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

    # G1_29_no_hand 기준 허리 관절 편차 억제
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
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
class G1RoughCurriculumCfg():
    """Rough 환경용 커리큘럼.

    험지 보행에 필요한 보상 가중치를 학습 단계별로 점진 조정한다.
    """
    # 10000 step마다 최대 난이도의 10%씩 증가 (거리 기반 아님)
    # terrain_levels = CurrTerm(
    #     func=mdp.terrain_levels_step_schedule,
    #     params={
    #         "step_interval": 8000*8,
    #         "percent_per_interval": 0.5,
    #         "min_steps": 8000*8-RESUME,
    #     },
    # ) 
    foot_clearance_weight = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "foot_clearance", "weight": 0.0, "num_steps": STEP-RESUME
        }
    )

    # 초기 50.0에서 num_steps 동안 선형 감쇠하여 0으로 수렴
    # min_steps 이전에는 weight=0, 이후 initial_weight에서 final_weight로 num_steps 동안 감쇠
    # feet_land_time_weight = CurrTerm(
    #     func=mdp.modify_reward_weight_linear_decay,
    #     params={
    #         "term_name": "feet_land_time",
    #         "initial_weight": 80.0,
    #         "final_weight": 2.0,
    #         "num_steps": 2.1*STEP-RESUME,
    #         "min_steps": 2*STEP-RESUME,  # 이 스텝 이후부터 감쇠 시작 (이전에는 weight=0)
    #     },
    # )

    # 발 접촉 힘 패널티를 일정 스텝 동안 점진 적용
    contact_forces_weight = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "contact_forces", "weight": -0.0000005, "num_steps": 2*STEP-RESUME}
    )

    # 액션 변화율 패널티를 일정 스텝 동안 점진 적용
    action_rate_l2_weight = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate_l2", "weight": -0.01, "num_steps": 2*STEP-RESUME}
    )

@configclass
class G1RoughEnvCfg(G1LocoEnvCfg):
    """Rough 학습용 환경 설정."""

    rewards: G1RoughRewards = G1RoughRewards()
    curriculum: G1RoughCurriculumCfg = G1RoughCurriculumCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.80)
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        # 초기에는 가장 쉬운 난이도에서 시작하도록 고정
        self.scene.terrain.terrain_generator.difficulty_range = (1.0, 1.0)
        self.scene.terrain.max_init_terrain_level = None
        # self.curriculum.terrain_levels = None
        # 리셋 시 초기 포즈/속도 랜덤화 범위
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
        # Rough 학습용 보상 가중치 미세 조정
        # no height scan
        # self.scene.height_scanner = None
        # reward for init model file
        # self.commands.base_velocity.ranges.lin_vel_x = (0.0, 2.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.01, 0.01)
        self.rewards.base_height.params["target_height"] = 0.76
        self.rewards.foot_clearance.weight = 0.75
        self.rewards.feet_land_time.weight = 0.0
        self.rewards.contact_forces.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.0001
        # self.rewards.dof_acc_l2.weight = -1.0e-7
        # self.rewards.joint_deviation_torso.weight = -0.5
        # self.rewards.dof_torques_l2.weight = -1.0e-6
        # self.rewards.joint_deviation_hip.weight = -1.0

        # 마찰 랜덤화 대상은 발 접촉 링크 중심으로 제한
        self.events.randomize_friction.params["asset_cfg"].body_names = [".*_ankle_roll_link"]
        # self.events.randomize_joint_param = None
        # self.events.randomize_link_mass = None
        # 과도한 난이도 상승 방지를 위해 일부 랜덤화 비활성화
        self.events.randomize_base_mass = None
        self.events.randomize_base_com = None

@configclass
class G1RoughEnvCfg_PLAY(G1RoughEnvCfg):
    """Rough 정책 확인(play)용 설정."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # self.curriculum.terrain_levels = None
        # 플레이 시에는 렌더링/확인 목적에 맞게 환경 수를 축소
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn only on easiest terrain (minimum difficulty)
        # self.scene.terrain.terrain_generator.difficulty_range = (1.0, 1.0)
        # self.scene.terrain.max_init_terrain_level = None
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

        # 플레이에서는 관측 노이즈를 끄고 정책 동작을 명확히 확인
        self.observations.policy.enable_corruption = False
