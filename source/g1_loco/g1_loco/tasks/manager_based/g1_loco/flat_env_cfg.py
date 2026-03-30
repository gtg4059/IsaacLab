# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flat(평지) 전용 환경 설정.

`g1_loco_env_cfg.py`의 공통 베이스를 상속하여,
지형을 평지로 고정하고 보상/커리큘럼을 평지 보행에 맞게 튜닝한다.
"""

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from . import mdp
from .g1_loco_env_cfg import G1LocoEnvCfg, RewardsCfg
from g1_loco.utils.unitree import G1_SHOE

STEP = 32000*8
RESUME = 0*8

@configclass
class G1FlatRewards(RewardsCfg):
    """Flat 환경용 보상 항목 정의.

    베이스 `RewardsCfg`를 확장/재정의하여 평지 보행 안정성에 맞게 조정한다.
    """

    # 목표 높이(월드 프레임) 유지 패널티
    base_height = RewTerm(func=mdp.base_height_l2, weight=-100.0, params={
        "target_height": 0.78,
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
class G1FlatCurriculumCfg():
    """Flat 환경용 커리큘럼.

    학습 진행(step)에 따라 일부 보상 가중치를 점진적으로 조정한다.
    """
    # 발 들림 보상은 초기에 꺼두고 점진 적용
    foot_clearance_weight = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "foot_clearance", "weight": 0.0, "num_steps": STEP-RESUME}
    )
    # 착지 시간 보상은 특정 시점 이후 선형 감쇠
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
    # 행동 변화율 패널티를 단계적으로 강화
    action_rate_l2_weight = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate_l2", "weight": -0.05, "num_steps": STEP-RESUME}
    )

@configclass
class G1FlatEnvCfg(G1LocoEnvCfg):
    """Flat 학습용 환경 설정."""

    rewards: G1FlatRewards = G1FlatRewards()
    curriculum: G1FlatCurriculumCfg = G1FlatCurriculumCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.robot = G1_SHOE.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        # Randomization
        self.events.randomize_friction.params["asset_cfg"].body_names = [".*_ankle_roll_link"]
        # Rewards
        self.rewards.action_rate_l2.weight = -0.001
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None

@configclass
class G1FlatEnvCfg_PLAY(G1FlatEnvCfg):
    """Flat 정책 확인(play)용 경량 설정."""

    curriculum: G1FlatCurriculumCfg = G1FlatCurriculumCfg()

    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()
        # 플레이 시에는 적은 환경만 실행해 시각 확인 용이하게 설정
        self.scene.num_envs = 8
        self.scene.env_spacing = 2.5
