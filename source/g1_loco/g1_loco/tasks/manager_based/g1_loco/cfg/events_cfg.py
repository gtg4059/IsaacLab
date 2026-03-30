# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event 설정: 도메인 랜덤화/리셋/외란."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .. import mdp


@configclass
class EventCfg:
    """도메인 랜덤화/리셋/외란 이벤트 정의.

    이벤트 모드:
    - startup: 에피소드 시작 전/초기화 시 적용
    - reset: 리셋 시점마다 적용
    - interval: 에피소드 중 주기적으로 적용
    """

    # interval: 주기적으로 외란을 줘서 정책 강건성 향상
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5)}},
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    # startup: 발 접촉면 마찰 랜덤화
    randomize_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
            "static_friction_range": (0.5, 1.0),
            "dynamic_friction_range": (0.5, 1.0),
            "restitution_range": (0.0, 0.4),
            "num_buckets": 256,
            "make_consistent": True,
        },
    )

    randomize_joint_param = EventTerm(
        func=mdp.randomize_joint_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.01, 1.0),
            "viscous_friction_distribution_params": (0.3, 1.5),
            "armature_distribution_params": (0.008, 0.06),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    randomize_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "mass_distribution_params": (-2.0, 5.0),
            "operation": "add",
        },
    )

    randomize_base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "com_range": {"x": (-0.12, 0.12), "y": (-0.12, 0.12), "z": (-0.08, 0.08)},
        },
    )
