# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity locomotion task용 이벤트 함수.

학습 중 일정 간격으로 상체 arm 관절에 무작위 joint target을 설정하는 이벤트 등.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def set_arm_joint_targets_random(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    position_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """지정된 관절에 default position 기준 offset 범위 내의 무작위 joint position target을 설정한다.

    default position은 로봇 asset(예: isaaclab_assets unitree.py)에 정의된 default joint 값
    (asset.data.default_joint_pos)을 의미한다.

    interval 이벤트와 함께 사용하면, 학습 중 일정 간격으로 상체 arm 등 지정 관절의
    PD 타겟만 무작위로 바꿔 움직이게 할 수 있다. (관절 상태를 직접 덮어쓰지 않음)

    Args:
        env: ManagerBasedEnv 인스턴스.
        env_ids: 이벤트가 적용될 환경 인덱스.
        position_range: default position에 더할 offset 범위 (min, max) [rad].
        asset_cfg: 로봇 asset 및 적용할 joint_names (또는 joint_ids) 설정.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # asset default (unitree.py 등에 정의된 값) 기준으로 offset 샘플링
    joint_pos = asset.data.default_joint_pos[env_ids][:, asset_cfg.joint_ids].clone()
    joint_pos += math_utils.sample_uniform(
        *position_range, joint_pos.shape, joint_pos.device
    )

    # soft limit 내로 clamp
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(
        joint_pos_limits[..., 0], joint_pos_limits[..., 1]
    )

    # joint position target만 설정 (시뮬 상태는 그대로, PD가 타겟으로 추종)
    asset.set_joint_position_target(
        joint_pos.view(len(env_ids), -1),
        joint_ids=asset_cfg.joint_ids,
        env_ids=env_ids,
    )
