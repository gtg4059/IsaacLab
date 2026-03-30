# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward 설정: 공통 베이스 보상 항목."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from .. import mdp


@configclass
class RewardsCfg:
    """기본 보상(패널티 중심) 정의.

    참고:
    - 본 클래스는 공통 베이스 역할을 하며,
      flat/rough 환경에서 보상 항목이 추가/재정의된다.
    """

    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)  # z축 속도 제곱 패널티
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)  # xy축 각속도 제곱 패널티
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-6)  # 관절 토크 제곱 패널티
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)  # 관절 가속도 제곱 패널티
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)  # 행동 변화율 패널티

    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)  # 기울기 패널티
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)
