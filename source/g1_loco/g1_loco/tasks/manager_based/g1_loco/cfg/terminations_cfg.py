# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination 설정: 에피소드 종료 조건."""

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from .. import mdp


@configclass
class TerminationsCfg:
    """에피소드 종료 조건 정의."""

    # 최대 에피소드 길이 도달 시 종료
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # 자세가 과도하게 기울어지면 종료
    base_contact = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.9},
    )
