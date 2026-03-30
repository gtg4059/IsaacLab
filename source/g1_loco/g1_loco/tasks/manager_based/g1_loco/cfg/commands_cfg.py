# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command 설정: 정책이 추종할 목표 명령."""

import math

from isaaclab.utils import configclass

from .. import mdp


@configclass
class CommandsCfg:
    """정책이 따라야 할 목표 명령(command) 정의."""

    # 속도 추종 태스크의 목표값 샘플링 범위/주기를 정의
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),
        rel_standing_envs=0.2,  # 속도 명령을 0으로 설정하는 확률
        rel_heading_envs=0.8,  # 헤딩 명령을 사용하는 확률
        heading_command=True,
        heading_control_stiffness=2.0,  # 헤딩 오차 보정 강도
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 2.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )
