# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action 설정: 정책 출력을 로봇 제어 명령으로 변환."""

from isaaclab.utils import configclass

from .. import mdp
from .common_cfg import G1_ACTUATED_JOINT_NAMES


@configclass
class ActionsCfg:
    """정책 출력(action)을 로봇 관절 명령으로 해석하는 규칙."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        # deploy 코드와 같은 순서로 joint_names를 설정
        joint_names=G1_ACTUATED_JOINT_NAMES,
        scale=0.25,
        use_default_offset=False,
        preserve_order=True,
    )
