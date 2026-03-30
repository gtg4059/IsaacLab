# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""하위 호환용 모듈.

기존 분리 구조에서 사용하던 `env_cfg` 경로를 유지하기 위해
`g1_loco_env_cfg`의 환경 클래스를 재노출한다.
"""

from ..g1_loco_env_cfg import G1LocoEnvCfg, G1LocoEnvCfg_PLAY

__all__ = ["G1LocoEnvCfg", "G1LocoEnvCfg_PLAY"]
