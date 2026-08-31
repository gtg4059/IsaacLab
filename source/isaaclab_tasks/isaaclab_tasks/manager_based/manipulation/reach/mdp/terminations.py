# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .rewards import ReachSuccessCriteria, reach_success_criteria

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_CRITERIA_KEYS = (
    "command_name",
    "asset_cfg",
    "max_distance",
    "max_angle_rad",
    "max_lin_vel",
    "max_ang_vel",
    "max_lin_acc",
    "max_ang_acc",
)


def reach_success(
    env: ManagerBasedRLEnv,
    reward_term_name: str = "reach_success_bonus",
) -> torch.Tensor:
    """Terminate when EE satisfies success thresholds from the reach bonus reward term."""
    term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    func = term_cfg.func
    if isinstance(func, ReachSuccessCriteria):
        return func.compute_success(env, **term_cfg.params)
    params = {key: term_cfg.params[key] for key in _CRITERIA_KEYS}
    return reach_success_criteria(env, **params)


def CRI_OVF(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.96,
) -> torch.Tensor:
    """Terminate when any collision point CRI exceeds ``threshold``."""
    asset: Articulation = env.scene[asset_cfg.name]
    result, _ = torch.max(asset.data.CRI, dim=1)
    return result > threshold
