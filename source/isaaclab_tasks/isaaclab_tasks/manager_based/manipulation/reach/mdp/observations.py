# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for the reach task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def ee_pose_in_base_frame(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """End effector pose (position 3 + quaternion 4 = 7) in the robot base frame.

    Uses the first body in asset_cfg.body_ids as the end effector.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos_w = asset.data.root_pos_w
    root_quat_w = asset.data.root_quat_w
    ee_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    ee_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
    )
    return torch.cat([ee_pos_b, ee_quat_b], dim=-1)
