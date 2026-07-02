# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def ee_pose_error_to_command(
    env: ManagerBasedEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Pose error between EE and target command in the robot base frame (6D: pos + axis-angle)."""
    asset: Articulation = env.scene[asset_cfg.name]

    root_pos_w = asset.data.root_pos_w
    root_quat_w = asset.data.root_quat_w
    ee_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    ee_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)

    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_quat_b = command[:, 3:7]

    pos_error = des_pos_b - ee_pos_b
    quat_error = math_utils.quat_mul(des_quat_b, math_utils.quat_inv(ee_quat_b))
    axis_angle_error = math_utils.axis_angle_from_quat(quat_error)

    return torch.cat([pos_error, axis_angle_error], dim=-1)


def collision_risk_index(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Collision Risk Index from Safetics CRI solver (via sfd_coreservice)."""
    asset: Articulation = env.scene[asset_cfg.name]
    # print("qd:", asset.data.joint_vel)
    # print("CRI:", asset.data.CRI)
    return asset.data.CRI


def CRI_optimize(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward shaping term that encourages CRI to stay near 0.97."""
    asset: Articulation = env.scene[asset_cfg.name]
    result, _ = torch.max(asset.data.CRI, dim=1)
    distance = (0.97 - result).abs()
    return torch.exp(-5.0 * distance)
