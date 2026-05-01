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
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


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


def ee_pose_error_to_command(
    env: ManagerBasedEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Pose error between the end effector and the target command, expressed in the robot base frame.

    Returns a 6-dim vector: [pos_error (3), axis_angle_error (3)].
    Both are computed in the base frame so the policy observes a consistent, robot-centric error signal.

    Args:
        env: The environment instance.
        command_name: Name of the pose command (must provide pos_b[:3] + quat_b[3:7]).
        asset_cfg: Scene entity config for the robot; first body_id is used as end effector.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # --- current EE pose in base frame ---
    root_pos_w = asset.data.root_pos_w
    root_quat_w = asset.data.root_quat_w
    ee_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    ee_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
    )

    # --- target pose in base frame (from command) ---
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_quat_b = command[:, 3:7]

    # --- position error: target - current (base frame) ---
    pos_error = des_pos_b - ee_pos_b

    # --- orientation error: relative rotation from EE to target, as axis-angle (3-dim) ---
    # q_err = q_des * q_ee^{-1}  (expressed in base frame)
    quat_error = math_utils.quat_mul(des_quat_b, math_utils.quat_inv(ee_quat_b))
    axis_angle_error = math_utils.axis_angle_from_quat(quat_error)

    return torch.cat([pos_error, axis_angle_error], dim=-1)

def collision_risk_index(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.CRI

def CRI_optimize(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    # max over joints → shape (num_envs,); scalar distance to target CRI, not vector norm over dim=1
    result, _ = torch.max(asset.data.CRI, dim=1)
    distance = (0.97-result).abs()
    return torch.exp(-5.0 * distance)

