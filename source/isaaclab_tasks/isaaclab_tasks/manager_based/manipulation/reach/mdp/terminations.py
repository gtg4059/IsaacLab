# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def CRI_OVF(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    result,_ = torch.max(asset.data.CRI,dim=1)
    return result>1


def reach(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="ee_link"),
    max_distance: float = 0.05,
    max_angle_deg: float = 5.7,
    max_lin_vel: float = 0.02,
) -> torch.Tensor:
    """Terminate when pose error is below thresholds and EE linear speed is low (task success).

    Position and quaternion errors follow the same transforms as
    :func:`position_orientation_command_error_fine_grained` in ``mdp.rewards``.
    ``quat_error_magnitude`` is in radians; ``max_angle_deg`` is converted internally.
    ``max_lin_vel`` is the maximum allowed L2 norm of the body linear velocity in world frame (m/s).
    Set ``max_lin_vel`` to a very large value to ignore velocity (legacy pose-only success).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)

    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    quat_err_rad = quat_error_magnitude(curr_quat_w, des_quat_w)

    bid = asset_cfg.body_ids[0]  # type: ignore
    lin_spd = torch.norm(asset.data.body_lin_vel_w[:, bid, :], dim=-1)

    max_angle_rad = math.radians(max_angle_deg)
    pose_ok = torch.logical_and(distance <= max_distance, quat_err_rad <= max_angle_rad)
    vel_ok = lin_spd <= max_lin_vel
    return torch.logical_and(pose_ok, vel_ok)