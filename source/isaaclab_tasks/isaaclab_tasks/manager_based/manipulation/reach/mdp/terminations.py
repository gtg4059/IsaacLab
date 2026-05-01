# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

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


def reach_satisfied(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="ee_link"),
    max_distance: float = 0.05,
    max_angle_rad: float = 0.1,
    max_lin_vel: float = 0.01,
) -> torch.Tensor:
    """Return whether each env satisfies the reach criteria (pose + low EE linear speed).

    Same geometry as :func:`reach` termination; use for rewards / events without ending the episode.
    ``quat_error_magnitude`` is in radians; ``max_angle_rad`` is a direct radian bound.
    ``max_lin_vel`` is the max L2 norm of EE linear velocity in world frame (m/s).
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

    pose_ok = torch.logical_and(distance <= max_distance, quat_err_rad <= max_angle_rad)
    vel_ok = lin_spd <= max_lin_vel
    return torch.logical_and(pose_ok, vel_ok)


def reach(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="ee_link"),
    max_distance: float = 0.05,
    max_angle_rad: float = 0.1,
    max_lin_vel: float = 0.01,
) -> torch.Tensor:
    """Episode termination when pose error and EE speed satisfy :func:`reach_satisfied`."""
    return reach_satisfied(
        env,
        command_name,
        asset_cfg,
        max_distance=max_distance,
        max_angle_rad=max_angle_rad,
        max_lin_vel=max_lin_vel,
    )