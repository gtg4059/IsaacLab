# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _body_idx_single(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> int:
    """Body index for EE-style terms."""
    if isinstance(asset_cfg.body_ids, list):
        return int(asset_cfg.body_ids[0])
    if isinstance(asset_cfg.body_ids, int):
        return int(asset_cfg.body_ids)
    entity: RigidObject = env.scene[asset_cfg.name]
    if asset_cfg.body_names is not None:
        keys = [asset_cfg.body_names] if isinstance(asset_cfg.body_names, str) else list(asset_cfg.body_names)
        ids, _ = entity.find_bodies(keys, preserve_order=asset_cfg.preserve_order)
        return int(ids[0])
    return 0


def reach_success_criteria(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    max_distance: float,
    max_angle_rad: float,
    max_lin_vel: float,
    max_ang_vel: float,
    max_lin_acc: float,
    max_ang_acc: float,
    command_b: torch.Tensor | None = None,
) -> torch.Tensor:
    """Boolean (num_envs,): EE meets pose, velocity, and acceleration tolerances."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = command_b if command_b is not None else env.command_manager.get_command(command_name)
    bid = _body_idx_single(env, asset_cfg)

    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, bid]
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)

    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, bid]
    quat_err_rad = quat_error_magnitude(curr_quat_w, des_quat_w)
    lin_spd = torch.norm(asset.data.body_lin_vel_w[:, bid, :], dim=-1)
    ang_spd = torch.norm(asset.data.body_ang_vel_w[:, bid, :], dim=-1)
    lin_acc = torch.norm(asset.data.body_lin_acc_w[:, bid, :], dim=-1)
    ang_acc = torch.norm(asset.data.body_ang_acc_w[:, bid, :], dim=-1)

    pose_ok = torch.logical_and(distance <= max_distance, quat_err_rad <= max_angle_rad)
    vel_ok = torch.logical_and(lin_spd <= max_lin_vel, ang_spd <= max_ang_vel)
    acc_ok = torch.logical_and(lin_acc <= max_lin_acc, ang_acc <= max_ang_acc)
    return torch.logical_and(torch.logical_and(pose_ok, vel_ok), acc_ok)


def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    bid = _body_idx_single(env, asset_cfg)
    curr_pos_w = asset.data.body_pos_w[:, bid]
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_orientation_command_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of position and orientation using exponential kernels."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    bid = _body_idx_single(env, asset_cfg)
    curr_pos_w = asset.data.body_pos_w[:, bid]
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)

    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, bid]
    return torch.exp(-2 * distance) * torch.exp(-0.5 * quat_error_magnitude(curr_quat_w, des_quat_w))


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    bid = _body_idx_single(env, asset_cfg)
    curr_pos_w = asset.data.body_pos_w[:, bid]
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    bid = _body_idx_single(env, asset_cfg)
    curr_quat_w = asset.data.body_quat_w[:, bid]
    return quat_error_magnitude(curr_quat_w, des_quat_w)
