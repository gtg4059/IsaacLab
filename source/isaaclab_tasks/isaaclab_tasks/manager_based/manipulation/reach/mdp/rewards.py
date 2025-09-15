# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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


def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)

def dual_position_command_error_left(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "dual_ee_pose",
) -> torch.Tensor:
    """Penalize tracking of the left hand position error using L2-norm.

    The function computes the position error between the desired position (from the dual command) and the
    current position of the left hand (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    dual_command = env.command_manager.get_command(command_name)
    
    # get left command (first 7 elements)
    left_command = dual_command[:, :7]
    
    # obtain the desired position for left hand
    left_des_pos_b = left_command[:, :3]
    
    # transform desired position to world frame
    left_des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, left_des_pos_b)
    
    # get current position for left hand
    left_curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    
    # compute position error for left hand
    left_pos_error = torch.norm(left_curr_pos_w - left_des_pos_w, dim=1)
    
    return left_pos_error

def dual_position_command_error_right(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "dual_ee_pose",
) -> torch.Tensor:
    """Penalize tracking of the right hand position error using L2-norm.

    The function computes the position error between the desired position (from the dual command) and the
    current position of the right hand (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    dual_command = env.command_manager.get_command(command_name)
    
    # get right command (next 7 elements)
    right_command = dual_command[:, 7:]
    
    # obtain the desired position for right hand
    right_des_pos_b = right_command[:, :3]
    
    # transform desired position to world frame
    right_des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, right_des_pos_b)
    
    # get current position for right hand
    right_curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    
    # compute position error for right hand
    right_pos_error = torch.norm(right_curr_pos_w - right_des_pos_w, dim=1)
    
    return right_pos_error

def dual_position_command_error_tanh_left(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "dual_ee_pose",
) -> torch.Tensor:
    """Reward tracking of the left hand position using the tanh kernel.

    The function computes the position error between the desired position (from the dual command) and the
    current position of the left hand (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    dual_command = env.command_manager.get_command(command_name)
    
    # get left command (first 7 elements)
    left_command = dual_command[:, :7]
    
    # obtain the desired position for left hand
    left_des_pos_b = left_command[:, :3]
    
    # transform desired position to world frame
    left_des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, left_des_pos_b)
    
    # get current position for left hand
    left_curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    
    # compute position error for left hand
    left_distance = torch.norm(left_curr_pos_w - left_des_pos_w, dim=1)
    
    # apply tanh kernel
    left_reward = 1 - torch.tanh(left_distance / std)
    
    return left_reward

def dual_position_command_error_tanh_right(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "dual_ee_pose",
) -> torch.Tensor:
    """Reward tracking of the right hand position using the tanh kernel.

    The function computes the position error between the desired position (from the dual command) and the
    current position of the right hand (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    dual_command = env.command_manager.get_command(command_name)
    
    # get right command (next 7 elements)
    right_command = dual_command[:, 7:]
    
    # obtain the desired position for right hand
    right_des_pos_b = right_command[:, :3]
    
    # transform desired position to world frame
    right_des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, right_des_pos_b)
    
    # get current position for right hand
    right_curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    
    # compute position error for right hand
    right_distance = torch.norm(right_curr_pos_w - right_des_pos_w, dim=1)
    
    # apply tanh kernel
    right_reward = 1 - torch.tanh(right_distance / std)
    
    return right_reward


def dual_orientation_command_error_left(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "dual_ee_pose",
) -> torch.Tensor:
    """Penalize tracking left hand orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the dual command) and the
    current orientation of the left hand (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    dual_command = env.command_manager.get_command(command_name)
    
    # get left command (first 7 elements)
    left_command = dual_command[:, :7]
    
    # obtain the desired orientation for left hand
    left_des_quat_b = left_command[:, 3:7]
    
    # transform desired orientation to world frame
    left_des_quat_w = quat_mul(asset.data.root_quat_w, left_des_quat_b)
    
    # get current orientation for left hand
    left_curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    
    # compute orientation error for left hand
    left_orientation_error = quat_error_magnitude(left_curr_quat_w, left_des_quat_w)
    
    return left_orientation_error

def dual_orientation_command_error_right(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "dual_ee_pose",
) -> torch.Tensor:
    """Penalize tracking right hand orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the dual command) and the
    current orientation of the right hand (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    dual_command = env.command_manager.get_command(command_name)
    
    # get right command (next 7 elements)
    right_command = dual_command[:, 7:]
    
    # obtain the desired orientation for right hand
    right_des_quat_b = right_command[:, 3:7]
    
    # transform desired orientation to world frame
    right_des_quat_w = quat_mul(asset.data.root_quat_w, right_des_quat_b)
    
    # get current orientation for right hand
    right_curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    
    # compute orientation error for right hand
    right_orientation_error = quat_error_magnitude(right_curr_quat_w, right_des_quat_w)
    
    return right_orientation_error