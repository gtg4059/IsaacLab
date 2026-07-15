# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reach-task-specific MDP event functions."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

from .rewards import reach_success_criteria


def reset_robot_joints_two_groups_by_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    primary_joint_names: str | list[str],
    primary_position_range: tuple[float, float],
    primary_velocity_range: tuple[float, float],
    secondary_joint_names: str | list[str],
    secondary_position_range: tuple[float, float],
    secondary_velocity_range: tuple[float, float],
):
    """Reset joint state from defaults with independent uniform offsets for two joint groups."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    n_env = joint_pos.shape[0]
    device = joint_pos.device

    def _names(names: str | list[str]) -> list[str]:
        return [names] if isinstance(names, str) else list(names)

    p_ids, _ = asset.find_joints(_names(primary_joint_names), preserve_order=True)
    s_ids, _ = asset.find_joints(_names(secondary_joint_names), preserve_order=True)

    if len(p_ids) > 0:
        j_p = torch.tensor(p_ids, device=device, dtype=torch.long)
        joint_pos[:, j_p] += math_utils.sample_uniform(*primary_position_range, (n_env, len(p_ids)), device)
        joint_vel[:, j_p] += math_utils.sample_uniform(*primary_velocity_range, (n_env, len(p_ids)), device)

    if len(s_ids) > 0:
        j_s = torch.tensor(s_ids, device=device, dtype=torch.long)
        joint_pos[:, j_s] += math_utils.sample_uniform(*secondary_position_range, (n_env, len(s_ids)), device)
        joint_vel[:, j_s] += math_utils.sample_uniform(*secondary_velocity_range, (n_env, len(s_ids)), device)

    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    asset.set_joint_position_target(joint_pos, env_ids=env_ids)
    asset.set_joint_velocity_target(joint_vel, env_ids=env_ids)


def reset_robot_joints_three_groups_by_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    primary_joint_names: str | list[str],
    primary_position_range: tuple[float, float],
    primary_velocity_range: tuple[float, float],
    secondary_joint_names: str | list[str],
    secondary_position_range: tuple[float, float],
    secondary_velocity_range: tuple[float, float],
    tertiary_joint_names: str | list[str],
    tertiary_position_range: tuple[float, float],
    tertiary_velocity_range: tuple[float, float],
):
    """Reset joint state from defaults with independent uniform offsets for three joint groups."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    n_env = joint_pos.shape[0]
    device = joint_pos.device

    def _names(names: str | list[str]) -> list[str]:
        return [names] if isinstance(names, str) else list(names)

    p_ids, _ = asset.find_joints(_names(primary_joint_names), preserve_order=True)
    s_ids, _ = asset.find_joints(_names(secondary_joint_names), preserve_order=True)
    t_ids, _ = asset.find_joints(_names(tertiary_joint_names), preserve_order=True)

    if len(p_ids) > 0:
        j_p = torch.tensor(p_ids, device=device, dtype=torch.long)
        joint_pos[:, j_p] += math_utils.sample_uniform(*primary_position_range, (n_env, len(p_ids)), device)
        joint_vel[:, j_p] += math_utils.sample_uniform(*primary_velocity_range, (n_env, len(p_ids)), device)

    if len(s_ids) > 0:
        j_s = torch.tensor(s_ids, device=device, dtype=torch.long)
        joint_pos[:, j_s] += math_utils.sample_uniform(*secondary_position_range, (n_env, len(s_ids)), device)
        joint_vel[:, j_s] += math_utils.sample_uniform(*secondary_velocity_range, (n_env, len(s_ids)), device)

    if len(t_ids) > 0:
        j_t = torch.tensor(t_ids, device=device, dtype=torch.long)
        joint_pos[:, j_t] += math_utils.sample_uniform(*tertiary_position_range, (n_env, len(t_ids)), device)
        joint_vel[:, j_t] += math_utils.sample_uniform(*tertiary_velocity_range, (n_env, len(t_ids)), device)

    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    asset.set_joint_position_target(joint_pos, env_ids=env_ids)
    asset.set_joint_velocity_target(joint_vel, env_ids=env_ids)


def reset_robot_joints_by_name_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    position_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]] | None = None,
):
    """Reset joint state from defaults with a per-joint uniform offset.

    Args:
        position_range: Mapping from joint name to ``(min, max)`` position offset.
        velocity_range: Mapping from joint name to ``(min, max)`` velocity offset.
            Joints missing from this dict default to ``(0.0, 0.0)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    n_env = joint_pos.shape[0]
    device = joint_pos.device
    if velocity_range is None:
        velocity_range = {}

    joint_names = list(position_range.keys())
    joint_ids, resolved_names = asset.find_joints(joint_names, preserve_order=True)
    if len(joint_ids) != len(joint_names):
        raise ValueError(
            "Could not resolve all joints in position_range."
            f" Requested={joint_names}, resolved={resolved_names}."
        )

    for joint_id, joint_name in zip(joint_ids, resolved_names):
        pos_lo, pos_hi = position_range[joint_name]
        vel_lo, vel_hi = velocity_range.get(joint_name, (0.0, 0.0))
        joint_pos[:, joint_id] += math_utils.sample_uniform(pos_lo, pos_hi, (n_env,), device)
        joint_vel[:, joint_id] += math_utils.sample_uniform(vel_lo, vel_hi, (n_env,), device)

    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    asset.set_joint_position_target(joint_pos, env_ids=env_ids)
    asset.set_joint_velocity_target(joint_vel, env_ids=env_ids)


def resample_ee_pose_command_on_reach(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    max_distance: float,
    max_angle_rad: float,
    max_lin_vel: float,
    max_ang_vel: float,
    max_lin_acc: float,
    max_ang_acc: float,
) -> None:
    """Resample the pose command for environments that satisfy reach success criteria."""
    from isaaclab.envs import ManagerBasedRLEnv

    if not isinstance(env, ManagerBasedRLEnv):
        return
    sat = reach_success_criteria(
        env,
        command_name,
        asset_cfg,
        max_distance=max_distance,
        max_angle_rad=max_angle_rad,
        max_lin_vel=max_lin_vel,
        max_ang_vel=max_ang_vel,
        max_lin_acc=max_lin_acc,
        max_ang_acc=max_ang_acc,
    )
    if env_ids is not None:
        reached = env_ids[sat[env_ids]]
    else:
        reached = torch.where(sat)[0]
    if reached.numel() == 0:
        return
    term = env.command_manager.get_term(command_name)
    term.reset(env_ids=reached)
