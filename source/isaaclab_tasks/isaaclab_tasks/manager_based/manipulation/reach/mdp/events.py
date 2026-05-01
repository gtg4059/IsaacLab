# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
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
    from isaaclab.envs import ManagerBasedEnv

from isaaclab.envs import ManagerBasedRLEnv

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
    """Reset joint state from defaults with independent uniform offsets for two joint groups.

    Applies offsets in one write so the two groups do not overwrite each other (unlike chaining
    :func:`isaaclab.envs.mdp.events.reset_joints_by_offset` twice).
    """
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
        joint_pos[:, j_p] += math_utils.sample_uniform(
            *primary_position_range, (n_env, len(p_ids)), device
        )
        joint_vel[:, j_p] += math_utils.sample_uniform(
            *primary_velocity_range, (n_env, len(p_ids)), device
        )

    if len(s_ids) > 0:
        j_s = torch.tensor(s_ids, device=device, dtype=torch.long)
        joint_pos[:, j_s] += math_utils.sample_uniform(
            *secondary_position_range, (n_env, len(s_ids)), device
        )
        joint_vel[:, j_s] += math_utils.sample_uniform(
            *secondary_velocity_range, (n_env, len(s_ids)), device
        )

    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


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
    """If an env satisfies reach criteria, resample that env's pose command (new target).

    Intended for ``EventTermCfg`` with ``mode="interval"`` and a short ``interval_range_s`` so this
    runs every control step *after* :meth:`CommandManager.compute`, keeping the episode alive while
    advancing goals on success.
    """
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
        sat_subset = sat[env_ids]
        reached = env_ids[sat_subset]
    else:
        reached = torch.where(sat)[0]
    if reached.numel() == 0:
        return
    term = env.command_manager.get_term(command_name)
    term.reset(env_ids=reached)
