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
