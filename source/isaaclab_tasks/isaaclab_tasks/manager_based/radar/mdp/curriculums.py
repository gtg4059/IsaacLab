# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.terrains import TerrainImporter


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.linalg.norm(asset.data.root_pos_w.torch[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.linalg.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())

def terrain_levels_vel_fix(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_ratio: float = 0.5,
    distance_buffer: float = 0.5,
    standing_vel_eps: float = 0.05,
) -> torch.Tensor:
    """Terrain curriculum with command-scaled promotion/demotion and a sub-terrain distance cap.

    Called on episode reset (before ``episode_length_buf`` is cleared). Uses the elapsed episode
    time per env and linear velocity command ``[:, :2]`` (terrain indicator excluded).

    Symmetric threshold ``distance_ratio`` on expected travel
    ``||cmd_xy|| * episode_time``:

    - **move_up**: ``distance >= expected * distance_ratio`` and
      ``distance <= size[0]/2 - distance_buffer`` (stay inside the sub-terrain cell).
    - **move_down**: ``distance < expected * distance_ratio``.
    - **Standing envs** (``is_standing_env`` or ``||cmd_xy|| < standing_vel_eps``): no level change.
    - Only envs with terrain command indicator ``== 1`` (``command[:, 3]``) are updated; others keep level.

    Returns:
        The mean terrain level for logging.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    env_ids_t = _resolve_curriculum_env_ids(env, env_ids)

    command = env.command_manager.get_command("base_velocity")
    cmd_speed = torch.norm(command[env_ids_t, :2], dim=1)
    # elapsed time in the episode that just ended (curriculum runs before buf reset)
    episode_time_s = env.episode_length_buf[env_ids_t].float() * env.step_dt
    distance = torch.norm(asset.data.root_pos_w[env_ids_t, :2] - env.scene.env_origins[env_ids_t, :2], dim=1)

    expected_distance = cmd_speed * episode_time_s

    is_standing = cmd_speed < standing_vel_eps
    cmd_term = env.command_manager.get_term("base_velocity")
    if hasattr(cmd_term, "is_standing_env"):
        is_standing = torch.logical_or(is_standing, cmd_term.is_standing_env[env_ids_t])

    sub_terrain_limit = max(
        distance_buffer, terrain.cfg.terrain_generator.size[0] * 0.9 - distance_buffer
    )
    distance_threshold = expected_distance * distance_ratio

    # ``UniformVelocityTerrainCommand``: ``command[:, 3]`` is 1.0 on ``terrain_indicator_one_sub_indices`` cells.
    if command.shape[-1] > 3:
        indicator_active = command[env_ids_t, 3] > 0.5
    else:
        indicator_active = torch.zeros(env_ids_t.numel(), dtype=torch.bool, device=env.device)
    # for i, env_id in enumerate(env_ids_t.tolist()):
    #     if not indicator_active[i]:
    #         continue
    #     print(
    #         f"[terrain_levels_vel_fix] env {env_id}: "
    #         f"expected_distance={expected_distance[i].item():.4f}, distance={distance[i].item():.4f}"
    #     )
    valid_episode = episode_time_s > 1e-6
    move_up = torch.logical_and(distance >= distance_threshold, distance <= sub_terrain_limit)
    move_up = torch.logical_and(move_up, torch.logical_and(~is_standing, valid_episode))
    move_up = torch.logical_and(move_up, indicator_active)

    move_down = torch.logical_and(distance < distance_threshold, ~is_standing)
    move_down = torch.logical_and(move_down, torch.logical_and(~move_up, valid_episode))
    move_down = torch.logical_and(move_down, indicator_active)
    # print(move_up, move_down)
    terrain.update_env_origins(env_ids_t, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())

def _resolve_curriculum_env_ids(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long).reshape(-1)
    return torch.as_tensor(list(env_ids), device=env.device, dtype=torch.long)