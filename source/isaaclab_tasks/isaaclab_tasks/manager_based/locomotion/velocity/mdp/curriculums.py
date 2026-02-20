# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def terrain_levels_vel_after_steps(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    min_steps: int = 20000,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terrain level curriculum that activates only after ``min_steps`` environment steps.

    Before ``min_steps``, terrain levels are not updated (stays at initial difficulty).
    After ``min_steps``, behaves identically to :func:`terrain_levels_vel`.

    Returns:
        The mean terrain level for the given environment ids.
    """
    terrain = env.scene.terrain
    assert terrain is not None, "Terrain curriculum requires scene.terrain."
    if env.common_step_counter < min_steps:
        return torch.mean(terrain.terrain_levels.float())
    return terrain_levels_vel(env, env_ids, asset_cfg)


def terrain_levels_step_schedule(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    step_interval: int = 10000,
    percent_per_interval: float = 0.1,
    min_steps: int = 0,
) -> torch.Tensor:
    """Terrain level curriculum that increases difficulty by a fixed percent of max every N steps.

    Ignores robot distance; every ``step_interval`` steps the target level increases by
    ``percent_per_interval * max_terrain_level`` (e.g. 10% every 10000 steps).
    All envs are set to the same target level.

    Returns:
        The mean terrain level (for logging).
    """
    t = env.scene.terrain
    assert t is not None, "Terrain curriculum requires scene.terrain."
    max_level = t.max_terrain_level
    if env.common_step_counter < min_steps:
        return torch.mean(t.terrain_levels.float())
    # target level: every step_interval add (percent_per_interval * max_level) levels, capped at max_level-1
    steps_effective = env.common_step_counter - min_steps
    num_intervals = steps_effective // step_interval
    levels_per_interval = max(1, round(percent_per_interval * max_level))
    target_level = min(max_level - 1, num_intervals * levels_per_interval)
    # set all envs to same target level and sync origins (global schedule)
    t.terrain_levels[:] = target_level
    t.env_origins[:] = t.terrain_origins[t.terrain_levels.long(), t.terrain_types]
    return torch.mean(t.terrain_levels.float())
