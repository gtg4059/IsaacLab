from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class modify_reward_weight(ManagerTermBase):
    """Curriculum that modifies the reward weight based on a step-wise schedule."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # obtain term configuration
        term_name = cfg.params["term_name"]
        self._term_cfg = env.reward_manager.get_term_cfg(term_name)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_name: str,
        weight: float,
        num_steps: int,
    ) -> float:
        # update term settings
        if env.common_step_counter > num_steps:
            self._term_cfg.weight = weight
            env.reward_manager.set_term_cfg(term_name, self._term_cfg)

        return self._term_cfg.weight
    
class modify_reward_weight_linear_decay(ManagerTermBase):
    """Curriculum that linearly decays reward weight from initial to final over num_steps.

    Weight behavior:
    - Before min_steps: weight = 0
    - After min_steps: linearly decays from initial_weight to final_weight over num_steps
    - Weight at step t (after min_steps): initial_weight + (final_weight - initial_weight) * min(1, (t - min_steps) / num_steps)
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        term_name = cfg.params["term_name"]
        self._term_cfg = env.reward_manager.get_term_cfg(term_name)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_name: str,
        initial_weight: float,
        final_weight: float = 0.0,
        num_steps: int = 1,
        min_steps: int = 0,
    ) -> float:
        # min_steps 이전에는 weight = 0 유지
        if env.common_step_counter < min_steps:
            weight = 0.0
        else:
            # min_steps 이후부터 num_steps 동안 initial_weight에서 final_weight로 감쇠
            steps_elapsed = env.common_step_counter - min_steps
            progress = min(1.0, steps_elapsed / max(1, num_steps))
            weight = initial_weight + (final_weight - initial_weight) * progress
        
        self._term_cfg.weight = float(weight)
        env.reward_manager.set_term_cfg(term_name, self._term_cfg)
        return self._term_cfg.weight

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