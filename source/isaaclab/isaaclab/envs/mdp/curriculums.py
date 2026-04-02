# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def modify_reward_weight(env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, weight: float, num_steps: int):
    """Curriculum that modifies a reward weight a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        weight: The weight of the reward term.
        num_steps: The number of steps after which the change should be applied.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.weight = weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)


def modify_reward_weight_linear(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    weight_start: float,
    weight_end: float,
    num_steps: int,
    hold_steps: int = 0,
) -> float:
    """Curriculum that linearly interpolates a reward weight over training steps.

    Uses :attr:`ManagerBasedRLEnv.common_step_counter` as the step index.

    * For ``step <= 0`` or ``step < hold_steps``: weight stays at ``weight_start``.
    * For ``hold_steps <= step < hold_steps + num_steps``: weight moves linearly from
      ``weight_start`` to ``weight_end``.
    * For ``step >= hold_steps + num_steps``: weight stays at ``weight_end``.

    If ``hold_steps`` is 0, behavior matches a ramp from step 0 over ``num_steps`` (then hold at end).

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        weight_start: Reward weight during the hold phase (should match the term's initial config weight).
        weight_end: Reward weight at the end of the ramp and held afterward.
        num_steps: Length of the linear ramp in environment steps. Must be at least 1.
        hold_steps: Keep ``weight_start`` for this many steps before the ramp begins. Defaults to 0.

    Returns:
        The reward weight applied this update (for logging).
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}.")
    if hold_steps < 0:
        raise ValueError(f"hold_steps must be >= 0, got {hold_steps}.")
    step = env.common_step_counter
    ramp_end = hold_steps + num_steps
    if step <= 0 or step < hold_steps:
        weight = weight_start
    elif step >= ramp_end:
        weight = weight_end
    else:
        t = (step - hold_steps) / num_steps
        weight = weight_start + t * (weight_end - weight_start)
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    term_cfg.weight = weight
    env.reward_manager.set_term_cfg(term_name, term_cfg)
    return weight
