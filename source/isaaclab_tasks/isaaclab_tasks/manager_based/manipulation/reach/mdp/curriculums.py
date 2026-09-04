# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum terms for CRI reach training."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Unified curriculum schedule (env steps; 1 PPO iter = 24 steps).
REACH_CRITERIA_HOLD_END = 24 * 3000
REACH_CRITERIA_DECAY_STEPS = 24 * 4000
CRI_OVF_REWARD_START = 24 * 7000
CRI_OVF_CROSSFADE_START = 24 * 15000
CRI_OVF_THRESHOLD_FINAL = 0.96
CRI_OVF_THRESHOLD_INITIAL = 2.0
TERM_PENALTY_INITIAL_WEIGHT = -200.0
TERM_PENALTY_FINAL_WEIGHT = -2000.0
TERM_PENALTY_SWITCH_STEP = 24 * 40000

_REACH_CRITERIA_PARAM_KEYS = (
    "max_distance",
    "max_angle_rad",
    "max_lin_vel",
    "max_ang_vel",
    "max_lin_acc",
    "max_ang_acc",
)


class reach_success_criteria_curriculum(ManagerTermBase):
    """Tighten reach success thresholds for bonus/penalty rewards (and optional event).

    Reward/event term configs hold the **final** (strictest) thresholds. This curriculum starts from
    relaxed values (``ease_factor`` times final).

    * Default: linear ``ease_factor → 1`` over ``num_steps`` after ``start_step``.
    * Piecewise (when ``mid_factor`` is set): ``ease_factor → mid_factor`` over ``num_steps``,
      hold ``mid_factor`` for ``hold_steps``, then ``mid_factor → 1`` over ``num_steps_final``.

    Set ``event_term_name`` to ``None`` when reach success ends the episode (no in-episode resample).
    ``terminations.reach_success`` reads thresholds from the reward term, so it tracks automatically.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        reward_term_names = cfg.params.get("reward_term_names")
        if reward_term_names is None:
            reward_term_names = [cfg.params.get("reward_term_name", "reach_success_bonus")]
        self._reward_term_names = list(reward_term_names)
        # Default keeps legacy in-episode resample; pass None for termination-on-reach envs.
        self._event_term_name = cfg.params.get("event_term_name", "resample_ee_pose_on_reach")
        self._reward_term_cfgs = {
            name: env.reward_manager.get_term_cfg(name) for name in self._reward_term_names
        }
        self._event_term_cfg = (
            env.event_manager.get_term_cfg(self._event_term_name) if self._event_term_name else None
        )

        primary_cfg = self._reward_term_cfgs[self._reward_term_names[0]]
        self._final_params = {key: primary_cfg.params[key] for key in _REACH_CRITERIA_PARAM_KEYS}
        self._ease_factor = cfg.params.get("ease_factor", 5.0)
        self._mid_factor = cfg.params.get("mid_factor")
        initial_override = cfg.params.get("initial_params")
        if initial_override is not None:
            self._initial_params = {
                key: initial_override.get(key, value * self._ease_factor)
                for key, value in self._final_params.items()
            }
        else:
            self._initial_params = {key: value * self._ease_factor for key, value in self._final_params.items()}
        self._mid_params = (
            {key: value * self._mid_factor for key, value in self._final_params.items()}
            if self._mid_factor is not None
            else None
        )

        self._start_step = cfg.params.get("start_step", 0)
        self._num_steps = cfg.params["num_steps"]
        self._hold_steps = cfg.params.get("hold_steps", 0)
        self._num_steps_final = cfg.params.get("num_steps_final", self._num_steps)
        self._last_params_key: tuple[float, ...] | None = None

        self._apply_params(self._initial_params)

    def _lerp(self, start: dict[str, float], end: dict[str, float], progress: float) -> dict[str, float]:
        return {key: start[key] + progress * (end[key] - start[key]) for key in _REACH_CRITERIA_PARAM_KEYS}

    def _interpolate_params(self, step: int) -> dict[str, float]:
        if step < self._start_step:
            return dict(self._initial_params)

        if self._mid_params is None:
            progress = min(1.0, (step - self._start_step) / max(self._num_steps, 1))
            return self._lerp(self._initial_params, self._final_params, progress)

        first_end = self._start_step + self._num_steps
        hold_end = first_end + self._hold_steps
        second_end = hold_end + self._num_steps_final

        if step < first_end:
            progress = min(1.0, (step - self._start_step) / max(self._num_steps, 1))
            return self._lerp(self._initial_params, self._mid_params, progress)
        if step < hold_end:
            return dict(self._mid_params)
        if step < second_end:
            progress = min(1.0, (step - hold_end) / max(self._num_steps_final, 1))
            return self._lerp(self._mid_params, self._final_params, progress)
        return dict(self._final_params)

    def _apply_params(self, params: dict[str, float]) -> None:
        params_key = tuple(round(params[key], 8) for key in _REACH_CRITERIA_PARAM_KEYS)
        if params_key == self._last_params_key:
            return

        env = self._env
        for key, value in params.items():
            for term_cfg in self._reward_term_cfgs.values():
                term_cfg.params[key] = value
            if self._event_term_cfg is not None:
                self._event_term_cfg.params[key] = value
        for name, term_cfg in self._reward_term_cfgs.items():
            env.reward_manager.set_term_cfg(name, term_cfg)
        if self._event_term_name is not None and self._event_term_cfg is not None:
            env.event_manager.set_term_cfg(self._event_term_name, self._event_term_cfg)
        self._last_params_key = params_key

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        num_steps: int,
        ease_factor: float = 5.0,
        start_step: int = 0,
        reward_term_name: str = "reach_success_bonus",
        reward_term_names: list[str] | None = None,
        event_term_name: str | None = "resample_ee_pose_on_reach",
        initial_params: dict[str, float] | None = None,
        mid_factor: float | None = None,
        hold_steps: int = 0,
        num_steps_final: int | None = None,
    ) -> dict[str, float]:
        params = self._interpolate_params(env.common_step_counter)
        self._apply_params(params)
        return params


class modify_reward_weight_linear(ManagerTermBase):
    """Curriculum that linearly interpolates a reward weight between two values over ``num_steps``."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._term_name = cfg.params["term_name"]
        self._term_cfg = env.reward_manager.get_term_cfg(self._term_name)
        self._initial_weight = cfg.params.get("initial_weight", self._term_cfg.weight)
        self._final_weight = cfg.params.get("final_weight", cfg.params.get("max_weight"))
        if self._final_weight is None:
            raise ValueError("Either 'final_weight' or 'max_weight' must be set in curriculum params.")
        self._start_step = cfg.params.get("start_step", 0)
        self._num_steps = cfg.params["num_steps"]
        self._last_weight: float | None = None

        self._apply_weight(self._initial_weight)

    def _compute_weight(self, step: int) -> float:
        if step < self._start_step:
            return self._initial_weight

        progress = min(1.0, (step - self._start_step) / self._num_steps)
        return self._initial_weight + progress * (self._final_weight - self._initial_weight)

    def _apply_weight(self, weight: float) -> None:
        if weight == self._last_weight:
            return
        self._term_cfg.weight = weight
        self._env.reward_manager.set_term_cfg(self._term_name, self._term_cfg)
        self._last_weight = weight

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_name: str,
        num_steps: int,
        initial_weight: float | None = None,
        final_weight: float | None = None,
        max_weight: float | None = None,
        start_step: int = 0,
    ) -> float:
        weight = self._compute_weight(env.common_step_counter)
        self._apply_weight(weight)
        return weight


def _linear_schedule(step: int, start_step: int, num_steps: int, initial_value: float, final_value: float) -> float:
    if step < start_step:
        return initial_value
    progress = min(1.0, (step - start_step) / num_steps)
    return initial_value + progress * (final_value - initial_value)


def linear_interpolate_by_step(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    data: float,
    initial_value: float,
    final_value: float,
    start_step: int,
    num_steps: int,
) -> float:
    """``modify_term_cfg`` helper that linearly interpolates a scalar by ``common_step_counter``."""
    return _linear_schedule(env.common_step_counter, start_step, num_steps, initial_value, final_value)


def cri_ovf_reward_weight_by_step(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    data: float,
    reward_start: int = CRI_OVF_REWARD_START,
    crossfade_start: int = CRI_OVF_CROSSFADE_START,
    initial_weight: float = -20.0,
    peak_weight: float = -800.0,
) -> float:
    """Step CRI_OVF reward weight: initial, then peak at reward_start, then 0 after crossfade."""
    step = env.common_step_counter
    if step < reward_start:
        return initial_weight
    if step < crossfade_start:
        return peak_weight
    return 0.0


def cri_ovf_threshold_by_step(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    data: float,
    crossfade_start: int = CRI_OVF_CROSSFADE_START,
    threshold_initial: float = CRI_OVF_THRESHOLD_INITIAL,
    threshold_final: float = CRI_OVF_THRESHOLD_FINAL,
) -> float:
    """Step OVF termination threshold: hold initial, then jump to final at crossfade_start."""
    if env.common_step_counter < crossfade_start:
        return threshold_initial
    return threshold_final


def command_range_step_by_step(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    data: tuple[float, float],
    switch_step: int,
    initial_range: tuple[float, float],
    final_range: tuple[float, float],
) -> tuple[float, float]:
    """Hold ``initial_range`` until ``switch_step``, then jump to ``final_range``."""
    if env.common_step_counter < switch_step:
        return (float(initial_range[0]), float(initial_range[1]))
    return (float(final_range[0]), float(final_range[1]))


def reward_weight_step_by_step(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    data: float,
    switch_step: int,
    initial_weight: float,
    final_weight: float,
) -> float:
    """Hold ``initial_weight`` until ``switch_step``, then jump to ``final_weight``."""
    if env.common_step_counter < switch_step:
        return initial_weight
    return final_weight


def hold_steps_by_step(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    data: int,
    switch_step: int,
    initial_hold_steps: int = 1,
    final_hold_steps: int = 8,
    interval_steps: int | None = None,
    increment: int = 2,
) -> int:
    """Hold ``initial_hold_steps`` until ``switch_step``, then raise toward ``final_hold_steps``.

    * ``interval_steps`` unset or ``<= 0``: jump once to ``final_hold_steps``.
    * Otherwise: at ``switch_step`` add ``increment``, then again every ``interval_steps``.

    This only changes the success gate. :class:`~isaaclab_tasks.manager_based.manipulation.reach.mdp.rewards.ReachSuccessCriteria`
    scales the bonus from ``hold_count`` (new max / ``hold_reward_ref``, plus stay).
    """
    initial = int(initial_hold_steps)
    final = int(final_hold_steps)
    if env.common_step_counter < switch_step:
        return initial
    if interval_steps is None or interval_steps <= 0:
        return final
    stages = (env.common_step_counter - switch_step) // int(interval_steps) + 1
    return int(min(final, initial + stages * int(increment)))


def termination_penalty_weight_by_step(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    data: float,
    switch_step: int = TERM_PENALTY_SWITCH_STEP,
    initial_weight: float = TERM_PENALTY_INITIAL_WEIGHT,
    final_weight: float = TERM_PENALTY_FINAL_WEIGHT,
) -> float:
    """Step termination penalty weight: hold initial, then jump to final at ``switch_step``."""
    return reward_weight_step_by_step(env, env_ids, data, switch_step, initial_weight, final_weight)
