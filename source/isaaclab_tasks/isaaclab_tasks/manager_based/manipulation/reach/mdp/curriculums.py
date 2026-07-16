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
    """Gradually tighten reach success thresholds for the bonus reward and command resample event.

    Reward/event term configs hold the **final** (strictest) thresholds. This curriculum starts from
    relaxed values and linearly interpolates toward those finals over ``num_steps``.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._reward_term_name = cfg.params.get("reward_term_name", "reach_success_bonus")
        self._event_term_name = cfg.params.get("event_term_name", "resample_ee_pose_on_reach")
        self._reward_term_cfg = env.reward_manager.get_term_cfg(self._reward_term_name)
        self._event_term_cfg = env.event_manager.get_term_cfg(self._event_term_name)

        self._final_params = {key: self._reward_term_cfg.params[key] for key in _REACH_CRITERIA_PARAM_KEYS}
        ease_factor = cfg.params.get("ease_factor", 5.0)
        initial_override = cfg.params.get("initial_params")
        if initial_override is not None:
            self._initial_params = {
                key: initial_override.get(key, value * ease_factor) for key, value in self._final_params.items()
            }
        else:
            self._initial_params = {key: value * ease_factor for key, value in self._final_params.items()}

        self._start_step = cfg.params.get("start_step", 0)
        self._num_steps = cfg.params["num_steps"]
        self._last_params_key: tuple[float, ...] | None = None

        self._apply_params(self._initial_params)

    def _interpolate_params(self, step: int) -> dict[str, float]:
        if step < self._start_step:
            return dict(self._initial_params)

        progress = min(1.0, (step - self._start_step) / self._num_steps)
        return {
            key: self._initial_params[key] + progress * (self._final_params[key] - self._initial_params[key])
            for key in _REACH_CRITERIA_PARAM_KEYS
        }

    def _apply_params(self, params: dict[str, float]) -> None:
        params_key = tuple(round(params[key], 8) for key in _REACH_CRITERIA_PARAM_KEYS)
        if params_key == self._last_params_key:
            return

        for key, value in params.items():
            self._reward_term_cfg.params[key] = value
            self._event_term_cfg.params[key] = value
        env = self._env
        env.reward_manager.set_term_cfg(self._reward_term_name, self._reward_term_cfg)
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
        event_term_name: str = "resample_ee_pose_on_reach",
        initial_params: dict[str, float] | None = None,
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


def termination_penalty_weight_by_step(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    data: float,
    switch_step: int = TERM_PENALTY_SWITCH_STEP,
    initial_weight: float = TERM_PENALTY_INITIAL_WEIGHT,
    final_weight: float = TERM_PENALTY_FINAL_WEIGHT,
) -> float:
    """Step termination penalty weight: hold initial, then jump to final at ``switch_step``."""
    if env.common_step_counter < switch_step:
        return initial_weight
    return final_weight
