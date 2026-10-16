# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum terms for CRI reach training."""

from __future__ import annotations

import math
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

    * Default: ``ease_factor → 1`` over ``num_steps`` after ``start_step``.
      ``decay_alpha=1`` is linear; ``decay_alpha=0.8`` is PCCL power cooling
      (slower at first, then faster toward the final gates).
    * Piecewise (when ``mid_factor`` is set): ``ease_factor → mid_factor`` over ``num_steps``,
      hold ``mid_factor`` for ``hold_steps``, then ``mid_factor → 1`` over ``num_steps_final``.

    Set ``event_term_name`` to ``None`` when reach success ends the episode (no in-episode resample).
    ``terminations.reach_success`` reads thresholds from the reward term, so it tracks automatically.
    Cooling always ends at the reward-term finals (1×).

    If ``mix_half_and_one_after`` is true after cooling, each env draws 2× vs 1×
    gates (mult 1.0 vs 0.5 on the cooled base) with probability 1/2 at reset.
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
        self._param_keys = tuple(key for key in _REACH_CRITERIA_PARAM_KEYS if key in primary_cfg.params)
        self._final_params = {key: primary_cfg.params[key] for key in self._param_keys}
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
        self._end_params = dict(self._final_params)

        self._start_step = cfg.params.get("start_step", 0)
        self._num_steps = cfg.params["num_steps"]
        self._hold_steps = cfg.params.get("hold_steps", 0)
        self._num_steps_final = cfg.params.get("num_steps_final", self._num_steps)
        self._decay_alpha = float(cfg.params.get("decay_alpha", 1.0))
        self._last_params_key: tuple[float, ...] | None = None
        self._hinge_term_name = cfg.params.get("hinge_term_name")
        self._hinge_param_name = cfg.params.get("hinge_param_name", "hinge_scale")
        self._hinge_zero_after = bool(cfg.params.get("hinge_zero_after", False))
        self._mix_half_and_one_after = bool(cfg.params.get("mix_half_and_one_after", False))
        self._hinge_term_cfg = (
            env.reward_manager.get_term_cfg(self._hinge_term_name) if self._hinge_term_name else None
        )
        self._last_hinge_scale: float | None = None
        self._last_mix_flag: bool | None = None

        self._apply_params(self._initial_params)
        self._apply_hinge_scale(1.0)
        self._apply_mix_flag(False)

    def _blend(self, start: dict[str, float], end: dict[str, float], k: int, span: int) -> dict[str, float]:
        """PCCL cooling: ``end + ((span-k)/span)**alpha * (start-end)``. ``alpha=1`` is linear."""
        remaining = max(span - max(k, 0), 0) / max(span, 1)
        weight = remaining ** self._decay_alpha
        blended = {}
        for key in self._param_keys:
            start_v = start[key]
            end_v = end[key]
            # inf - inf is NaN. An unlimited gate (max_lin_vel=inf) stays unlimited.
            if not (math.isfinite(start_v) and math.isfinite(end_v)):
                blended[key] = start_v if not math.isfinite(start_v) else end_v
            else:
                blended[key] = end_v + weight * (start_v - end_v)
        return blended

    def _interpolate_params(self, step: int) -> dict[str, float]:
        if step < self._start_step:
            return dict(self._initial_params)

        if self._mid_params is None:
            return self._blend(self._initial_params, self._end_params, step - self._start_step, self._num_steps)

        first_end = self._start_step + self._num_steps
        hold_end = first_end + self._hold_steps
        second_end = hold_end + self._num_steps_final

        if step < first_end:
            return self._blend(self._initial_params, self._mid_params, step - self._start_step, self._num_steps)
        if step < hold_end:
            return dict(self._mid_params)
        if step < second_end:
            return self._blend(self._mid_params, self._end_params, step - hold_end, self._num_steps_final)
        return dict(self._end_params)

    def _apply_params(self, params: dict[str, float]) -> None:
        params_key = tuple(round(params[key], 8) for key in self._param_keys)
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

    def _cooling_end_step(self) -> int:
        if self._mid_params is None:
            return self._start_step + self._num_steps
        return self._start_step + self._num_steps + self._hold_steps + self._num_steps_final

    def _apply_hinge_scale(self, scale: float) -> None:
        if self._hinge_term_cfg is None:
            return
        scale = max(float(scale), 0.0)
        if self._last_hinge_scale is not None and abs(scale - self._last_hinge_scale) < 1e-12:
            return
        self._hinge_term_cfg.params[self._hinge_param_name] = scale
        self._env.reward_manager.set_term_cfg(self._hinge_term_name, self._hinge_term_cfg)
        self._last_hinge_scale = scale

    def _apply_mix_flag(self, enabled: bool) -> None:
        if self._last_mix_flag is enabled:
            return
        for term_cfg in self._reward_term_cfgs.values():
            term_cfg.params["mix_half_and_one"] = enabled
        for name, term_cfg in self._reward_term_cfgs.items():
            self._env.reward_manager.set_term_cfg(name, term_cfg)
        self._last_mix_flag = enabled

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
        decay_alpha: float = 1.0,
        hinge_term_name: str | None = None,
        hinge_param_name: str = "hinge_scale",
        hinge_zero_after: bool = False,
        mix_half_and_one_after: bool = False,
    ) -> dict[str, float]:
        params = self._interpolate_params(env.common_step_counter)
        self._apply_params(params)
        cooled = env.common_step_counter >= self._cooling_end_step()
        hinge = 0.0 if (self._hinge_zero_after and cooled) else 1.0
        self._apply_hinge_scale(hinge)
        self._apply_mix_flag(bool(self._mix_half_and_one_after and cooled))
        out = dict(params)
        if self._hinge_term_cfg is not None:
            out[self._hinge_param_name] = hinge
        out["mix_half_and_one"] = float(self._mix_half_and_one_after and cooled)
        return out


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


def spline_offset_std_by_step(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    data: float,
    hold_start: int,
    hold_end: int,
    cool_steps: int,
    amplitude: float,
) -> float:
    """Std of the once-per-episode joint-velocity sample.

    Zero until ``hold_start`` so a reach can exist before the path is bent.
    ``amplitude`` until ``hold_end``, then a linear cool to 0. The event stops
    drawing random numbers once this returns to 0 after the hold.
    """
    if amplitude <= 0.0:
        return 0.0
    step = env.common_step_counter
    if step < hold_start:
        return 0.0
    if step < hold_end:
        return amplitude
    if cool_steps <= 0:
        return 0.0
    progress = min(1.0, (step - hold_end) / cool_steps)
    return amplitude * (1.0 - progress)


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
    num_steps: int | None = None,
    decay_alpha: float = 2.0,
) -> float:
    """Hold the OVF threshold, then cool toward ``threshold_final``.

    Without ``num_steps`` the value jumps at ``crossfade_start``. With it, cooling
    matches reach-gate PCCL: ``final + ((span-k)/span)**alpha * (initial-final)``.
    ``decay_alpha=2`` moves most of the drop in the first half of the window.
    """
    step = env.common_step_counter
    if step < crossfade_start:
        return threshold_initial
    if num_steps is None or num_steps <= 0:
        return threshold_final
    remaining = max(num_steps - (step - crossfade_start), 0) / num_steps
    weight = remaining ** decay_alpha
    return threshold_final + weight * (threshold_initial - threshold_final)


def reach_bonus_time_decay_s(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    data: float,
    start_step: int,
    num_steps: int,
    final_s: float = 6.0,
    initial_s: float = 1.0e6,
    decay_alpha: float = 1.0,
) -> float:
    """Cool the reach-bonus time constant from ``initial_s`` down to ``final_s``.

    The bonus multiplies by ``exp(-t/τ)`` with ``t`` the time since the current
    target was sampled. Shorter ``t`` always pays more; ``final_s`` is the
    time constant, not a target arrival. ``τ`` is scheduled on the rate
    ``1/τ``: zero until ``start_step``, then ``(1/final) * (1 - remaining**alpha)``.
    ``decay_alpha=1`` raises that rate linearly. ``initial_s`` is large enough
    that the multiplier stays at 1.
    """
    step = env.common_step_counter
    if num_steps <= 0 or step < start_step:
        return float(initial_s)
    if step >= start_step + num_steps:
        return float(final_s)
    remaining = (num_steps - (step - start_step)) / num_steps
    rate = (1.0 / float(final_s)) * (1.0 - remaining ** decay_alpha)
    if rate <= 1.0 / float(initial_s):
        return float(initial_s)
    return 1.0 / rate


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


class resample_on_reset_curriculum(ManagerTermBase):
    """Draw a new target and arm on every early reset, then keep a failed attempt.

    Before ``switch_step``, ``keep_command_until_success`` and ``keep_failed_spawn``
    are false, so a reset samples ``ee_pose`` and the joint pose. At ``switch_step``
    both become true: reach success still samples, and an OVF or horizon timeout
    restores that attempt's spawn and target.

    Curriculum runs at the start of reset, so the flags are already set when the
    reset event and the command term run.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._switch_step = int(cfg.params["switch_step"])
        self._command_name = cfg.params.get("command_name", "ee_pose")
        self._event_term_name = cfg.params.get("event_term_name", "reset_robot_joints")
        self._keeping: bool | None = None

    def _apply(self, keep: bool) -> None:
        if keep == self._keeping:
            return
        command = self._env.command_manager.get_term(self._command_name)
        command.cfg.keep_command_until_success = keep
        event_cfg = self._env.event_manager.get_term_cfg(self._event_term_name)
        event_cfg.params["keep_failed_spawn"] = keep
        self._keeping = keep

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        switch_step: int,
        command_name: str = "ee_pose",
        event_term_name: str = "reset_robot_joints",
    ) -> float:
        keep = env.common_step_counter >= self._switch_step
        self._apply(keep)
        return float(keep)


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
