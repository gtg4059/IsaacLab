# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum terms for pick-and-place lift training."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class lift_place_down_curriculum(ManagerTermBase):
    """Two-phase lift curriculum controlled by ``env.common_step_counter``.

    Phase 1 (steps ``<= num_steps``, default 3000 PPO iters x 24 steps/env):
        - ``target_pose`` samples from ``air_ranges`` only.
        - ``object_dropped_after_lift`` termination is active.

    Phase 2 (steps ``> num_steps``):
        - ``target_pose`` starts aerial on reset; switches to floor after aerial reach + 1/3 episode time.
        - ``object_dropped_after_lift`` termination is disabled (place-down allowed).
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._phase2_applied = False

    def __call__(self, env: ManagerBasedRLEnv, env_ids: Sequence[int], num_steps: int = 3000 * 24) -> float:
        if env.common_step_counter > num_steps and not self._phase2_applied:
            self._apply_phase2(env)
        return 1.0 if env.common_step_counter <= num_steps else 2.0

    def _apply_phase2(self, env: ManagerBasedRLEnv) -> None:
        self._phase2_applied = True

        drop_cfg = env.termination_manager.get_term_cfg("object_dropped_after_lift")
        if hasattr(drop_cfg.func, "set_enabled"):
            drop_cfg.func.set_enabled(False)

        cmd_term = env.command_manager.get_term("target_pose")
        if hasattr(cmd_term, "set_air_only"):
            cmd_term.set_air_only(False)
        # Avoid timer-based air resample after handoff; targets change only on reset or lift handoff.
        cmd_term.cfg.resampling_time_range = (1000.0, 1000.0)

        print(
            f"[lift_place_down_curriculum] Switched to phase 2 at common_step={env.common_step_counter}: "
            "sequential air-then-floor target_pose + object_dropped_after_lift disabled."
        )
