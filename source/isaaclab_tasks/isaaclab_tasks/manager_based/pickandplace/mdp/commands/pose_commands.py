# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pose command generators for pick-and-place lift training."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand
from isaaclab.utils.math import combine_frame_transforms, quat_from_euler_xyz, quat_unique

from isaaclab_tasks.manager_based.pickandplace.mdp.phase import (
    reached_aerial_handoff_pose,
    should_handoff_to_floor_target,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .pose_commands_cfg import AlternatingLiftPoseCommandCfg


class AlternatingLiftPoseCommand(UniformPoseCommand):
    """Lift ``target_pose`` command with optional in-episode air-to-floor handoff.

    Phase 1 (``air_only``): sample aerial targets only (full episode).

    Phase 2 (sequential place-down): on reset sample an aerial target; switch to a floor-near target
    in the same episode once the object reaches the aerial target **and** at least one third of the
    episode has elapsed.
    """

    cfg: AlternatingLiftPoseCommandCfg

    def __init__(self, cfg: AlternatingLiftPoseCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._air_only = not cfg.start_in_phase2
        self._in_floor_phase = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._air_reached_at_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)

    @property
    def in_floor_phase(self) -> torch.Tensor:
        """Per-env mask: True after the aerial target handoff to the floor place target."""
        return self._in_floor_phase

    def set_air_only(self, air_only: bool) -> None:
        """Phase 1: aerial targets only. Phase 2: pass ``False`` for sequential air then floor."""
        self._air_only = air_only
        if air_only:
            self._in_floor_phase[:] = False
            self._air_reached_at_step[:] = -1

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None or isinstance(env_ids, slice):
            self._in_floor_phase[:] = False
            self._air_reached_at_step[:] = -1
        else:
            if not isinstance(env_ids, torch.Tensor):
                env_ids = torch.tensor(list(env_ids), device=self.device, dtype=torch.long)
            self._in_floor_phase[env_ids] = False
            self._air_reached_at_step[env_ids] = -1
        return super().reset(env_ids)

    def _resample_command(self, env_ids: Sequence[int]):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(list(env_ids), device=self.device, dtype=torch.long)

        if self._air_only:
            self._sample_pose_from_ranges(env_ids, self.cfg.air_ranges)
            return

        # Sequential mode: resample air for pre-handoff envs, floor for post-handoff envs.
        air_ids = env_ids[~self._in_floor_phase[env_ids]]
        floor_ids = env_ids[self._in_floor_phase[env_ids]]
        if len(air_ids) > 0:
            self._sample_pose_from_ranges(air_ids, self.cfg.air_ranges)
        if len(floor_ids) > 0:
            self._sample_pose_from_ranges(floor_ids, self.cfg.floor_ranges)

    def _update_command(self):
        if self._air_only:
            return

        if self.cfg.handoff_air_dwell_s is not None:
            reached = reached_aerial_handoff_pose(
                self._env,
                command_pos_w=self.pose_command_w[:, :3],
                minimal_height=self.cfg.handoff_min_height,
                goal_distance_threshold=self.cfg.handoff_goal_distance,
            )
            pending_air = (~self._in_floor_phase) & (self._air_reached_at_step < 0) & reached
            if torch.any(pending_air):
                self._air_reached_at_step[pending_air] = self._env.episode_length_buf[pending_air]

            dwell_steps = max(1, int(self.cfg.handoff_air_dwell_s / self._env.step_dt))
            handoff_mask = (~self._in_floor_phase) & (self._air_reached_at_step >= 0) & (
                self._env.episode_length_buf - self._air_reached_at_step >= dwell_steps
            )
        else:
            handoff_mask = (~self._in_floor_phase) & should_handoff_to_floor_target(
                self._env,
                command_pos_w=self.pose_command_w[:, :3],
                minimal_height=self.cfg.handoff_min_height,
                goal_distance_threshold=self.cfg.handoff_goal_distance,
                time_fraction=self.cfg.handoff_time_fraction,
            )
        if not torch.any(handoff_mask):
            return

        handoff_ids = handoff_mask.nonzero(as_tuple=False).squeeze(-1)
        self._sample_pose_from_ranges(handoff_ids, self.cfg.floor_ranges)
        self._in_floor_phase[handoff_ids] = True
        self._sync_pose_command_world(handoff_ids)

    def _sync_pose_command_world(self, env_ids: torch.Tensor) -> None:
        """Refresh world-frame command used by debug visualization."""
        self.pose_command_w[env_ids, :3], self.pose_command_w[env_ids, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w[env_ids],
            self.robot.data.root_quat_w[env_ids],
            self.pose_command_b[env_ids, :3],
            self.pose_command_b[env_ids, 3:],
        )

    def _sample_pose_from_ranges(self, env_ids: torch.Tensor, ranges) -> None:
        r = torch.empty(len(env_ids), device=self.device)
        self.pose_command_b[env_ids, 0] = r.uniform_(*ranges.pos_x)
        self.pose_command_b[env_ids, 1] = r.uniform_(*ranges.pos_y)
        self.pose_command_b[env_ids, 2] = r.uniform_(*ranges.pos_z)

        euler_angles = torch.zeros_like(self.pose_command_b[env_ids, :3])
        euler_angles[:, 0].uniform_(*ranges.roll)
        euler_angles[:, 1].uniform_(*ranges.pitch)
        euler_angles[:, 2].uniform_(*ranges.yaw)
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        self.pose_command_b[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat
