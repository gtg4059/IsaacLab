# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom action terms for pick-and-place PLAY evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.pickandplace.mdp.phase import reached_command_target

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class PlayAutoReleaseGripperAction(BinaryJointPositionAction):
    """Binary gripper action that auto-opens after a delay once the EE reaches the floor target in PLAY."""

    cfg: "PlayAutoReleaseGripperActionCfg"

    def __init__(self, cfg: "PlayAutoReleaseGripperActionCfg", env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._gripper_released = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._floor_reached_at_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)

        pending = ~self._gripper_released
        if torch.any(pending):
            newly_released = pending & self._should_release_gripper()
            if torch.any(newly_released):
                self._gripper_released |= newly_released

        if torch.any(self._gripper_released):
            self._processed_actions[self._gripper_released] = self._open_command
            self._raw_actions[self._gripper_released] = 1.0

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._gripper_released[:] = False
            self._floor_reached_at_step[:] = -1
        else:
            self._gripper_released[env_ids] = False
            self._floor_reached_at_step[env_ids] = -1

    def _should_release_gripper(self) -> torch.Tensor:
        cmd_term = self._env.command_manager.get_term(self.cfg.command_name)
        if not hasattr(cmd_term, "in_floor_phase"):
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        in_floor_phase = cmd_term.in_floor_phase
        if not torch.any(in_floor_phase):
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        at_floor_target = in_floor_phase & reached_command_target(
            self._env,
            command_pos_w=cmd_term.pose_command_w[:, :3],
            goal_distance_threshold=self.cfg.release_goal_distance,
            ee_body_name=self.cfg.ee_body_name,
        )

        pending_floor = (self._floor_reached_at_step < 0) & at_floor_target
        if torch.any(pending_floor):
            self._floor_reached_at_step[pending_floor] = self._env.episode_length_buf[pending_floor]

        delay_steps = max(1, int(self.cfg.release_delay_s / self._env.step_dt))
        return in_floor_phase & (self._floor_reached_at_step >= 0) & (
            self._env.episode_length_buf - self._floor_reached_at_step >= delay_steps
        )


@configclass
class PlayAutoReleaseGripperActionCfg(BinaryJointPositionActionCfg):
    """Configuration for PLAY gripper auto-release on floor target reach."""

    class_type: type[ActionTerm] = PlayAutoReleaseGripperAction

    command_name: str = "target_pose"
    """Command term that tracks the sequential air-then-floor targets."""

    ee_body_name: str = "panda_hand"
    """End-effector body used for reach checks."""

    release_goal_distance: float = 0.08
    """Distance threshold (m) for detecting arrival at the floor target."""

    release_delay_s: float = 0.5
    """Seconds to wait at the floor target before opening the gripper."""
