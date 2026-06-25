# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class object_dropped_after_lift(ManagerTermBase):
    """Terminate when the object was lifted above a threshold and then dropped back to the table."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._was_lifted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._enabled = cfg.params.get("enabled", True)

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable this termination (used by lift curriculum phase 2)."""
        self._enabled = enabled

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None or isinstance(env_ids, slice):
            self._was_lifted[:] = False
            return
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(list(env_ids), device=self.device, dtype=torch.long)
        self._was_lifted[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        minimal_lift_height: float = 0.04,
        table_height_threshold: float = 0.04,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        enabled: bool = True,
    ) -> torch.Tensor:
        if not (self._enabled and enabled):
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        obj: RigidObject = env.scene[object_cfg.name]
        height = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]

        self._was_lifted |= height > minimal_lift_height
        return self._was_lifted & (height < table_height_threshold)
