# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CRI-filter joint velocity action for the reach CRI-F environment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.joint_actions import JointVelocityAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from isaaclab.envs.mdp.actions.actions_cfg import JointVelocityActionCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class JointVelocityCriFilterAction(JointVelocityAction):
    """Joint velocity action that evaluates CRI with ``run_cri_filter`` only.

    Each tick: one ``run_cri_filter(q, qd_RL)``. Next obs is that CRI. First obs is 0.
    ``filter_enabled`` only toggles Newton (s=1 vs s*); CRI is always from ``run_cri_filter``.
    """

    cfg: "JointVelocityCriFilterActionCfg"

    def __init__(self, cfg: "JointVelocityCriFilterActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset.data.enable_cri_filter_mode(cri_limit=cfg.cri_limit, filter_enabled=cfg.filter_enabled)
        n = self.num_envs
        device = self.device
        self._qd_cmd = torch.zeros(n, self.action_dim, device=device)
        self._ep_count = torch.zeros(n, device=device)
        self._ep_s_sum = torch.zeros(n, device=device)
        self._ep_s_min = torch.ones(n, device=device)
        self._ep_s_lt1 = torch.zeros(n, device=device)
        self._ep_max_cri_pre = torch.zeros(n, device=device)
        self._ep_qd_err = torch.zeros(n, device=device)
        self._ep_parallel_err = torch.zeros(n, device=device)
        self._last_lib_min_scale = 1.0

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        qd_rl = self._processed_actions
        cri_pre, s, qd_cmd = self._asset.data.apply_cri_filter(qd_rl)
        self._qd_cmd.copy_(qd_rl if not self.cfg.filter_enabled else qd_cmd)
        self._last_lib_min_scale = float(self._asset.data._cri_filter_lib_min_scale)

        self._ep_count += 1.0
        cri_max = cri_pre.amax(dim=-1) if cri_pre.dim() > 1 else cri_pre
        if cri_max.dim() > 1:
            cri_max = cri_max.amax(dim=-1)
        self._ep_max_cri_pre = torch.maximum(self._ep_max_cri_pre, cri_max.to(dtype=self._ep_max_cri_pre.dtype))
        if self.cfg.filter_enabled:
            self._ep_s_sum += s
            self._ep_s_min = torch.minimum(self._ep_s_min, s)
            self._ep_s_lt1 += (s < 1.0 - 1e-6).to(dtype=s.dtype)
            qd_err = torch.linalg.norm(qd_cmd - s.unsqueeze(-1) * qd_rl, dim=-1)
            self._ep_qd_err += qd_err
            self._ep_parallel_err += qd_err
        else:
            self._ep_s_sum += 1.0

    def apply_actions(self):
        self._asset.set_joint_velocity_target(self._qd_cmd, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        count = self._ep_count[env_ids]
        denom = torch.clamp(count, min=1.0)
        extras = {
            "Episode_CRI_Filter/s_mean": torch.mean(self._ep_s_sum[env_ids] / denom),
            "Episode_CRI_Filter/s_min": torch.mean(self._ep_s_min[env_ids]),
            "Episode_CRI_Filter/frac_s_lt_1": torch.mean(self._ep_s_lt1[env_ids] / denom),
            "Episode_CRI_Filter/max_cri_pre": torch.mean(self._ep_max_cri_pre[env_ids]),
            "Episode_CRI_Filter/qd_cmd_err": torch.mean(self._ep_qd_err[env_ids] / denom),
            "Episode_CRI_Filter/parallel_err": torch.mean(self._ep_parallel_err[env_ids] / denom),
            "Episode_CRI_Filter/lib_min_scale": torch.tensor(
                self._last_lib_min_scale, device=self.device, dtype=torch.float32
            ),
            "Episode_CRI_Filter/enabled": torch.tensor(
                float(self.cfg.filter_enabled), device=self.device, dtype=torch.float32
            ),
            "Episode_CRI_Filter/solves_per_step": torch.tensor(
                float(self._asset.data.cri_filter_solves_per_step), device=self.device, dtype=torch.float32
            ),
        }
        self._qd_cmd[env_ids] = 0.0
        self._ep_count[env_ids] = 0.0
        self._ep_s_sum[env_ids] = 0.0
        self._ep_s_min[env_ids] = 1.0
        self._ep_s_lt1[env_ids] = 0.0
        self._ep_max_cri_pre[env_ids] = 0.0
        self._ep_qd_err[env_ids] = 0.0
        self._ep_parallel_err[env_ids] = 0.0
        return extras


@configclass
class JointVelocityCriFilterActionCfg(JointVelocityActionCfg):
    """Configuration for :class:`JointVelocityCriFilterAction`."""

    class_type: type[ActionTerm] = JointVelocityCriFilterAction

    cri_limit: float = 1.0
    """CRI filter limit passed to ``set_cri_filter_limit``. Defaults to 1.0."""

    filter_enabled: bool = False
    """Newton scale. False: CRI via ``run_cri_filter`` with s=1. True: apply s*qd_RL."""
