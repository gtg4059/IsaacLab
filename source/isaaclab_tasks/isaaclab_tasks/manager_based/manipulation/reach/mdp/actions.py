# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CBF-filter joint actions for the reach CRI-F environment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.joint_actions import JointVelocityAction, RelativeJointPositionAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from isaaclab.envs.mdp.actions.actions_cfg import JointVelocityActionCfg, RelativeJointPositionActionCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class JointVelocityCriFilterAction(JointVelocityAction):
    """Joint velocity action that evaluates CRI with CBF ``run_cri_filter`` only.

    Each tick: one ``run_cri_filter(q, qd_RL)``. Command is library ``qd_cmd``.
    Next obs is that CRI. First obs is 0. Time-scale ``s`` is not applied.
    """

    cfg: "JointVelocityCriFilterActionCfg"

    def __init__(self, cfg: "JointVelocityCriFilterActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset.data.enable_cri_filter_mode(
            cri_limit=cfg.cri_limit, filter_enabled=cfg.filter_enabled, cbf_alpha=cfg.cbf_alpha
        )
        n = self.num_envs
        device = self.device
        self._qd_cmd = torch.zeros(n, self.action_dim, device=device)
        self._ep_count = torch.zeros(n, device=device)
        self._ep_delta_sum = torch.zeros(n, device=device)
        self._ep_delta_max = torch.zeros(n, device=device)
        self._ep_max_cri_pre = torch.zeros(n, device=device)
        self._last_cbf_alpha = float(cfg.cbf_alpha)
        self._last_approach_limit = float(self._asset.data.cri_filter_approach_limit)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        qd_rl = self._processed_actions
        cri_pre, delta, qd_cmd = self._asset.data.apply_cri_filter(qd_rl)
        self._qd_cmd.copy_(qd_rl if not self.cfg.filter_enabled else qd_cmd)
        self._last_cbf_alpha = float(self._asset.data.cri_filter_cbf_alpha)
        self._last_approach_limit = float(self._asset.data.cri_filter_approach_limit)

        self._ep_count += 1.0
        cri_max = cri_pre.amax(dim=-1) if cri_pre.dim() > 1 else cri_pre
        if cri_max.dim() > 1:
            cri_max = cri_max.amax(dim=-1)
        self._ep_max_cri_pre = torch.maximum(self._ep_max_cri_pre, cri_max.to(dtype=self._ep_max_cri_pre.dtype))
        delta = delta.to(dtype=self._ep_delta_sum.dtype)
        self._ep_delta_sum += delta
        self._ep_delta_max = torch.maximum(self._ep_delta_max, delta)

    def apply_actions(self):
        self._asset.set_joint_velocity_target(self._qd_cmd, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        count = self._ep_count[env_ids]
        denom = torch.clamp(count, min=1.0)
        extras = {
            "Episode_CRI_Filter/delta_mean": torch.mean(self._ep_delta_sum[env_ids] / denom),
            "Episode_CRI_Filter/delta_max": torch.mean(self._ep_delta_max[env_ids]),
            "Episode_CRI_Filter/max_cri_pre": torch.mean(self._ep_max_cri_pre[env_ids]),
            "Episode_CRI_Filter/cbf_alpha": torch.tensor(
                self._last_cbf_alpha, device=self.device, dtype=torch.float32
            ),
            "Episode_CRI_Filter/approach_limit": torch.tensor(
                self._last_approach_limit, device=self.device, dtype=torch.float32
            ),
            "Episode_CRI_Filter/cri_limit": torch.tensor(
                float(self.cfg.cri_limit), device=self.device, dtype=torch.float32
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
        self._ep_delta_sum[env_ids] = 0.0
        self._ep_delta_max[env_ids] = 0.0
        self._ep_max_cri_pre[env_ids] = 0.0
        return extras


@configclass
class JointVelocityCriFilterActionCfg(JointVelocityActionCfg):
    """Configuration for :class:`JointVelocityCriFilterAction`."""

    class_type: type[ActionTerm] = JointVelocityCriFilterAction

    cri_limit: float = 0.96
    """Hard CRI cap passed to ``set_cri_filter_limit``."""

    cbf_alpha: float = 0.02
    """CBF boundary coefficient α. Approach limit is ``cri_limit * (1 - α)``."""

    filter_enabled: bool = True
    """If True, apply library ``qd_cmd``. If False, command ``qd_RL`` and still log CRI."""


class JointPositionCriFilterAction(RelativeJointPositionAction):
    """Delta-position action filtered in velocity space by CBF ``run_cri_filter``.

    Policy output is a joint delta ``Δq`` (OpenPI π0.5-DROID style). Each tick:

    1. ``qd_nom = Δq / dt``
    2. one ``run_cri_filter(q, qd_nom)`` → ``qd_cmd``
    3. ``q_cmd = q + qd_cmd * dt`` sent as a position target

    The library QP stays relative-degree 1 (velocity). Time-scale ``s`` is not applied.
    """

    cfg: "JointPositionCriFilterActionCfg"

    def __init__(self, cfg: "JointPositionCriFilterActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset.data.enable_cri_filter_mode(
            cri_limit=cfg.cri_limit, filter_enabled=cfg.filter_enabled, cbf_alpha=cfg.cbf_alpha
        )
        n = self.num_envs
        device = self.device
        self._q_cmd = torch.zeros(n, self.action_dim, device=device)
        self._ep_count = torch.zeros(n, device=device)
        self._ep_delta_sum = torch.zeros(n, device=device)
        self._ep_delta_max = torch.zeros(n, device=device)
        self._ep_max_cri_pre = torch.zeros(n, device=device)
        self._last_cbf_alpha = float(cfg.cbf_alpha)
        self._last_approach_limit = float(self._asset.data.cri_filter_approach_limit)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        dq_rl = self._processed_actions
        dt = self._env.step_dt
        qd_nom = dq_rl / dt
        cri_pre, delta, qd_cmd = self._asset.data.apply_cri_filter(qd_nom)
        qd_out = qd_nom if not self.cfg.filter_enabled else qd_cmd
        q = self._asset.data.joint_pos[:, self._joint_ids]
        self._q_cmd.copy_(q + qd_out * dt)
        self._last_cbf_alpha = float(self._asset.data.cri_filter_cbf_alpha)
        self._last_approach_limit = float(self._asset.data.cri_filter_approach_limit)

        self._ep_count += 1.0
        cri_max = cri_pre.amax(dim=-1) if cri_pre.dim() > 1 else cri_pre
        if cri_max.dim() > 1:
            cri_max = cri_max.amax(dim=-1)
        self._ep_max_cri_pre = torch.maximum(self._ep_max_cri_pre, cri_max.to(dtype=self._ep_max_cri_pre.dtype))
        delta = delta.to(dtype=self._ep_delta_sum.dtype)
        self._ep_delta_sum += delta
        self._ep_delta_max = torch.maximum(self._ep_delta_max, delta)

    def apply_actions(self):
        self._asset.set_joint_position_target(self._q_cmd, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        count = self._ep_count[env_ids]
        denom = torch.clamp(count, min=1.0)
        extras = {
            "Episode_CRI_Filter/delta_mean": torch.mean(self._ep_delta_sum[env_ids] / denom),
            "Episode_CRI_Filter/delta_max": torch.mean(self._ep_delta_max[env_ids]),
            "Episode_CRI_Filter/max_cri_pre": torch.mean(self._ep_max_cri_pre[env_ids]),
            "Episode_CRI_Filter/cbf_alpha": torch.tensor(
                self._last_cbf_alpha, device=self.device, dtype=torch.float32
            ),
            "Episode_CRI_Filter/approach_limit": torch.tensor(
                self._last_approach_limit, device=self.device, dtype=torch.float32
            ),
            "Episode_CRI_Filter/cri_limit": torch.tensor(
                float(self.cfg.cri_limit), device=self.device, dtype=torch.float32
            ),
            "Episode_CRI_Filter/enabled": torch.tensor(
                float(self.cfg.filter_enabled), device=self.device, dtype=torch.float32
            ),
            "Episode_CRI_Filter/solves_per_step": torch.tensor(
                float(self._asset.data.cri_filter_solves_per_step), device=self.device, dtype=torch.float32
            ),
        }
        self._q_cmd[env_ids] = 0.0
        self._ep_count[env_ids] = 0.0
        self._ep_delta_sum[env_ids] = 0.0
        self._ep_delta_max[env_ids] = 0.0
        self._ep_max_cri_pre[env_ids] = 0.0
        return extras


@configclass
class JointPositionCriFilterActionCfg(RelativeJointPositionActionCfg):
    """Configuration for :class:`JointPositionCriFilterAction`."""

    class_type: type[ActionTerm] = JointPositionCriFilterAction

    cri_limit: float = 0.96
    """Hard CRI cap passed to ``set_cri_filter_limit``."""

    cbf_alpha: float = 0.02
    """CBF boundary coefficient α. Approach limit is ``cri_limit * (1 - α)``."""

    filter_enabled: bool = True
    """If True, apply library ``qd_cmd`` then integrate to ``q_cmd``. If False, ``q + Δq``."""
