# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

from isaaclab_tasks.manager_based.manipulation.reach.mdp.observations import command_origin_pose_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _body_idx_single(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> int:
    """Body index for EE-style terms."""
    if isinstance(asset_cfg.body_ids, list):
        return int(asset_cfg.body_ids[0])
    if isinstance(asset_cfg.body_ids, int):
        return int(asset_cfg.body_ids)
    entity: RigidObject = env.scene[asset_cfg.name]
    if asset_cfg.body_names is not None:
        keys = [asset_cfg.body_names] if isinstance(asset_cfg.body_names, str) else list(asset_cfg.body_names)
        ids, _ = entity.find_bodies(keys, preserve_order=asset_cfg.preserve_order)
        return int(ids[0])
    return 0


def reach_success_criteria(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    max_distance: float,
    max_angle_rad: float,
    max_lin_vel: float,
    max_ang_vel: float,
    max_lin_acc: float,
    max_ang_acc: float,
    command_b: torch.Tensor | None = None,
) -> torch.Tensor:
    """Boolean (num_envs,): EE meets pose, velocity, and acceleration tolerances."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = command_b if command_b is not None else env.command_manager.get_command(command_name)
    bid = _body_idx_single(env, asset_cfg)

    des_pos_b = command[:, :3]
    origin_pos_w, origin_quat_w = command_origin_pose_w(env, command_name, asset)
    des_pos_w, _ = combine_frame_transforms(origin_pos_w, origin_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, bid]
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)

    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(origin_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, bid]
    quat_err_rad = quat_error_magnitude(curr_quat_w, des_quat_w)
    lin_spd = torch.norm(asset.data.body_lin_vel_w[:, bid, :], dim=-1)
    ang_spd = torch.norm(asset.data.body_ang_vel_w[:, bid, :], dim=-1)
    lin_acc = torch.norm(asset.data.body_lin_acc_w[:, bid, :], dim=-1)
    ang_acc = torch.norm(asset.data.body_ang_acc_w[:, bid, :], dim=-1)

    pose_ok = torch.logical_and(distance <= max_distance, quat_err_rad <= max_angle_rad)
    vel_ok = torch.logical_and(lin_spd <= max_lin_vel, ang_spd <= max_ang_vel)
    acc_ok = torch.logical_and(lin_acc <= max_lin_acc, ang_acc <= max_ang_acc)
    return torch.logical_and(torch.logical_and(pose_ok, vel_ok), acc_ok)


class ReachSuccessCriteria(ManagerTermBase):
    """Reach success: ramped pose with dwell, then pose+vel+acc.

    Position tolerance goes linearly from ``pos_ease_factor * max_distance`` to
    ``max_distance`` over ``pos_ramp_steps``. Until ``vel_switch_step``,
    velocity/acceleration are ignored and success requires ``hold_steps``
    consecutive in-tolerance steps. After that, configured pose+vel+acc apply
    with no dwell.

    Terminations run before rewards, so :meth:`compute_success` is idempotent per
    ``env.common_step_counter``.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._hold_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._last_success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._updated_step = -1

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None or isinstance(env_ids, slice):
            self._hold_count.zero_()
            self._last_success.zero_()
            return
        self._hold_count[env_ids] = 0
        self._last_success[env_ids] = False

    def _ramped_distance(self, step: int, max_distance: float, pos_ease_factor: float, pos_ramp_steps: int | None) -> float:
        if pos_ramp_steps is None or pos_ramp_steps <= 0:
            return max_distance
        progress = min(1.0, step / pos_ramp_steps)
        scale = pos_ease_factor + progress * (1.0 - pos_ease_factor)
        return max_distance * scale

    def _active_gates(
        self,
        env: ManagerBasedRLEnv,
        max_distance: float,
        max_angle_rad: float,
        max_lin_vel: float,
        max_ang_vel: float,
        max_lin_acc: float,
        max_ang_acc: float,
        pos_ramp_steps: int | None,
        vel_switch_step: int | None,
        pos_ease_factor: float,
        hold_steps: int,
    ) -> tuple[float, float, float, float, float, float, int]:
        step = env.common_step_counter
        distance = self._ramped_distance(step, max_distance, pos_ease_factor, pos_ramp_steps)
        if vel_switch_step is None or step >= vel_switch_step:
            return max_distance, max_angle_rad, max_lin_vel, max_ang_vel, max_lin_acc, max_ang_acc, 0
        return distance, max_angle_rad, float("inf"), float("inf"), float("inf"), float("inf"), hold_steps

    def compute_success(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg,
        max_distance: float,
        max_angle_rad: float,
        max_lin_vel: float,
        max_ang_vel: float,
        max_lin_acc: float,
        max_ang_acc: float,
        vel_switch_step: int | None = None,
        pos_ramp_steps: int | None = None,
        pos_ease_factor: float = 3.0,
        hold_steps: int = 0,
        command_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._updated_step == env.common_step_counter:
            return self._last_success

        distance, angle, lin_vel, ang_vel, lin_acc, ang_acc, hold = self._active_gates(
            env,
            max_distance,
            max_angle_rad,
            max_lin_vel,
            max_ang_vel,
            max_lin_acc,
            max_ang_acc,
            pos_ramp_steps,
            vel_switch_step,
            pos_ease_factor,
            hold_steps,
        )
        instant = reach_success_criteria(
            env,
            command_name=command_name,
            asset_cfg=asset_cfg,
            max_distance=distance,
            max_angle_rad=angle,
            max_lin_vel=lin_vel,
            max_ang_vel=ang_vel,
            max_lin_acc=lin_acc,
            max_ang_acc=ang_acc,
            command_b=command_b,
        )
        if hold <= 0:
            self._hold_count.zero_()
            self._last_success[:] = instant
        else:
            self._hold_count[:] = torch.where(instant, self._hold_count + 1, torch.zeros_like(self._hold_count))
            self._last_success[:] = self._hold_count >= hold
        self._updated_step = env.common_step_counter
        return self._last_success

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg,
        max_distance: float,
        max_angle_rad: float,
        max_lin_vel: float,
        max_ang_vel: float,
        max_lin_acc: float,
        max_ang_acc: float,
        vel_switch_step: int | None = None,
        pos_ramp_steps: int | None = None,
        pos_ease_factor: float = 3.0,
        hold_steps: int = 0,
    ) -> torch.Tensor:
        return self.compute_success(
            env,
            command_name=command_name,
            asset_cfg=asset_cfg,
            max_distance=max_distance,
            max_angle_rad=max_angle_rad,
            max_lin_vel=max_lin_vel,
            max_ang_vel=max_ang_vel,
            max_lin_acc=max_lin_acc,
            max_ang_acc=max_ang_acc,
            vel_switch_step=vel_switch_step,
            pos_ramp_steps=pos_ramp_steps,
            pos_ease_factor=pos_ease_factor,
            hold_steps=hold_steps,
        ).float()


class reach_success_bonus(ManagerTermBase):
    """Sparse reach bonus that doubles for each success before episode reset.

    On the rising edge of :func:`reach_success_criteria`, returns ``2**(n-1)`` where ``n`` is the
    number of successes in the current episode (1, 2, 4, ...). The counter resets with the env.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._success_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._prev_success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None or isinstance(env_ids, slice):
            self._success_count.zero_()
            self._prev_success.zero_()
            return
        self._success_count[env_ids] = 0
        self._prev_success[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg,
        max_distance: float,
        max_angle_rad: float,
        max_lin_vel: float,
        max_ang_vel: float,
        max_lin_acc: float,
        max_ang_acc: float,
    ) -> torch.Tensor:
        success = reach_success_criteria(
            env,
            command_name=command_name,
            asset_cfg=asset_cfg,
            max_distance=max_distance,
            max_angle_rad=max_angle_rad,
            max_lin_vel=max_lin_vel,
            max_ang_vel=max_ang_vel,
            max_lin_acc=max_lin_acc,
            max_ang_acc=max_ang_acc,
        )
        event = success & ~self._prev_success
        self._prev_success[:] = success
        self._success_count[event] += 1

        reward = torch.zeros(self.num_envs, device=self.device)
        if torch.any(event):
            reward[event] = torch.pow(2.0, (self._success_count[event] - 1).to(dtype=torch.float32))
        return reward


class timeout_no_reach_penalty(ManagerTermBase):
    """Sparse penalty when episode-length timeout fires without any reach success.

    Uses the ``time_out`` termination term only (not ``time_outs``), so
    ``reach_success`` marked ``time_out=True`` is never penalized. Success is
    read from ``reward_term_name`` when that term is :class:`ReachSuccessCriteria`.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._ever_reached = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None or isinstance(env_ids, slice):
            self._ever_reached.zero_()
            return
        self._ever_reached[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        reward_term_name: str = "reach_success_bonus",
    ) -> torch.Tensor:
        term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        func = term_cfg.func
        if isinstance(func, ReachSuccessCriteria):
            success = func.compute_success(env, **term_cfg.params)
        else:
            success = reach_success_criteria(env, **term_cfg.params)
        self._ever_reached |= success

        if "time_out" in env.termination_manager.active_terms:
            timed_out = env.termination_manager.get_term("time_out")
        else:
            timed_out = env.termination_manager.time_outs
        return (timed_out & ~self._ever_reached).float()


def is_alive_time_ramp(
    env: ManagerBasedRLEnv,
    ramp_start_s: float = 12.0,
    initial: float = 0.2,
    final: float = 1.0,
    ramp_end_s: float | None = None,
) -> torch.Tensor:
    """Living-cost magnitude: hold ``initial`` until ``ramp_start_s``, then lerp to ``final``.

    Pair with a negative reward weight (typically ``-1.0``) so the applied penalty is
    ``-initial`` then ramps to ``-final`` at ``ramp_end_s`` (default: episode length).
    Zero on terminated (non-timeout) envs, matching :func:`isaaclab.envs.mdp.is_alive`.
    """
    t = env.episode_length_buf.float() * env.step_dt
    end = env.max_episode_length_s if ramp_end_s is None else float(ramp_end_s)
    span = max(end - ramp_start_s, 1e-6)
    progress = ((t - ramp_start_s) / span).clamp(min=0.0, max=1.0)
    magnitude = initial + (final - initial) * progress
    alive = (~env.termination_manager.terminated).float()
    return magnitude * alive


def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    origin_pos_w, origin_quat_w = command_origin_pose_w(env, command_name, asset)
    des_pos_w, _ = combine_frame_transforms(origin_pos_w, origin_quat_w, des_pos_b)
    bid = _body_idx_single(env, asset_cfg)
    curr_pos_w = asset.data.body_pos_w[:, bid]
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_orientation_command_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of position and orientation using exponential kernels."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_pos_b = command[:, :3]
    origin_pos_w, origin_quat_w = command_origin_pose_w(env, command_name, asset)
    des_pos_w, _ = combine_frame_transforms(origin_pos_w, origin_quat_w, des_pos_b)
    bid = _body_idx_single(env, asset_cfg)
    curr_pos_w = asset.data.body_pos_w[:, bid]
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)

    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(origin_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, bid]
    # Workspace-scale basin: 1/e at ~0.5 m, ~1.0 rad.
    return torch.exp(-1.0 * distance) * torch.exp(-0.5 * quat_error_magnitude(curr_quat_w, des_quat_w))

def position_orientation_command_error_fine_grained(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward fine-grained tracking of position and orientation using tighter exponential kernels."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_pos_b = command[:, :3]
    origin_pos_w, origin_quat_w = command_origin_pose_w(env, command_name, asset)
    des_pos_w, _ = combine_frame_transforms(origin_pos_w, origin_quat_w, des_pos_b)
    bid = _body_idx_single(env, asset_cfg)
    curr_pos_w = asset.data.body_pos_w[:, bid]
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)

    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(origin_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, bid]
    # Last-mile basin (~4x tighter): 1/e at ~0.125 m, ~0.25 rad, toward 3 cm / 0.1 rad.
    return torch.exp(-6.0 * distance) * torch.exp(-3.0 * quat_error_magnitude(curr_quat_w, des_quat_w))


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    origin_pos_w, origin_quat_w = command_origin_pose_w(env, command_name, asset)
    des_pos_w, _ = combine_frame_transforms(origin_pos_w, origin_quat_w, des_pos_b)
    bid = _body_idx_single(env, asset_cfg)
    curr_pos_w = asset.data.body_pos_w[:, bid]
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_b = command[:, 3:7]
    _, origin_quat_w = command_origin_pose_w(env, command_name, asset)
    des_quat_w = quat_mul(origin_quat_w, des_quat_b)
    bid = _body_idx_single(env, asset_cfg)
    curr_quat_w = asset.data.body_quat_w[:, bid]
    return quat_error_magnitude(curr_quat_w, des_quat_w)


def CRI_OVF_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    limit: float = 0.96,
    sigma: float = 20.0,
) -> torch.Tensor:
    """Penalize CRI with an exp barrier up to ``limit``, then linear to OVF threshold.

    Same ``max(CRI)`` readout as the OVF termination. Below ``limit`` this is the
    original exponential barrier (0 at CRI=0, ~1 at ``limit``). Above ``limit``,
    a linear term grows from 0 to 1 as CRI approaches the live OVF termination
    threshold (2.0 early, 0.96 after curriculum). Use a negative reward weight.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cri, _ = torch.max(asset.data.CRI, dim=1)
    headroom = (limit - cri).clamp(min=0.0)
    exp_pen = torch.exp(-sigma * headroom) - torch.exp(cri.new_tensor(-sigma * limit))

    try:
        threshold = float(env.termination_manager.get_term_cfg("OVF").params["threshold"])
    except (ValueError, KeyError, AttributeError):
        threshold = limit
    excess = (cri - limit).clamp(min=0.0)
    span = max(threshold - limit, 1e-6)
    lin_pen = (excess / span).clamp(max=1.0)
    return exp_pen + lin_pen
