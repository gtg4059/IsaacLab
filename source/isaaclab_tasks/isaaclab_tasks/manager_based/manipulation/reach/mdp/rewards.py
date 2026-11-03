# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import math

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

from isaaclab_tasks.manager_based.manipulation.reach.mdp.observations import command_elapsed_s, command_origin_pose_w

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
    max_ang_vel: float | None = None,
    max_lin_acc: float = float("inf"),
    max_ang_acc: float = float("inf"),
    command_b: torch.Tensor | None = None,
) -> torch.Tensor:
    """Boolean (num_envs,): EE meets pose and combined-twist tolerances.

    Twist is Jawale's ``||(v, ω)||_2`` vs ``max_lin_vel``. Acceleration
    limits default to disabled (``inf``).
    """
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
    # Jawale: ||(v, ω)||_2. ``max_ang_vel`` is unused; keep it equal to ``max_lin_vel``.
    twist = torch.sqrt(lin_spd.square() + ang_spd.square())

    pose_ok = torch.logical_and(distance <= max_distance, quat_err_rad <= max_angle_rad)
    vel_ok = twist <= max_lin_vel
    ok = torch.logical_and(pose_ok, vel_ok)
    if math.isfinite(max_lin_acc) or math.isfinite(max_ang_acc):
        lin_acc = torch.norm(asset.data.body_lin_acc_w[:, bid, :], dim=-1)
        ang_acc = torch.norm(asset.data.body_ang_acc_w[:, bid, :], dim=-1)
        acc_ok = torch.logical_and(lin_acc <= max_lin_acc, ang_acc <= max_ang_acc)
        ok = torch.logical_and(ok, acc_ok)
    return ok


class ReachSuccessCriteria(ManagerTermBase):
    """In-gate tracker: pose+vel, optional consecutive hold count.

    Position tolerance goes linearly from ``pos_ease_factor * max_distance`` to
    ``max_distance`` over ``pos_ramp_steps``. Until ``vel_switch_step``,
    velocity/acceleration are ignored.

    ``hold_steps`` / ``hold_square_max`` of 0 means no hold: every in-gate step
    is success and pays 1. With a positive hold, intermediate steps pay
    ``hold_step_bonus`` and the success step returns 1 (applied ``weight * dt``).

    :meth:`compute_success` is True on each in-gate step when hold is 0, else
    when the consecutive count reaches the hold. Idempotent per
    ``env.common_step_counter``.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._hold_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._last_success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._last_shaped = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        # 1.0 = cooled base (e.g. 2× final); 0.5 = 1× final. Used only when mix is on.
        self._gate_mult = torch.ones(env.num_envs, dtype=torch.float, device=env.device)
        self._mix_armed = False
        self._updated_step = -1
        self._rewarded_step = -1
        self._path_step = -1
        n = env.num_envs
        dev = env.device
        self._path_len = torch.zeros(n, device=dev)
        self._start_dist = torch.zeros(n, device=dev)
        self._started = torch.zeros(n, dtype=torch.bool, device=dev)
        self._arrived = torch.zeros(n, dtype=torch.bool, device=dev)
        self._success_time = torch.zeros(n, device=dev)
        self._detour = torch.zeros(n, device=dev)
        self._start_pos = torch.zeros(n, 3, device=dev)
        self._goal_pos = torch.zeros(n, 3, device=dev)
        self._mid_yz = torch.zeros(n, 2, device=dev)
        self._mid_err = torch.full((n,), float("inf"), device=dev)
        self._last_time_mean = 0.0
        self._last_time_std = 0.0
        self._last_detour_mean = 0.0
        self._last_detour_std = 0.0
        self._last_mid_spread = 0.0
        self._last_start_dist = 0.0
        self._last_far_time = 0.0
        self._last_near_time = 0.0

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        logs = self._arrival_logs(env_ids)
        if env_ids is None or isinstance(env_ids, slice):
            self._hold_count.zero_()
            self._last_success.zero_()
            self._last_shaped.zero_()
            self._path_len.zero_()
            self._start_dist.zero_()
            self._started.zero_()
            self._arrived.zero_()
            self._success_time.zero_()
            self._detour.zero_()
            self._start_pos.zero_()
            self._goal_pos.zero_()
            self._mid_yz.zero_()
            self._mid_err.fill_(float("inf"))
            if self._mix_armed:
                self._resample_gate_mult(None)
            return logs
        self._hold_count[env_ids] = 0
        self._last_success[env_ids] = False
        self._last_shaped[env_ids] = 0.0
        self._path_len[env_ids] = 0.0
        self._start_dist[env_ids] = 0.0
        self._started[env_ids] = False
        self._arrived[env_ids] = False
        self._success_time[env_ids] = 0.0
        self._detour[env_ids] = 0.0
        self._start_pos[env_ids] = 0.0
        self._goal_pos[env_ids] = 0.0
        self._mid_yz[env_ids] = 0.0
        self._mid_err[env_ids] = float("inf")
        if self._mix_armed:
            self._resample_gate_mult(env_ids)
        return logs

    def _arrival_logs(self, env_ids: Sequence[int] | None) -> dict[str, float]:
        """Mean and spread of arrival time and path detour, over successes in this reset.

        Arrival time, success count, and the near/far split use only arrivals
        whose rewind count is still zero.
        """
        rewinds = getattr(self._env, "_ovf_rewind_count", None)
        if env_ids is None or isinstance(env_ids, slice):
            arrived = self._arrived
            times_all = self._success_time
            detours_all = self._detour
            mids_all = self._mid_yz
            dists_all = self._start_dist
            rewind_at = rewinds
        else:
            arrived = self._arrived[env_ids]
            times_all = self._success_time[env_ids]
            detours_all = self._detour[env_ids]
            mids_all = self._mid_yz[env_ids]
            dists_all = self._start_dist[env_ids]
            rewind_at = None if rewinds is None else rewinds[env_ids]
        n = int(arrived.sum().item())
        far_n = 0
        if n > 0:
            detours = detours_all[arrived]
            mids = mids_all[arrived]
            dists = dists_all[arrived]
            self._last_detour_mean = float(detours.mean())
            self._last_start_dist = float(dists.mean())
            far_n = int((dists >= 0.6).sum().item())
            if n > 1:
                self._last_detour_std = float(detours.std(unbiased=False))
                center = mids.mean(dim=0)
                self._last_mid_spread = float(torch.sqrt((mids - center).square().sum(dim=-1).mean()))
            else:
                self._last_detour_std = 0.0
                self._last_mid_spread = 0.0
        # Time and success count: arrivals that never rewound. Short is < 0.6 m.
        if rewind_at is None:
            clean = arrived
        else:
            clean = arrived & (rewind_at <= 0)
        clean_n = int(clean.sum().item())
        if clean_n > 0:
            clean_times = times_all[clean]
            clean_dists = dists_all[clean]
            self._last_time_mean = float(clean_times.mean())
            self._last_time_std = float(clean_times.std(unbiased=False)) if clean_n > 1 else 0.0
            far = clean_dists >= 0.6
            near = ~far
            if int(far.sum().item()) > 0:
                self._last_far_time = float(clean_times[far].mean())
            if int(near.sum().item()) > 0:
                self._last_near_time = float(clean_times[near].mean())
        if rewinds is None:
            rewind_mean = 0.0
        elif env_ids is None or isinstance(env_ids, slice):
            rewind_mean = float(rewinds.mean())
            rewinds.zero_()
        else:
            rewind_mean = float(rewinds[env_ids].mean())
            rewinds[env_ids] = 0.0
        return {
            "Metrics/reach/success_time_s": self._last_time_mean,
            "Metrics/reach/success_time_std_s": self._last_time_std,
            "Metrics/reach/detour": self._last_detour_mean,
            "Metrics/reach/detour_std": self._last_detour_std,
            "Metrics/reach/midpoint_spread_m": self._last_mid_spread,
            "Metrics/reach/successes": float(clean_n),
            "Metrics/reach/ovf_rewinds": rewind_mean,
            "Metrics/reach/start_dist_m": self._last_start_dist,
            "Metrics/reach/success_time_far_s": self._last_far_time,
            "Metrics/reach/far_successes": float(far_n),
            "Metrics/reach/success_time_near_s": self._last_near_time,
        }

    def _note_path(self, env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> None:
        """Integrate EE path length, and record straight-line distance on the first step."""
        if self._path_step == env.common_step_counter:
            return
        self._path_step = env.common_step_counter
        asset: RigidObject = env.scene[asset_cfg.name]
        bid = _body_idx_single(env, asset_cfg)
        curr_pos = asset.data.body_pos_w[:, bid]
        lin_spd = torch.norm(asset.data.body_lin_vel_w[:, bid, :], dim=-1)
        self._path_len += lin_spd * float(env.step_dt)
        fresh = ~self._started
        if bool(fresh.any()):
            command = env.command_manager.get_command(command_name)
            origin_pos_w, origin_quat_w = command_origin_pose_w(env, command_name, asset)
            des_pos_w, _ = combine_frame_transforms(origin_pos_w, origin_quat_w, command[:, :3])
            dist = torch.norm(curr_pos - des_pos_w, dim=1)
            self._start_dist = torch.where(fresh, dist, self._start_dist)
            self._start_pos = torch.where(fresh.unsqueeze(-1), curr_pos, self._start_pos)
            self._goal_pos = torch.where(fresh.unsqueeze(-1), des_pos_w, self._goal_pos)
            self._mid_err = torch.where(fresh, torch.full_like(self._mid_err, float("inf")), self._mid_err)
            self._started |= fresh
        self._update_midpoint(curr_pos)

    def _update_midpoint(self, curr_pos: torch.Tensor) -> None:
        """Keep the EE offset, in the plane normal to start→goal, closest to halfway."""
        delta = self._goal_pos - self._start_pos
        dist = torch.norm(delta, dim=-1, keepdim=True).clamp_min(1.0e-3)
        forward = delta / dist
        up = torch.zeros_like(forward)
        up[:, 2] = 1.0
        side = torch.cross(up, forward, dim=-1)
        side_norm = torch.norm(side, dim=-1, keepdim=True)
        alt = torch.zeros_like(forward)
        alt[:, 0] = 1.0
        side = torch.where(side_norm > 1.0e-4, side, torch.cross(alt, forward, dim=-1))
        side = side / torch.norm(side, dim=-1, keepdim=True).clamp_min(1.0e-6)
        lift = torch.cross(forward, side, dim=-1)
        rel = curr_pos - self._start_pos
        along = (rel * forward).sum(dim=-1)
        progress = along / dist.squeeze(-1)
        lat = torch.stack(((rel * side).sum(dim=-1), (rel * lift).sum(dim=-1)), dim=-1)
        err = (progress - 0.5).abs()
        closer = self._started & (err < self._mid_err)
        self._mid_err = torch.where(closer, err, self._mid_err)
        self._mid_yz = torch.where(closer.unsqueeze(-1), lat, self._mid_yz)

    def _resample_gate_mult(self, env_ids: Sequence[int] | None) -> None:
        """50/50 2× base (mult=1) vs 1× final (mult=0.5) on the cooled gate params."""
        if env_ids is None or isinstance(env_ids, slice):
            pick_one = torch.bernoulli(torch.full_like(self._gate_mult, 0.5)).bool()
            self._gate_mult.fill_(0.5)
            self._gate_mult[pick_one] = 1.0
            return
        n = len(env_ids)
        pick_one = torch.bernoulli(torch.full((n,), 0.5, device=self._gate_mult.device)).bool()
        mult = torch.full((n,), 0.5, device=self._gate_mult.device)
        mult[pick_one] = 1.0
        self._gate_mult[env_ids] = mult

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
        max_ang_vel: float | None,
        max_lin_acc: float,
        max_ang_acc: float,
        pos_ramp_steps: int | None,
        vel_switch_step: int | None,
        pos_ease_factor: float,
        hold_steps: int,
    ) -> tuple[float, float, float, float, float, float, int]:
        step = env.common_step_counter
        hold = max(int(hold_steps), 0)
        distance = self._ramped_distance(step, max_distance, pos_ease_factor, pos_ramp_steps)
        if vel_switch_step is None or step >= vel_switch_step:
            return max_distance, max_angle_rad, max_lin_vel, max_ang_vel, max_lin_acc, max_ang_acc, hold
        return distance, max_angle_rad, float("inf"), float("inf"), float("inf"), float("inf"), hold

    def compute_success(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg,
        max_distance: float,
        max_angle_rad: float,
        max_lin_vel: float,
        max_ang_vel: float | None = None,
        max_lin_acc: float = float("inf"),
        max_ang_acc: float = float("inf"),
        vel_switch_step: int | None = None,
        pos_ramp_steps: int | None = None,
        pos_ease_factor: float = 3.0,
        hold_steps: int = 0,
        hold_square_max: int = 10,
        command_b: torch.Tensor | None = None,
        mix_half_and_one: bool = False,
        **_unused,
    ) -> torch.Tensor:
        if self._updated_step == env.common_step_counter:
            return self._last_success
        if max_ang_vel is None:
            max_ang_vel = max_lin_vel

        if mix_half_and_one and not self._mix_armed:
            self._mix_armed = True
            self._resample_gate_mult(None)
        elif not mix_half_and_one:
            self._mix_armed = False
            self._gate_mult.fill_(1.0)

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
        if self._mix_armed:
            distance = distance * self._gate_mult
            angle = angle * self._gate_mult
            lin_vel = lin_vel * self._gate_mult
            ang_vel = ang_vel * self._gate_mult
        if hold <= 0:
            hold = max(int(hold_square_max), 0)
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
        self._hold_count[:] = torch.where(instant, self._hold_count + 1, torch.zeros_like(self._hold_count))
        if hold > 0:
            self._last_success[:] = self._hold_count >= hold
        else:
            self._last_success[:] = instant
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
        max_ang_vel: float | None = None,
        max_lin_acc: float = float("inf"),
        max_ang_acc: float = float("inf"),
        vel_switch_step: int | None = None,
        pos_ramp_steps: int | None = None,
        pos_ease_factor: float = 3.0,
        hold_steps: int = 0,
        hold_square_max: int = 10,
        hold_step_bonus: float = 0.1,
        mix_half_and_one: bool = False,
    ) -> torch.Tensor:
        self._note_path(env, command_name, asset_cfg)
        self.compute_success(
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
            hold_square_max=hold_square_max,
            mix_half_and_one=mix_half_and_one,
        )
        if self._rewarded_step == env.common_step_counter:
            return self._last_shaped

        count = self._hold_count
        hold = max(int(hold_steps), 0)
        if hold <= 0:
            hold = max(int(hold_square_max), 0)
        if hold <= 0:
            self._last_shaped[:] = self._last_success.float()
        else:
            increment = torch.where(
                (count > 0) & (count < hold),
                torch.full_like(self._last_shaped, float(hold_step_bonus)),
                torch.zeros_like(self._last_shaped),
            )
            success = torch.where(count == hold, torch.ones_like(self._last_shaped), torch.zeros_like(self._last_shaped))
            self._last_shaped[:] = increment + success
        arrived_now = self._last_success & ~self._arrived
        if bool(arrived_now.any()):
            elapsed = command_elapsed_s(env, command_name)
            detour = self._path_len / self._start_dist.clamp_min(1.0e-3)
            self._success_time = torch.where(arrived_now, elapsed, self._success_time)
            self._detour = torch.where(arrived_now, detour, self._detour)
            self._arrived |= arrived_now
        self._rewarded_step = env.common_step_counter
        return self._last_shaped


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
        max_ang_vel: float | None = None,
        max_lin_acc: float = float("inf"),
        max_ang_acc: float = float("inf"),
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
    """Reward position and orientation as a sum, so angle still pays while the hand is far.

    Position stays a tight kernel (1/e at 1/6 m). Orientation uses a wide kernel
    (1/e at 1 rad) and is not multiplied by the position term. The average keeps
    the peak at 1, so an existing positive weight means the same maximum rate.
    """
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
    angle = quat_error_magnitude(curr_quat_w, des_quat_w)
    return 0.5 * (torch.exp(-6.0 * distance) + torch.exp(-angle))


def position_orientation_twist_command_error_fine_grained(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    max_lin_vel: float = 0.10,
) -> torch.Tensor:
    """Scale the pose sum by combined twist so the peak is a stop in the pose.

    Twist is ``||(v, ω)||_2`` and is 1/e at ``max_lin_vel``. The factor is 1
    only when stopped, so the peak rate stays the reward weight.
    """
    pose = position_orientation_command_error_fine_grained(env, command_name, asset_cfg)
    asset: RigidObject = env.scene[asset_cfg.name]
    bid = _body_idx_single(env, asset_cfg)
    lin_spd = torch.norm(asset.data.body_lin_vel_w[:, bid, :], dim=-1)
    ang_spd = torch.norm(asset.data.body_ang_vel_w[:, bid, :], dim=-1)
    twist = torch.sqrt(lin_spd.square() + ang_spd.square())
    twist_scale = max(float(max_lin_vel), 1.0e-6)
    return pose * torch.exp(-twist / twist_scale)


def distance_linear_approach_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    support_distance: float = 0.08,
    max_angle_rad: float = 0.4,
    max_lin_vel: float = 0.08,
) -> torch.Tensor:
    """Mean of pose/twist slacks toward zero error inside ``support_distance``.

    Pay/no-pay is ``d <= support_distance`` only. Slack widths are the cfg
    8× bounds and do not follow the success curriculum. Twist is Jawale's
    ``||(v, ω)||_2`` vs ``max_lin_vel``. Over a bound that axis is 0 in the
    mean. Peak 1 only at ``(d, θ, twist) = 0``.
    """
    support_d = max(float(support_distance), 1e-6)
    eps_d = support_d
    max_angle = max(float(max_angle_rad), 1e-6)
    max_twist = max(float(max_lin_vel), 1e-6)

    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    bid = _body_idx_single(env, asset_cfg)

    des_pos_b = command[:, :3]
    origin_pos_w, origin_quat_w = command_origin_pose_w(env, command_name, asset)
    des_pos_w, _ = combine_frame_transforms(origin_pos_w, origin_quat_w, des_pos_b)
    distance = torch.norm(asset.data.body_pos_w[:, bid] - des_pos_w, dim=1)

    des_quat_w = quat_mul(origin_quat_w, command[:, 3:7])
    ang_err = quat_error_magnitude(asset.data.body_quat_w[:, bid], des_quat_w)
    lin_spd = torch.norm(asset.data.body_lin_vel_w[:, bid, :], dim=-1)
    ang_spd = torch.norm(asset.data.body_ang_vel_w[:, bid, :], dim=-1)
    twist = torch.sqrt(lin_spd.square() + ang_spd.square())

    s_d = torch.clamp(1.0 - distance / eps_d, min=0.0, max=1.0)
    s_th = torch.clamp(1.0 - ang_err / max_angle, min=0.0, max=1.0)
    s_tw = torch.clamp(1.0 - twist / max_twist, min=0.0, max=1.0)
    shaped = (s_d + s_th + s_tw) / 3.0
    return torch.where(distance <= support_d, shaped, torch.zeros_like(shaped))


def pose_twist_linear_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    reward_term_name: str = "reach_success_bonus",
    support_distance: float = 0.08,
    ori_scale: float = 1.0,
    lin_vel_scale: float = 1.0,
    ang_vel_scale: float = 1.0,
    hinge_scale: float = 1.0,
    outer_flat_scale: float = 1.2,
) -> torch.Tensor:
    """Overflow hinge vs ``hinge_scale`` × live success ε inside 8 cm.

    Live ε comes from ``reward_term_name``. Inside ``support_distance`` (8 cm):
    while ``hinge_scale > 0``, ``relu(d-ε)+…``; after cooling (scale 0), raw
    in the ball. Outside 8 cm always pays flat distance-only
    ``outer_flat_scale × support_distance`` from step 0.
    """
    gates = env.reward_manager.get_term_cfg(reward_term_name).params
    scale = max(float(hinge_scale), 0.0)
    eps_d = float(gates["max_distance"]) * scale
    eps_th = float(gates["max_angle_rad"]) * scale
    eps_v = float(gates["max_lin_vel"]) * scale
    eps_w = float(gates["max_ang_vel"]) * scale
    outer_d = float(support_distance)

    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    bid = _body_idx_single(env, asset_cfg)

    des_pos_b = command[:, :3]
    origin_pos_w, origin_quat_w = command_origin_pose_w(env, command_name, asset)
    des_pos_w, _ = combine_frame_transforms(origin_pos_w, origin_quat_w, des_pos_b)
    distance = torch.norm(asset.data.body_pos_w[:, bid] - des_pos_w, dim=1)

    des_quat_w = quat_mul(origin_quat_w, command[:, 3:7])
    ang_err = quat_error_magnitude(asset.data.body_quat_w[:, bid], des_quat_w)
    lin_spd = torch.norm(asset.data.body_lin_vel_w[:, bid, :], dim=-1)
    ang_spd = torch.norm(asset.data.body_ang_vel_w[:, bid, :], dim=-1)
    twist = torch.sqrt(lin_spd.square() + ang_spd.square())

    in_support = distance <= outer_d
    in_gate = reach_success_criteria(
        env,
        command_name=command_name,
        asset_cfg=asset_cfg,
        max_distance=eps_d,
        max_angle_rad=eps_th,
        max_lin_vel=eps_v,
        max_ang_vel=eps_w,
    )
    overflow = (
        torch.relu(distance - eps_d)
        + float(ori_scale) * torch.relu(ang_err - eps_th)
        + float(lin_vel_scale) * torch.relu(twist - eps_v)
    )
    outer_flat = torch.full_like(distance, float(outer_flat_scale) * outer_d)
    if scale <= 0.0:
        inner = torch.where(in_gate, torch.zeros_like(overflow), overflow)
    else:
        inner = torch.where(in_gate | ~in_support, torch.zeros_like(overflow), overflow)
    return torch.where(in_support, inner, outer_flat)


def position_orientation_command_error_last_centimeter(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    pos_scale: float = 50.0,
    ori_scale: float = 20.0,
    pos_cutoff: float = 0.03,
    ori_cutoff_basin: float = 0.15,
    still_cutoff: float = 0.02,
    ori_cutoff: float = 0.1,
    lin_vel_scale: float = 1.0 / 0.03,
    ang_vel_scale: float = 1.0 / 0.03,
    still_mix: float = 0.2,
    approach_vel: float = 0.03,
    approach_floor: float = 0.15,
) -> torch.Tensor:
    """Last-centimeter basin at 1.5x support, strong pose kernels at the gates.

    Zero outside the 1.5x pose box (``pos_cutoff`` / ``ori_cutoff_basin``).
    Pose 1/e is at the hold gates (2 cm) and half-gate ori (0.05 rad) so
    0.10–0.11 rad still has slope. Still kernels 1/e at 1.5x twist
    (0.03 m/s / 0.03 rad/s). Still is only paid inside the hold gates
    ``still_cutoff`` / ``ori_cutoff``; elsewhere the approach kernel
    (closing speed vs ``approach_vel``) is used.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_pos_b = command[:, :3]
    origin_pos_w, origin_quat_w = command_origin_pose_w(env, command_name, asset)
    des_pos_w, _ = combine_frame_transforms(origin_pos_w, origin_quat_w, des_pos_b)
    bid = _body_idx_single(env, asset_cfg)
    curr_pos_w = asset.data.body_pos_w[:, bid]
    offset = curr_pos_w - des_pos_w
    distance = torch.norm(offset, dim=1)

    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(origin_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, bid]
    ang_err = quat_error_magnitude(curr_quat_w, des_quat_w)

    lin_vel = asset.data.body_lin_vel_w[:, bid, :]
    lin_spd = torch.norm(lin_vel, dim=-1)
    ang_spd = torch.norm(asset.data.body_ang_vel_w[:, bid, :], dim=-1)

    pose = torch.exp(-pos_scale * distance) * torch.exp(-ori_scale * ang_err)
    still = torch.exp(-lin_vel_scale * lin_spd) * torch.exp(-ang_vel_scale * ang_spd)
    mix = min(max(float(still_mix), 0.0), 1.0)
    still_kernel = pose * (mix + (1.0 - mix) * still)

    radial = offset / distance.clamp(min=1e-6).unsqueeze(-1)
    closing = -(lin_vel * radial).sum(dim=-1)
    ref = max(float(approach_vel), 1e-6)
    approach = (closing / ref).clamp(min=0.0, max=1.0)
    floor = min(max(float(approach_floor), 0.0), 1.0)
    approach_kernel = pose * (floor + (1.0 - floor) * approach)

    in_hold = (distance < still_cutoff) & (ang_err < ori_cutoff)
    kernel = torch.where(in_hold, still_kernel, approach_kernel)
    in_basin = (distance < pos_cutoff) & (ang_err < ori_cutoff_basin)
    return torch.where(in_basin, kernel, torch.zeros_like(kernel))


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


def CRI_OVF_lin(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    limit: float = 0.96,
) -> torch.Tensor:
    """Zero at or below ``limit``, then linear up to 1 at the live OVF threshold.

    Same ``max(CRI)`` readout as :func:`CRI_OVF_exp` and the OVF termination.
    No penalty on the approach below ``limit``. Above it, the value is
    ``(CRI - limit) / (threshold - limit)``, clamped to 1. Use a negative reward weight.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cri, _ = torch.max(asset.data.CRI, dim=1)
    try:
        threshold = float(env.termination_manager.get_term_cfg("OVF").params["threshold"])
    except (ValueError, KeyError, AttributeError):
        threshold = limit
    excess = (cri - limit).clamp(min=0.0)
    span = max(threshold - limit, 1e-6)
    return (excess / span).clamp(max=1.0)


def ovf_rewind_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """1 on a step ``rewind_arm_on_cri_limit`` will rewind, otherwise 0.

    Reward runs before the interval event, so this repeats that event's test:
    stored spawn exists, max CRI is above ``terminations.OVF``, and the env is
    not already resetting. Weight 15 pays +1 on the return, one step of the clock.
    """
    n = env.num_envs
    if not hasattr(env, "_reach_spawn_joint_pos"):
        return torch.zeros(n, device=env.scene["robot"].device)
    term = getattr(env, "termination_manager", None)
    names = getattr(term, "active_terms", None)
    if term is None or names is None or "OVF" not in names:
        return torch.zeros(n, device=env.scene["robot"].device)
    asset: Articulation = env.scene["robot"]
    cri, _ = torch.max(asset.data.CRI, dim=1)
    threshold = float(term.get_term_cfg("OVF").params["threshold"])
    crossed = cri > threshold
    reset_buf = getattr(env, "reset_buf", None)
    if reset_buf is not None:
        crossed = crossed & ~reset_buf
    return crossed.float()


class clean_attempt_time_bonus(ManagerTermBase):
    """Pay back the attempt's ``is_alive`` clock on the rewind step.

    Other steps return 0. The bank stores the raw ``is_alive`` value of each
    step in the current attempt, including the rewind step. The reward manager
    multiplies by ``weight * dt``. Weight +5 therefore returns +5 per second of
    that attempt when ``is_alive`` weight is -5, and the two sums cancel.
    Each rewind pays back only the attempt that just ended.

    Rewind is predicted the same way as :func:`ovf_rewind_bonus`, before the
    interval event runs. Success and timeout do not pay back: the bank is
    cleared on episode reset after those rewards have already been applied.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._bank = torch.zeros(env.num_envs, device=env.device)
        self._last = torch.zeros(env.num_envs, device=env.device)
        self._step = -1

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None or isinstance(env_ids, slice):
            self._bank.zero_()
        else:
            self._bank[env_ids] = 0.0
        return {}

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        if self._step == env.common_step_counter:
            return self._last
        self._step = env.common_step_counter
        alive = _is_alive_raw(env)
        rewind = _rewind_this_step(env)
        refund = self._bank + alive
        self._last = torch.where(rewind, refund, torch.zeros_like(refund))
        self._bank = torch.where(rewind, torch.zeros_like(self._bank), self._bank + alive)
        return self._last


def _is_alive_raw(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Same raw value as the ``is_alive`` reward, before its weight and dt."""
    try:
        params = env.reward_manager.get_term_cfg("is_alive").params
    except (ValueError, KeyError, AttributeError):
        params = {}
    return is_alive_time_ramp(env, **params)


def _rewind_this_step(env: ManagerBasedRLEnv) -> torch.Tensor:
    """True where ``rewind_arm_on_cri_limit`` will rewind after this reward."""
    n = env.num_envs
    device = env.scene["robot"].device
    if not hasattr(env, "_reach_spawn_joint_pos"):
        return torch.zeros(n, dtype=torch.bool, device=device)
    term = getattr(env, "termination_manager", None)
    names = getattr(term, "active_terms", None)
    if term is None or names is None or "OVF" not in names:
        return torch.zeros(n, dtype=torch.bool, device=device)
    asset: Articulation = env.scene["robot"]
    cri, _ = torch.max(asset.data.CRI, dim=1)
    threshold = float(term.get_term_cfg("OVF").params["threshold"])
    crossed = cri > threshold
    reset_buf = getattr(env, "reset_buf", None)
    if reset_buf is not None:
        crossed = crossed & ~reset_buf
    return crossed
