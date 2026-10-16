# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reach-task-specific MDP event functions."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

from .rewards import reach_success_criteria


def reset_robot_joints_two_groups_by_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    primary_joint_names: str | list[str],
    primary_position_range: tuple[float, float],
    primary_velocity_range: tuple[float, float],
    secondary_joint_names: str | list[str],
    secondary_position_range: tuple[float, float],
    secondary_velocity_range: tuple[float, float],
):
    """Reset joint state from defaults with independent uniform offsets for two joint groups."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    n_env = joint_pos.shape[0]
    device = joint_pos.device

    def _names(names: str | list[str]) -> list[str]:
        return [names] if isinstance(names, str) else list(names)

    p_ids, _ = asset.find_joints(_names(primary_joint_names), preserve_order=True)
    s_ids, _ = asset.find_joints(_names(secondary_joint_names), preserve_order=True)

    if len(p_ids) > 0:
        j_p = torch.tensor(p_ids, device=device, dtype=torch.long)
        joint_pos[:, j_p] += math_utils.sample_uniform(*primary_position_range, (n_env, len(p_ids)), device)
        joint_vel[:, j_p] += math_utils.sample_uniform(*primary_velocity_range, (n_env, len(p_ids)), device)

    if len(s_ids) > 0:
        j_s = torch.tensor(s_ids, device=device, dtype=torch.long)
        joint_pos[:, j_s] += math_utils.sample_uniform(*secondary_position_range, (n_env, len(s_ids)), device)
        joint_vel[:, j_s] += math_utils.sample_uniform(*secondary_velocity_range, (n_env, len(s_ids)), device)

    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    asset.set_joint_position_target(joint_pos, env_ids=env_ids)
    asset.set_joint_velocity_target(joint_vel, env_ids=env_ids)


def reset_robot_joints_three_groups_by_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    primary_joint_names: str | list[str],
    primary_position_range: tuple[float, float],
    primary_velocity_range: tuple[float, float],
    secondary_joint_names: str | list[str],
    secondary_position_range: tuple[float, float],
    secondary_velocity_range: tuple[float, float],
    tertiary_joint_names: str | list[str],
    tertiary_position_range: tuple[float, float],
    tertiary_velocity_range: tuple[float, float],
):
    """Reset joint state from defaults with independent uniform offsets for three joint groups."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    n_env = joint_pos.shape[0]
    device = joint_pos.device

    def _names(names: str | list[str]) -> list[str]:
        return [names] if isinstance(names, str) else list(names)

    p_ids, _ = asset.find_joints(_names(primary_joint_names), preserve_order=True)
    s_ids, _ = asset.find_joints(_names(secondary_joint_names), preserve_order=True)
    t_ids, _ = asset.find_joints(_names(tertiary_joint_names), preserve_order=True)

    if len(p_ids) > 0:
        j_p = torch.tensor(p_ids, device=device, dtype=torch.long)
        joint_pos[:, j_p] += math_utils.sample_uniform(*primary_position_range, (n_env, len(p_ids)), device)
        joint_vel[:, j_p] += math_utils.sample_uniform(*primary_velocity_range, (n_env, len(p_ids)), device)

    if len(s_ids) > 0:
        j_s = torch.tensor(s_ids, device=device, dtype=torch.long)
        joint_pos[:, j_s] += math_utils.sample_uniform(*secondary_position_range, (n_env, len(s_ids)), device)
        joint_vel[:, j_s] += math_utils.sample_uniform(*secondary_velocity_range, (n_env, len(s_ids)), device)

    if len(t_ids) > 0:
        j_t = torch.tensor(t_ids, device=device, dtype=torch.long)
        joint_pos[:, j_t] += math_utils.sample_uniform(*tertiary_position_range, (n_env, len(t_ids)), device)
        joint_vel[:, j_t] += math_utils.sample_uniform(*tertiary_velocity_range, (n_env, len(t_ids)), device)

    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    asset.set_joint_position_target(joint_pos, env_ids=env_ids)
    asset.set_joint_velocity_target(joint_vel, env_ids=env_ids)


def _keep_failed_spawn(env: ManagerBasedEnv, env_ids: torch.Tensor) -> torch.Tensor:
    """True when this reset is an OVF or a horizon timeout, and not a reach success.

    The termination flags are still the ones from the step that triggered the reset.
    """
    keep = torch.zeros(env_ids.shape[0], dtype=torch.bool, device=env_ids.device)
    term = getattr(env, "termination_manager", None)
    names = getattr(term, "active_terms", None)
    if term is None or names is None:
        return keep
    failed = torch.zeros(env.num_envs, dtype=torch.bool, device=env_ids.device)
    if "OVF" in names:
        failed = failed | term.get_term("OVF").to(device=env_ids.device)
    if "time_out" in names:
        failed = failed | term.get_term("time_out").to(device=env_ids.device)
    if "reach_success" in names:
        failed = failed & ~term.get_term("reach_success").to(device=env_ids.device)
    return failed[env_ids]


def reset_robot_joints_by_name_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    position_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]] | None = None,
    keep_failed_spawn: bool = True,
):
    """Reset joint state from defaults with a per-joint uniform offset.

    With ``keep_failed_spawn``, OVF and the horizon timeout restore the joint
    pose from the start of that attempt, so neither failure draws a new arm.
    Reach success draws a new pose, which is then the spawn restored by a
    later failure. The target is kept separately by ``keep_command_until_success``.
    The early-training curriculum sets this flag false so every reset draws a
    new arm.

    Args:
        position_range: Mapping from joint name to ``(min, max)`` position offset.
        velocity_range: Mapping from joint name to ``(min, max)`` velocity offset.
            Joints missing from this dict default to ``(0.0, 0.0)``.
        keep_failed_spawn: Restore the stored spawn after OVF or timeout.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "_reach_spawn_joint_pos"):
        env._reach_spawn_joint_pos = asset.data.default_joint_pos.clone()
        env._reach_spawn_joint_vel = asset.data.default_joint_vel.clone()

    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    n_env = joint_pos.shape[0]
    device = joint_pos.device
    if velocity_range is None:
        velocity_range = {}

    joint_names = list(position_range.keys())
    joint_ids, resolved_names = asset.find_joints(joint_names, preserve_order=True)
    if len(joint_ids) != len(joint_names):
        raise ValueError(
            "Could not resolve all joints in position_range."
            f" Requested={joint_names}, resolved={resolved_names}."
        )

    for joint_id, joint_name in zip(joint_ids, resolved_names):
        pos_lo, pos_hi = position_range[joint_name]
        vel_lo, vel_hi = velocity_range.get(joint_name, (0.0, 0.0))
        joint_pos[:, joint_id] += math_utils.sample_uniform(pos_lo, pos_hi, (n_env,), device)
        joint_vel[:, joint_id] += math_utils.sample_uniform(vel_lo, vel_hi, (n_env,), device)

    if keep_failed_spawn:
        keep = _keep_failed_spawn(env, env_ids)
    else:
        keep = torch.zeros(env_ids.shape[0], dtype=torch.bool, device=env_ids.device)
    if bool(keep.any()):
        joint_pos[keep] = env._reach_spawn_joint_pos[env_ids[keep]]
        joint_vel[keep] = env._reach_spawn_joint_vel[env_ids[keep]]

    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    fresh = ~keep
    if bool(fresh.any()):
        env._reach_spawn_joint_pos[env_ids[fresh]] = joint_pos[fresh]
        env._reach_spawn_joint_vel[env_ids[fresh]] = joint_vel[fresh]

    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    asset.set_joint_position_target(joint_pos, env_ids=env_ids)
    asset.set_joint_velocity_target(joint_vel, env_ids=env_ids)


def _rewind_reach_spawn(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Put the arm back at the stored spawn. The episode is not reset."""
    if env_ids.numel() == 0 or not hasattr(env, "_reach_spawn_joint_pos"):
        return
    asset: Articulation = env.scene["robot"]
    joint_pos = env._reach_spawn_joint_pos[env_ids]
    joint_vel = env._reach_spawn_joint_vel[env_ids]
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    asset.set_joint_position_target(joint_pos, env_ids=env_ids)
    asset.set_joint_velocity_target(joint_vel, env_ids=env_ids)
    env.sim.forward()


def rewind_arm_on_cri_limit(env: ManagerBasedEnv, env_ids: torch.Tensor | None) -> None:
    """Rewind to the stored spawn when CRI crosses the OVF threshold.

    The episode continues and the target stays. The threshold is
    ``terminations.OVF``. Envs that reset on this step are left as the reset
    wrote them. ``_ovf_rewind_count`` is what the reach metrics read.
    """
    if not hasattr(env, "_reach_spawn_joint_pos"):
        return
    term = getattr(env, "termination_manager", None)
    names = getattr(term, "active_terms", None)
    if term is None or names is None or "OVF" not in names:
        return
    asset: Articulation = env.scene["robot"]
    cri, _ = torch.max(asset.data.CRI, dim=1)
    threshold = float(term.get_term_cfg("OVF").params["threshold"])
    crossed = cri > threshold
    if env_ids is not None:
        selected = torch.zeros_like(crossed)
        selected[env_ids] = True
        crossed = crossed & selected
    reset_buf = getattr(env, "reset_buf", None)
    if reset_buf is not None:
        crossed = crossed & ~reset_buf
    if not hasattr(env, "_ovf_rewind_count"):
        env._ovf_rewind_count = torch.zeros(env.num_envs, device=cri.device)
    env._ovf_rewind_count[crossed] += 1.0
    _rewind_reach_spawn(env, crossed.nonzero(as_tuple=False).flatten())


def resample_ee_pose_command_on_reach(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    max_distance: float,
    max_angle_rad: float,
    max_lin_vel: float,
    max_ang_vel: float | None = None,
    max_lin_acc: float = float("inf"),
    max_ang_acc: float = float("inf"),
) -> None:
    """Resample the pose command for environments that satisfy reach success criteria.

    ``max_lin_vel`` is the combined twist ``||(v, ω)||_2``. ``max_ang_vel`` is unused.
    """
    from isaaclab.envs import ManagerBasedRLEnv

    if not isinstance(env, ManagerBasedRLEnv):
        return
    sat = reach_success_criteria(
        env,
        command_name,
        asset_cfg,
        max_distance=max_distance,
        max_angle_rad=max_angle_rad,
        max_lin_vel=max_lin_vel,
        max_ang_vel=max_ang_vel,
        max_lin_acc=max_lin_acc,
        max_ang_acc=max_ang_acc,
    )
    if env_ids is not None:
        reached = env_ids[sat[env_ids]]
    else:
        reached = torch.where(sat)[0]
    if reached.numel() == 0:
        return
    term = env.command_manager.get_term(command_name)
    term.reset(env_ids=reached)


class early_spline_velocity_offset(ManagerTermBase):
    """One joint-velocity sample per episode.

    Drawn at reset and held until the next reset. The action multiplies it by
    sin(pi * t / T), so the added speed is 0 at the start and at arrival.
    ``std`` is in rad/s. While the curriculum keeps it at 0, nothing is drawn.
    After a positive std has been used and std returns to 0, drawing stops.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._offset: torch.Tensor | None = None
        self._retired = False
        self._saw_positive = False

    def __call__(self, env, env_ids, std: float):
        if self._retired:
            return
        if float(std) <= 0.0:
            if self._offset is not None:
                action = env.action_manager.get_term("arm_action")
                action._spline_offset = None
                self._offset = None
            if self._saw_positive:
                self._retired = True
            return
        self._saw_positive = True
        action = env.action_manager.get_term("arm_action")
        if self._offset is None:
            self._offset = torch.zeros(env.num_envs, action.action_dim, device=self.device)
            action._spline_offset = self._offset
        count = env.num_envs if isinstance(env_ids, slice) else len(env_ids)
        self._offset[env_ids] = torch.randn(count, action.action_dim, device=self.device) * float(std)
