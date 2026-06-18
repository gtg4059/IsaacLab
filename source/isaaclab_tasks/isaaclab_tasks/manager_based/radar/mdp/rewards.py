# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import math
import torch

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor, RayCaster


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    from isaaclab.sensors import ContactSensor

    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt).torch[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time.torch[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    from isaaclab.sensors import ContactSensor

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    from isaaclab.sensors import ContactSensor

    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    )
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned
    robot frame using an exponential kernel.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w.torch), asset.data.root_lin_vel_w.torch[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w.torch[:, 2]
    )
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.linalg.norm(command[:, :2], dim=1) < command_threshold)

def base_height_log(
    env,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    from isaaclab.assets import Articulation
    from isaaclab.sensors import RayCaster

    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty with log scaling
    sq = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    return torch.clip(torch.log1p(5 * sq) / 2, max=0.3)

def contact_forces_minimize(env, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize contact forces as the amount of violations of the net contact force."""
    from isaaclab.sensors import ContactSensor

    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # compute the violation
    violation = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] - threshold
    return torch.sum((violation.clip(min=0.0)) ** 2, dim=1)

# def foot_clearance_reward(
#     env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
# ) -> torch.Tensor:
#     """Reward the swinging feet for clearing a specified height off the ground"""
#     asset: RigidObject = env.scene[asset_cfg.name]
#     foot_z_target_error = torch.square(asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2] - target_height)
#     foot_velocity_tanh = torch.tanh(
#         tanh_mult * torch.linalg.norm(asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2], dim=2)
#     )
#     reward = foot_z_target_error * foot_velocity_tanh
#     return torch.exp(-torch.sum(reward, dim=1) / std)

def feet_gait_walk_adaptive(
    env: "ManagerBasedRLEnv",
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.55,
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
    v_gate: float = 0.1,
    w_gate: float = 0.07,
    mode_scales: dict[str, float] | None = None,
    t_max: float = 1.0,
    t_min: float = 0.45,
    v_floor: float = 0.2,
    exponent: float = 1.0 / 3.0,
) -> torch.Tensor:
    """속도 적응형 gait-clock vs contact agreement (§ 5.6 Pillar 1 LIPM 리듬).

    ``feet_gait`` 의 고정 ``period`` 를 속도 조건부 ``T(v)`` 로 교체한 Phase 6.2
    구현. 수식 (§ 5.6.3 Alexander power-law)::

        T_target(v) = clamp( t_max · (v_floor / max(|v|, v_floor))^exponent,
                             t_min, t_max )

    STAND/TRANSLATE/CURVE 시만 활성 (``feet_gait`` 와 동일한 ``command_threshold``
    gate). SPIN 은 ``feet_gait_spin`` 이 담당하므로 본 함수에서는 자연히 꺼진다.

    ``mode_scales`` 는 CURVE 에서 직선 보행 리듬 완화 등 기존 용법을 그대로 이어받는다.

    Args:
        env: Isaac Lab 환경.
        offset: feet 간 phase offset (예: ``[0.0, 0.5]``).
        sensor_cfg: feet contact sensor 설정.
        threshold: stance 비율 (기본 0.55, duty factor β).
        command_name: velocity 명령 term 이름.
        command_threshold: ``‖v_cmd_xy‖`` 가 이보다 작으면 reward = 0 (STAND 분리).
        v_gate, w_gate: ``mode_scales`` 평가용 게이트 (``_gates.mode_scale`` 규약).
        mode_scales: 모드별 배율 (예: ``{"curve": 0.6}``). ``None`` 이면 미적용.
        t_max: 저속 (|v| ≤ ``v_floor``) 에서 step period 천장 [s].
        t_min: 고속 포화 step period 바닥 [s].
        v_floor: ``T(v)`` 계산용 속도 하한 [m/s]. 분모 0 방지 + STAND 연속성.
        exponent: Alexander power-law 지수 (기본 1/3).

    Returns:
        ``[num_envs]`` reward. ``_feet_gait_phase_reward_adaptive`` 와 동일 범위.
    """
    cmd = env.command_manager.get_command(command_name)
    v_norm = torch.linalg.norm(cmd[:, :2], dim=1)  # [num_envs]

    # T_target(v) — in-place chain 으로 메모리 할당 최소화.
    T = v_norm.clamp_min(v_floor)
    T = (v_floor / T).pow_(exponent).mul_(t_max).clamp_(t_min, t_max)  # [num_envs]

    reward = _feet_gait_phase_reward_adaptive(env, T, offset, sensor_cfg, threshold)

    # command gate (‖v_xy‖ < command_threshold 이면 STAND — reward 0).
    reward = reward * (v_norm > command_threshold)

    if mode_scales is not None:
        reward = reward * mode_scale(env, command_name, v_gate, w_gate, mode_scales)
    return reward

def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact_time = contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    asset = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    reward = torch.where(
        torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=-1, keepdim=True) > 0.05, reward, 0
    )
    return torch.exp(-torch.sum(reward, dim=1) / std) * single_stance


def _feet_gait_phase_reward_adaptive(
    env: "ManagerBasedRLEnv",
    T: torch.Tensor,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Contact vs stance-clock agreement with **per-env adaptive period**.

    ``_feet_gait_phase_reward`` 의 scalar ``period`` 를 env 별 tensor ``T`` 로 확장한
    근사 구현. 명령이 자주 리샘플되면 phase 에 step 불연속이 발생할 수 있으나,
    UniformMixed command 의 `resampling_time_range` 가 수 초 단위라 학습에는
    영향이 작다. 정확한 phase 적분이 필요하면 env 별 ``phase_buf`` 상태를 도입해
    승격한다 (설계 문서 § 5.6.4 의 2차 구현).

    Args:
        env: Isaac Lab 환경.
        T: ``[num_envs]`` float tensor, env 별 target step period [s].
        offset: feet 간 phase offset (예: ``[0.0, 0.5]``).
        sensor_cfg: feet contact sensor 설정.
        threshold: stance 비율 (phase < threshold 인 구간을 stance 로 간주).

    Returns:
        ``[num_envs]`` reward — 계획된 stance-clock 과 실제 contact 일치 개수
        (feet 수 만큼 최대, ``_feet_gait_phase_reward`` 와 동일 범위).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    # per-env elapsed time; episode_length_buf 는 int → float 캐스팅 후 T 로 정규화.
    elapsed = env.episode_length_buf.to(T.dtype) * env.step_dt  # [num_envs]
    global_phase = (elapsed / T).fmod_(1.0).unsqueeze(1)  # [num_envs, 1]

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i, offset_ in enumerate(offset):
        phase = (global_phase[:, 0] + offset_) % 1.0  # [num_envs]
        is_stance = phase < threshold
        reward += ~(is_stance ^ is_contact[:, i])
    return reward
    
MODE_STAND = "stand"
MODE_TRANSLATE = "translate"
MODE_SPIN = "spin"
MODE_CURVE = "curve"
MODES: tuple[str, ...] = (MODE_STAND, MODE_TRANSLATE, MODE_SPIN, MODE_CURVE)

# Phase 6.4.x P1 α — TRANSLATE blend endpoint key 와 보간 exponent.
# SSOT (§ 5.5.14.1.a): w_lateral = (θ / (π/2)) ** 2.
TRANSLATE_LATERAL_KEY = "translate_lateral"
TRANSLATE_BLEND_EXPONENT = 2.0

def mode_scale(
    env: "ManagerBasedRLEnv",
    command_name: str,
    v_gate: float,
    w_gate: float,
    scales: dict[str, float],
) -> torch.Tensor:
    """모드별 스칼라 가중치를 환경별 ``float`` 텐서로 펼친다.

    ``scales`` 에 누락된 모드는 기본값 ``1.0`` 으로 처리한다. 리워드 함수에서
    ``reward = base_reward * mode_scale(...)`` 형태로 곱해 모드별 강도를 조절할 때 사용한다.

    **Phase 6.4.x P1 α (Soft blend)**: TRANSLATE 샘플에 대해서만 두 endpoint
    ``scales["translate"]`` (= forward_ish 기본값) 와 ``scales[TRANSLATE_LATERAL_KEY]``
    (= 측방 endpoint) 를 ``w_lateral = (θ/(π/2))**2`` 로 연속 보간한다.
    ``TRANSLATE_LATERAL_KEY`` 가 ``scales`` 에 없으면 ``"translate"`` 값으로 fallback
    (Phase 6.3 동작과 동일 → backward-compatible).

    Args:
        env: 환경 인스턴스.
        command_name: 속도 명령 텀 이름.
        v_gate, w_gate: :func:`mode_mask` 와 동일.
        scales: ``{"stand": 1.0, "translate": 1.0, "translate_lateral": 0.4, "spin": 0.3,
            "curve": 0.7}`` 처럼 모드명 → 배율. 누락된 키는 ``1.0`` (단, ``translate_lateral``
            은 ``translate`` 값으로 fallback).

    Returns:
        ``(num_envs,)`` shape 의 ``float`` 텐서.
    """
    masks = mode_mask(env, command_name, v_gate, w_gate)
    result = torch.ones(env.num_envs, device=env.device, dtype=torch.float32)

    # STAND/SPIN/CURVE — 기존 동작 유지 (단일 scalar).
    for mode in (MODE_STAND, MODE_SPIN, MODE_CURVE):
        s = float(scales.get(mode, 1.0))
        if s == 1.0:
            continue
        result = torch.where(masks[mode], torch.full_like(result, s), result)

    # TRANSLATE — P1 α 보간. lateral endpoint 누락 시 forward 값으로 fallback (no-op).
    s_fwd = float(scales.get(MODE_TRANSLATE, 1.0))
    s_lat = float(scales.get(TRANSLATE_LATERAL_KEY, s_fwd))
    if s_fwd == 1.0 and s_lat == 1.0:
        return result  # 두 endpoint 모두 1.0 → translate 처방 무변경 (early-return).

    cmd = env.command_manager.get_command(command_name)
    w_lat = translate_blend_weight(cmd[:, :2])
    blended = (1.0 - w_lat) * s_fwd + w_lat * s_lat
    result = torch.where(masks[MODE_TRANSLATE], blended, result)
    return result

def mode_mask(
    env: "ManagerBasedRLEnv",
    command_name: str,
    v_gate: float,
    w_gate: float,
) -> dict[str, torch.Tensor]:
    """명령 기반으로 환경별 boolean 마스크를 반환한다.

    Args:
        env: 환경 인스턴스.
        command_name: 속도 명령 텀 이름 (예: ``"base_velocity"``). 첫 두 성분은 ``(vx, vy)``, 세 번째는 ``omega_z``.
        v_gate: 평면 속도 노름 임계 [m/s]. 이하이면 "정지에 가깝다" 로 간주.
        w_gate: 각속도 절대값 임계 [rad/s]. 이하이면 "회전이 없다" 로 간주.

    Returns:
        각 모드명 → ``(num_envs,)`` shape 의 ``bool`` 텐서. 네 마스크는 배타적·완전하므로
        어느 프레임에서든 정확히 하나만 ``True`` 다.
    """
    cmd = env.command_manager.get_command(command_name)
    planar = torch.linalg.norm(cmd[:, :2], dim=1)
    omega = torch.abs(cmd[:, 2])

    v_big = planar > v_gate
    w_big = omega > w_gate

    return {
        MODE_STAND: (~v_big) & (~w_big),
        MODE_TRANSLATE: v_big & (~w_big),
        MODE_SPIN: (~v_big) & w_big,
        MODE_CURVE: v_big & w_big,
    }

def translate_blend_weight(
    cmd_planar_vec: torch.Tensor,
    *,
    eps: float = 1e-6,
    exponent: float = TRANSLATE_BLEND_EXPONENT,
) -> torch.Tensor:
    """TRANSLATE 명령에 대한 측방 blend weight ``w_lateral ∈ [0, 1]`` 를 계산한다.

    ``θ = atan2(|cmd_vy|, |cmd_vx|)`` 를 ``[0, π/2]`` 로 두고
    ``w = (θ / (π/2)) ** exponent`` 를 반환한다. 직진 (``θ=0``) 에서 ``w=0``,
    측방 (``θ=π/2``) 에서 ``w=1``. exponent ``2`` 는 SSOT 권장값으로 직진 근처에서 flat,
    측방 근처에서 steep 한 shape 를 만든다 (45° 에서 ``w=0.25``).

    Args:
        cmd_planar_vec: ``(num_envs, 2)`` shape 의 명령 ``(cmd_vx, cmd_vy)`` 텐서.
        eps: ``θ`` 분모 안정화용 작은 값. ``cmd_planar_vec`` 가 0 벡터인 STAND 샘플에서
            ``atan2(0, 0) = 0`` 이지만 수치 안정성용.
        exponent: 보간 곡선 지수. SSOT 기본값 ``2.0``.

    Returns:
        ``(num_envs,)`` shape 의 float 텐서, ``[0, 1]``. STAND/SPIN/CURVE 샘플에 대해서도
        값이 정의되지만 (cmd 자체로부터 계산) ``mode_scale`` 안에서는 TRANSLATE 마스크로
        제한 적용된다.
    """
    abs_vx = torch.abs(cmd_planar_vec[:, 0])
    abs_vy = torch.abs(cmd_planar_vec[:, 1])
    theta = torch.atan2(abs_vy, abs_vx + eps)  # [0, π/2]
    w = theta / (math.pi / 2.0)
    return torch.clamp(w, min=0.0, max=1.0).pow(exponent)