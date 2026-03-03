# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generator that does nothing."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import UniformForceCommandCfg

# Phase 정의 (정수형)
PHASE_INTERVAL = 0
PHASE_RAMP = 1
PHASE_ACTIVE = 2
PHASE_SETTLING = 3


class UniformForceCommand(CommandTerm):
    """Command generator for generating force commands uniformly.
    """

    cfg: UniformForceCommandCfg
    """Configuration for the command generator."""
    def __init__(self, cfg: UniformForceCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator."""
        super().__init__(cfg, env)

        # check configuration


        # obtain the robot asset and body indices (single or multiple)
        self.robot: Articulation = env.scene[cfg.asset_name]
        if cfg.body_names is not None:
            body_keys = cfg.body_names
        elif cfg.body_name is not None:
            body_keys = [cfg.body_name]
        else:
            raise ValueError("UniformForceCommandCfg: set either body_name or body_names.")
        self.body_indices = list(self.robot.find_bodies(body_keys, preserve_order=True)[0])
        self.body_idx = self.body_indices[0]  # for debug vis / first body

        # crete buffers
        # -- commands: (fx, fy, fz)
        self.force_command = torch.zeros(self.num_envs, 3, device=self.device)
        self.force_current = torch.zeros_like(self.force_command)
        
        self.is_force_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # time buffers: interval->ramp->duration->settling
        self.force_timer = torch.zeros(self.num_envs, device=self.device)
        self.force_duration = torch.zeros(self.num_envs, device=self.device)

        self.interval_timer = torch.zeros(self.num_envs, device=self.device)
        self.interval = torch.zeros(self.num_envs, device=self.device)

        self.ramp_timer = torch.zeros(self.num_envs, device=self.device)
        self.ramp_duration = torch.zeros(self.num_envs, device=self.device)

        self.settling_timer = torch.zeros(self.num_envs, device=self.device)
        self.settling_duration = torch.zeros(self.num_envs, device=self.device)

        # phase: interval / ramp / active / settling
        self.phase = torch.full((self.num_envs,), PHASE_INTERVAL, dtype=torch.long, device=self.device)

        # -- metrics
        self.metrics["force_active_ratio"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["force_norm"] = torch.zeros(self.num_envs, device=self.device)

        # -- initial interval sample
        self.interval.uniform_(*cfg.ranges.interval_range_s)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformForceCommand:\n"
        msg += "\tCommand dimension: N/A\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        msg += f"\tForce Range fx: {self.cfg.ranges.force_range_fx}, fy: {self.cfg.ranges.force_range_fy}, fz: {self.cfg.ranges.force_range_fz}\n"
        msg += f"\tDuration Range: {self.cfg.ranges.duration_range_s}\n"
        msg += f"\tActive Envs: {self.is_force_active.sum().item()}/{self.num_envs}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Current applied force (base frame). Shape is (num_envs, 3)."""
        return self.force_current

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # logs data 
        # force_active_ratio는 외력이 가해지는 비율-시간
        # force_norm_mean는 가해지는 외력의 평균 크기
        # active = self.is_force_active
        self.metrics["force_active_ratio"] = self.is_force_active.float()

        self.metrics["force_norm"][:] = torch.linalg.norm(self.force_current, dim=1)
        # if active.any():
        #     self.metrics["force_norm"] = (
        #         torch.linalg.norm(self.force_current[active], dim=1)
        #     )
        # else:
        #     self.metrics["force_norm"].zero_()


    def _resample_command(self, env_ids: Sequence[int]):
        """sample force commands in a event form"""

        # force event를 위한 envs 
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        ready = self.interval_timer[env_ids_t] >= self.interval[env_ids_t]
        ready_envs = env_ids_t[ready]

        if ready_envs.numel() == 0:
            return

        # force를 주는 events
        r = torch.rand(len(ready_envs), device=self.device)
        apply_mask = r < self.cfg.apply_probability
        apply_envs = ready_envs[apply_mask]

        if apply_envs.numel() > 0:
            ## force components (base frame): fx, fy, fz — like vx, vy, vz for velocity
            fx = torch.empty(len(apply_envs), device=self.device).uniform_(*self.cfg.ranges.force_range_fx)
            fy = torch.empty(len(apply_envs), device=self.device).uniform_(*self.cfg.ranges.force_range_fy)
            fz = torch.empty(len(apply_envs), device=self.device).uniform_(*self.cfg.ranges.force_range_fz)
            self.force_command[apply_envs, 0] = fx
            self.force_command[apply_envs, 1] = fy
            self.force_command[apply_envs, 2] = fz
            self.force_current[apply_envs].zero_()
            self.is_force_active[apply_envs] = True

            ## duration
            self.force_duration[apply_envs].uniform_(*self.cfg.ranges.duration_range_s)
            # self.force_timer[apply_envs].zero_()

            self.ramp_duration[apply_envs] = 0.3 * self.force_duration[apply_envs]
            self.settling_duration[apply_envs] = 0.3 * self.force_duration[apply_envs]

            self.force_timer[apply_envs].zero_()
            self.ramp_timer[apply_envs].zero_()
            self.settling_timer[apply_envs].zero_()

            self.phase[apply_envs] = PHASE_RAMP

        # resample interval
        self.interval[ready_envs].uniform_(*self.cfg.ranges.interval_range_s)
        self.interval_timer[ready_envs].zero_()

    def _update_command(self):
        dt = self._env.step_dt
        self.interval_timer += dt
        
        # 활성화된 환경만 타이머 업데이트
        active_mask = self.is_force_active
        self.force_timer[active_mask] += dt

        # 1. RAMP PHASE
        ramp_mask = (self.phase == PHASE_RAMP) & active_mask
        if ramp_mask.any():
            ratio = torch.clamp(self.force_timer[ramp_mask] / self.ramp_duration[ramp_mask], 0.0, 1.0)
            # 계단 함수 로직 (제공해주신 방식 유지)
            scale = torch.where(ratio < 0.33, 0.0, torch.where(ratio < 0.66, 0.5, 1.0))
            self.force_current[ramp_mask] = scale.unsqueeze(1) * self.force_command[ramp_mask]
            
            # 다음 페이즈로 전환
            done_ramp = self.force_timer[ramp_mask] >= self.ramp_duration[ramp_mask]
            if done_ramp.any():
                self.phase[ramp_mask][done_ramp] = PHASE_ACTIVE

        # 2. ACTIVE PHASE
        active_phase_mask = (self.phase == PHASE_ACTIVE) & active_mask
        if active_phase_mask.any():
            self.force_current[active_phase_mask] = self.force_command[active_phase_mask]
            # 전체 duration에서 ramp와 settling 시간을 뺀 만큼 유지하거나 적절히 판단
            expired = self.force_timer[active_phase_mask] >= (self.force_duration[active_phase_mask] - self.settling_duration[active_phase_mask])
            if expired.any():
                self.phase[active_phase_mask][expired] = PHASE_SETTLING

        # 3. SETTLING PHASE
        settling_mask = (self.phase == PHASE_SETTLING) & active_mask
        if settling_mask.any():
            # Settling 전용 타이머 계산 (Active 종료 시점부터의 시간)
            settling_time = self.force_timer[settling_mask] - (self.force_duration[settling_mask] - self.settling_duration[settling_mask])
            ratio = torch.clamp(settling_time / self.settling_duration[settling_mask], 0.0, 1.0)
            scale = torch.where(ratio < 0.33, 1.0, torch.where(ratio < 0.66, 0.5, 0.0))
            self.force_current[settling_mask] = scale.unsqueeze(1) * self.force_command[settling_mask]

            # 종료 처리
            done = settling_time >= self.settling_duration[settling_mask]
            if done.any():
                done_indices = settling_mask.nonzero(as_tuple=False).flatten()[done]
                self.reset(done_indices)


    def reset(self, env_ids: Sequence[int] | None = None):
        # 부모 reset 호출 (metrics 처리 + _resample 호출)
        extras = super().reset(env_ids)

        # env_ids 정규화
        # if env_ids is None:
        #     env_ids = slice(None)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device)

        # force 관련 내부 상태 완전 초기화
        self.force_command[env_ids] = 0.0
        self.force_current[env_ids] = 0.0
        self.is_force_active[env_ids] = False

        self.force_timer[env_ids] = 0.0
        self.ramp_timer[env_ids] = 0.0
        self.settling_timer[env_ids] = 0.0

        self.phase[env_ids] = PHASE_INTERVAL
        self.interval[env_ids].uniform_(*self.cfg.ranges.interval_range_s)

        return extras
    

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "applied_force_visualizer"):
                # -- current applied force state
                self.applied_force_visualizer = VisualizationMarkers(self.cfg.applied_force_visualizer_cfg)
            # set their visibility to true
            self.applied_force_visualizer.set_visibility(True)
        else:
            if hasattr(self, "applied_force_visualizer"):
                self.applied_force_visualizer.set_visibility(False)            


    def _debug_vis_callback(self, event):
        # check if robot is initialized
        if not self.robot.is_initialized:
            return
        # body 위치 (world frame) — __init__에서 이미 계산한 첫 body 인덱스 사용 (body_name/body_names 둘 다 대응)
        body_pos_w = self.robot.data.body_pos_w[:, self.body_idx]

        # 시각화를 위해 약간 위로 띄움 (겹침 방지)
        vis_pos_w = body_pos_w.clone()
        vis_pos_w[:, 2] += getattr(self.cfg, "debug_vis_height_offset", 0.0)

        # force command (base frame)
        force_b = self.force_current  # (num_envs, 3)

        # arrow scale & orientation 계산
        arrow_scale, arrow_quat = self._resolve_force_to_arrow(force_b)

        # marker 표시
        self.applied_force_visualizer.visualize(
            translations=vis_pos_w,
            orientations=arrow_quat,
            scales=arrow_scale,
        )
    """
    Internal helpers.
    """

    def _resolve_force_to_arrow(
        self, force_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert force vector (base frame) to arrow scale and quaternion (world frame)."""

        # 기본 marker scale
        default_scale = self.applied_force_visualizer.cfg.markers["arrow"].scale
        arrow_scale = (
            torch.tensor(default_scale, device=self.device)
            .repeat(force_b.shape[0], 1)
        )

        # arrow size about force magnitude
        force_norm = torch.linalg.norm(force_b, dim=1)
        arrow_scale[:, 0] *= force_norm * self.cfg.force_vis_scale

        # force direction base_frame
        force_dir = force_b / (force_norm.unsqueeze(1) + 1e-6)

        # base frame에서 force 방향을 향하는 quaternion
        # 기본 arrow가 +X 방향을 보고 있다고 가정
        heading = torch.atan2(force_dir[:, 1], force_dir[:, 0])
        pitch = torch.atan2(
            -force_dir[:, 2],
            torch.linalg.norm(force_dir[:, :2], dim=1),
        )
        zeros = torch.zeros_like(heading)

        arrow_quat_b = math_utils.quat_from_euler_xyz(zeros, pitch, heading)
        # base → world 변환
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat_w = math_utils.quat_mul(base_quat_w, arrow_quat_b)

        return arrow_scale, arrow_quat_w