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
        self.num_bodies = len(self.body_indices)

        # create buffers: per-body force (각 링크마다 범위 내 랜덤 힘)
        # -- (num_envs, num_bodies, 3) for fx, fy, fz per link
        self.force_command = torch.zeros(self.num_envs, self.num_bodies, 3, device=self.device)
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
        """Net (합력) applied force in base frame. Shape is (num_envs, 3). 보상/관측용."""
        return self.force_current.sum(dim=1)

    @property
    def command_per_body(self) -> torch.Tensor:
        """Per-link force in base frame. Shape is (num_envs, num_bodies, 3). 이벤트 적용용."""
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

        self.metrics["force_norm"][:] = torch.linalg.norm(self.force_current.sum(dim=1), dim=1)
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
            ## 각 링크마다 범위 내 독립 랜덤 힘 (base frame fx, fy, fz)
            n_apply = len(apply_envs)
            for b in range(self.num_bodies):
                fx = torch.empty(n_apply, device=self.device).uniform_(*self.cfg.ranges.force_range_fx)
                fy = torch.empty(n_apply, device=self.device).uniform_(*self.cfg.ranges.force_range_fy)
                fz = torch.empty(n_apply, device=self.device).uniform_(*self.cfg.ranges.force_range_fz)
                self.force_command[apply_envs, b, 0] = fx
                self.force_command[apply_envs, b, 1] = fy
                self.force_command[apply_envs, b, 2] = fz
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
            scale = torch.where(ratio < 0.33, 0.0, torch.where(ratio < 0.66, 0.5, 1.0))
            self.force_current[ramp_mask] = scale.unsqueeze(1).unsqueeze(2) * self.force_command[ramp_mask]
            
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
            settling_time = self.force_timer[settling_mask] - (self.force_duration[settling_mask] - self.settling_duration[settling_mask])
            ratio = torch.clamp(settling_time / self.settling_duration[settling_mask], 0.0, 1.0)
            scale = torch.where(ratio < 0.33, 1.0, torch.where(ratio < 0.66, 0.5, 0.0))
            self.force_current[settling_mask] = scale.unsqueeze(1).unsqueeze(2) * self.force_command[settling_mask]

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
        if not self.robot.is_initialized:
            return
        # 링크별로 다른 힘이 적용되므로 각 body 위치에 해당 링크의 힘 크기/방향으로 화살표 표시
        offset = getattr(self.cfg, "debug_vis_height_offset", 0.0)
        pos_list = []
        scale_list = []
        quat_list = []
        for b in range(self.num_bodies):
            force_b = self.force_current[:, b, :]  # (num_envs, 3)
            arrow_scale, arrow_quat = self._resolve_force_to_arrow(force_b)
            pos_w = self.robot.data.body_pos_w[:, self.body_indices[b]].clone()
            pos_w[:, 2] += offset
            pos_list.append(pos_w)
            scale_list.append(arrow_scale)
            quat_list.append(arrow_quat)
        vis_pos_w = torch.stack(pos_list, dim=0).permute(1, 0, 2).reshape(-1, 3)
        vis_scale = torch.stack(scale_list, dim=0).permute(1, 0, 2).reshape(-1, 3)
        vis_quat = torch.stack(quat_list, dim=0).permute(1, 0, 2).reshape(-1, 4)
        self.applied_force_visualizer.visualize(
            translations=vis_pos_w,
            orientations=vis_quat,
            scales=vis_scale,
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