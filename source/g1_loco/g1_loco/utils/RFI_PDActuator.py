from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

from isaaclab.utils import DelayBuffer, LinearInterpolation
from isaaclab.utils.types import ArticulationActions

from isaaclab.actuators.actuator_base import ActuatorBase

if TYPE_CHECKING:
    from .RFI_PDActuatorCfg import (
        RFI_PDActuatorCfg,
    )


"""
Explicit Actuator Models.
"""
class RFI_PDActuator(ActuatorBase):

    cfg: RFI_PDActuatorCfg
    """The configuration for the actuator model."""

    """
    Operations.
    """
    def __init__(self, cfg: RFI_PDActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # parse configuration
        # self._rfi = self.r.uniform_(*self.cfg.rfi) 
        self.r = torch.zeros_like(self.computed_effort)
        # instantiate the delay buffers
        self.positions_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.velocities_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.efforts_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        # all of the envs
        self._ALL_INDICES = torch.arange(self._num_envs, dtype=torch.long, device=self._device)

    def reset(self, env_ids: Sequence[int]):
        self._rfi = self.r.uniform_(*self.cfg.rfi) 
        # number of environments (since env_ids can be a slice)
        if env_ids is None or env_ids == slice(None):
            num_envs = self._num_envs
        else:
            num_envs = len(env_ids)
        # set a new random delay for environments in env_ids
        time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self._device,
        )
        # set delays
        self.positions_delay_buffer.set_time_lag(time_lags, env_ids)
        self.velocities_delay_buffer.set_time_lag(time_lags, env_ids)
        self.efforts_delay_buffer.set_time_lag(time_lags, env_ids)
        # reset buffers
        self.positions_delay_buffer.reset(env_ids)
        self.velocities_delay_buffer.reset(env_ids)
        self.efforts_delay_buffer.reset(env_ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # apply delay based on the delay the model for all the setpoints
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)
        # compute errors
        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel
        # calculate the desired joint torques
        self.computed_effort = self.stiffness * error_pos + self.damping * error_vel + control_action.joint_efforts + self._rfi * self.effort_limit
        # clip the torques based on the motor limits
        self.applied_effort = self._clip_effort(self.computed_effort)
        # set the computed actions back into the control action
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action