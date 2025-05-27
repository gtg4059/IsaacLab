# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.string as string_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul, compute_pose_error
from dataclasses import MISSING
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

    from . import actions_cfg


class JointAction(ActionTerm):
    r"""Base class for joint actions.

    This action term performs pre-processing of the raw actions using affine transformations (scale and offset).
    These transformations can be configured to be applied to a subset of the articulation's joints.

    Mathematically, the action term is defined as:

    .. math::

       \text{action} = \text{offset} + \text{scaling} \times \text{input action}

    where :math:`\text{action}` is the action that is sent to the articulation's actuated joints, :math:`\text{offset}`
    is the offset applied to the input action, :math:`\text{scaling}` is the scaling applied to the input
    action, and :math:`\text{input action}` is the input action from the user.

    Based on above, this kind of action transformation ensures that the input and output actions are in the same
    units and dimensions. The child classes of this action term can then map the output action to a specific
    desired command of the articulation's joints (e.g. position, velocity, etc.).
    """

    cfg: actions_cfg.JointActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor | float
    """The scaling factor applied to the input action."""
    _offset: torch.Tensor | float
    """The offset applied to the input action."""
    _clip: torch.Tensor
    """The clip applied to the input action."""

    def __init__(self, cfg: actions_cfg.JointActionCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        # log the resolved joint names for debugging
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )
        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints and not self.cfg.preserve_order:
            self._joint_ids = slice(None)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # parse scale
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.scale, self._joint_names)
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")
        # parse offset
        if isinstance(cfg.offset, (float, int)):
            self._offset = float(cfg.offset)
        elif isinstance(cfg.offset, dict):
            self._offset = torch.zeros_like(self._raw_actions)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.offset, self._joint_names)
            self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported offset type: {type(cfg.offset)}. Supported types are float and dict.")
        # parse clip
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, self.action_dim, 1
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, self._joint_names)
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        self._processed_actions = self._raw_actions * self._scale + self._offset
        # clip actions
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0


class JointPositionAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: actions_cfg.JointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.JointPositionActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

    def apply_actions(self):
        # set position targets
        # print("self._joint_ids:",self._joint_ids)
        # print("self._joint_names:",self._joint_names)
        # print("self.processed_actions:",self.processed_actions)
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)


class RelativeJointPositionAction(JointAction):
    r"""Joint action term that applies the processed actions to the articulation's joints as relative position commands.

    Unlike :class:`JointPositionAction`, this action term applies the processed actions as relative position commands.
    This means that the processed actions are added to the current joint positions of the articulation's joints
    before being sent as position commands.

    This means that the action applied at every step is:

    .. math::

         \text{applied action} = \text{current joint positions} + \text{processed actions}

    where :math:`\text{current joint positions}` are the current joint positions of the articulation's joints.
    """

    cfg: actions_cfg.RelativeJointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.RelativeJointPositionActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use zero offset for relative position
        if cfg.use_zero_offset:
            self._offset = 0.0

    def apply_actions(self):
        # add current joint positions to the processed actions
        current_actions = self.processed_actions + self._asset.data.joint_pos[:, self._joint_ids]
        # set position targets
        self._asset.set_joint_position_target(current_actions, joint_ids=self._joint_ids)


class JointVelocityAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as velocity commands."""

    cfg: actions_cfg.JointVelocityActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.JointVelocityActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint velocity as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_vel[:, self._joint_ids].clone()

    def apply_actions(self):
        # set joint velocity targets
        self._asset.set_joint_velocity_target(self.processed_actions, joint_ids=self._joint_ids)


class JointEffortAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as effort commands."""

    cfg: actions_cfg.JointEffortActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.JointEffortActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def apply_actions(self):
        # set joint effort targets
        self._asset.set_joint_effort_target(self.processed_actions, joint_ids=self._joint_ids)

class JointVelocityActionScale(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as velocity commands."""

    cfg: actions_cfg.JointVelocityActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.JointVelocityActionScaleCfg, env: ManagerBasedRLEnv):
        # initialize the action term
        super().__init__(cfg, env)
        self.asset: RigidObject = env.scene[cfg.asset_cfg.name]
        self.command = env.command_manager.get_command(cfg.command_name)
        #self.target = torch.zeros_like(self.processed_actions)
        self.body_ids, self.body_names = self._asset.find_bodies(cfg.body)
        # use default joint velocity as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_vel[:, self._joint_ids].clone()

    def apply_actions(self):
        # des_quat_b = self.command[:, 3:7]
        # des_quat_w = quat_mul(self.asset.data.root_state_w[:, 3:7], des_quat_b)
        # curr_quat_w = self.asset.data.body_state_w[:, self.body_ids[0], 3:7]  # type: ignore
        # angle = quat_error_magnitude(curr_quat_w, des_quat_w)
        # # obtain the desired and current positions
        # des_pos_b = self.command[:, :3]
        # des_pos_w, _ = combine_frame_transforms(self.asset.data.root_state_w[:, :3], self.asset.data.root_state_w[:, 3:7], des_pos_b)
        # curr_pos_w = self.asset.data.body_state_w[:, self.body_ids[0], :3]
        # # print(curr_pos_w - des_pos_w)
        # distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
        # x = distance+2*angle
        des_pos_b = self.command[:, :3]
        des_pos_w, _ = combine_frame_transforms(self.asset.data.root_state_w[:, :3], self.asset.data.root_state_w[:, 3:7], des_pos_b)
        curr_pos_w = self.asset.data.body_state_w[:, self.body_ids[0], :3]  # type: ignore
        distance = torch.norm(curr_pos_w - des_pos_w, dim=1)

        # obtain the desired and current orientations
        des_quat_b = self.command[:, 3:7]
        des_quat_w = quat_mul(self.asset.data.root_state_w[:, 3:7], des_quat_b)
        curr_quat_w = self.asset.data.body_state_w[:, self.body_ids[0], 3:7]  # type: ignore
        result = (1 - torch.tanh(quat_error_magnitude(curr_quat_w, des_quat_w)*2))*(1 - torch.tanh(distance / 0.1))
        # print(result)
        # set joint velocity targets
        self._asset.set_joint_velocity_target(self.processed_actions*torch.where(result<0.92,1,0).view(-1, 1), joint_ids=self._joint_ids)

class JointEffortAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as effort commands."""

    cfg: actions_cfg.JointEffortActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.JointEffortActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def apply_actions(self):
        # set joint effort targets
        self._asset.set_joint_effort_target(self.processed_actions, joint_ids=self._joint_ids)

class JointEffortActionScale(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as effort commands."""

    cfg: actions_cfg.JointEffortActionScaleCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.JointEffortActionScaleCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # extract the asset (to enable type hinting)
        self.asset: RigidObject = env.scene[cfg.asset_cfg.name]
        self.command = env.command_manager.get_command(cfg.command_name)
        self.target = torch.zeros_like(self.processed_actions)
        self.body_ids, self.body_names = self._asset.find_bodies(cfg.body)
        # self.body_id = cfg.asset_cfg.body_ids[0]

        
    def apply_actions(self):

        des_quat_b = self.command[:, 3:7]
        des_quat_w = quat_mul(self.asset.data.root_state_w[:, 3:7], des_quat_b)
        curr_quat_w = self.asset.data.body_state_w[:, self.body_ids[0], 3:7]  # type: ignore
        angle = quat_error_magnitude(curr_quat_w, des_quat_w)
        # obtain the desired and current positions
        des_pos_b = self.command[:, :3]
        des_pos_w, _ = combine_frame_transforms(self.asset.data.root_state_w[:, :3], self.asset.data.root_state_w[:, 3:7], des_pos_b)
        curr_pos_w = self.asset.data.body_state_w[:, self.body_ids[0], :3]
        # print(curr_pos_w - des_pos_w)
        distance = torch.norm(curr_pos_w - des_pos_w, dim=1)

        
        # print("distance")
        # print(torch.tanh(distance).shape)
        # print("self.processed_actions")
        # print(self.processed_actions.shape)
        # set joint effort targets
        # self._asset.set_joint_effort_target(self.processed_actions*(torch.tanh(3*distance)).view(-1, 1), joint_ids=self._joint_ids)
        self._asset.set_joint_effort_target(self.processed_actions*(torch.tanh(2*distance+angle)).view(-1, 1), joint_ids=self._joint_ids)