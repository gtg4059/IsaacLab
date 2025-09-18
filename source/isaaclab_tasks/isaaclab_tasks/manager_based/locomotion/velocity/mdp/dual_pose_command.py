# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for dual pose tracking."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .dual_pose_command_cfg import DualPoseCommandCfg


class DualPoseCommand(CommandTerm):
    """Command generator for generating dual pose commands with shared and symmetric parameters.

    This command generator creates pose commands for two end-effectors (left and right hands)
    where:
    - pos_x, pos_z, roll, pitch, yaw are shared between both hands
    - pos_y and yaw are symmetric (left: +pos_y, +yaw; right: -pos_y, -yaw)
    - Object position is set to the middle of both hand positions in y-axis
    """

    cfg: DualPoseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: DualPoseCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # extract the robot and body indices for both hands
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.left_body_idx = self.robot.find_bodies(cfg.left_body_name)[0][0]
        self.right_body_idx = self.robot.find_bodies(cfg.right_body_name)[0][0]

        # create buffers for both hands
        # -- commands: (x, y, z, qw, qx, qy, qz) in root frame
        self.left_pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.left_pose_command_b[:, 3] = 1.0
        self.left_pose_command_w = torch.zeros_like(self.left_pose_command_b)
        
        self.right_pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.right_pose_command_b[:, 3] = 1.0
        self.right_pose_command_w = torch.zeros_like(self.right_pose_command_b)
        
        # -- shared parameters buffer
        self.shared_params = torch.zeros(self.num_envs, 5, device=self.device)  # pos_x, pos_z, roll, pitch, yaw
        
        # -- metrics
        self.metrics["left_position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["left_orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["right_position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["right_orientation_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "DualPoseCommand:\n"
        msg += f"\tLeft command dimension: {tuple(self.left_command.shape[1:])}\n"
        msg += f"\tRight command dimension: {tuple(self.right_command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def left_command(self) -> torch.Tensor:
        """The desired left hand pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        """
        return self.left_pose_command_b

    @property
    def right_command(self) -> torch.Tensor:
        """The desired right hand pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        """
        return self.right_pose_command_b

    @property
    def command(self) -> torch.Tensor:
        """The combined command for both hands. Shape is (num_envs, 14).

        First 7 elements are for left hand, next 7 elements are for right hand.
        """
        return torch.cat([self.left_pose_command_b, self.right_pose_command_b], dim=1)

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # transform commands from base frame to simulation world frame
        # -- left hand
        self.left_pose_command_w[:, :3], self.left_pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.left_pose_command_b[:, :3],
            self.left_pose_command_b[:, 3:],
        )
        # -- right hand
        self.right_pose_command_w[:, :3], self.right_pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.right_pose_command_b[:, :3],
            self.right_pose_command_b[:, 3:],
        )
        
        # compute the errors
        # -- left hand
        left_pos_error, left_rot_error = compute_pose_error(
            self.left_pose_command_w[:, :3],
            self.left_pose_command_w[:, 3:],
            self.robot.data.body_pos_w[:, self.left_body_idx],
            self.robot.data.body_quat_w[:, self.left_body_idx],
        )
        self.metrics["left_position_error"] = torch.norm(left_pos_error, dim=-1)
        self.metrics["left_orientation_error"] = torch.norm(left_rot_error, dim=-1)
        
        # -- right hand
        right_pos_error, right_rot_error = compute_pose_error(
            self.right_pose_command_w[:, :3],
            self.right_pose_command_w[:, 3:],
            self.robot.data.body_pos_w[:, self.right_body_idx],
            self.robot.data.body_quat_w[:, self.right_body_idx],
        )
        self.metrics["right_position_error"] = torch.norm(right_pos_error, dim=-1)
        self.metrics["right_orientation_error"] = torch.norm(right_rot_error, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        # sample shared parameters
        r = torch.empty(len(env_ids), device=self.device)
        self.shared_params[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)  # pos_x
        self.shared_params[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_z)  # pos_z
        self.shared_params[env_ids, 2] = r.uniform_(*self.cfg.ranges.roll)   # roll
        self.shared_params[env_ids, 3] = r.uniform_(*self.cfg.ranges.pitch)  # pitch
        self.shared_params[env_ids, 4] = r.uniform_(*self.cfg.ranges.yaw)    # yaw
        
        # sample pos_y (will be used symmetrically)
        pos_y = torch.empty(len(env_ids), device=self.device)
        pos_y.uniform_(*self.cfg.ranges.pos_y)
        
        # set left hand pose
        self.left_pose_command_b[env_ids, 0] = self.shared_params[env_ids, 0]  # pos_x
        self.left_pose_command_b[env_ids, 1] = pos_y                           # pos_y (positive)
        self.left_pose_command_b[env_ids, 2] = self.shared_params[env_ids, 1]  # pos_z
        
        # set right hand pose
        self.right_pose_command_b[env_ids, 0] = self.shared_params[env_ids, 0]  # pos_x
        self.right_pose_command_b[env_ids, 1] = -pos_y                          # pos_y (negative, symmetric)
        self.right_pose_command_b[env_ids, 2] = self.shared_params[env_ids, 1]  # pos_z
        
        # set orientations
        # -- left hand orientation
        left_euler_angles = torch.zeros_like(self.left_pose_command_b[env_ids, :3])
        left_euler_angles[:, 0] = self.shared_params[env_ids, 2]  # roll
        left_euler_angles[:, 1] = self.shared_params[env_ids, 3]  # pitch
        left_euler_angles[:, 2] = self.shared_params[env_ids, 4]  # yaw (positive)
        left_quat = quat_from_euler_xyz(left_euler_angles[:, 0], left_euler_angles[:, 1], left_euler_angles[:, 2])
        self.left_pose_command_b[env_ids, 3:] = quat_unique(left_quat) if self.cfg.make_quat_unique else left_quat
        
        # -- right hand orientation
        right_euler_angles = torch.zeros_like(self.right_pose_command_b[env_ids, :3])
        right_euler_angles[:, 0] = self.shared_params[env_ids, 2]   # roll
        right_euler_angles[:, 1] = self.shared_params[env_ids, 3]   # pitch
        right_euler_angles[:, 2] = -self.shared_params[env_ids, 4]  # yaw (negative, symmetric)
        right_quat = quat_from_euler_xyz(right_euler_angles[:, 0], right_euler_angles[:, 1], right_euler_angles[:, 2])
        self.right_pose_command_b[env_ids, 3:] = quat_unique(right_quat) if self.cfg.make_quat_unique else right_quat

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "left_goal_pose_visualizer"):
                # -- left goal pose
                self.left_goal_pose_visualizer = VisualizationMarkers(self.cfg.left_goal_pose_visualizer_cfg)
                # -- left current body pose
                self.left_current_pose_visualizer = VisualizationMarkers(self.cfg.left_current_pose_visualizer_cfg)
                # -- right goal pose
                self.right_goal_pose_visualizer = VisualizationMarkers(self.cfg.right_goal_pose_visualizer_cfg)
                # -- right current body pose
                self.right_current_pose_visualizer = VisualizationMarkers(self.cfg.right_current_pose_visualizer_cfg)
            # set their visibility to true
            self.left_goal_pose_visualizer.set_visibility(True)
            self.left_current_pose_visualizer.set_visibility(True)
            self.right_goal_pose_visualizer.set_visibility(True)
            self.right_current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "left_goal_pose_visualizer"):
                self.left_goal_pose_visualizer.set_visibility(False)
                self.left_current_pose_visualizer.set_visibility(False)
                self.right_goal_pose_visualizer.set_visibility(False)
                self.right_current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- left goal pose
        self.left_goal_pose_visualizer.visualize(self.left_pose_command_w[:, :3], self.left_pose_command_w[:, 3:])
        # -- left current body pose
        left_body_link_pose_w = self.robot.data.body_link_pose_w[:, self.left_body_idx]
        self.left_current_pose_visualizer.visualize(left_body_link_pose_w[:, :3], left_body_link_pose_w[:, 3:7])
        # -- right goal pose
        self.right_goal_pose_visualizer.visualize(self.right_pose_command_w[:, :3], self.right_pose_command_w[:, 3:])
        # -- right current body pose
        right_body_link_pose_w = self.robot.data.body_link_pose_w[:, self.right_body_idx]
        self.right_current_pose_visualizer.visualize(right_body_link_pose_w[:, :3], right_body_link_pose_w[:, 3:7])
