# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions for object positioning based on hand positions."""

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import EventTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def set_object_position_between_hands(
    env,
    env_ids: torch.Tensor,
    object_name: str = "object",
    left_hand_pos_y: torch.Tensor | None = None,
    right_hand_pos_y: torch.Tensor | None = None,
    object_pos_x: float | None = None,
    object_pos_z: float | None = None,
) -> None:
    """Set object position to be between the two hands in y-axis.
    
    Args:
        env: The environment object.
        env_ids: Environment IDs to update.
        object_name: Name of the object to position.
        left_hand_pos_y: Left hand y position in base frame.
        right_hand_pos_y: Right hand y position in base frame.
        object_pos_x: Object x position in base frame.
        object_pos_z: Object z position in base frame.
    """
    # Get the object
    object_asset: RigidObject = env.scene[object_name]
    
    # Calculate middle y position between hands
    if left_hand_pos_y is not None and right_hand_pos_y is not None:
        middle_y = (left_hand_pos_y[env_ids] + right_hand_pos_y[env_ids]) / 2.0
    else:
        # Default to center if hand positions are not provided
        middle_y = torch.zeros(len(env_ids), device=env.device)
    
    # Get DualPoseCommandCfg distribution values for pos_x and pos_z
    if "dual_ee_pose" in env.command_manager._terms:
        dual_command = env.command_manager.get_term("dual_ee_pose") 
        pos_x_range = dual_command.cfg.ranges.pos_x
        pos_z_range = dual_command.cfg.ranges.pos_z
        
        # Sample random values within the distribution ranges
        if object_pos_x is None:
            object_pos_x_tensor = torch.empty(len(env_ids), device=env.device).uniform_(*pos_x_range)
        else:
            object_pos_x_tensor = torch.full((len(env_ids),), object_pos_x, device=env.device)
            
        if object_pos_z is None:
            object_pos_z_tensor = torch.empty(len(env_ids), device=env.device).uniform_(*pos_z_range)
        else:
            object_pos_z_tensor = torch.full((len(env_ids),), object_pos_z, device=env.device)
    else:
        # Fallback to default values if dual_ee_pose is not available
        if object_pos_x is None:
            object_pos_x_tensor = torch.full((len(env_ids),), 0.37, device=env.device)
        else:
            object_pos_x_tensor = torch.full((len(env_ids),), object_pos_x, device=env.device)
            
        if object_pos_z is None:
            object_pos_z_tensor = torch.full((len(env_ids),), 0.86, device=env.device)
        else:
            object_pos_z_tensor = torch.full((len(env_ids),), object_pos_z, device=env.device)
    
    # Simple approach: Set object position directly in world frame
    # Get robot position and add object position offset
    robot = env.scene["robot"]
    
    # Set object position in world frame (simple addition)
    object_pos_w = robot.data.root_pos_w[env_ids].clone()
    object_pos_w[:, 0] += object_pos_x_tensor  # x offset
    object_pos_w[:, 1] += middle_y      # y offset (between hands)
    object_pos_w[:, 2] += object_pos_z_tensor  # z offset
    
    # Use robot's orientation for object
    object_quat_w = robot.data.root_quat_w[env_ids].clone()
    
    # Combine position and orientation into 7D tensor (pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z)
    object_pose_w = torch.cat([object_pos_w, object_quat_w], dim=1)
    
    # Set object pose in world frame for specific environments
    # Directly set the pose data instead of using write_root_pose_to_sim
    object_asset.data.root_link_pose_w[env_ids] = object_pose_w


def position_object_between_hands_event(env, env_ids: torch.Tensor, object_name: str = "object", object_pos_x: float | None = None, object_pos_z: float | None = None) -> None:
    """Event function for positioning object between hands.
    
    Args:
        env: The environment object.
        env_ids: Environment IDs to update.
        object_name: Name of the object to position.
        object_pos_x: Object x position in base frame.
        object_pos_z: Object z position in base frame.
    """
    # Get hand positions from dual pose command
    if "dual_ee_pose" in env.command_manager._terms:
        dual_command = env.command_manager.get_term("dual_ee_pose")
        left_hand_pos_y = dual_command.left_pose_command_b[:, 1]
        right_hand_pos_y = dual_command.right_pose_command_b[:, 1]
    else:
        left_hand_pos_y = None
        right_hand_pos_y = None
    
    # Set object position
    set_object_position_between_hands(
        env=env,
        env_ids=env_ids,
        object_name=object_name,
        left_hand_pos_y=left_hand_pos_y,
        right_hand_pos_y=right_hand_pos_y,
        object_pos_x=object_pos_x,
        object_pos_z=object_pos_z,
    )
