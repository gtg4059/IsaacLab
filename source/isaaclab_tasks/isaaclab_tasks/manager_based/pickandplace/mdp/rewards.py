# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ee_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_body_name: str = "panda_hand",
) -> torch.Tensor:
    """Reward tracking ``target_pose`` with the end-effector body (not the object center).

    Active only after the object is lifted above ``minimal_height`` (same gate as lift goal tracking).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)

    body_ids, _ = robot.find_bodies(ee_body_name)
    hand_pos_w = robot.data.body_pos_w[:, body_ids[0]]
    distance = torch.norm(des_pos_w - hand_pos_w, dim=1)

    lifted = obj.data.root_pos_w[:, 2] > minimal_height
    return lifted * (1.0 - torch.tanh(distance / std))


def ee_orientation_command_error(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_body_name: str = "panda_hand",
) -> torch.Tensor:
    """Penalize orientation error between the EE body and ``target_pose`` command.

    Active only after the object is lifted above ``minimal_height`` (same gate as ``ee_goal_distance``).
    Returns the shortest-path quaternion error magnitude in radians.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(robot.data.root_quat_w, des_quat_b)

    body_ids, _ = robot.find_bodies(ee_body_name)
    curr_quat_w = robot.data.body_quat_w[:, body_ids[0]]
    orientation_error = quat_error_magnitude(curr_quat_w, des_quat_w)

    lifted = obj.data.root_pos_w[:, 2] > minimal_height
    return lifted * orientation_error
