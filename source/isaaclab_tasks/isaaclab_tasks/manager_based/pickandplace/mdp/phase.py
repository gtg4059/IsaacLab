# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase helpers for lift place-down handoff within a single episode."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reached_aerial_handoff_pose(
    env: ManagerBasedRLEnv,
    command_pos_w: torch.Tensor,
    minimal_height: float = 0.04,
    goal_distance_threshold: float = 0.08,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_body_name: str = "panda_hand",
) -> torch.Tensor:
    """True when the object is lifted and the EE is at the aerial command."""
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    height = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    lifted = height > minimal_height

    body_ids, _ = robot.find_bodies(ee_body_name)
    hand_pos_w = robot.data.body_pos_w[:, body_ids[0]]
    goal_distance = torch.norm(hand_pos_w - command_pos_w, dim=1)
    reached = goal_distance < goal_distance_threshold

    return lifted & reached


def should_handoff_to_floor_target(
    env: ManagerBasedRLEnv,
    command_pos_w: torch.Tensor,
    minimal_height: float = 0.04,
    goal_distance_threshold: float = 0.08,
    time_fraction: float = 1.0 / 3.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_body_name: str = "panda_hand",
) -> torch.Tensor:
    """True when ``panda_hand`` reaches the aerial target AND at least ``time_fraction`` of the episode elapsed."""
    reached = reached_aerial_handoff_pose(
        env,
        command_pos_w=command_pos_w,
        minimal_height=minimal_height,
        goal_distance_threshold=goal_distance_threshold,
        object_cfg=object_cfg,
        robot_cfg=robot_cfg,
        ee_body_name=ee_body_name,
    )

    min_steps = max(1, int(env.max_episode_length * time_fraction))
    time_elapsed = env.episode_length_buf >= min_steps

    return reached & time_elapsed


def reached_command_target(
    env: ManagerBasedRLEnv,
    command_pos_w: torch.Tensor,
    goal_distance_threshold: float = 0.08,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_body_name: str = "panda_hand",
) -> torch.Tensor:
    """True when the end-effector is within ``goal_distance_threshold`` of ``command_pos_w``."""
    robot: Articulation = env.scene[robot_cfg.name]

    body_ids, _ = robot.find_bodies(ee_body_name)
    hand_pos_w = robot.data.body_pos_w[:, body_ids[0]]
    goal_distance = torch.norm(hand_pos_w - command_pos_w, dim=1)
    return goal_distance < goal_distance_threshold
