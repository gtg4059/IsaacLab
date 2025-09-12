# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for dual pose command generators."""

from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass

from .dual_pose_command import DualPoseCommand


@configclass
class DualPoseCommandCfg(CommandTermCfg):
    """Configuration for dual pose command generator."""

    class_type: type = DualPoseCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    left_body_name: str = MISSING
    """Name of the left hand body in the asset for which the commands are generated."""

    right_body_name: str = MISSING
    """Name of the right hand body in the asset for which the commands are generated."""

    make_quat_unique: bool = False
    """Whether to make the quaternion unique or not. Defaults to False.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the dual pose commands."""

        pos_x: tuple[float, float] = MISSING
        """Range for the x position (in m). Shared between both hands."""

        pos_y: tuple[float, float] = MISSING
        """Range for the y position (in m). Will be used symmetrically (left: +pos_y, right: -pos_y)."""

        pos_z: tuple[float, float] = MISSING
        """Range for the z position (in m). Shared between both hands."""

        roll: tuple[float, float] = MISSING
        """Range for the roll angle (in rad). Shared between both hands."""

        pitch: tuple[float, float] = MISSING
        """Range for the pitch angle (in rad). Shared between both hands."""

        yaw: tuple[float, float] = MISSING
        """Range for the yaw angle (in rad). Will be used symmetrically (left: +yaw, right: -yaw)."""

    ranges: Ranges = MISSING
    """Ranges for the commands."""

    # Visualization markers for left hand
    left_goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/left_goal_pose")
    """The configuration for the left goal pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    left_current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/left_body_pose"
    )
    """The configuration for the left current pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    # Visualization markers for right hand
    right_goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/right_goal_pose")
    """The configuration for the right goal pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    right_current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/right_body_pose"
    )
    """The configuration for the right current pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.1, 0.1, 0.1)
    left_goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    left_current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    right_goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    right_current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
