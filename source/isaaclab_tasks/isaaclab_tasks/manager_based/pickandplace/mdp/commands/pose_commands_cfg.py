# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for pick-and-place pose command generators."""

from dataclasses import MISSING

from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.utils import configclass

from .pose_commands import AlternatingLiftPoseCommand


@configclass
class AlternatingLiftPoseCommandCfg(UniformPoseCommandCfg):
    """Aerial lift target, then optional in-episode switch to floor-near place target."""

    class_type: type = AlternatingLiftPoseCommand

    air_ranges: UniformPoseCommandCfg.Ranges = MISSING
    """Ranges for aerial lift targets (sampled on reset / before handoff)."""

    floor_ranges: UniformPoseCommandCfg.Ranges = MISSING
    """Ranges for table-near targets (sampled after lift completes in the same episode)."""

    start_in_phase2: bool = False
    """If True, use sequential air-then-floor within each episode (PLAY / curriculum phase 2)."""

    handoff_min_height: float = 0.04
    """Object must be lifted above this height for aerial reach check."""

    handoff_goal_distance: float = 0.08
    """Max object–aerial-target distance (m) for reach check."""

    handoff_time_fraction: float = 1.0 / 3.0
    """Minimum elapsed episode fraction before floor handoff (e.g. 1/3 of play time)."""

    handoff_air_dwell_s: float | None = None
    """If set, hold the aerial target for this many seconds after reach before floor handoff (PLAY)."""
