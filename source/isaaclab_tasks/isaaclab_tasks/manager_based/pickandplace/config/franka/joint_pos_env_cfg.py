# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.pickandplace import mdp as pnp_mdp
from isaaclab_tasks.manager_based.pickandplace.config.franka.franka_scene_cfg import configure_franka_cube_scene
from isaaclab_tasks.manager_based.pickandplace.pickandplace_env_cfg import PickandplaceLiftEnvCfg


def configure_franka_pnp_actions(cfg, *, auto_release_gripper: bool = False) -> None:
    """Configure unified 8D actions (7 arm + 1 gripper) on any PnP env cfg."""
    cfg.actions.arm_action = pnp_mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
    )
    gripper_action_cls = pnp_mdp.PlayAutoReleaseGripperActionCfg if auto_release_gripper else pnp_mdp.BinaryJointPositionActionCfg
    cfg.actions.gripper_action = gripper_action_cls(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


def configure_lift_play_phase2(cfg) -> None:
    """Configure lift PLAY to match post-curriculum (phase 2) training behavior."""
    cfg.commands.target_pose.start_in_phase2 = True
    cfg.commands.target_pose.resampling_time_range = (1000.0, 1000.0)
    cfg.commands.target_pose.handoff_min_height = 0.04
    cfg.commands.target_pose.handoff_goal_distance = 0.08
    cfg.commands.target_pose.handoff_air_dwell_s = 1.0
    cfg.terminations.object_dropped_after_lift.params["enabled"] = False
    cfg.curriculum.place_down = None
    cfg.episode_length_s = 12.0
    if hasattr(cfg.actions.gripper_action, "release_goal_distance"):
        cfg.actions.gripper_action.release_goal_distance = cfg.commands.target_pose.handoff_goal_distance
        cfg.actions.gripper_action.ee_body_name = cfg.commands.target_pose.body_name
        cfg.actions.gripper_action.release_delay_s = 0.5


@configclass
class FrankaCubePickandplaceLiftEnvCfg(PickandplaceLiftEnvCfg):
    """Franka pick-and-place: pick, grasp, lift, and place-down."""

    def __post_init__(self):
        super().__post_init__()
        configure_franka_cube_scene(self)
        configure_franka_pnp_actions(self)
        self.commands.target_pose.body_name = "panda_hand"


@configclass
class FrankaCubePickandplaceLiftEnvCfg_PLAY(FrankaCubePickandplaceLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        configure_franka_pnp_actions(self, auto_release_gripper=True)
        configure_lift_play_phase2(self)
