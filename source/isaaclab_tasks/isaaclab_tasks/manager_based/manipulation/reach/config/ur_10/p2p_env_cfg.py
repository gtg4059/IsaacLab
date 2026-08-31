# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .joint_pos_env_cfg import UR10ReachEnvCfg
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm

from isaaclab_assets import UR10_CFG  # isort: skip
import isaaclab.sim as sim_utils

##
# Environment configuration
##

# --task Isaac-Reach-UR10-P2P-v0

# CRI OVF curriculum schedule (env steps; 1 PPO iter = 48 steps).
CRI_OVF_REWARD_START = 48 * 1000
CRI_OVF_REWARD_STEPS = 48 * 2000
CRI_OVF_CROSSFADE_START = 48 * 3500
CRI_OVF_CROSSFADE_STEPS = 48 * 2000

# termination_penalty: hold -200, then ramp to -2000.
TERM_PENALTY_INITIAL_WEIGHT = -200.0
TERM_PENALTY_FINAL_WEIGHT = -2000.0
TERM_PENALTY_RAMP_START = 48 * 12000
TERM_PENALTY_RAMP_STEPS = 48 * 2000


@configclass
class P2PRewardsCfg:
    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="ee_link"), "command_name": "ee_pose"},
    )
    end_effector_pos_orientation_tracking = RewTerm(
        func=mdp.position_orientation_command_error,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="ee_link"), "command_name": "ee_pose"},
    )
    # end_effector_pos_orientation_tracking_fine_grained = RewTerm(
    #     func=mdp.position_orientation_command_error_fine_grained,
    #     weight=16.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names="ee_link"), "command_name": "ee_pose"},
    # )
    # Final weight; curriculum holds TERM_PENALTY_INITIAL_WEIGHT then ramps here.
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=TERM_PENALTY_FINAL_WEIGHT)
    action_rate = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.01)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-5)
    reach_success_bonus = RewTerm(
        func=mdp.reach_success_criteria,
        weight=1200.0,
        params={
            "command_name": "ee_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names="ee_link"),
            "max_distance": 0.03,
            "max_angle_rad": 0.1,
            "max_lin_vel": 0.01,
            "max_ang_vel": 0.1,
            "max_lin_acc": 0.01,
            "max_ang_acc": 0.1,
        },
    )

@configclass
class P2PEventCfg:
    reset_robot_joints = EventTerm(
        func=mdp.reset_robot_joints_two_groups_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "primary_joint_names": ["shoulder_lift_joint"],
            "primary_position_range": (0.0, 0.0),
            "primary_velocity_range": (0.0, 0.0),
            "secondary_joint_names": [
                "shoulder_pan_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            "secondary_position_range": (0.0, 0.0),
            "secondary_velocity_range": (0.0, 0.0),
        },
    )


    resample_ee_pose_on_reach = EventTerm(
        func=mdp.resample_ee_pose_command_on_reach,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        is_global_time=True,
        params={
            "command_name": "ee_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names="ee_link"),
            "max_distance": 0.03,
            "max_angle_rad": 0.1,
            "max_lin_vel": 0.01,
            "max_ang_vel": 0.1,
            "max_lin_acc": 0.01,
            "max_ang_acc": 0.1,
        },
    )

@configclass
class P2PCurriculumCfg:
    reach_success_criteria = CurrTerm(
        func=mdp.reach_success_criteria_curriculum,
        params={
            "ease_factor": 5.0,
            "start_step": 48 * 2000,
            "num_steps": 48 * 2000,
        },
    )
    termination_penalty = CurrTerm(
        func=mdp.modify_reward_weight_linear,
        params={
            "term_name": "termination_penalty",
            "initial_weight": TERM_PENALTY_INITIAL_WEIGHT,
            "final_weight": TERM_PENALTY_FINAL_WEIGHT,
            "start_step": TERM_PENALTY_RAMP_START,
            "num_steps": TERM_PENALTY_RAMP_STEPS,
        },
    )


@configclass
class P2PTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    reach_success = DoneTerm(
        func=mdp.reach_success,
        params={"reward_term_name": "reach_success_bonus"},
        time_out=True,
    )
    OVF = DoneTerm(
        func=mdp.CRI_OVF,
        params={"threshold": 0.96},
    )

@configclass
class UR10ReachP2PEnvCfg(UR10ReachEnvCfg):
    """UR10 reach with a single point-to-point goal per episode (no in-episode resample)."""
    terminations: P2PTerminationsCfg = P2PTerminationsCfg()
    events: P2PEventCfg = P2PEventCfg()
    curriculum: P2PCurriculumCfg = P2PCurriculumCfg()
    rewards: P2PRewardsCfg = P2PRewardsCfg()
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UR10_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
            spawn=UR10_CFG.spawn.replace(
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                    solver_position_iteration_count=4,   # Franka/Kinova 참고
                    solver_velocity_iteration_count=0,
                ),
            ),
        )

        # Parent UR10ReachEnvCfg injects tertiary_* ranges for the 3-group continuous-reach
        # reset; P2P uses reset_robot_joints_two_groups_by_offset, so drop those keys.
        for key in (
            "tertiary_joint_names",
            "tertiary_position_range",
            "tertiary_velocity_range",
        ):
            self.events.reset_robot_joints.params.pop(key, None)
        self.events.reset_robot_joints.params["primary_position_range"] = (0.0, 0.0)
        self.events.reset_robot_joints.params["secondary_position_range"] = (0.0, 0.0)
        self.events.reset_robot_joints.params["primary_velocity_range"] = (0.0, 0.0)
        self.events.reset_robot_joints.params["secondary_velocity_range"] = (0.0, 0.0)

        self.commands.ee_pose.ranges.pos_r = (0.6, 0.6)
        self.commands.ee_pose.ranges.pos_z = (0.6, 0.6)
        self.commands.ee_pose.ranges.pos_th = (0.0, 0.0)
        self.commands.ee_pose.ranges.roll = (0.0, 0.0)
        self.commands.ee_pose.ranges.pitch = (0.0, 0.0)
        self.commands.ee_pose.ranges.yaw = (0.0, 0.0)


@configclass
class UR10ReachP2PEnvCfg_PLAY(UR10ReachP2PEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.viewer.eye = (4.5, 4.5, 4.5)
        self.observations.policy.enable_corruption = False
        # Keep final (strict) reach / CRI thresholds; do not ease them for evaluation.
        self.curriculum.reach_success_criteria = None
        self.curriculum.termination_penalty = None
        self.terminations.OVF.params["threshold"] = 0.96
        if hasattr(self.rewards, "CRI_OVF"):
            self.rewards.CRI_OVF.params.pop("threshold", None)
            self.rewards.CRI_OVF.params["limit"] = 0.96
