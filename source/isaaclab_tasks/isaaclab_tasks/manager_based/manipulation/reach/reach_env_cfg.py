# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp

##
# Scene definition
##


@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # mount
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
        ),
    )

    # robots
    robot: ArticulationCfg = MISSING


##
# MDP settings
##

# Tolerances for ``mdp.reach_success_criteria`` — keep ``reach_success_bonus`` and
# ``resample_ee_pose_on_reach`` in sync (one source of truth for numbers).
REACH_SUCCESS_MAX_DISTANCE = 0.03
REACH_SUCCESS_MAX_ANGLE_RAD = 0.1
REACH_SUCCESS_MAX_LIN_VEL = 0.01
REACH_SUCCESS_MAX_ANG_VEL = 0.1
REACH_SUCCESS_MAX_LIN_ACC = 0.01
REACH_SUCCESS_MAX_ANG_ACC = 0.1


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    ee_pose = mdp.UniformPoseTrigCommandCfg(
        asset_name="robot",
        body_name="ee_link",
        debug_vis=True,
        resampling_time_range=(24,24),
        ranges=mdp.UniformPoseTrigCommandCfg.PolarRanges(
            pos_th=MISSING,
            # pos_r=(0.6,0.6),
            # pos_z=(0.6,0.6),
            pos_r=(0.4,0.8),
            pos_z=(0.4, 0.8),
            roll=MISSING,
            pitch=MISSING,  # depends on end-effector axis
            yaw=MISSING
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = MISSING
    # arm_action = mdp.JointPositionActionCfg(
    #     asset_name="robot", 
    #     scale=0.25, # deploy code와 같은 순서로 scale을 설정
    # )
    gripper_action: ActionTerm | None = None


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        ee_pose_error = ObsTerm(
            func=mdp.ee_pose_error_to_command,
            params={"command_name": "ee_pose", "asset_cfg": SceneEntityCfg("robot", body_names="ee_link")},
        )
        CRI = ObsTerm(func=mdp.collision_risk_index)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.001, n_max=0.001))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.001, n_max=0.001))
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    # observation groups
    policy: PolicyCfg = PolicyCfg()


# @configclass
# class EventCfg:
#     """Configuration for events."""

#     reset_robot_joints = EventTerm(
#         func=mdp.reset_joints_by_scale,
#         mode="reset",
#         params={
#             "position_range": (0.5, 1.5),
#             "velocity_range": (0.0, 0.0),
#         },
#     )

@configclass
class EventCfg:
    """Configuration for events."""

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

    # Every control step: resample ``ee_pose`` for envs that satisfy reach (episode continues).
    resample_ee_pose_on_reach = EventTerm(
        func=mdp.resample_ee_pose_command_on_reach,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        is_global_time=True,
        params={
            "command_name": "ee_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names="ee_link"),
            "max_distance": REACH_SUCCESS_MAX_DISTANCE,
            "max_angle_rad": REACH_SUCCESS_MAX_ANGLE_RAD,
            "max_lin_vel": REACH_SUCCESS_MAX_LIN_VEL,
            "max_ang_vel": REACH_SUCCESS_MAX_ANG_VEL,
            "max_lin_acc": REACH_SUCCESS_MAX_LIN_ACC,
            "max_ang_acc": REACH_SUCCESS_MAX_ANG_ACC,
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

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
    #     weight=4.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names="ee_link"), "command_name": "ee_pose"},
    # )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-2000.0)
    # alive = RewTerm(func=mdp.is_alive, weight=1.2)

    # action penalty (initial weight; curriculum ramps toward stronger penalty)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    # dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-6)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-5)

    # CRI_optimize = RewTerm(func=mdp.CRI_optimize, weight=10.0)

    # Bonus when EE is within tolerances and slow (numeric tolerances: module constants above).
    reach_success_bonus = RewTerm(
        func=mdp.reach_success_criteria,
        weight=600.0,
        params={
            "command_name": "ee_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names="ee_link"),
            "max_distance": REACH_SUCCESS_MAX_DISTANCE,
            "max_angle_rad": REACH_SUCCESS_MAX_ANGLE_RAD,
            "max_lin_vel": REACH_SUCCESS_MAX_LIN_VEL,
            "max_ang_vel": REACH_SUCCESS_MAX_ANG_VEL,
            "max_lin_acc": REACH_SUCCESS_MAX_LIN_ACC,
            "max_ang_acc": REACH_SUCCESS_MAX_ANG_ACC,
        },
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Success criteria live in ``mdp.reach_success_criteria`` / ``reach_success_bonus`` only (no ``reach`` term).
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    OVF = DoneTerm(func=mdp.CRI_OVF)

# @configclass
# class CurriculumCfg:
#     """Curriculum terms for the MDP."""

#     # Linear ramp after hold_steps: weight_start is fixed until then, then ramps to weight_end over num_steps.
#     termination_penalty = CurrTerm(
#         func=mdp.modify_reward_weight_linear,
#         params={
#             "term_name": "termination_penalty",
#             "weight_start": -2.0,
#             "weight_end": -400.0,
#             "hold_steps": 1 * 16,
#             "num_steps": 12000 * 16,
#         },
#     )



##
# Environment configuration
##


@configclass
class ReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: ReachSceneCfg = ReachSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.episode_length_s = 24.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 1.0 / 60.0
