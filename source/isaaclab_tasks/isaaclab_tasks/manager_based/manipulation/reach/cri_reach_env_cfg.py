# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CRI-aware reach environment (UR_CRI_recurr style)."""

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

@configclass
class CRIReachSceneCfg(InteractiveSceneCfg):
    """Scene for CRI reach training."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
        ),
    )

    robot: ArticulationCfg = MISSING


@configclass
class CRICommandsCfg:
    """Polar pose commands for CRI reach."""

    ee_pose = mdp.UniformPoseTrigCommandCfg(
        asset_name="robot",
        body_name="ee_link",
        debug_vis=True,
        resampling_time_range=(24, 24),
        ranges=mdp.UniformPoseTrigCommandCfg.PolarRanges(
            pos_th=MISSING,
            pos_r=(0.4, 0.8),
            pos_z=(0.4, 0.8),
            roll=MISSING,
            pitch=MISSING,
            yaw=MISSING,
        ),
    )


@configclass
class CRIActionsCfg:
    arm_action: ActionTerm = MISSING
    gripper_action: ActionTerm | None = None


@configclass
class CRIObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
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
            self.concatenate_terms = True
            self.history_length = 5

    policy: PolicyCfg = PolicyCfg()


@configclass
class CRIEventCfg:
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
class CRIRewardsCfg:
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
    end_effector_pos_orientation_tracking_fine_grained = RewTerm(
        func=mdp.position_orientation_command_error_fine_grained,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="ee_link"), "command_name": "ee_pose"},
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-2000.0)
    action_rate = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.1)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-5)
    # dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-6)
    # alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # CRI_OVF = RewTerm(func=mdp.CRI_OVF, weight=-20.0)
    reach_success_bonus = RewTerm(
        func=mdp.reach_success_criteria,
        weight=600.0,
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
class CRICurriculumCfg:
    # termination_penalty = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "termination_penalty", "weight": -2000.0, "num_steps": 24*40000},
    # )
    # action_rate = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "action_rate", "weight": -0.1, "num_steps": 24*40000},
    # )
    # dof_acc_l2 = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "dof_acc_l2", "weight": -1.0e-5, "num_steps": 24*40000},
    # )
    termination_penalty = CurrTerm(
        func=mdp.modify_reward_weight_linear,
        params={
            "term_name": "termination_penalty",
            "initial_weight": -1000.0,
            "max_weight": -2000.0,
            "start_step": 24 * 100,
            "num_steps": 24 * 12100,
        },
    )
    # CRI_OVF = CurrTerm(
    #     func=mdp.modify_reward_weight_linear,
    #     params={
    #         "term_name": "CRI_OVF",
    #         "initial_weight": -20.0,
    #         "max_weight": -800.0,
    #         # reach_success_criteria: start 24*1000 + duration 24*5000
    #         "start_step": 24 * 8000,
    #         "num_steps": 24 * 8000,
    #     },
    # )
    # reach_success_criteria = CurrTerm(
    #     func=mdp.reach_success_criteria_curriculum,
    #     params={
    #         # Reward/event cfg values are finals; curriculum eases in over this many env steps.
    #         "num_steps": 24 * 4000,
    #         "ease_factor": 5.0,
    #         "start_step": 24 * 2000,
    #     },
    # )


@configclass
class CRITerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    OVF = DoneTerm(func=mdp.CRI_OVF)


@configclass
class CRIReachEnvCfg(ManagerBasedRLEnvCfg):
    """Reach environment with CRI observation/termination (UR_CRI_recurr)."""

    scene: CRIReachSceneCfg = CRIReachSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: CRIObservationsCfg = CRIObservationsCfg()
    actions: CRIActionsCfg = CRIActionsCfg()
    commands: CRICommandsCfg = CRICommandsCfg()
    rewards: CRIRewardsCfg = CRIRewardsCfg()
    terminations: CRITerminationsCfg = CRITerminationsCfg()
    events: CRIEventCfg = CRIEventCfg()
    # curriculum: CRICurriculumCfg = CRICurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 24.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.sim.dt = 1.0 / 60.0
