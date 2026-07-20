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

# CRI OVF curriculum schedule (env steps; 1 PPO iter = 24 steps).
CRI_OVF_REWARD_START = 24 * 12000
CRI_OVF_CROSSFADE_START = 24 * 20000
CRI_OVF_THRESHOLD_INITIAL = 2.0
CRI_OVF_THRESHOLD_FINAL = 0.96
REACH_SUCCESS_CRITERIA_START = 24 * 80000
FINE_GRAINED_TRACKING_WEIGHT = 8.0

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
        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.001, n_max=0.001))
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-0.001, n_max=0.001))
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.concatenate_terms = True
            self.history_length = 5

    policy: PolicyCfg = PolicyCfg()


@configclass
class CRIEventCfg:
    reset_robot_joints = EventTerm(
        func=mdp.reset_robot_joints_by_name_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": {
                "shoulder_pan_joint": (0.0, 0.0),
                "shoulder_lift_joint": (0.0, 0.0),
                "elbow_joint": (0.0, 0.0),
                "wrist_1_joint": (0.0, 0.0),
                "wrist_2_joint": (0.0, 0.0),
                "wrist_3_joint": (0.0, 0.0),
            },
            "velocity_range": {
                "shoulder_pan_joint": (0.0, 0.0),
                "shoulder_lift_joint": (0.0, 0.0),
                "elbow_joint": (0.0, 0.0),
                "wrist_1_joint": (0.0, 0.0),
                "wrist_2_joint": (0.0, 0.0),
                "wrist_3_joint": (0.0, 0.0),
            },
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
        weight=FINE_GRAINED_TRACKING_WEIGHT,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="ee_link"), "command_name": "ee_pose"},
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    action_rate = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.2)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-5)
    # CRI_OVF = RewTerm(
    #     func=mdp.CRI_OVF,
    #     weight=-50.0,
    #     params={"threshold": 0.96},
    # )
    reach_success_bonus = RewTerm(
        func=mdp.reach_success_bonus,
        weight=400.0,
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
    reach_success_criteria = CurrTerm(
        func=mdp.reach_success_criteria_curriculum,
        params={
            "ease_factor": 5.0,
            "start_step": REACH_SUCCESS_CRITERIA_START,
            "num_steps": 24 * 40000,
        },
    )
    # When reach success criteria curriculum starts, drop fine-grained tracking reward.
    fine_grained_tracking_weight = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.end_effector_pos_orientation_tracking_fine_grained.weight",
            "modify_fn": mdp.reward_weight_step_by_step,
            "modify_params": {
                "switch_step": REACH_SUCCESS_CRITERIA_START,
                "initial_weight": FINE_GRAINED_TRACKING_WEIGHT,
                "final_weight": 1.0,
            },
        },
    )
    termination_penalty = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.termination_penalty.weight",
            "modify_fn": mdp.termination_penalty_weight_by_step,
            "modify_params": {
                "switch_step": 24 * 40000,
                "initial_weight": -200.0,
                "final_weight": -2000.0,
            },
        },
    )
    # cri_ovf_reward_weight = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "rewards.CRI_OVF.weight",
    #         "modify_fn": mdp.cri_ovf_reward_weight_by_step,
    #         "modify_params": {
    #             "reward_start": CRI_OVF_REWARD_START,
    #             "crossfade_start": CRI_OVF_CROSSFADE_START,
    #         },
    #     },
    # )
    # cri_ovf_term_threshold = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "terminations.OVF.params.threshold",
    #         "modify_fn": mdp.cri_ovf_threshold_by_step,
    #         "modify_params": {
    #             "crossfade_start": CRI_OVF_CROSSFADE_START,
    #             "threshold_initial": CRI_OVF_THRESHOLD_INITIAL,
    #             "threshold_final": CRI_OVF_THRESHOLD_FINAL,
    #         },
    #     },
    # )


@configclass
class CRITerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    OVF = DoneTerm(
        func=mdp.CRI_OVF,
        params={"threshold": 0.96},
    )


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
    curriculum: CRICurriculumCfg = CRICurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 24.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.sim.dt = 1.0 / 60.0
