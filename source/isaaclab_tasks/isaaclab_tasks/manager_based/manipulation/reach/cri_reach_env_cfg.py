# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CRI-aware reach environment (UR_CRI_recurr style).

Reach success ends the episode via ``terminations.reach_success`` (no in-episode target resample).
"""

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

# CRI OVF curriculum schedule (env steps; 1 PPO iter = 48 steps).
# CRI_OVF_REWARD_START = 48 * 12000
# CRI_OVF_CROSSFADE_START = 48 * 20000
# REACH_SUCCESS_CRITERIA_START = 48 * 00000
# REACH_CRITERIA_RAMP_STEPS = 48 * 40000

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
        origin_body_name="base_link",
        debug_vis=False,
        resampling_time_range=(48, 48),
        ranges=mdp.UniformPoseTrigCommandCfg.PolarRanges(
            pos_th=MISSING,
            pos_r=(0.01, 0.8),
            # pos_z / exclude_pos_z: metres above base_link (mount), not world or table z.
            pos_z=(0.01, 0.8),
            exclude_pos_r=(0.0, 0.4),
            exclude_pos_z=(0.0, 0.6),
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


@configclass
class CRIRewardsCfg:
    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="ee_link"), "command_name": "ee_pose"},
    )
    end_effector_pos_orientation_tracking = RewTerm(
        func=mdp.position_orientation_command_error,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="ee_link"), "command_name": "ee_pose"},
    )
    end_effector_pos_orientation_tracking_fine_grained = RewTerm(
        func=mdp.position_orientation_command_error_fine_grained,
        weight=4.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="ee_link"), "command_name": "ee_pose"},
    )
    # OVF only (reach_success / time_out are time_out=True). Keep 3x timeout cost
    # so avoiding a late timeout by crashing is never the cheaper option.
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-600.0)
    action_rate = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.1)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    CRI_OVF = RewTerm(
        func=mdp.CRI_OVF_exp,
        weight=-10.0,
        params={"limit": 0.96, "sigma": 20.0},
    )
    # Constant -0.2 until curriculum raises params.final; then 12–48 s ramps -0.2 → -1.0.
    # Weight is -1 so the term value is the applied living cost.
    is_alive = RewTerm(
        func=mdp.is_alive_time_ramp,
        weight=-1.0,
        params={"ramp_start_s": 12.0, "initial": 0.2, "final": 0.2},
    )
    # Sparse bonus on the success step; episode then ends via terminations.reach_success.
    # Fixed pose+vel+acc gates from the first step (no ease, dwell, or vel switch).
    reach_success_bonus = RewTerm(
        func=mdp.ReachSuccessCriteria,
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
class CRICurriculumCfg:
    # reach_success_criteria = CurrTerm(
    #     func=mdp.reach_success_criteria_curriculum,
    #     params={
    #         "ease_factor": 3.0,
    #         "start_step": REACH_SUCCESS_CRITERIA_START,
    #         "num_steps": REACH_CRITERIA_RAMP_STEPS,
    #         "reward_term_names": ["reach_success_bonus"],
    #         # Thresholds live on reward terms; terminations.reach_success reads them from there.
    #         "event_term_name": None,
    #     },
    # )
    # Phase 1: fine-grained pulls into the 3 cm basin; timeout penalty stays 0 so
    # the only sparse target is reach_success_bonus. Phase 2: drop the sit subsidy
    # and turn on timeout (1/2 of OVF -600) so lingering at the goal is not free.
    fine_grained_tracking_weight = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.end_effector_pos_orientation_tracking_fine_grained.weight",
            "modify_fn": mdp.reward_weight_step_by_step,
            "modify_params": {
                "switch_step": 48 * 12000,
                "initial_weight": 4.0,
                "final_weight": 1.0,
            },
        },
    )
    # Phase 1: same constant living cost as the 08-29 run. Phase 2: enable the
    # in-episode 12–48 s ramp (-0.2 → -1.0) once reach_success is already moving.
    is_alive_ramp_final = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.is_alive.params.final",
            "modify_fn": mdp.reward_weight_step_by_step,
            "modify_params": {
                "switch_step": 48 * 12000,
                "initial_weight": 0.2,
                "final_weight": 1.5,
            },
        },
    )

    # termination_penalty = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "rewards.termination_penalty.weight",
    #         "modify_fn": mdp.termination_penalty_weight_by_step,
    #         "modify_params": {
    #             "switch_step": 48 * 16000,
    #             "initial_weight": -600.0,
    #             "final_weight": -1800.0,
    #         },
    #     },
    # )
    # # CRI OVF termination을 위한의 soft penalty로서 사용되는 보상이지만
    # # penalty가 초기에 너무 크면 OVF termination쪽으로 학습되버린다
    # cri_ovf_penalty = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "rewards.CRI_OVF.weight",
    #         "modify_fn": mdp.termination_penalty_weight_by_step,
    #         "modify_params": {
    #             "switch_step": 48 * 240000,
    #             "initial_weight": -0.0,
    #             "final_weight": -10.0,
    #         },
    #     },
    # )
    # reach 성공 판정이 줄어들 때 도달을 유도하는 보상들이 커지면서, 
    # 도달하지를 않고 도달 직전에서 위치를 유도하며 멈추는 현상이 발생한다
    # 이를 방지하기 위한 세션을 유지에 대한 패널티로서, 목표의 도달을 유도한다
    # is_alive_penalty = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "rewards.is_alive.weight",
    #         "modify_fn": mdp.termination_penalty_weight_by_step,
    #         "modify_params": {
    #             "switch_step": 48 * 16000,
    #             "initial_weight": -0.2,
    #             "final_weight": -1.0,
    #         },
    #     },
    # )
    # # cri_ovf_reward_weight = CurrTerm(
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
    cri_ovf_term_threshold = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "terminations.OVF.params.threshold",
            "modify_fn": mdp.cri_ovf_threshold_by_step,
            "modify_params": {
                "crossfade_start": 48 * 6000,
                "threshold_initial": 2.0,
                "threshold_final": 0.96,
            },
        },
    )


@configclass
class CRITerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Mark as time_out so rewards.is_terminated (OVF etc.) does not penalize success.
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
class CRIReachEnvCfg(ManagerBasedRLEnvCfg):
    """Reach environment with CRI observation; episode resets on reach success or failure."""

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
        self.episode_length_s = 48.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.sim.dt = 1.0 / 60.0
