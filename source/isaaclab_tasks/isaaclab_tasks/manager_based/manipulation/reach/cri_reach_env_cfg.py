# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CRI-aware reach environment (UR_CRI_recurr style).

Episodes end on timeout or OVF only. In-gate hold is rewarded by streak and
total time in the pose/vel/acc basin; there is no reach-success reset.
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
            # Horizontal cylinder r<0.35 is excluded via pos_r min. max_pos_norm clips
            # the (r, z) box to a 1 m ball about base_link. pos_z is metres above
            # base_link (mount); -0.90 is world floor (-1.05) plus 15 cm clearance.
            pos_r=(0.35, 1.0),
            pos_z=(-0.90, 1.0),
            max_pos_norm=1.0,
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
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="ee_link"), "command_name": "ee_pose"},
    )
    # Off until last_centimeter_weight @ iter 24000. 8 cm cutoff; 1/e at 4 cm.
    # end_effector_pos_orientation_tracking_last_centimeter = RewTerm(
    #     func=mdp.position_orientation_command_error_last_centimeter,
    #     weight=0.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names="ee_link"), "command_name": "ee_pose"},
    # )
    # OVF only (time_out is time_out=True; reach no longer terminates).
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-600.0)
    action_rate = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.1)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    CRI_OVF = RewTerm(
        func=mdp.CRI_OVF_exp,
        weight=-10.0,
        params={"limit": 0.96, "sigma": 20.0},
    )
    # Hold 0.3 until 6 s, then lerp to params.final by 10 s (flat after that).
    # Weight -1 so the term value is the applied living cost. final stays 0.3
    # until is_alive_ramp_final. After iter 24000, final=1.2 (~18 / s): sit
    # cost shows up around the success p50, leaving ~38 s to explore / OVF.
    # Full sit ~756 + timeout -400 => park ~-1156. Explore-then-OVF at
    # 14–20 s is ~-744 to -852, and that ranking holds through ~37 s.
    # is_alive = RewTerm(
    #     func=mdp.is_alive_time_ramp,
    #     weight=-1.0,
    #     params={"ramp_start_s": 6.0, "ramp_end_s": 10.0, "initial": 0.3, "final": 0.3},
    # )
    # Off until timeout_no_reach_weight @ iter 24000. time_out term only
    # (reach_success is time_out=True and is not penalized).
    # timeout_no_reach = RewTerm(
    #     func=mdp.timeout_no_reach_penalty,
    #     weight=0.0,
    #     params={"reward_term_name": "reach_success_bonus"},
    # )
    # No hold_steps gate and no reach termination. Pays more as the episode-max
    # consecutive hold and the in-gate total grow (each / max_episode_length).
    # Full-episode hold ≈ 2 * 1200 * dt; a break-and-reenter streak only
    # increases stay, not progress, unless it beats the max.
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
            "stay_scale": 1.0,
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
    # fine_grained_tracking_weight = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "rewards.end_effector_pos_orientation_tracking_fine_grained.weight",
    #         "modify_fn": mdp.reward_weight_step_by_step,
    #         "modify_params": {
    #             "switch_step": 48 * 12000,
    #             "initial_weight": 1.0,
    #             "final_weight": 0.5,
    #         },
    #     },
    # )
    # Phase 1: no living (cfg 0.0 / 0.0). Phase 2: in-episode 8–20 s ramp 0.3 → 0.9.
    # is_alive_ramp_initial = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "rewards.is_alive.params.initial",
    #         "modify_fn": mdp.reward_weight_step_by_step,
    #         "modify_params": {
    #             "switch_step": 48 * 6000,
    #             "initial_weight": 0.0,
    #             "final_weight": 0.3,
    #         },
    #     },
    # )
    # is_alive_ramp_final = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "rewards.is_alive.params.final",
    #         "modify_fn": mdp.reward_weight_step_by_step,
    #         "modify_params": {
    #             "switch_step": 48 * 24000,
    #             "initial_weight": 0.3,
    #             "final_weight": 1.2,
    #         },
    #     },
    # )
    # # Same switch as is_alive_ramp_final. Pulls 8 cm → 3 cm after living
    # # pressure turns on; FG alone is flat at the current ~20 cm plateau.
    # last_centimeter_weight = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "rewards.end_effector_pos_orientation_tracking_last_centimeter.weight",
    #         "modify_fn": mdp.reward_weight_step_by_step,
    #         "modify_params": {
    #             "switch_step": 48 * 24000,
    #             "initial_weight": 0.0,
    #             "final_weight": 2.0,
    #         },
    #     },
    # )
    # # Same switch as is_alive_ramp_final. After 10 s, sitting burns 1.2/step
    # # so park loss is front-loaded; -400 is only the 48 s backstop.
    # # Park ~-1156 vs explore-OVF through ~37 s. Reach +1200 still dominates.
    # # Early suicide (~-609) is the cheap give-up; CRI_OVF -10 and tracking
    # # keep that from being the default.
    # timeout_no_reach_weight = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "rewards.timeout_no_reach.weight",
    #         "modify_fn": mdp.reward_weight_step_by_step,
    #         "modify_params": {
    #             "switch_step": 48 * 24000,
    #             "initial_weight": 0.0,
    #             "final_weight": -400.0,
    #         },
    #     },
    # )
    # Hold length is not a curriculum. Reward already scales with episode-max
    # consecutive hold and in-gate total; no interval / hold_steps schedule.
    # reach_settle_hold = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "rewards.reach_success_bonus.params.hold_steps",
    #         "modify_fn": mdp.hold_steps_by_step,
    #         "modify_params": {
    #             "switch_step": 48 * 10000,
    #             "initial_hold_steps": 2,
    #             "final_hold_steps": 8,
    #             "interval_steps": 48 * 2000,
    #             "increment": 2,
    #         },
    #     },
    # )

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
    # ee_pose_pos_r = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "commands.ee_pose.ranges.pos_r",
    #         "modify_fn": mdp.command_range_step_by_step,
    #         "modify_params": {
    #             "switch_step": 48 * 4000,
    #             "initial_range": (0.001, 0.04),
    #             "final_range": (0.001, 0.8),
    #         },
    #     },
    # )
    # # Same step as ee_pose_pos_r: keep CRI soft while the workspace is a thin cylinder,
    # # then raise it when pos_r opens to 0.8.
    # cri_ovf_reward_weight = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "rewards.CRI_OVF.weight",
    #         "modify_fn": mdp.reward_weight_step_by_step,
    #         "modify_params": {
    #             "switch_step": 48 * 4000,
    #             "initial_weight": -0.0,
    #             "final_weight": -2.0,
    #         },
    #     },
    # )
    cri_ovf_term_threshold = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "terminations.OVF.params.threshold",
            "modify_fn": mdp.cri_ovf_threshold_by_step,
            "modify_params": {
                "crossfade_start": 48 * 24000,
                "threshold_initial": 2.0,
                "threshold_final": 0.96,
            },
        },
    )


@configclass
class CRITerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Reach success does not end the episode; hold reward accumulates until
    # timeout or OVF. Keep the term commented so play can still opt in.
    # reach_success = DoneTerm(
    #     func=mdp.reach_success,
    #     params={"reward_term_name": "reach_success_bonus"},
    #     time_out=True,
    # )
    OVF = DoneTerm(
        func=mdp.CRI_OVF,
        params={"threshold": 0.96},
    )


@configclass
class CRIReachEnvCfg(ManagerBasedRLEnvCfg):
    """Reach environment with CRI observation; episode resets on timeout or OVF."""

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
        self.episode_length_s = 25.6
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.sim.dt = 1.0 / 60.0
