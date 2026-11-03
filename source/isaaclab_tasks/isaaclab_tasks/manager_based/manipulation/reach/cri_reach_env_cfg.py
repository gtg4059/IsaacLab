# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CRI-aware reach environment (UR_CRI_recurr style).

Episodes end on reach or the horizon. Crossing the CRI limit does not end the
episode: the arm rewinds inside the episode and the return continues. A horizon
timeout keeps the target and draws a new arm. Reach success draws both.
"""

from dataclasses import MISSING

from pxr import Usd, UsdPhysics

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


def _deactivate_colliders(prim_path: str, keep_suffixes: tuple[str, ...] = ()) -> int:
    """Remove colliders from the physics scene without flattening visual instances.

    Leaving shapes in place with ``collisionEnabled`` false keeps them in the GPU
    collision pipeline and hangs ``sim.reset``. Deactivated prims are skipped.
    An articulation with every shape removed also hangs ``sim.reset`` at 8192
    environments, so ``keep_suffixes`` can retain one primitive.
    """
    stage = sim_utils.get_current_stage()
    root = stage.GetPrimAtPath(prim_path)
    pending = [root]
    while pending:
        child = pending.pop()
        if child.IsInstance():
            if "collision" in child.GetName().lower():
                child.SetInstanceable(False)
                pending.extend(child.GetChildren())
            continue
        pending.extend(child.GetChildren())
    colliders = [prim for prim in Usd.PrimRange(root) if prim.HasAPI(UsdPhysics.CollisionAPI)]
    deactivated = 0
    for prim in colliders:
        path = prim.GetPath().pathString
        if any(path.endswith(suffix) for suffix in keep_suffixes):
            continue
        prim.SetActive(False)
        deactivated += 1
    if not colliders:
        raise RuntimeError(f"No collision prim found under '{prim_path}'.")
    return deactivated


@sim_utils.clone
def _spawn_usd_no_collision(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a USD asset and drop its colliders before physics is created."""
    prim = sim_utils.spawn_from_usd.__wrapped__(prim_path, cfg, translation, orientation, **kwargs)
    _deactivate_colliders(prim_path)
    return prim


@sim_utils.clone
def _spawn_ur10_base_collider_only(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Drop UR10 link colliders except the base cylinder.

    PhysX GPU setup hangs at 8192 environments when the articulation has no shapes.
    """
    prim = sim_utils.spawn_from_usd.__wrapped__(prim_path, cfg, translation, orientation, **kwargs)
    _deactivate_colliders(prim_path, keep_suffixes=("/base_link/cylinder",))
    return prim


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

    # Visual stand only. The collision cube is deactivated so PhysX never cooks it.
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd",
            scale=(2.0, 2.0, 2.0),
            func=_spawn_usd_no_collision,
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
        # Timeout keeps this target. Reach success draws a new one.
        keep_command_until_success=True,
        ranges=mdp.UniformPoseTrigCommandCfg.PolarRanges(
            pos_th=MISSING,
            # Horizontal cylinder r<0.35 is excluded via pos_r min. max_pos_norm clips
            # the (r, z) box to a 1 m ball about base_link. pos_z is metres above
            # base_link (mount); -0.90 is world floor (-1.05) plus 15 cm clearance.
            pos_r=(0.4, 1.1),
            pos_z=(0.2, 1.0),
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
        # IndustReal / Isaac Lab gear: 목표·EE는 외수용 → mm·deg
        ee_pose_error = ObsTerm(
            func=mdp.ee_pose_error_to_command,
            params={"command_name": "ee_pose", "asset_cfg": SceneEntityCfg("robot", body_names="ee_link")},
            noise=Unoise(n_min=-0.002, n_max=0.002),  # ±2 mm, ±0.002 rad
        )
        CRI = ObsTerm(
            func=mdp.collision_risk_index,
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=mdp.last_action)
        # Seconds since this target was sampled. Actor and critic share this group.
        # goal_elapsed_s = ObsTerm(func=mdp.goal_elapsed_s, params={"command_name": "ee_pose"})

        def __post_init__(self):
            self.enable_corruption = True
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
            # Timeout and reach success draw a new arm. The target is kept
            # separately by keep_command_until_success.
            "keep_failed_spawn": False,
        },
    )
    # Every control step. Crossing the OVF threshold rewinds to the spawn.
    # The episode and the target stay. ovf_rewind_bonus pays on that step.
    rewind_on_cri = EventTerm(
        func=mdp.rewind_arm_on_cri_limit,
        mode="interval",
        interval_range_s=(0.0, 0.0),
    )


@configclass
class CRIRewardsCfg:
    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-0.4,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="ee_link"), "command_name": "ee_pose"},
    )

    end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="ee_link"), "command_name": "ee_pose"},
    )

    # end_effector_pos_orientation_tracking = RewTerm(
    #     func=mdp.position_orientation_command_error,
    #     weight=0.5,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names="ee_link"), "command_name": "ee_pose"},
    # )

    # Position + angle only. Peak is the weight, including while the hand is still moving.
    # end_effector_pos_orientation_tracking_fine_grained = RewTerm(
    #     func=mdp.position_orientation_command_error_fine_grained,
    #     weight=0.5,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names="ee_link"), "command_name": "ee_pose"},
    # )

    # CRI_OVF = RewTerm(
    #     func=mdp.CRI_OVF_lin,
    #     weight=-2.0,
    #     params={"limit": 0.96},
    # )
    # +1 on the return for each OVF rewind (weight * dt, dt = 1/15).
    # ovf_rewind_bonus_weight holds this at 0 until iter 18000.
    ovf_rewind_bonus = RewTerm(func=mdp.ovf_rewind_bonus, weight=0.0)
    # Same pose sum, times exp(-||(v,ω)|| / 0.10). Peak is the weight only when stopped.
    # end_effector_pos_orientation_twist_tracking_fine_grained = RewTerm(
    #     func=mdp.position_orientation_twist_command_error_fine_grained,
    #     weight=2.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="ee_link"),
    #         "command_name": "ee_pose",
    #         "max_lin_vel": 0.10,
    #     },
    # )
    # Pay iff d <= support_distance (8 cm). Mean of d / θ / combined-twist
    # slacks toward 0. Twist is ||(v, ω)||_2 vs 0.08 (8× of Jawale 0.02).
    # Slack widths stay at 8×; they do not cool.
    # Weight 0 until pose_twist_linear_weight (48 * 7000) sets 2.0.
    # end_effector_pose_twist_linear = RewTerm(
    #     func=mdp.distance_linear_approach_reward,
    #     weight=0.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="ee_link"),
    #         "command_name": "ee_pose",
    #         "support_distance": 0.01 * 8.0,
    #         "max_angle_rad": 0.05 * 8.0,
    #         "max_lin_vel": 0.01 * 8.0,  # ||(v, ω)||_2, 8× of Jawale 0.02
    #     },
    # )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=0.0)
    action_rate = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.1)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    # −1 per step from the start (weight −15, step_dt = 1/15).
    is_alive = RewTerm(
        func=mdp.is_alive_time_ramp,
        weight=-0.0,
        params={"ramp_start_s": 8.0, "ramp_end_s": 16.0, "initial": 1.0, "final": 1.0},
    )
    # 0 until a rewind. Weight +5 pays back is_alive's −5/s on that attempt.
    # A clean arrival keeps the clock. clean_attempt_time_weight sets +5.
    clean_attempt_time = RewTerm(func=mdp.clean_attempt_time_bonus, weight=0.0)
    # One-shot +1 on the arrival step. weight * dt ≈ +40.
    # Staying in-gate does not keep paying.
    # Ha et al., CoRL 2020, curriculum level 1: 10 cm and 0.20 rad. No twist gate.
    reach_success_bonus = RewTerm(
        func=mdp.ReachSuccessCriteria,
        weight=600.0,
        params={
            "command_name": "ee_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names="ee_link"),
            "max_distance": 0.25,
            "max_angle_rad": 0.25,
            "max_lin_vel": 0.20,  # ||(v, ω)||_2
            "hold_square_max": 1,
        },
    )


@configclass
class CRICurriculumCfg:
    # Off. That term kept the failed attempt's spawn as well as the target.
    # Timeout now keeps only the target and draws a new arm.
    # resample_on_reset = CurrTerm(
    #     func=mdp.resample_on_reset_curriculum,
    #     params={
    #         "switch_step": 48 * 20000,
    #         "command_name": "ee_pose",
    #         "event_term_name": "reset_robot_joints",
    #     },
    # )
    # After the 16-45-34 reach phase and the OVF cool (iter 6000), ramp to −1 per step.
    # is_alive_weight = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "rewards.is_alive.weight",
    #         "modify_fn": mdp.linear_interpolate_by_step,
    #         "modify_params": {
    #             "start_step": 48 * 18000,
    #             "num_steps": 48 * 1000,
    #             "initial_value": 0.0,
    #             "final_value": -5.0,
    #         },
    #     },
    # )
    # ovf_rewind_bonus_weight = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "rewards.ovf_rewind_bonus.weight",
    #         "modify_fn": mdp.linear_interpolate_by_step,
    #         "modify_params": {
    #             "start_step": 48 * 18000,
    #             "num_steps": 48 * 1000,
    #             "initial_value": 0.0,
    #             "final_value": 2.0,
    #         },
    #     },
    # )

    is_alive_weight = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.is_alive.weight",
            "modify_fn": mdp.reward_weight_step_by_step,
            "modify_params": {
                "switch_step": 48 * 18000,
                "initial_weight": -0.0,
                "final_weight": -5.0,
            },
        },
    )

    # # No rewind bonus until iter 18000, then +2/15 on the rewind step.
    # ovf_rewind_bonus_weight = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "rewards.ovf_rewind_bonus.weight",
    #         "modify_fn": mdp.reward_weight_step_by_step,
    #         "modify_params": {
    #             "switch_step": 48 * 18000,
    #             "initial_weight": 0.0,
    #             "final_weight": 0.0,
    #         },
    #     },
    # )
    # After iter 120000, a rewind pays back is_alive (−5/s) for that attempt.
    clean_attempt_time_weight = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.clean_attempt_time.weight",
            "modify_fn": mdp.reward_weight_step_by_step,
            "modify_params": {
                "switch_step": 48 * 120000,
                "initial_weight": 0.0,
                "final_weight": 5.0,
            },
        },
    )
    # Hold action-rate at -0.1 until the same step, then drop it to -0.001.
    action_rate_weight = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.action_rate.weight",
            "modify_fn": mdp.reward_weight_step_by_step,
            "modify_params": {
                "switch_step": 48 * 120000,
                "initial_weight": -0.1,
                "final_weight": -0.001,
            },
        },
    )
    # Hold the travel penalties until iter 19000, then drop them.
    # Orientation matching and action-rate both lengthen the path.
    # orientation_tracking_weight = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "rewards.end_effector_orientation_tracking.weight",
    #         "modify_fn": mdp.reward_weight_step_by_step,
    #         "modify_params": {
    #             "switch_step": 48 * 18000,
    #             "initial_weight": -0.2,
    #             "final_weight": 0.0,
    #         },
    #     },
    # )
    # # At iter 19000, strengthen the distance term so the path shortens.
    # position_tracking_weight = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "rewards.end_effector_position_tracking.weight",
    #         "modify_fn": mdp.reward_weight_step_by_step,
    #         "modify_params": {
    #             "switch_step": 48 * 18000,
    #             "initial_weight": -0.4,
    #             "final_weight": -4.0,
    #         },
    #     },
    # )

    # vasan_reset_cost_weight = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "rewards.vasan_reset_cost.weight",
    #         "modify_fn": mdp.linear_interpolate_by_step,
    #         "modify_params": {
    #             "start_step": 48 * 18000,
    #             "num_steps": 48 * 1000,
    #             "initial_value": -0.15,
    #             "final_value": -5.0,
    #         },
    #     },
    # )
    # Quiet until iter 800, so the reach bonus can turn on before paths are bent.
    # Hold 0.05 rad/s through iter 3500 (arrival time was still falling in that
    # stretch). Cool to 0 by iter 4000, before the OVF line starts moving.
    # The sample is once per episode; sin(pi t / T) makes it 0 at arrival.
    # early_spline_velocity_offset = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "events.early_spline_velocity_offset.params.std",
    #         "modify_fn": mdp.spline_offset_std_by_step,
    #         "modify_params": {
    #             "hold_start": 48 * 500,
    #             "hold_end": 48 * 1000,
    #             "cool_steps": 48 * 500,
    #             "amplitude": 0.05,
    #         },
    #     },
    # )
    # cri_ovf_term_threshold = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "terminations.OVF.params.threshold",
    #         "modify_fn": mdp.cri_ovf_threshold_by_step,
    #         "modify_params": {
    #             "crossfade_start": 48 * 1000,
    #             "num_steps": 48 * 1000,
    #             "decay_alpha": 2.0,
    #             "threshold_initial": 1.3,
    #             "threshold_final": 0.96,
    #         },
    #     },
    # )
    
    # 4× until iter 2000 (0.40 m, 0.80 rad, twist 0.40), then to 1× by iter 3000.
    # reach_success_criteria = CurrTerm(
    #     func=mdp.reach_success_criteria_curriculum,
    #     params={
    #         "ease_factor": 2.0,
    #         "decay_alpha": 2.0,
    #         "start_step": 48 * 2000,
    #         "num_steps": 48 * 1000,
    #         "reward_term_names": ["reach_success_bonus"],
    #         "event_term_name": None,
    #         "mix_half_and_one_after": False,
    #     },
    # )

    # Off. Decay on the reach bonus is the time pressure.
    # is_alive_weight = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "rewards.is_alive.weight",
    #         "modify_fn": mdp.linear_interpolate_by_step,
    #         "modify_params": {
    #             "start_step": 48 * 12000,
    #             "num_steps": 48 * 8000,
    #             "initial_value": 0.0,
    #             "final_value": -4.0,
    #         },
    #     },
    # )
    # pose_twist_linear_weight = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "rewards.end_effector_pose_twist_linear.weight",
    #         "modify_fn": mdp.reward_weight_step_by_step,
    #         "modify_params": {
    #             "switch_step": 48 * 7000,
    #             "initial_weight": 0.0,
    #             "final_weight": 1.0,
    #         },
    #     },
    # )

@configclass
class CRITerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Success resets the robot. time_out=True so the termination penalty does not apply.
    reach_success = DoneTerm(
        func=mdp.reach_success,
        params={"reward_term_name": "reach_success_bonus"},
        time_out=False,
    )
    # Not an episode end. rewind_on_cri reads this threshold and rewinds.
    OVF = DoneTerm(
        func=mdp.cri_boundary_continue,
        params={"threshold": 0.96},
    )


@configclass
class CRIReachEnvCfg(ManagerBasedRLEnvCfg):
    """Reach environment with CRI observation. Episodes end on reach, timeout, or OVF."""
    # 8192 is the training width. The base cylinder kept in
    # _spawn_ur10_base_collider_only is what stops PhysX from hanging here.
    scene: CRIReachSceneCfg = CRIReachSceneCfg(num_envs=8192, env_spacing=2.5)
    observations: CRIObservationsCfg = CRIObservationsCfg()
    actions: CRIActionsCfg = CRIActionsCfg()
    commands: CRICommandsCfg = CRICommandsCfg()
    rewards: CRIRewardsCfg = CRIRewardsCfg()
    terminations: CRITerminationsCfg = CRITerminationsCfg()
    events: CRIEventCfg = CRIEventCfg()
    curriculum: CRICurriculumCfg = CRICurriculumCfg()
    # curriculum: None = None

    def __post_init__(self):
        self.k = 3.2
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 10*self.k
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.sim.dt = 1.0 / 60.0
