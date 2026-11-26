# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CRI-aware reach environment (UR_CRI_recurr style).

Episodes end on reach or the horizon. Max CRI above 1.99 rewinds the arm to
the attempt's start pose and does not end the episode. From 0.96 up to that
line the arm stays and the step pays the soft penalty. A horizon timeout keeps
the target and restores that attempt's start pose. Reach success draws a new
arm and a new target.
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
            noise=Unoise(n_min=-0.01, n_max=0.01),  # ±2 mm, ±0.002 rad
        )
        CRI = ObsTerm(
            func=mdp.collision_risk_index,
        )
        # Isaac Lab reach 기본. ±0.01 rad, ±0.01 rad/s.
        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.001, n_max=0.001))
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-0.01, n_max=0.01))
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
            # A timeout restores this attempt's start pose. Reach success draws
            # a new arm. The target is kept by keep_command_until_success.
            "keep_failed_spawn": True,
        },
    )
    # Max CRI above terminations.OVF (1.99) puts the arm back at the spawn.
    # The episode keeps running.
    rewind_on_cri = EventTerm(
        func=mdp.rewind_arm_on_cri_limit,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={"enabled": True},
    )


@configclass
class CRIRewardsCfg:
    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="ee_link"), "command_name": "ee_pose"},
    )

    end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="ee_link"), "command_name": "ee_pose"},
    )

    # Soft line at 0.96. y = 0 at or below it. Above it, y = (max CRI - 0.96) + 0.1.
    # Crossing 1.99 rewinds the arm and does not end the episode. Weight steps
    # from −0.5 to −15 at 48 * 2000. Return is weight * y * dt.
    ovf_rewind_bonus = RewTerm(
        func=mdp.ovf_rewind_bonus,
        weight=-0.5,
        params={"threshold": 0.96},
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=0.0)
    action_rate = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.1)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    # 0 until is_alive_weight sets −5 at 48 * 120000.
    is_alive = RewTerm(
        func=mdp.is_alive_time_ramp,
        weight=-0.0,
        params={"ramp_start_s": 8.0, "ramp_end_s": 16.0, "initial": 1.0, "final": 1.0},
    )
    # 0 until clean_attempt_time_weight. Paid on a reach, for attempts rewound
    # earlier in that episode. A horizon timeout pays nothing.
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
            "max_distance": 0.05,
            "max_angle_rad": 0.05,
            # ||(v, ω)||_2. Ignored until vel_switch_step, then this bound.
            "max_lin_vel": 0.10,
            "vel_switch_step": 48 * 120000,
            "hold_square_max": 1,
        },
    )

@configclass
class CRICurriculumCfg:
    # # Penalty magnitude grows from 0 to −0.5. 48 steps is one iteration.
    # ovf_rewind_bonus_weight = CurrTerm(
    #     func=mdp.modify_reward_weight_linear,
    #     params={
    #         "term_name": "ovf_rewind_bonus",
    #         "initial_weight": -15.0,
    #         "final_weight": -15.0,
    #         "start_step": 48 * 8000,
    #         "num_steps": 48 * 8000,
    #     },
    # )

    ovf_rewind_bonus_weight = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.ovf_rewind_bonus.weight",
            "modify_fn": mdp.reward_weight_step_by_step,
            "modify_params": {
                "switch_step": 48 * 2000,
                "initial_weight": -0.5,
                "final_weight": -25.0,
            },
        },
    )

    # ovf_rewind_bonus_weight = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "rewards.ovf_rewind_bonus.weight",
    #         "modify_fn": mdp.reward_weight_step_by_step,
    #         "modify_params": {
    #             "switch_step": 48 * 12000,
    #             "initial_weight": -0.5,
    #             "final_weight": -0.0,
    #         },
    #     },
    # )

    # # Soft until this step, then rewind to the spawn on a crossing.
    # rewind_on_cri = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "events.rewind_on_cri.params.enabled",
    #         "modify_fn": mdp.reward_weight_step_by_step,
    #         "modify_params": {
    #             "switch_step": 48 * 10000,
    #             "initial_weight": 0.0,
    #             "final_weight": 1.0,
    #         },
    #     },
    # )
    # # Penalty while the rewind is off. Weight 0 once the rewind turns on.
    # ovf_rewind_bonus_weight = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "rewards.ovf_rewind_bonus.weight",
    #         "modify_fn": mdp.reward_weight_step_by_step,
    #         "modify_params": {
    #             "switch_step": 48 * 10000,
    #             "initial_weight": -15.0,
    #             "final_weight": 0.0,
    #         },
    #     },
    # )
    is_alive_weight = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.is_alive.weight",
            "modify_fn": mdp.reward_weight_step_by_step,
            "modify_params": {
                "switch_step": 48 * 8000,
                "initial_weight": -0.0,
                "final_weight": -0.5,
            },
        },
    )
    # clean_attempt_time_weight = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "rewards.clean_attempt_time.weight",
    #         "modify_fn": mdp.reward_weight_step_by_step,
    #         "modify_params": {
    #             "switch_step": 48 * 72000,
    #             "initial_weight": 0.0,
    #             "final_weight": 2.0,
    #         },
    #     },
    # )
    # Hold action-rate at -0.1 until 48 * 120000, then drop it to -0.001.
    # action_rate_weight = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "rewards.action_rate.weight",
    #         "modify_fn": mdp.reward_weight_step_by_step,
    #         "modify_params": {
    #             "switch_step": 48 * 120000,
    #             "initial_weight": -0.1,
    #             "final_weight": -0.001,
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
    # Not an episode end. rewind_on_cri reads this threshold and returns the
    # arm to the spawn. The soft penalty is rewards.ovf_rewind_bonus at 0.96.
    OVF = DoneTerm(
        func=mdp.cri_boundary_continue,
        params={"threshold": 1.99},
    )


@configclass
class CRIReachEnvCfg(ManagerBasedRLEnvCfg):
    """Reach environment with CRI observation. Episodes end on reach or timeout."""
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
