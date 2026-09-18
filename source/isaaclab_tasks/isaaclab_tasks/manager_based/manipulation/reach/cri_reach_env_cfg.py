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
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="ee_link"), "command_name": "ee_pose"},
    )
    # Pay iff d <= support_distance (8 cm). Mean of d / θ / combined-twist
    # slacks toward 0. Twist is ||(v, ω)||_2 vs 0.08 (8× of Jawale 0.02).
    # Slack widths stay at 8×; they do not cool.
    # Weight 0 until pose_twist_linear_weight (48 * 7000) sets 2.0.
    end_effector_pose_twist_linear = RewTerm(
        func=mdp.distance_linear_approach_reward,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="ee_link"),
            "command_name": "ee_pose",
            "support_distance": 0.01 * 8.0,
            "max_angle_rad": 0.05 * 8.0,
            "max_lin_vel": 0.01 * 8.0,  # ||(v, ω)||_2, 8× of Jawale 0.02
        },
    )
    # OVF only (time_out is time_out=True; reach no longer terminates).
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)
    action_rate = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.1)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    CRI_OVF = RewTerm(
        func=mdp.CRI_OVF_exp,
        weight=-6.0,
        params={"limit": 0.96, "sigma": 12.0},
    )
    # Constant −0.3 / s (dt cancels). initial=final so the 8–16 s ramp is unused.
    # 32 s timeout ≈ −9.6 vs one-shot ≈ +40. OVF steps are 0 (not alive).
    is_alive = RewTerm(
        func=mdp.is_alive_time_ramp,
        weight=-1.0,
        params={"ramp_start_s": 8.0, "ramp_end_s": 16.0, "initial": 0.3, "final": 0.3},
    )
    # One-shot: +1 on the first in-gate step of a streak (re-entry pays again).
    # Applied weight * dt ≈ 40. Staying in-gate does not keep paying.
    reach_success_bonus = RewTerm(
        func=mdp.ReachSuccessCriteria,
        weight=600.0,
        params={
            "command_name": "ee_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names="ee_link"),
            # 1× finals. Curriculum cools 8× → 2×: 2 cm / 0.1 rad /
            # ||(v, ω)||_2 < 0.02 (Jawale).
            "max_distance": 0.01,
            "max_angle_rad": 0.05,
            "max_lin_vel": 0.01,  # combined twist; cools to 0.02
            "hold_square_max": 1,
        },
    )


@configclass
class CRICurriculumCfg:
    cri_ovf_term_threshold = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "terminations.OVF.params.threshold",
            "modify_fn": mdp.cri_ovf_threshold_by_step,
            "modify_params": {
                "crossfade_start": 48 * 4000,
                "threshold_initial": 2.0,
                "threshold_final": 0.96,
            },
        },
    )

    # Cool 8× → 2× (2 cm / 0.1 rad / ||(v, ω)||_2 < 0.02) and hold.
    reach_success_criteria = CurrTerm(
        func=mdp.reach_success_criteria_curriculum,
        params={
            "ease_factor": 8.0,
            "cool_end_factor": 2.0,
            "decay_alpha": 2.0,
            "start_step": 48 * 7000,
            "num_steps": 48 * 9000,
            "reward_term_names": ["reach_success_bonus"],
            "event_term_name": None,
            "mix_half_and_one_after": False,
        },
    )
    pose_twist_linear_weight = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.end_effector_pose_twist_linear.weight",
            "modify_fn": mdp.reward_weight_step_by_step,
            "modify_params": {
                "switch_step": 48 * 7000,
                "initial_weight": 0.0,
                "final_weight": 1.0,
            },
        },
    )

@configclass
class CRITerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Success no longer resets; hold bonus can fire more than once per episode.
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
        self.k = 3.2
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 10*self.k
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.sim.dt = 1.0 / 60.0
