# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pick-and-place environment configuration.

Unified interface:

- **Action (8D):** 7 arm joints + 1 binary gripper command.
- **Observation (36D):** joint_pos (9) + joint_vel (9) + target_pose (7) + object_position (3) + actions (8).

Training task (``Isaac-Pickandplace-Cube-Franka-v0``):

1. **Curriculum phase 1** – air-only lift, ``object_dropped_after_lift`` active.
2. **Curriculum phase 2** – in-episode air target then floor place-down, drop termination off.

Place (manual gripper open) remains outside the RL env.
"""

from dataclasses import MISSING

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_tasks.manager_based.pickandplace import mdp as pnp_mdp

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the pick-and-place scene with a robot and an object."""

    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


def _configure_pickandplace_sim(env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Apply shared simulation settings for pick-and-place training."""
    env_cfg.decimation = 2
    env_cfg.sim.dt = 0.01
    env_cfg.sim.render_interval = env_cfg.decimation
    env_cfg.sim.physx.bounce_threshold_velocity = 0.01
    env_cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
    env_cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
    env_cfg.sim.physx.friction_correlation_distance = 0.00625


##
# Unified obs / action / command
##


@configclass
class LiftCommandsCfg:
    """Lift ``target_pose``: air on reset; phase 2 adds in-episode switch to floor after lift."""

    target_pose = pnp_mdp.AlternatingLiftPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        air_ranges=pnp_mdp.AlternatingLiftPoseCommandCfg.Ranges(
            pos_x=(0.35, 0.65),
            pos_y=(-0.3, 0.3),
            pos_z=(0.22, 0.55),
            roll=(math.pi, math.pi),
            pitch=(0.0, 0.0),
            yaw=(-0.5, 0.5),
        ),
        floor_ranges=pnp_mdp.AlternatingLiftPoseCommandCfg.Ranges(
            pos_x=(0.35, 0.65),
            pos_y=(-0.3, 0.3),
            pos_z=(0.14, 0.18),
            roll=(math.pi, math.pi),
            pitch=(0.0, 0.0),
            yaw=(-0.0, 0.0),
        ),
        # Unused by AlternatingLiftPoseCommand; kept for config compatibility.
        ranges=pnp_mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.35, 0.65),
            pos_y=(-0.3, 0.3),
            pos_z=(0.0, 0.55),
            roll=(math.pi, math.pi),
            pitch=(0.0, 0.0),
            yaw=(-0.5, 0.5),
        ),
    )


@configclass
class PnPActionsCfg:
    """Unified action space: 7-DOF arm + binary gripper (8D total)."""

    arm_action: pnp_mdp.JointPositionActionCfg = MISSING
    gripper_action: pnp_mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class PnPObservationsCfg:
    """Unified observation space (36D, no CRI).

    Layout: joint_pos (9) | joint_vel (9) | target_pose (7) | object_position (3) | actions (8)
    """

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=pnp_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=pnp_mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        target_pose = ObsTerm(func=pnp_mdp.generated_commands, params={"command_name": "target_pose"})
        object_position = ObsTerm(func=pnp_mdp.object_position_in_robot_root_frame)
        actions = ObsTerm(func=pnp_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


##
# Lift (pick, grasp, lift, place-down curriculum)
##


@configclass
class LiftEventCfg:
    """Reset events for the lift phase."""

    reset_all = EventTerm(func=pnp_mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=pnp_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (-0.03, -0.03)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class LiftRewardsCfg:
    """Reward terms for the lift phase."""

    reaching_object = RewTerm(func=pnp_mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)
    lifting_object = RewTerm(func=pnp_mdp.object_is_lifted, params={"minimal_height": 0.04}, weight=15.0)
    object_goal_tracking = RewTerm(
        func=pnp_mdp.ee_goal_distance,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "target_pose", "ee_body_name": "panda_hand"},
        weight=16.0,
    )
    object_goal_tracking_fine_grained = RewTerm(
        func=pnp_mdp.ee_goal_distance,
        params={"std": 0.05, "minimal_height": 0.04, "command_name": "target_pose", "ee_body_name": "panda_hand"},
        weight=5.0,
    )
    end_effector_orientation_tracking = RewTerm(
        func=pnp_mdp.ee_orientation_command_error,
        params={"minimal_height": 0.04, "command_name": "target_pose", "ee_body_name": "panda_hand"},
        weight=-1.0,
    )
    action_rate = RewTerm(func=pnp_mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=pnp_mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # termination_penalty = RewTerm(func=pnp_mdp.is_terminated, weight=-100.0)


@configclass
class LiftTerminationsCfg:
    """Termination terms for the lift phase."""

    time_out = DoneTerm(func=pnp_mdp.time_out, time_out=True)
    object_dropping = DoneTerm(
        func=pnp_mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )
    object_dropped_after_lift = DoneTerm(
        func=pnp_mdp.object_dropped_after_lift,
        params={
            "minimal_lift_height": 0.04,
            "table_height_threshold": 0.04,
            "object_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class LiftCurriculumCfg:
    """Curriculum terms for the lift phase."""

    action_rate = CurrTerm(
        func=pnp_mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 20000}
    )
    joint_vel = CurrTerm(
        func=pnp_mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 20000}
    )
    place_down = CurrTerm(func=pnp_mdp.lift_place_down_curriculum)


@configclass
class PickandplaceLiftEnvCfg(ManagerBasedRLEnvCfg):
    """Pick, grasp, lift, and place-down (two-phase curriculum)."""

    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: PnPObservationsCfg = PnPObservationsCfg()
    actions: PnPActionsCfg = PnPActionsCfg()
    commands: LiftCommandsCfg = LiftCommandsCfg()
    rewards: LiftRewardsCfg = LiftRewardsCfg()
    terminations: LiftTerminationsCfg = LiftTerminationsCfg()
    events: LiftEventCfg = LiftEventCfg()
    curriculum: LiftCurriculumCfg = LiftCurriculumCfg()

    def __post_init__(self):
        _configure_pickandplace_sim(self)
        # Long enough for lift + in-episode place-down in curriculum phase 2.
        self.episode_length_s = 12.0
