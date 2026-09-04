# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CBF-filter reach environment (UR10).

CRI is not computed here via AtMotionState / ``collision_risk_index``.
The policy CRI term is the previous-step ``run_cri_filter`` cache
(:func:`mdp.cri_filter_pre`). Reset / first obs: CRI=0. Each later tick
solves once; the next policy obs is that CRI. The policy action is a joint
delta ``Δq`` (OpenPI π0.5-DROID). The CBF-QP still filters velocity
``qd_nom = Δq / dt``; the plant command is the integrated position
``q + qd_cmd * dt``. Time-scale ``s`` is not an observation or control
input. ``filter_enabled`` toggles the CBF filter.

Launch from the IsaacLab repo root:

* Train: ``./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Reach-UR10-CRI-F-v0``
* Play: ``./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Reach-UR10-CRI-F-Play-v0 --num_envs 1``
"""

import math

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.cri_reach_env_cfg import CRIReachEnvCfg

from isaaclab_assets import UR10_DC_MOTOR_CFG  # isort: skip


@configclass
class CRIFObservationsCfg:
    """Same policy terms as :class:`CRIObservationsCfg`. No time-scale ``s``."""

    @configclass
    class PolicyCfg(ObsGroup):
        ee_pose_error = ObsTerm(
            func=mdp.ee_pose_error_to_command,
            params={"command_name": "ee_pose", "asset_cfg": SceneEntityCfg("robot", body_names="ee_link")},
        )
        CRI = ObsTerm(func=mdp.cri_filter_pre)
        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.001, n_max=0.001))
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-0.001, n_max=0.001))
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.concatenate_terms = True
            self.history_length = 5

    policy: PolicyCfg = PolicyCfg()


@configclass
class CRIFReachEnvCfg(CRIReachEnvCfg):
    """UR10 reach env whose CRI and command come only from the CBF ``run_cri_filter``."""

    observations: CRIFObservationsCfg = CRIFObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UR10_DC_MOTOR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.commands.ee_pose.body_name = "ee_link"
        self.actions.arm_action = mdp.JointPositionCriFilterActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=0.05,
            use_zero_offset=True,
            cri_limit=0.96,
            cbf_alpha=0.02,
            filter_enabled=True,
        )

        self.events.reset_robot_joints.params["position_range"] = {
            "shoulder_pan_joint": (-math.pi, math.pi),
            "shoulder_lift_joint": (-math.pi, 0.0),
            "elbow_joint": (-math.pi * 2 / 3, math.pi * 2 / 3),
            "wrist_1_joint": (-math.pi, math.pi),
            "wrist_2_joint": (-math.pi, math.pi),
            "wrist_3_joint": (-math.pi, math.pi),
        }
        self.events.reset_robot_joints.params["velocity_range"] = {
            "shoulder_pan_joint": (0.0, 0.0),
            "shoulder_lift_joint": (0.0, 0.0),
            "elbow_joint": (0.0, 0.0),
            "wrist_1_joint": (0.0, 0.0),
            "wrist_2_joint": (0.0, 0.0),
            "wrist_3_joint": (0.0, 0.0),
        }

        self.commands.ee_pose.ranges.pos_th = (-math.pi, math.pi)
        self.commands.ee_pose.ranges.roll = (-math.pi, math.pi)
        self.commands.ee_pose.ranges.pitch = (-math.pi, math.pi)
        self.commands.ee_pose.ranges.yaw = (-math.pi, math.pi)


@configclass
class CRIFReachEnvCfg_PLAY(CRIFReachEnvCfg):
    """Single-env play variant of :class:`CRIFReachEnvCfg`."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.viewer.eye = (4.5, 4.5, 4.5)
        self.observations.policy.enable_corruption = False
        self.terminations.OVF.params["threshold"] = 0.96
        self.commands.ee_pose.debug_vis = True
        if hasattr(self.rewards, "CRI_OVF"):
            self.rewards.CRI_OVF.params["limit"] = 0.96
