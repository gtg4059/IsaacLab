# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR10 CRI reach with the direction filter wired but not applied.

Scene, robot, commands, rewards, events, terminations, and curriculum are
:class:`UR10ReachEnvCfg`. The arm action still calls ``run_cri_filter`` so the
CRI observation stays :func:`mdp.cri_filter_pre`. ``filter_enabled`` is False,
so the joint command is the policy velocity. Set it True to slow the whole
linear velocity at every collision point over the limit.

Launch from the IsaacLab repo root:

* Train: ``./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Reach-UR10-CRI-F-v0``
* Play: ``./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Reach-UR10-CRI-F-Play-v0 --num_envs 1``
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.config.ur_10.joint_pos_env_cfg import (
    UR10ReachEnvCfg,
    UR10ReachEnvCfg_PLAY,
)


@configclass
class CRIFObservationsCfg:
    """Policy observation for CRI-F. Actor and critic share this group."""

    @configclass
    class PolicyCfg(ObsGroup):
        ee_pose_error = ObsTerm(
            func=mdp.ee_pose_error_to_command,
            params={"command_name": "ee_pose", "asset_cfg": SceneEntityCfg("robot", body_names="ee_link")},
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )
        # Previous step's CRI of the sent command. Filter off: that is the raw forward.
        CRI = ObsTerm(func=mdp.cri_filter_pre)
        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.001, n_max=0.001))
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-0.001, n_max=0.001))
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    policy: PolicyCfg = PolicyCfg()


def _use_direction_filter(cfg) -> None:
    """Solve the CRI-level QP on every force and pressure row."""
    cfg.actions.arm_action = mdp.JointVelocityCriFilterActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True,
        cri_limit=0.96,
        direction_gain=0.9,
        speed_enabled=False,
        strong_cri=0.92,
        strong_target=0.0095,
        filter_enabled=False,
    )


@configclass
class CRIFReachEnvCfg(UR10ReachEnvCfg):
    """:class:`UR10ReachEnvCfg` with the filter action off and :class:`CRIFObservationsCfg`."""

    observations: CRIFObservationsCfg = CRIFObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _use_direction_filter(self)


@configclass
class CRIFReachEnvCfg_PLAY(UR10ReachEnvCfg_PLAY):
    """Play variant. Filter off, and no spawn rewind. This checkpoint is still the soft trajectory."""

    observations: CRIFObservationsCfg = CRIFObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _use_direction_filter(self)
        # Soft regime: a CRI crossing does not pull the arm back to the spawn,
        # and a timeout does not restore that spawn.
        if hasattr(self.events, "rewind_on_cri"):
            self.events.rewind_on_cri.params["enabled"] = False
        self.events.reset_robot_joints.params["keep_failed_spawn"] = False
