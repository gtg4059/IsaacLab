# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CRI-filter reach environment.

CRI is not computed here via AtMotionState / ``collision_risk_index``.
The policy CRI term is the previous-step ``run_cri_filter`` cache
(:func:`mdp.cri_filter_pre`). Reset / first obs: CRI=0, s=1. Each later tick
solves once; the next policy obs is that CRI. ``filter_enabled`` only toggles
Newton (s). Other terms match :class:`CRIObservationsCfg` plus ``s``.
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.cri_reach_env_cfg import CRIReachEnvCfg


@configclass
class CRIFObservationsCfg:
    """Same policy terms as :class:`CRIObservationsCfg`, plus filter scale ``s``."""

    @configclass
    class PolicyCfg(ObsGroup):
        ee_pose_error = ObsTerm(
            func=mdp.ee_pose_error_to_command,
            params={"command_name": "ee_pose", "asset_cfg": SceneEntityCfg("robot", body_names="ee_link")},
        )
        CRI = ObsTerm(func=mdp.cri_filter_pre)
        cri_scale = ObsTerm(func=mdp.cri_filter_scale)
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
    """Reach env whose CRI comes only from ``run_cri_filter``."""

    observations: CRIFObservationsCfg = CRIFObservationsCfg()
