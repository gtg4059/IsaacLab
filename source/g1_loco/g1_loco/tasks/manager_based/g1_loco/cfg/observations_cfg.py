# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation 설정: policy/critic 입력 벡터 정의."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from .. import mdp
from .common_cfg import G1_ACTUATED_JOINT_NAMES


def _robot_joint_cfg() -> SceneEntityCfg:
    """관절 관측에서 재사용하는 로봇 엔티티 설정을 생성한다."""
    return SceneEntityCfg("robot", joint_names=G1_ACTUATED_JOINT_NAMES, preserve_order=True)


@configclass
class ObservationsCfg:
    """관측 벡터 정의.

    - policy: 실제 정책 네트워크 입력
    - critic: 가치함수 학습용 입력(학습 안정성 향상을 위해 더 풍부할 수 있음)
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """정책(Actor) 입력 관측."""

        # 기본 각속도 관측
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), scale=0.25)
        # 중력 방향 관측
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        # 관절 위치/속도 관측
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": _robot_joint_cfg()},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": _robot_joint_cfg()},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            scale=0.05,
        )
        # 이전 행동 및 목표 속도 명령 관측
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            scale=(2.0, 2.0, 0.25),
        )

        def __post_init__(self):
            # 관측 노이즈를 활성화해 강건성(robustness) 확보
            self.enable_corruption = True
            # 개별 관측 항목을 하나의 벡터로 이어 붙여 네트워크에 입력
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """가치함수(Critic) 입력 관측."""

        # 정책 학습 안정화를 위해 critic에는 추가 관측(base_lin_vel)을 포함
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1), scale=2.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": _robot_joint_cfg()},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": _robot_joint_cfg()},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            scale=(2.0, 2.0, 0.25),
        )

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
