# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG, RANDOM_ROUGH_TERRAINS_CFG  # isort: skip

#########################################################
# 기본적으로 unitree_SDK에서 지정하는 모터 순서를 사용하며 
# 스케일 조정은 unitree_rl_gym과 동일하게 이루어지도록 설정
#########################################################


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=RANDOM_ROUGH_TERRAINS_CFG, # 거친 바닥만 사용하는 terrain 설정 
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    
    base_velocity = mdp.UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(5.0, 10.0),
            rel_standing_envs=0.2, # 속도 명령을 0으로 설정하는 확률
            rel_heading_envs=0.8, # 헤딩 명령을 사용하는 확률
            heading_command=True,
            heading_control_stiffness=2.0, # 헤딩 명령을 사용할 때 헤딩 오차를 얼마나 빠르게 보정할지
            debug_vis=True, # 디버그 시각화 활성화
            ranges=mdp.UniformVelocityCommandCfg.Ranges( # 속도 명령의 범위
                lin_vel_x=(-1.0, 2.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
            ),
        )
    

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", 
        # deploy code와 같은 순서로 joint_names를 설정
        joint_names=[
                     'left_hip_pitch_joint', 
                     'left_hip_roll_joint', 
                     'left_hip_yaw_joint', 
                     'left_knee_joint', 
                     'left_ankle_pitch_joint', 
                     'left_ankle_roll_joint', 
                     'right_hip_pitch_joint', 
                     'right_hip_roll_joint', 
                     'right_hip_yaw_joint', 
                     'right_knee_joint', 
                     'right_ankle_pitch_joint', 
                     'right_ankle_roll_joint',
                     # G1_29_no_hand
                    "waist_yaw_joint",
                    "waist_roll_joint",
                    "waist_pitch_joint",
                    # G1 상체 arm: 하체·허리만 제어 시 주석 처리 (상체는 interval 이벤트로 랜덤 타겟)
                    # "left_shoulder_pitch_joint",
                    # "left_shoulder_roll_joint",
                    # "left_shoulder_yaw_joint",
                    # "left_elbow_joint",
                    # "left_wrist_roll_joint",
                    # "left_wrist_pitch_joint",
                    # "left_wrist_yaw_joint",
                    # "right_shoulder_pitch_joint",
                    # "right_shoulder_roll_joint",
                    # "right_shoulder_yaw_joint",
                    # "right_elbow_joint",
                    # "right_wrist_roll_joint",
                    # "right_wrist_pitch_joint",
                    # "right_wrist_yaw_joint",
                     ], 
        scale=0.25, # deploy code와 같은 순서로 scale을 설정
        use_default_offset=False, # deploy code와 같이 offset 제거
        preserve_order=True, # 위의 명시된 순서를 고정시키는 옵션
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # 기본 각속도 관측
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2),scale=0.25)
        # 중력 관측
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        # 관절 위치 관측 (원본 코드는 joint_pos_rel을 사용하지만, 이 코드는 joint_pos를 사용)
        joint_pos = ObsTerm(func=mdp.joint_pos, 
                            params={"asset_cfg": SceneEntityCfg("robot",
                                    joint_names=[
                                                'left_hip_pitch_joint', 
                                                'left_hip_roll_joint', 
                                                'left_hip_yaw_joint', 
                                                'left_knee_joint', 
                                                'left_ankle_pitch_joint', 
                                                'left_ankle_roll_joint', 
                                                'right_hip_pitch_joint', 
                                                'right_hip_roll_joint', 
                                                'right_hip_yaw_joint', 
                                                'right_knee_joint', 
                                                'right_ankle_pitch_joint', 
                                                'right_ankle_roll_joint',
                                                # G1_29_no_hand
                                                "waist_yaw_joint",
                                                "waist_roll_joint",
                                                "waist_pitch_joint",
                                                "left_shoulder_pitch_joint",
                                                "left_shoulder_roll_joint",
                                                "left_shoulder_yaw_joint",
                                                "left_elbow_joint",
                                                "left_wrist_roll_joint",
                                                "left_wrist_pitch_joint",
                                                "left_wrist_yaw_joint",
                                                "right_shoulder_pitch_joint",
                                                "right_shoulder_roll_joint",
                                                "right_shoulder_yaw_joint",
                                                "right_elbow_joint",
                                                "right_wrist_roll_joint",
                                                "right_wrist_pitch_joint",
                                                "right_wrist_yaw_joint",
                                                ],
                                    preserve_order=True,
                                    )},
                            noise=Unoise(n_min=-0.01, n_max=0.01),scale=1.0)
        # 관절 속도 관측 (원본 코드는 joint_vel_rel을 사용하지만, 이 코드는 joint_vel을 사용)
        joint_vel = ObsTerm(func=mdp.joint_vel, 
                            params={"asset_cfg": SceneEntityCfg("robot",
                                    joint_names=[
                                                'left_hip_pitch_joint', 
                                                'left_hip_roll_joint', 
                                                'left_hip_yaw_joint', 
                                                'left_knee_joint', 
                                                'left_ankle_pitch_joint', 
                                                'left_ankle_roll_joint', 
                                                'right_hip_pitch_joint', 
                                                'right_hip_roll_joint', 
                                                'right_hip_yaw_joint', 
                                                'right_knee_joint', 
                                                'right_ankle_pitch_joint', 
                                                'right_ankle_roll_joint',
                                                # G1_29_no_hand
                                                "waist_yaw_joint",
                                                "waist_roll_joint",
                                                "waist_pitch_joint",
                                                "left_shoulder_pitch_joint",
                                                "left_shoulder_roll_joint",
                                                "left_shoulder_yaw_joint",
                                                "left_elbow_joint",
                                                "left_wrist_roll_joint",
                                                "left_wrist_pitch_joint",
                                                "left_wrist_yaw_joint",
                                                "right_shoulder_pitch_joint",
                                                "right_shoulder_roll_joint",
                                                "right_shoulder_yaw_joint",
                                                "right_elbow_joint",
                                                "right_wrist_roll_joint",
                                                "right_wrist_pitch_joint",
                                                "right_wrist_yaw_joint",
                                                ],
                                    preserve_order=True,
                                    )},
                            noise=Unoise(n_min=-1.5, n_max=1.5),scale=0.05)
        actions = ObsTerm(func=mdp.last_action) # 마지막 행동 관측
        #########################################################################################
        # 속도 명령 관측
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"},scale=(2.0,2.0,0.25))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        # observation terms (order preserved)
        # policy group과 critic group이 나뉘어진 것은 현실에서 쉽게 얻을 수 없고, 
        # 시뮬레이션에서 쉽게 얻을 수 있는 데이터를 사용해 학습 퀄리티를 높이기 위함임
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1),scale=2.0) # 변화점: 기본 직선 속도 관측 추가
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2),scale=0.25)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(func=mdp.joint_pos, 
                            params={"asset_cfg": SceneEntityCfg("robot",
                                    joint_names=[
                                                'left_hip_pitch_joint', 
                                                'left_hip_roll_joint', 
                                                'left_hip_yaw_joint', 
                                                'left_knee_joint', 
                                                'left_ankle_pitch_joint', 
                                                'left_ankle_roll_joint', 
                                                'right_hip_pitch_joint', 
                                                'right_hip_roll_joint', 
                                                'right_hip_yaw_joint', 
                                                'right_knee_joint', 
                                                'right_ankle_pitch_joint', 
                                                'right_ankle_roll_joint',
                                                # G1_29_no_hand
                                                "waist_yaw_joint",
                                                "waist_roll_joint",
                                                "waist_pitch_joint",
                                                "left_shoulder_pitch_joint",
                                                "left_shoulder_roll_joint",
                                                "left_shoulder_yaw_joint",
                                                "left_elbow_joint",
                                                "left_wrist_roll_joint",
                                                "left_wrist_pitch_joint",
                                                "left_wrist_yaw_joint",
                                                "right_shoulder_pitch_joint",
                                                "right_shoulder_roll_joint",
                                                "right_shoulder_yaw_joint",
                                                "right_elbow_joint",
                                                "right_wrist_roll_joint",
                                                "right_wrist_pitch_joint",
                                                "right_wrist_yaw_joint",
                                                ],
                                    preserve_order=True,
                                    )},
                            noise=Unoise(n_min=-0.01, n_max=0.01),scale=1.0)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, 
                            params={"asset_cfg": SceneEntityCfg("robot",
                                    joint_names=[
                                                'left_hip_pitch_joint', 
                                                'left_hip_roll_joint', 
                                                'left_hip_yaw_joint', 
                                                'left_knee_joint', 
                                                'left_ankle_pitch_joint', 
                                                'left_ankle_roll_joint', 
                                                'right_hip_pitch_joint', 
                                                'right_hip_roll_joint', 
                                                'right_hip_yaw_joint', 
                                                'right_knee_joint', 
                                                'right_ankle_pitch_joint', 
                                                'right_ankle_roll_joint',
                                                # G1_29_no_hand
                                                "waist_yaw_joint",
                                                "waist_roll_joint",
                                                "waist_pitch_joint",
                                                "left_shoulder_pitch_joint",
                                                "left_shoulder_roll_joint",
                                                "left_shoulder_yaw_joint",
                                                "left_elbow_joint",
                                                "left_wrist_roll_joint",
                                                "left_wrist_pitch_joint",
                                                "left_wrist_yaw_joint",
                                                "right_shoulder_pitch_joint",
                                                "right_shoulder_roll_joint",
                                                "right_shoulder_yaw_joint",
                                                "right_elbow_joint",
                                                "right_wrist_roll_joint",
                                                "right_wrist_pitch_joint",
                                                "right_wrist_yaw_joint",
                                                ],
                                    preserve_order=True,
                                    )},
                            noise=Unoise(n_min=-1.5, n_max=1.5),scale=0.05)
        actions = ObsTerm(func=mdp.last_action)
        #########################################################################################
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"},scale=(2.0,2.0,0.25))

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# 대부분의 randomization을 담당하는 부분
# notion의 "[정리] 휴머노이드 걷기 학습" 참조
@configclass
class EventCfg:
    """Configuration for events."""

    # interval
    push_robot = EventTerm( # 랜덤한 시간에 랜덤 속도로 로봇을 미는 이벤트
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5)}},
    )

    # startup 시 상체 arm joint target을 unitree.py default(asset default_joint_pos)로 설정
    set_arm_joint_targets_startup = EventTerm(
        func=mdp.set_arm_joint_targets_random,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "left_shoulder_pitch_joint",
                    "left_shoulder_roll_joint",
                    "left_shoulder_yaw_joint",
                    "left_elbow_joint",
                    "left_wrist_roll_joint",
                    "left_wrist_pitch_joint",
                    "left_wrist_yaw_joint",
                    "right_shoulder_pitch_joint",
                    "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint",
                    "right_elbow_joint",
                    "right_wrist_roll_joint",
                    "right_wrist_pitch_joint",
                    "right_wrist_yaw_joint",
                ],
                preserve_order=True,
            ),
            "position_range": (0.0, 0.0),  # unitree.py default 그대로 (offset 0)
        },
    )

    # 학습 중 일정 간격으로 상체 arm joint target을 무작위로 설정 (하체·허리만 제어 시)
    set_arm_joint_targets_interval = EventTerm(
        func=mdp.set_arm_joint_targets_random,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "left_shoulder_pitch_joint",
                    "left_shoulder_roll_joint",
                    "left_shoulder_yaw_joint",
                    "left_elbow_joint",
                    "left_wrist_roll_joint",
                    "left_wrist_pitch_joint",
                    "left_wrist_yaw_joint",
                    "right_shoulder_pitch_joint",
                    "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint",
                    "right_elbow_joint",
                    "right_wrist_roll_joint",
                    "right_wrist_pitch_joint",
                    "right_wrist_yaw_joint",
                ],
                preserve_order=True,
            ),
            "position_range": (0.0, 0.0),  # 커리큘럼으로 0 → ±0.6 rad 점진 증가
        },
    )

    reset_base = EventTerm( # 로봇의 초기 상태를 초기화하는 이벤트
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    # startup
    randomize_friction = EventTerm( # epoch마다 로봇의 링크에 랜덤 마찰력을 부여하는 이벤트
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=['pelvis', 
                                                             'left_hip_pitch_link', 
                                                             'right_hip_pitch_link', 
                                                             'waist_yaw_link', 
                                                             'left_hip_roll_link', 
                                                             'right_hip_roll_link', 
                                                             'waist_roll_link', 
                                                             'left_hip_yaw_link', 
                                                             'right_hip_yaw_link', 
                                                             'torso_link', 
                                                             'left_knee_link', 
                                                             'right_knee_link', 
                                                             'left_shoulder_pitch_link', 
                                                             'right_shoulder_pitch_link', 
                                                             'left_ankle_pitch_link', 
                                                             'right_ankle_pitch_link', 
                                                             'left_shoulder_roll_link', 
                                                             'right_shoulder_roll_link', 
                                                             'left_ankle_roll_link', 
                                                             'right_ankle_roll_link', 
                                                             'left_shoulder_yaw_link', 
                                                             'right_shoulder_yaw_link', 
                                                             'left_elbow_link', 
                                                             'right_elbow_link', 
                                                             'left_wrist_roll_link', 
                                                             'right_wrist_roll_link', 
                                                             'left_wrist_pitch_link', 
                                                             'right_wrist_pitch_link', 
                                                             'left_wrist_yaw_link', 
                                                             'right_wrist_yaw_link']),
            "static_friction_range": (0.5, 1.3),
            "dynamic_friction_range": (0.5, 1.3),
            "restitution_range": (0.0, 0.4),
            "num_buckets": 256,
            "make_consistent": True
        },
    )

    randomize_joint_param = EventTerm( # 로봇이 재생성될때마다 관절 마찰력, 점성, 관절 질량을 랜덤하게 변경하는 이벤트
        func=mdp.randomize_joint_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.01, 1.0),
            "viscous_friction_distribution_params": (0.3, 1.5),
            "armature_distribution_params": (0.008,0.06),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    randomize_link_mass = EventTerm( # epoch마다 로봇의 링크 질량을 랜덤하게 변경하는 이벤트
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=['left_hip_pitch_link', 
                                                             'right_hip_pitch_link', 
                                                             'waist_yaw_link', 
                                                             'left_hip_roll_link', 
                                                             'right_hip_roll_link', 
                                                             'waist_roll_link', 
                                                             'left_hip_yaw_link', 
                                                             'right_hip_yaw_link', 
                                                             'torso_link', 
                                                             'left_knee_link', 
                                                             'right_knee_link', 
                                                             'left_shoulder_pitch_link', 
                                                             'right_shoulder_pitch_link', 
                                                             'left_ankle_pitch_link', 
                                                             'right_ankle_pitch_link', 
                                                             'left_shoulder_roll_link', 
                                                             'right_shoulder_roll_link', 
                                                             'left_ankle_roll_link', 
                                                             'right_ankle_roll_link', 
                                                             'left_shoulder_yaw_link', 
                                                             'right_shoulder_yaw_link', 
                                                             'left_elbow_link', 
                                                             'right_elbow_link', 
                                                             'left_wrist_roll_link', 
                                                             'right_wrist_roll_link', 
                                                             'left_wrist_pitch_link', 
                                                             'right_wrist_pitch_link',]),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    randomize_base_mass = EventTerm( # epoch마다 로봇의 기반 질량을 랜덤하게 변경하는 이벤트
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "mass_distribution_params": (-2., 5.),
            "operation": "add",
        },
    )

    randomize_base_com = EventTerm( # epoch마다 로봇의 기반 질량 중심을 랜덤하게 변경하는 이벤트
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "com_range": {"x": (-0.12, 0.12), "y": (-0.12, 0.12), "z": (-0.08, 0.08)},
        },
    )

    randomize_pd_gains = EventTerm( # 로봇이 재생성될때마다 PD 게인을 랜덤하게 변경하는 이벤트
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


    randomize_motor_zero_offset = EventTerm( # 로봇이 재생성될때마다 초기 모터 오프셋을 랜덤하게 변경하는 이벤트
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.05, 0.05),
            "velocity_range": (-0.0, 0.0),
        },
    )




@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm( # xy축 속도 추적 보상
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm( # 각속도 추적 보상
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2) # z축 속도 제곱 패널티
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05) # xy축 각속도 제곱 패널티
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-6) # 관절 토크 제곱 패널티
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-6) # 관절 가속도 제곱 패널티
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.001) # 행동 속도 제곱 패널티
    undesired_contacts = RewTerm( # 원하지 않는 접촉 패널티
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    )
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0) # 평형 어긋남에 대한 패널티
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)


@configclass
class TerminationsCfg:  # 로봇 종료 및 재생성 조건 설정
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True) # 시간 초과
    robot_dropping = DoneTerm( # 로봇이 지정 높이 아래로 떨어지면 종료
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.55, "asset_cfg": SceneEntityCfg("robot")}
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel) # 지형 난이도 조절


##
# Environment configuration
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5) # 로봇 갯수와 간격 설정 
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4 # 물리 업데이트 주기 설정 -> 200/4 = 50Hz
        self.episode_length_s = 20.0 # 에피소드 길이 설정
        # simulation settings
        self.sim.dt = 0.005 # 시뮬레이션 업데이트 주기 설정 -> 0.005s = 200Hz
        self.sim.render_interval = self.decimation # 렌더링 주기 설정 -> 50Hz
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
