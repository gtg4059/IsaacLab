# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, ISAAC_NUCLEUS_DIR

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_newton.sensors import ContactSensorCfg as NewtonContactSensorCfg
from isaaclab_ovphysx.sensors import ContactSensorCfg as OvPhysXContactSensorCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sensors import ContactSensorCfg as PhysXContactSensorCfg

from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.managers import CurriculumTermCfg as CurrTerm

from isaaclab_assets.robots.unitree import UNITREE_G1_29DOF_CFG


from . import mdp
from isaaclab_tasks.utils import PresetCfg, preset
from .mdp import G1_ACTUATED_JOINT_NAMES

##
# Pre-defined configs
##


def _robot_joint_cfg() -> SceneEntityCfg:
    """관절 관측에서 재사용하는 로봇 엔티티 설정을 생성한다."""
    return SceneEntityCfg("robot", joint_names=G1_ACTUATED_JOINT_NAMES, preserve_order=True)


def _contact_forces_cfg() -> SceneEntityCfg:
    """접촉력 관측에서 재사용하는 contact sensor body 선택 설정."""
    return SceneEntityCfg("contact_forces")

##
# Scene definition
##

@configclass
class RoughPhysicsCfg(PresetCfg):
    """Shared physics preset for all rough-terrain locomotion envs."""

    default = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=200,
            nconmax=100,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
        num_substeps=1,
        debug_mode=False,
        # 1 cm shape margin is the single most important Newton setting for rough
        # terrain — without it, non-Anymal-D robots fail to learn stable contact
        # on triangle-mesh terrain. See isaaclab_newton 0.5.22 changelog.
        default_shape_cfg=NewtonShapeCfg(margin=0.01),
    )
    physx = default


##
# Scene definition
##


@configclass
class VelocityEnvContactSensorCfg(PresetCfg):
    default = PhysXContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    newton_mjwarp = NewtonContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    physx = default
    ovphysx = OvPhysXContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

@configclass
class HeightScanEncoderCfg:
    """``height_scanner`` raw scan을 MLP latent로 압축하는 encoder 설정.

    ``RadarEnvCfg.height_scan_encoder`` 와 ``height_scan_encoded`` observation term 이
    동일한 값을 참조한다 (scene cfg 가 아님 — ``InteractiveScene`` asset 으로 등록되지 않음).
    """

    latent_dim: int = 64
    hidden_dims: tuple[int, int] = (256, 128)
    clip: tuple[float, float] = (-1.0, 1.0)
    offset: float = 0.5

@configclass
class RadarSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
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
    robot: ArticulationCfg = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = VelocityEnvContactSensorCfg()
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
    )


##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """정책(Actor) 입력 관측."""

        # h_p_t: root height
        base_pos_z = ObsTerm(func=mdp.base_pos_z, scale=0.8)
        # p_p_t: local body position (env-frame root position)
        root_pos_w = ObsTerm(
            func=mdp.root_pos_w,
            scale=(0.125, 0.125, 0.8),
        )
        # R_p_t: local body rotation (root quaternion, wxyz)
        root_quat_w = ObsTerm(
            func=mdp.root_quat_w,
            scale=1.0,
            params={"make_quat_unique": True},
        )
        # v_p_t: local body linear velocity
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0)
        # ω_p_t: local body angular velocity
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        # c_p_t: body contact force (base frame)
        body_contact_forces = ObsTerm(
            func=mdp.body_contact_forces_b,
            params={"sensor_cfg": _contact_forces_cfg()},
            scale=0.00125,
        )
        # q_t: joint positions
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": _robot_joint_cfg()},
            scale=1.0,
        )
        # q̇_t: joint velocities
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": _robot_joint_cfg()},
            scale=0.05,
        )
        # a_{t-1}: last action (default joint bias 포함)
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            scale=(2.0, 2.0, 0.25),
        )
        # e_t: robot-centric height map (encoder latent)
        height_scan = ObsTerm(
            func=mdp.height_scan_encoded,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "clip": HeightScanEncoderCfg().clip,
                "offset": HeightScanEncoderCfg().offset,
                "latent_dim": HeightScanEncoderCfg().latent_dim,
                "hidden_dims": HeightScanEncoderCfg().hidden_dims,
            },
        )
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """정책(Actor) 입력 관측."""

        # h_p_t: root height
        base_pos_z = ObsTerm(func=mdp.base_pos_z, scale=0.8)
        # p_p_t: local body position (env-frame root position)
        root_pos_w = ObsTerm(
            func=mdp.root_pos_w,
            scale=(0.125, 0.125, 0.8),
        )
        # R_p_t: local body rotation (root quaternion, wxyz)
        root_quat_w = ObsTerm(
            func=mdp.root_quat_w,
            scale=1.0,
            params={"make_quat_unique": True},
        )
        # v_p_t: local body linear velocity
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0)
        # ω_p_t: local body angular velocity
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        # c_p_t: body contact force (base frame)
        body_contact_forces = ObsTerm(
            func=mdp.body_contact_forces_b,
            params={"sensor_cfg": _contact_forces_cfg()},
            scale=0.00125,
        )
        # q_t: joint positions
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": _robot_joint_cfg()},
            scale=1.0,
        )
        # q̇_t: joint velocities
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": _robot_joint_cfg()},
            scale=0.05,
        )
        # a_{t-1}: last action (default joint bias 포함)
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            scale=(2.0, 2.0, 0.25),
        )
        # e_t: robot-centric height map (encoder latent)
        height_scan = ObsTerm(
            func=mdp.height_scan_encoded,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "clip": HeightScanEncoderCfg().clip,
                "offset": HeightScanEncoderCfg().offset,
                "latent_dim": HeightScanEncoderCfg().latent_dim,
                "hidden_dims": HeightScanEncoderCfg().hidden_dims,
            },
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
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

    base_com = preset(
        default=EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
            },
        ),
        newton_mjwarp=None,
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    # -- penalties
    # base_height = RewTerm(
    #     func=mdp.base_height_log,
    #     weight=-100.0,
    #     params={
    #         "target_height": 0.75,
    #         "sensor_cfg": SceneEntityCfg("height_scanner"),
    #     },
    # )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time,
    #     weight=0.125,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    #         "command_name": "base_velocity",
    #         "threshold": 0.5,
    #     },
    # )
    feet_gait_walk = RewTerm(
        func=mdp.feet_gait_walk_adaptive,
        weight=0.2,  # V2 변경 (×1.5)
        params={
            "offset": [0.0, 0.5],
            "threshold": 0.45,
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "v_gate": 0.1,
            "w_gate": 0.07,
            "mode_scales": {"curve": 0.4},
            "t_max": 1.0,
            "t_min": 0.45,
            "v_floor": 0.2,
            "exponent": 1.0/3.0,
        },
    )
    foot_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=0.25,
        params={
            "std": 0.05,
            "target_height": 0.2,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    # # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    # ".*_shoulder_roll_joint",
                    # ".*_elbow_joint",
                    ".*_knee_joint",
                    ".*_ankle_pitch_joint",
                    ".*_ankle_roll_joint",
                ],
            )
        },
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    # ".*_shoulder_pitch_joint",
                    # ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    # ".*_elbow_joint",
                    ".*_wrist_roll_joint",
                    ".*_wrist_pitch_joint",
                    ".*_wrist_yaw_joint",
                ],
            )
        },
    )
    joint_deviation_shoulders = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    # ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    # ".*_wrist_roll_joint",
                    # ".*_wrist_pitch_joint",
                    # ".*_wrist_yaw_joint",
                ],
            )
        },
    )

    # G1_29_no_hand
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist_roll_joint",
                    "waist_pitch_joint",
                    "waist_yaw_joint",
                ],
            )
        },
    )
    contact_forces = RewTerm(
        func=mdp.contact_forces_minimize,
        weight=-1.0e-7,
        params={
            "threshold": 0.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"), "threshold": 1.0},
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel_fix)

##
# Environment configuration
##


@configclass
class RadarEnvCfg(ManagerBasedRLEnvCfg):
    height_scan_encoder: HeightScanEncoderCfg = HeightScanEncoderCfg()
    # Simulation settings — shared physics preset (PhysX + MJWarp) for all rough-terrain envs
    sim: SimulationCfg = SimulationCfg(physics=RoughPhysicsCfg())
    # Scene settings
    scene: RadarSceneCfg = RadarSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.scene.robot.spawn.activate_contact_sensors = True
        # 높이 스캐너 기준 프레임을 torso_link로 맞춤
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        # 학습 시작 시 초기 지형 레벨을 제한해 안정적 초기 수렴 유도
        self.scene.terrain.max_init_terrain_level = 0

        # general settings
        self.decimation = 4  # 물리 업데이트 주기 설정 -> 200/4 = 50Hz
        self.episode_length_s = 20.0  # 에피소드 길이 설정

        # simulation settings
        self.sim.dt = 0.005  # 시뮬레이션 업데이트 주기 설정 -> 0.005s = 200Hz
        self.sim.render_interval = self.decimation  # 렌더링 주기 설정 -> 50Hz
        self.sim.physics_material = self.scene.terrain.physics_material

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.decimation * self.sim.dt

        # terrain 커리큘럼 항목이 있으면 지형 생성기 커리큘럼을 켠다.
        # (파생 cfg에서 curriculum 유무에 따라 자동으로 on/off)
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
        # Scene
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.80)
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

        enc = self.height_scan_encoder
        height_scan_params = {
            "clip": enc.clip,
            "offset": enc.offset,
            "latent_dim": enc.latent_dim,
            "hidden_dims": enc.hidden_dims,
        }
        self.observations.policy.height_scan.params.update(height_scan_params)
        self.observations.critic.height_scan.params.update(height_scan_params)

        # self.events.push_robot = None
        self.events.push_robot.params["velocity_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}
        self.events.randomize_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]

        # self.rewards.base_height.params["target_height"] = 0.76
        # self.rewards.base_height.weight = 0.0
        # self.rewards.contact_forces.weight = 0.0
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -1.0
        # self.rewards.feet_air_time.weight = 0.1
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )
        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        )
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        self.terminations.base_contact = None

        self.rewards.action_rate_l2.weight = -0.005      # Isaac rough 초기값보다 완화

        # flat terrain generator
        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None
        # self.curriculum = None


@configclass
class RadarEnvCfg_PLAY(RadarEnvCfg):
    """경량 디버그/검증용 Play 설정 (env 수·지형 축소)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None