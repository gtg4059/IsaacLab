# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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
from isaaclab.devices import DevicesCfg

import torch
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.devices import Se2Keyboard
from isaaclab.devices.keyboard.se2_keyboard import Se2KeyboardCfg

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG, RANDOM_ROUGH_TERRAINS_CFG  # isort: skip

##
# Scene definition
##
from isaaclab.devices.gamepad import Se2bGamepad, Se2bGamepadCfg

def gamepad_commands(env: ManagerBasedRLEnv) -> torch.Tensor:
    """게임패드(조이스틱) 입력을 받아옴."""
    if not hasattr(env, "gamepad"):
        env.gamepad = Se2bGamepad(Se2bGamepadCfg(
            v_x_sensitivity=1.0, v_y_sensitivity=1.0, omega_z_sensitivity=0.25, dead_zone=0.1
        ))
        env.gamepad.reset()
    command = env.gamepad.advance()
    return torch.tensor([command[0],-command[1],-command[2]], device=env.device, dtype=torch.float32).unsqueeze(0).repeat(env.num_envs, 1)

def keyboard_commands(env: ManagerBasedRLEnv) -> torch.Tensor:
    """키보드로부터 명령을 받아옵니다."""
    if not hasattr(env, "keyboard"):
        env.keyboard = Se2Keyboard(Se2KeyboardCfg(
            v_x_sensitivity=1.0, v_y_sensitivity=1.0, omega_z_sensitivity=1.0
            )
        )
        env.keyboard.reset()
    
    command = env.keyboard.advance()
    return torch.tensor(command, device=env.device, dtype=torch.float32).unsqueeze(0).repeat(env.num_envs, 1)

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=RANDOM_ROUGH_TERRAINS_CFG,
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

    
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


    # # #Set Cube as object
    # object = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Object",
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=[0.32, 10.0, 0.82],# 0.37, 0, 0.82
    #         rot=[1.0, 0.0 ,0.0, 0.0]),
    #     spawn=sim_utils.UsdFileCfg(
    #         # white wing-box
    #         usd_path="./source/isaaclab_assets/data/Assets/Wingbox.usd",
    #         scale=(11.5,7.78,5.357),# 380,250,150
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.8),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             # kinematic_enabled=True,
    #             solver_position_iteration_count=8, #16,
    #             solver_velocity_iteration_count=1,
    #             max_angular_velocity=1000.0,
    #             max_linear_velocity=1000.0,
    #             max_depenetration_velocity=5.0,
    #             disable_gravity=False,
    #         ),
    #     ),
    # )
    

##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    base_velocity = mdp.UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(5.0, 10.0),
            rel_standing_envs=0.2,
            rel_heading_envs=0.8,
            heading_command=True,
            heading_control_stiffness=2.0,
            debug_vis=True,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 2.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
            ),
        )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", 
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
        scale=0.25, 
        use_default_offset=False,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class RunCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
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
        #####################################################################################
        velocity_commands = ObsTerm(func=keyboard_commands,scale=(2.0,2.0,0.25))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    Run: RunCfg = RunCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # # interval
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(5.0, 10.0),
    #     params={"velocity_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5)}},
    # )

    reset_base = EventTerm(
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
    randomize_friction = EventTerm(
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
            "static_friction_range": (0.2, 1.3),
            "dynamic_friction_range": (0.2, 1.3),
            "restitution_range": (0.0, 0.4),
            "num_buckets": 256,
            "make_consistent": True
        },
    )

    randomize_joint_param = EventTerm(
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

    randomize_link_mass = EventTerm(
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

    randomize_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "mass_distribution_params": (-2., 5.),
            "operation": "add",
        },
    )

    randomize_base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "com_range": {"x": (-0.12, 0.12), "y": (-0.12, 0.12), "z": (-0.08, 0.08)},
        },
    )

    randomize_pd_gains = EventTerm(
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


    randomize_motor_zero_offset = EventTerm(
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
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    lin_vel_xy_l2 = RewTerm(func=mdp.lin_vel_xy_l2, weight=-100.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    )
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)
    alive = RewTerm(func=mdp.is_alive, weight=20.0)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    robot_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.60, "asset_cfg": SceneEntityCfg("robot")}
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=1, env_spacing=1)
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
        self.decimation = 4
        self.episode_length_s = 20000.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.bounce_threshold_velocity = 0.1
        # eye: tuple[float, float, float] = (7.5, 7.5, 7.5),
        # lookat: tuple[float, float, float] = (0, 0, 0),
        self.viewer.eye = (2.0, 2.0, 3.0)#(-2.0, 0, 5.0)
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
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