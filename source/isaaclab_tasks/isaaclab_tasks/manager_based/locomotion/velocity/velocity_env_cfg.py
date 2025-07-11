# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
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
from isaaclab.managers import SceneEntityCfg, ActionTermCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns, TiledCameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaaclab.markers.config import FRAME_MARKER_CFG

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

    # Set Cube as object
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.36, 0, 0.70], rot=[1, 0, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path="./source/isaaclab_assets/data/Robots/DexCube.usd",
            scale=(3.10,4.14, 2.84),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.4),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # kinematic_enabled=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            # activate_contact_sensors=True,
        ),
    )

    # obj_init = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Object_init",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[0.36, 0, 0.70], rot=[1, 0, 0, 0]),
    #     spawn=sim_utils.DomeLightCfg(
    #         intensity=0.0,
    #         texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
    #     ),
    # )

    # add cube
    object_init: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object_init",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.36, 0, 0.70], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 0.01, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
    )

    # mount
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.36, 0, 0.60], rot=[1, 0, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path="./source/isaaclab_assets/data/Robots/DexCube.usd", scale=(4.0, 4.0, 1.00),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.6),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=True,
            ),
        ),
    )

    # camera = TiledCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/torso_link/d435_link/camera",
    #     update_period=0.5,
    #     height=480,
    #     width=640,
    #     debug_vis=True,
    #     data_types=["instance_id_segmentation_fast"],
    #     colorize_instance_id_segmentation=True,
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
    #     ),
    #     offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    # )

    # contact_table = ContactSensorCfg(
    #         prim_path="{ENV_REGEX_NS}/Object",
    #         debug_vis=False,
    #         update_period=0.0,
    #         filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"],
    #         track_air_time=True,
    #     )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(#0.84, 0.86
            pos_x=(0.0, 0.0), pos_y=(-0.0, 0.0), pos_z=(0.0, 0.0), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
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
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2),scale=0.25)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, 
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
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})# 3
        # object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame, params={"object_cfg": SceneEntityCfg("object_init")})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1),scale=2.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2),scale=0.25)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, 
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
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})# 3
        # object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame, params={"object_cfg": SceneEntityCfg("object_init")})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    Critic: CriticCfg = CriticCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "static_friction_range": (0.1, 1.25),
            "dynamic_friction_range": (0.1, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True
        },
    )

    physics_material_hand = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_wrist_yaw_link",
                                                            "R_.*",
                                                            "L_.*",
                                                            ]),
            "static_friction_range": (2.0, 2.0),
            "dynamic_friction_range": (2.0, 2.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    physics_material_obj = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (2.0, 2.0),
            "dynamic_friction_range": (2.0, 2.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "mass_distribution_params": (-1., 3.),
            "operation": "add",
        },
    )

    add_obj_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (-0.2, 0.2),
            "operation": "add",
        },
    )

    # set_robot_joints_targets = EventTerm(
    #     func=mdp.reset_joints_targets,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot",
    #             joint_names=[
    #                         'L_thumb_proximal_yaw_joint',
    #                          'R_thumb_proximal_yaw_joint',
    #                         'L_thumb_proximal_pitch_joint',
    #                         'R_thumb_proximal_pitch_joint',
    #                         '.*_proximal_joint',
    #                         '.*_thumb_intermediate_joint',
    #                         '.*_thumb_distal_joint',
    #                         ],
    #             preserve_order=True,
    #         )
    #     },
    # )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform_init,
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

    reset_table = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("table"),
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (-0.0, 0.0)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
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

    # interval
    # push_object = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(10.0, 15.0),
    #     params={
    #         "asset_cfg": SceneEntityCfg("object"),
    #         "velocity_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5)}
    #         },
    # )

    reset_box_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.03, 0.03), "yaw": (-0.0, 0.0)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    lin_vel_xy_l2 = RewTerm(func=mdp.lin_vel_xy_l2, weight=-10.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    )
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)
    is_alive = RewTerm(func=mdp.is_alive,weight=10.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces",body_names="torso_link"), "threshold": 40.0},
    # )
    base_contact2 = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces",body_names="pelvis"), "threshold": 40.0},
    )
    base_contact3 = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces",body_names=".*_hip_roll_link"), "threshold": 3.0},
    )
    # base_contact4 = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces",body_names=".*_wrist_pitch_link"), "threshold": 10.0},
    # )
    # base_contact5 = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces",body_names=".*_elbow_link"), "threshold": 10.0},
    # )
    # base_contact6 = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces",body_names=[
    #                                                               ".*_thumb_intermediate",
    #                                                               ".*_index_intermediate",
    #                                                               ".*_middle_intermediate",
    #                                                               ".*_pinky_intermediate",
    #                                                               ".*_ring_intermediate",
    #                                                               ]), "threshold": 20.0},
    # )
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.6, "asset_cfg": SceneEntityCfg("object")}
    )
    robot_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("robot")}
    )
    bad_position = DoneTerm(
        func=mdp.bad_position, params={"limit_dist": 0.5, "asset_cfg": SceneEntityCfg("robot")}
    )
    # set_robot_joints_targets = DoneTerm(
    #     func=mdp.reset_joints_targets,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot",
    #             joint_names=[
    #                         'L_thumb_proximal_yaw_joint',
    #                          'R_thumb_proximal_yaw_joint',
    #                         'L_thumb_proximal_pitch_joint',
    #                         'R_thumb_proximal_pitch_joint',
    #                         '.*_proximal_joint',
    #                         '.*_thumb_intermediate_joint',
    #                         '.*_thumb_distal_joint',
    #                         ],
    #             preserve_order=True,
    #         )
    #     },
    # )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    # table_delete = CurrTerm(func=mdp.delete_table)


##
# Environment configuration
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
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
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        # if self.scene.height_scanner is not None:
        #     self.scene.height_scanner.update_period = self.decimation * self.sim.dt
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
