# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

##
# Pre-defined configs
##
from isaaclab_assets import G1_DEX_FIX, G1_DEX # isort: skip


@configclass
class G1Rewards(RewardsCfg):
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    foot_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=5.0,
        params={
            "std": 0.05,
            "target_height": 0.08,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    contact_forces = RewTerm(
        func=mdp.contact_forces_minimize,
        weight=-0.00000005,
        params={
            "threshold": 250.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )

    # pickup reward
    reaching_object= RewTerm(
        func=mdp.object_ee_distance, 
        params={
            "std": 0.3,
            "asset_cfg":SceneEntityCfg("robot", body_names=[".*_middle_proximal"]),
        }, 
        weight=0.05
    )
 
    object_contact = RewTerm(
        func=mdp.object_is_contacted, 
        weight=0.1,
        params={"threshold": 0.4,"sensor_cfg": SceneEntityCfg("contact_forces", body_names=
                                                              ["left_wrist_yaw_link",
                                                                "right_wrist_yaw_link",
                                                                ],preserve_order=True,
            )
        }, 
    )
    flat_orientation_obj = RewTerm(func=mdp.flat_orientation_obj, weight=5.0)
    object_goal_distance = RewTerm(func=mdp.object_goal_distance, 
                               weight=1.0,
                               params={"std": 0.3,
                                       "minimal_height": 0.72,
        }, 
    )
    # same motion
    motion_equality_shoulder1 = RewTerm(
        func=mdp.motion_equality_cons,
        weight=0.5,
        params={
            "std": 0.2,"asset_cfg": SceneEntityCfg("robot", joint_names=".*_shoulder_yaw_joint"),
        },
    )

    motion_equality_shoulder2 = RewTerm(
        func=mdp.motion_equality_pros,
        weight=0.5,
        params={
            "std": 0.2,"asset_cfg": SceneEntityCfg("robot", joint_names=".*_shoulder_pitch_joint"),
        },
    )

    motion_equality_elbow = RewTerm(
        func=mdp.motion_equality_pros,
        weight=0.5,
        params={
            "std": 0.2,"asset_cfg": SceneEntityCfg("robot", joint_names=".*_elbow_joint"),
        },
    )

    motion_equality_wrist = RewTerm(
        func=mdp.motion_equality_pros,
        weight=0.5,
        params={
            "std": 0.2,"asset_cfg": SceneEntityCfg("robot", joint_names=".*_wrist_pitch_joint"),
        },
    )

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_roll_joint",
                    ".*_wrist_yaw_joint",
                    ".*_wrist_roll_joint",
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                ],
            )
        },
    )

    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )

    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
            ".*_ankle_pitch_joint", 
            ".*_ankle_roll_joint",
            ".*_knee_joint"
            ])},
    )

    # joint_deviation_arms2 = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.2,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 ".*_shoulder_pitch_joint",
    #                 ".*_shoulder_yaw_joint",
    #                 ".*_elbow_joint",
    #             ],
    #         )
    #     },
    # )
   
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
            "waist_roll_joint",
            "waist_pitch_joint",
            "waist_yaw_joint",
        ])},
    )
    set_robot_joints_forces = RewTerm(
        func=mdp.reset_joints_forces,
        weight=-0.00001,
        params={
            "asset_cfg": SceneEntityCfg("robot",
                joint_names=['.*_proximal_joint'],
                preserve_order=True,
            )
        },
    )



@configclass
class G1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: G1Rewards = G1Rewards()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = G1_DEX_FIX.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        # self.scene.camera.prim_path = "{ENV_REGEX_NS}/Robot/torso_link/d435_link"
        
        # self.scene.contact_table = ContactSensorCfg(
        #     prim_path="{ENV_REGEX_NS}/Object",
        #     debug_vis=False,
        #     update_period=0.0,
        #     filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"],
        # )
        # Randomization
        # self.events.push_robot = None
        # self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (-0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        
        # Rewards
        self.rewards.undesired_contacts = None
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )
        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        )



@configclass
class G1RoughEnvCfg_PLAY(G1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # self.commands.base_velocity.ranges.x = (-5.0, 5.0)
        # self.commands.base_velocity.ranges.y = (-5.0, 5.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
