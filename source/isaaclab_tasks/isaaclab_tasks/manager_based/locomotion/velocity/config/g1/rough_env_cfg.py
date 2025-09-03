# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_tasks.manager_based.manipulation.reach.mdp as manipulation_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

##
# Pre-defined configs
##
from isaaclab_assets import G1_DEX_FIX, G1_DEX_FIX_D  # isort: skip


@configclass
class G1Rewards(RewardsCfg):
    """Reward terms for the MDP."""

    # base_position_l2 = RewTerm(func=mdp.base_position_l2, weight=-100.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    base_height_l2 = RewTerm(func=mdp.base_height_l2, weight=-50.0, params={
            "target_height": 0.78, 
        }
    )
    # # pickup reward
    # reaching_object= RewTerm(
    #     func=mdp.object_ee_distance, 
    #     params={
    #         "std": 0.12,
    #         "asset_cfg":SceneEntityCfg("robot", body_names=[".*_middle_proximal"]),
    #         # "asset_cfg":SceneEntityCfg("robot", body_names=[".*_wrist_yaw_link"]),
    #     }, 
    #     weight=0.1
    # )
    
    # flat_orientation_obj = RewTerm(func=mdp.flat_orientation_obj, weight=0.7)
 
    # object_contact = RewTerm(
    #     func=mdp.object_is_contacted, 
    #     weight=0.05,
    #     params={"threshold": 0.4,"sensor_cfg": SceneEntityCfg("contact_forces", body_names=
    #                                                           [
    #                                                               "left_wrist_yaw_link",
    #                                                               "right_wrist_yaw_link",
    #                                                             #   "left_wrist_pitch_link",
    #                                                             #   "right_wrist_pitch_link",
    #                                                             #   "L_thumb_proximal",
    #                                                             #   "R_thumb_proximal",
    #                                                             #   ".*_thumb_intermediate",
    #                                                             #   ".*_index_intermediate",
    #                                                             #   ".*_middle_intermediate",
    #                                                             #   ".*_pinky_intermediate",
    #                                                             #   ".*_ring_intermediate",
    #                                                             #   "left_ankle_roll_link",
    #                                                             #   "right_ankle_roll_link"
    #                                                               ],preserve_order=True,
    #         )
    #     }, 
    # )

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    foot_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=0.5,
        params={
            "std": 0.05,
            "target_height": 0.08,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
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

    # same motion
    motion_equality_shoulder1 = RewTerm(
        func=mdp.motion_equality_cons,
        weight=-0.1,
        params={
            "std": 0.2,"asset_cfg": SceneEntityCfg("robot", joint_names=".*_shoulder_yaw_joint"),
        },
    )

    motion_equality_shoulder2 = RewTerm(
        func=mdp.motion_equality_pros,
        weight=-0.1,
        params={
            "std": 0.2,"asset_cfg": SceneEntityCfg("robot", joint_names=".*_shoulder_pitch_joint"),
        },
    )

    motion_equality_elbow = RewTerm(
        func=mdp.motion_equality_pros,
        weight=-0.1,
        params={
            "std": 0.2,"asset_cfg": SceneEntityCfg("robot", joint_names=".*_elbow_joint"),
        },
    )

    motion_equality_wrist1 = RewTerm(
        func=mdp.motion_equality_cons,
        weight=-0.1,
        params={
            "std": 0.2,"asset_cfg": SceneEntityCfg("robot", joint_names=".*_wrist_roll_joint"),
        },
    )

    motion_equality_wrist2 = RewTerm(
        func=mdp.motion_equality_pros,
        weight=-0.1,
        params={
            "std": 0.2,"asset_cfg": SceneEntityCfg("robot", joint_names=".*_wrist_pitch_joint"),
        },
    )

    motion_equality_wrist3 = RewTerm(
        func=mdp.motion_equality_cons,
        weight=-0.1,
        params={
            "std": 0.2,"asset_cfg": SceneEntityCfg("robot", joint_names=".*_wrist_yaw_joint"),
        },
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.75,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )

    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_knee_joint",".*_ankle_roll_joint"])},
    )

    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_roll_joint",
                    # ".*_shoulder_pitch_joint",
                    # ".*_shoulder_yaw_joint",
                    # ".*_elbow_joint",
                    # ".*_wrist_yaw_joint",
                    # ".*_wrist_pitch_joint",
                    ".*_wrist_roll_joint",
                ],
            )
        },
    )

    joint_deviation_arms2 = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    # ".*_shoulder_roll_joint",
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_yaw_joint",
                    ".*_wrist_pitch_joint",
                    # ".*_wrist_roll_joint",
                ],
            )
        },
    )
   
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
            "waist_roll_joint",
            "waist_pitch_joint",
            "waist_yaw_joint",
        ])},
    )

    # joint_deviation_elbow = RewTerm(
    #     func=mdp.joint_pos,
    #     weight=-2.0,
    #     params={
    #         "target": -1.0,
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_elbow_joint"])
    #     },
    # )

    # joint_deviation_wrist_pitch = RewTerm(
    #     func=mdp.joint_pos,
    #     weight=-2.0,
    #     params={
    #         "target": 1.5,
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wrist_pitch_joint"])
    #     },
    # )
    
    # set_robot_joints_targets = RewTerm(
    #     func=mdp.reset_joints_targets,
    #     weight=-0.00001,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot",
    #             joint_names=[
    #                 # 'L_thumb_proximal_yaw_joint',
    #                 #          'R_thumb_proximal_yaw_joint',
    #                 #         'L_thumb_proximal_pitch_joint',
    #                 #         'R_thumb_proximal_pitch_joint',
    #                         '.*_proximal_joint',
    #                 #         '.*_thumb_intermediate_joint',
    #                 #         '.*_thumb_distal_joint',
    #                         ],
    #             preserve_order=True,
    #         )
    #     },
    # )

    # set_robot_joints_forces = RewTerm(
    #     func=mdp.reset_joints_forces,
    #     weight=-0.00001,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot",
    #             joint_names=['.*_proximal_joint'],
    #             preserve_order=True,
    #         )
    #     },
    # )

    # delete_table = RewTerm(
    #     func=mdp.delete_table,
    #     weight=-0.00001,
    # )
    left_ee_pos_tracking = RewTerm(
        func=manipulation_mdp.position_command_error,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="L_middle_proximal"),
            "command_name": "left_ee_pose",
        },
    )
    left_ee_pos_tracking_fine_grained = RewTerm(
        func=manipulation_mdp.position_command_error_tanh,
        weight=0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="L_middle_proximal"),
            "std": 0.05,
            "command_name": "left_ee_pose",
        },
    )
    left_end_effector_orientation_tracking = RewTerm(
        func=manipulation_mdp.orientation_command_error,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="L_middle_proximal"),
            "command_name": "left_ee_pose",
        },
    )
    right_ee_pos_tracking = RewTerm(
        func=manipulation_mdp.position_command_error,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="R_middle_proximal"),
            "command_name": "right_ee_pose",
        },
    )
    right_ee_pos_tracking_fine_grained = RewTerm(
        func=manipulation_mdp.position_command_error_tanh,
        weight=0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="R_middle_proximal"),
            "std": 0.05,
            "command_name": "right_ee_pose",
        },
    )
    right_end_effector_orientation_tracking = RewTerm(
        func=manipulation_mdp.orientation_command_error,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="R_middle_proximal"),
            "command_name": "right_ee_pose",
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

        # Randomization
        # self.events.push_robot = None
        # self.events.add_base_mass = None
        # self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.00, 0.00), "y": (-0.00, 0.00), "yaw": (-0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        # self.events.base_com = None

        # Rewards
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.action_rate_l2.weight = -0.005
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

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
