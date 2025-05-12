# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, FrameTransformer
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat, subtract_frame_transforms, euler_xyz_from_quat, quat_mul, quat_error_magnitude
from isaaclab.assets import RigidObject, Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    double_stance = torch.sum(in_contact.int(), dim=1) == 2
    reward = torch.min(torch.where(double_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    return reward



def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # print("contacts:",contacts)
    # print("body_vel:",body_vel)
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_pos_w[:, :3])
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)

def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # euler_w = euler_xyz_from_quat(asset.data.root_quat_w)[2]
    # ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2]-euler_w)

    # lin_vel_error = torch.sum(
    #     torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_pos_w[:, :2]), dim=1
    # )
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)# torch.where(lin_vel_error<0.1,torch.exp(-ang_vel_error / std**2),0)

def track_pos_exp(
    env, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]

    COM_error = torch.sum(
        torch.square(asset.data.root_pos_w[:, :2]-env.scene.env_origins[:, :2]), dim=1
    )
    # print(asset.data.root_pos_w[:, :2])
    return torch.exp(-COM_error / std)# torch.where(lin_vel_error<0.1,torch.exp(-ang_vel_error / std**2),0)

def track_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return 100*torch.exp(-ang_vel_error / std**2)/(1+2*torch.exp(lin_vel_error))


"""
lift
"""
def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)

def object_is_contacted(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    double_stance = torch.sum(in_contact.int(), dim=1) >= 8
    # print(in_contact)
    reward = torch.min(torch.where(double_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # print("reward:",reward*double_stance)
    # print("torch.sum(in_contact.int(), dim=1):",torch.sum(in_contact.int(), dim=1))
    return reward#*torch.sum(in_contact.int(), dim=1)

def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    asset = env.scene[asset_cfg.name]

    des_pos_b = object.data.root_pos_w
    curr_pos_w1 = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    curr_pos_w2 = asset.data.body_state_w[:, asset_cfg.body_ids[1], :3]  # type: ignore
    distance1 = torch.norm(curr_pos_w1 - des_pos_b, dim=1)
    distance2 = torch.norm(curr_pos_w2 - des_pos_b, dim=1)

    # obtain the desired and current orientations
    des_quat_b = object.data.root_quat_w
    curr_quat_w1 = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    curr_quat_w2 = asset.data.body_state_w[:, asset_cfg.body_ids[1], 3:7]  # type: ignore
    angle1 = quat_error_magnitude(curr_quat_w1, des_quat_b)
    angle2 = quat_error_magnitude(curr_quat_w2, des_quat_b)

    result1 = (1 - torch.tanh((angle1)/std))+(1 - torch.tanh(torch.abs(distance1-0.1)/(std)))
    result2 = (1 - torch.tanh((angle2)/std))+(1 - torch.tanh(torch.abs(distance2-0.1)/(std)))

    # print("distance1:",distance1)
    # print("angle1:",angle1)
    # print("distance2:",distance2)
    # print("angle2:",angle2)
    # print("box:",euler_xyz_from_quat(des_quat_b))
    # print("1:",euler_xyz_from_quat(curr_quat_w1))
    # print("2:",euler_xyz_from_quat(curr_quat_w2))
    # print("box:",(des_quat_b))
    # print("1:",(curr_quat_w1))
    # print("2:",(curr_quat_w2))
    return result1*result2 #torch.tanh(object_ee_distance1 / std)-torch.tanh(object_ee_distance2 / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    # distance = torch.abs(object.data.root_pos_w[:, 2]-torch.ones_like(object.data.root_pos_w[:, 2])*(height))
    distance = torch.norm((object.data.root_pos_w-env.scene.env_origins)-env.command_manager.get_command(command_name)[:,:3], dim=1)
    angle = quat_error_magnitude(object.data.root_quat_w,env.command_manager.get_command(command_name)[:,3:7])
    # print("height:",object.data.root_pos_w[:, 2])
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)*(1 - torch.tanh(distance / std))*(1 - torch.tanh(angle/std))

def flat_orientation_obj(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    return torch.sum(torch.square(object.data.projected_gravity_b[:, :2]), dim=1)*torch.where(object.data.root_pos_w[:, 2] > 0.83, 1.0, 0.0)

def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    # print("object:",object.data.root_state_w[:, :7])
    return object_pos_b

##############################################################################

def motion_equality_pros(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    curr_pos_w1 = asset.data.joint_pos[:, asset_cfg.joint_ids[0]]
    curr_pos_w2 = asset.data.joint_pos[:, asset_cfg.joint_ids[1]]
    # print(asset.data.joint_names)
    # print("joint:",asset.data.joint_pos[:,:])
    return 1 - torch.tanh(torch.abs(curr_pos_w1-curr_pos_w2) / std)

def motion_equality_cons(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    curr_pos_w1 = asset.data.joint_pos[:, asset_cfg.joint_ids[0]]
    curr_pos_w2 = asset.data.joint_pos[:, asset_cfg.joint_ids[1]]
    return 1 - torch.tanh(torch.abs(curr_pos_w1+curr_pos_w2) / std)