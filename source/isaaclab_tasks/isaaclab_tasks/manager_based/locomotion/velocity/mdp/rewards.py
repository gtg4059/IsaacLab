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
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, FrameTransformer
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat, combine_frame_transforms, euler_xyz_from_quat
from isaaclab.assets import RigidObject

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
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < 0.1
    #print("reward:",reward)
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
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_pos_w[:, :3])
    # print("vel_yaw:",vel_yaw)
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    # print("torch.exp(-lin_vel_error / std**2):",torch.exp(-lin_vel_error / std**2))
    # print("lin_vel_error:",torch.exp(-lin_vel_error / std**2))
    # print("command:",env.command_manager.get_command(command_name)[:, :2])
    # print("lin_vel_error2:",asset.data.root_pos_w[:, :2])
    # print("torch.exp(-lin_vel_error / std**2):",env.command_manager.get_command(command_name)[:, :2],asset.data.root_pos_w[:, :2],torch.exp(-lin_vel_error / std**2))
    return torch.exp(-lin_vel_error / std**2)

def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # euler_w = euler_xyz_from_quat(asset.data.root_quat_w)[2]
    # # print("euler_w:",euler_w)
    # ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2]-euler_w)

    # lin_vel_error = torch.sum(
    #     torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_pos_w[:, :2]), dim=1
    # )
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    # print("command_name:",env.command_manager.get_command(command_name)[:, 2])
    # print("asset.data.root_ang_vel_w[:, 2]:",asset.data.root_ang_vel_w[:, 2])
    # print("ang_vel_error:",ang_vel_error)
    # print("torch.exp(-ang_vel_error / std**2):",torch.exp(-ang_vel_error / std**2))
    # print("torch.exp(-ang_vel_error / std**2):",torch.exp(-ang_vel_error / std**2))
    return torch.exp(-ang_vel_error / std**2)# torch.where(lin_vel_error<0.1,torch.exp(-ang_vel_error / std**2),0)

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
    # print("command_manager:",env.command_manager.get_command(command_name)[:, :3])
    #print("command_manager:",env.command_manager.get_command(command_name)[:, 2])
    return 100*torch.exp(-ang_vel_error / std**2)/(1+2*torch.exp(lin_vel_error))


"""
lift
"""
def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    print(object.data.root_pos_w[:, 2])
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)

def object_is_contacted(
    env: ManagerBasedRLEnv, threshold: float, 
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_arm"),
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
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
    # ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    # print("asset_cfg.body_ids:",asset_cfg.body_ids)
    curr_pos_w1 = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]
    curr_pos_w2 = asset.data.body_state_w[:, asset_cfg.body_ids[1], :3]
    # ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance1 = torch.norm(cube_pos_w - curr_pos_w1, dim=1)
    object_ee_distance2 = torch.norm(cube_pos_w - curr_pos_w2, dim=1)

    # print("object_ee_distance:",object_ee_distance1,object_ee_distance2)
    return 0.7 - object_ee_distance1 - object_ee_distance2 #torch.tanh(object_ee_distance1 / std)-torch.tanh(object_ee_distance2 / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    roll = euler_xyz_from_quat(object.data.root_quat_w)[0]
    pitch = euler_xyz_from_quat(object.data.root_quat_w)[1]
    # rewarded if the object is lifted above the threshold
    object: RigidObject = env.scene[object_cfg.name]
    distance = torch.abs(object.data.root_pos_w[:, 2]-torch.ones_like(object.data.root_pos_w[:, 2])*(minimal_height+0.2))
    # print(object.data.root_pos_w[:, 2])
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)*(1 - torch.tanh(distance / std)*(1-torch.tanh((torch.abs(roll)+torch.abs(pitch))/std)))

# def motion_equality(
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ) -> torch.Tensor:
#     """Reward the agent for tracking the goal pose using tanh-kernel."""
#     # extract the used quantities (to enable type-hinting)
#     robot = env.scene[asset_cfg.name]

#     curr_pos_w1 = robot.data.body_state_w[:, asset_cfg.body_ids[0], :3]
#     curr_pos_w2 = robot.data.body_state_w[:, asset_cfg.body_ids[1], :3]
    
#     return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)*(1 - torch.tanh(distance / std)*(1-torch.tanh((torch.abs(roll)+torch.abs(pitch))/std)))