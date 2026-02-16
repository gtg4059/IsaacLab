# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import yaw_quat, subtract_frame_transforms, quat_error_magnitude
from isaaclab.assets import RigidObject, Articulation
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

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
    single_stance = torch.sum(in_contact.int(), dim=1) == 1

    reward = torch.min(torch.where(double_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.where(torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) < 0.05,torch.clamp(reward, max=threshold),0)
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

def foot_clearance_reward(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    target_height: float, std: float
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    double_stance = torch.sum(in_contact.int(), dim=1) == 2 
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    # print(single_stance)
    return torch.exp(-torch.sum(reward, dim=1) / std)*single_stance

def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
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
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)

"""
lift
"""
def object_is_lifted(
    env: ManagerBasedRLEnv,std:float,minimal_height: float, height: float, 
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    # print(object.data.root_pos_w[:, 2])
    distance = torch.abs(object.data.root_pos_w[:,2]-height*torch.ones_like((object.data.root_pos_w[:,2])))
    # return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)
    # print(object.data.root_pos_w[:,2])
    return ((1 - torch.tanh(torch.abs(distance)/std))+5*(1 - torch.tanh(torch.abs(distance)/std**2)))*torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)

def object_is_contacted(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact_force = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids],dim=2)
    # not_allow_contact = contact_force > 12
    contact = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids],dim=2)>0.01
    # return torch.sum(contact.int()-0.00002*contact_force**2, dim=1)
    # print("body_names:",sensor_cfg.body_names)
    # print("contact:",contact)
    # print("contact_force:",contact_force)
    # print(0.00005*not_allow_contact*contact_force**2)
    # print(contact_force)
    return torch.sum(contact.int()-0.0000001*(contact_force**2), dim=1)

def table_not_contacted(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_table"),
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    # contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    # return torch.sqrt(torch.sqrt(air_time[:,0]))
    # contact_force = torch.norm(contact_sensor.data.force_matrix_w[:, 0],dim=2)#N,B,M,3
    discontact = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids],dim=2) < 1e-8
    # print("discontact:",discontact.shape)
    # print(0.00002*contact_force**2)
    return discontact.squeeze_()

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
    # -1.2359
    des_pos_b = object.data.root_pos_w 
    # -0.9818
    curr_pos_w1 = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    # -1.5668
    curr_pos_w2 = asset.data.body_state_w[:, asset_cfg.body_ids[1], :3]  # type: ignore
    #-0.9818-(-1.2359+0.18) = 
    distance1 = torch.norm(curr_pos_w1 - (des_pos_b+torch.tensor([0.0,0.12,0.0],device="cuda:0").repeat(env.num_envs,1)), dim=1, p=2)# 0.12
    #-1.5668-(-1.2359-0.18) = 
    distance2 = torch.norm(curr_pos_w2 - (des_pos_b-torch.tensor([0.0,0.12,0.0],device="cuda:0").repeat(env.num_envs,1)), dim=1, p=2)
    # print("curr_pos_w1:",curr_pos_w1)
    # print("curr_pos_w2:",curr_pos_w2)
    # print("des_pos_b:",des_pos_b)
    # obtain the desired and current orientations
    des_quat_b = object.data.root_quat_w
    curr_quat_w1 = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    curr_quat_w2 = asset.data.body_state_w[:, asset_cfg.body_ids[1], 3:7]  # type: ignore
    angle1 = quat_error_magnitude(des_quat_b-curr_quat_w1, torch.tensor([0.7073883, 0,0,-0.7068252],device="cuda:0").repeat(env.num_envs,1))#-pi
    angle2 = quat_error_magnitude(des_quat_b-curr_quat_w2, torch.tensor([0.7073883, 0,0, 0.7068252],device="cuda:0").repeat(env.num_envs,1))#pi
    # result1 = (1 - torch.tanh(torch.abs(angle1)/(std)))*(1 - torch.tanh(torch.abs(distance1-0.18)/(std**2)))
    # result2 = (1 - torch.tanh(torch.abs(angle2)/(std)))*(1 - torch.tanh(torch.abs(distance2-0.18)/(std**2)))
    dist = torch.sqrt((1 - torch.tanh(torch.abs(distance1)/(std)))*(1 - torch.tanh(torch.abs(distance2)/(std))))+5*torch.sqrt((1 - torch.tanh(torch.abs(distance1)/(std**2)))*(1 - torch.tanh(torch.abs(distance2)/(std**2))))
    angle = torch.sqrt((1 - torch.tanh(torch.abs(angle1/(std*2))))*(1 - torch.tanh(torch.abs(angle2/(std*2)))))
    # print("distance1:",distance1)
    # print("distance2:",distance2)
    return dist#+0.3*angle


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_table"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    robot: RigidObject = env.scene[asset_cfg.name]
    # contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # air_time = contact_sensor.data.current_air_time[:, 0]
    # in_air = air_time > 0.0
    object_pos_b, object_quat_b = subtract_frame_transforms(
        robot.data.body_pos_w[:,asset_cfg.body_ids[0]], robot.data.body_quat_w[:, asset_cfg.body_ids[0]], object.data.root_pos_w[:, :3]
    )
    distance = torch.norm(torch.abs(object_pos_b[:, :3]-torch.tensor([0.0, 0.0, 0.48],device="cuda:0").repeat(env.num_envs,1)),dim=1)
    # roll = math_utils.wrap_to_pi(euler_xyz_from_quat(object.data.root_quat_w)[0])
    # pitch = math_utils.wrap_to_pi(euler_xyz_from_quat(object.data.root_quat_w)[1])
    # yaw = math_utils.wrap_to_pi(euler_xyz_from_quat(object.data.root_quat_w)[2])
    # distance = torch.norm((object.data.root_pos_w-robot.data.root_pos_w)-env.command_manager.get_command(command_name)[:,:3], dim=1)
    # angle = torch.sqrt(roll**2+pitch**2+yaw**2)
    # print((object.data.root_pos_w-robot.data.root_pos_w))
    # print("distance:",((1 - torch.tanh(torch.abs(distance)/(std)))+5*(1 - torch.tanh(torch.abs(distance)/(std**2)))))
    # print("object_pos_b:",object_pos_b)
    # print("distance:",distance)
    # print("angle:",roll,pitch,yaw)
    # print(object_pos_b[:, :3]-torch.tensor([0.25, 0.0, 0.08],device="cuda:0").repeat(env.num_envs,1))
    return ((1 - torch.tanh(torch.abs(distance)/std)))*torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)

def flat_orientation_obj(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    # return -torch.sum(torch.square(object.data.projected_gravity_b[:, :2]), dim=1)#*torch.where(object.data.root_pos_w[:, 2] > 0.83, 1.0, 0.0)
    return -torch.sum(torch.square(object.data.projected_gravity_b[:, :2]), dim=1)#*torch.where(object.data.root_pos_w[:, 2] > 0.83, 1.0, 0.0)


def object_state_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, object_quat_b = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    # print("object:",object.data.root_pos_w[:, 2])
    return torch.concat((object_pos_b,object_quat_b),dim=1)

def object_is_contacted_obs(
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
    return 0.5*in_contact#*torch.sum(in_contact.int(), dim=1)

def shoulder_roll_limit(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*_shoulder_roll_joint"]),
) -> torch.Tensor:
    """
    Penalize if specified shoulder roll joints are out of custom limits.
    asset_cfg.joint_names로 관절을 지정할 수 있음.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # joint_deviation_l1 함수처럼 joint_ids를 사용
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    
    # 관절별로 패널티 계산
    penalties = torch.zeros(env.num_envs, device=asset.data.joint_pos.device)
    
    # joint_names가 있으면 이름으로 분기 처리
    if asset_cfg.joint_names:
        for i, name in enumerate(asset_cfg.joint_names):
            if i < joint_pos.shape[1]:  # 인덱스 범위 확인
                joint = joint_pos[:, i]
                # 왼쪽은 0.1 미만, 오른쪽은 -0.1 초과일 때 패널티 (이름에 따라 분기)
                if "left" in name:
                    penalties -= (joint < 0.1).float()
                elif "right" in name:
                    penalties -= (joint > -0.1).float()
    
    return penalties

def foot_flat_contact(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    full foot(ankle_roll_joint) contact
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]

    foot_orientations = asset.data.body_quat_w[:, asset_cfg.body_ids]  # [N, num_feet, 4]
    num_envs, num_feet = foot_orientations.shape[:2]
    foot_orientations_flat = foot_orientations.view(-1, 4)

    from isaaclab.utils.math import euler_xyz_from_quat
    roll, pitch, yaw = euler_xyz_from_quat(foot_orientations_flat)
    foot_pitch = pitch.view(num_envs, num_feet)

    pitch_reward = torch.exp(-torch.abs(foot_pitch) * 5.0)

    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = (contact_time > 0.0).float()

    pitch_reward_total = torch.sum(pitch_reward * in_contact, dim=1)


    # forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    # strong_contact = forces.norm(dim=-1).max(dim=1)[0] > 1.0  # [N, num_feet]

    z_axis_world = asset.data.body_state_w[:, asset_cfg.body_ids, 2]  # [N, num_feet]
    flatness = torch.abs(z_axis_world)

    # contact_reward_total = torch.sum(flatness * strong_contact.float(), dim=1)
    contact_reward_total = torch.sum(flatness, dim=1)

    total_reward = 0.6 * pitch_reward_total + 0.3 * contact_reward_total

    return total_reward


def feet_air_time_balance(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    threshold: float, 
    sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward balanced air time between left and right feet.
    
    This function rewards the agent for having similar air time between left and right feet,
    promoting balanced gait patterns.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Get air time for both feet
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]  # [N, 2]
    
    # Compute difference between left and right foot air time
    left_air_time = air_time[:, 0]   # Left foot
    right_air_time = air_time[:, 1]  # Right foot
    
    # Balance reward: penalize large differences
    air_time_diff = torch.abs(left_air_time - right_air_time)
    balance_reward = torch.exp(-air_time_diff / threshold)  # Higher reward for smaller differences
    
    # Only reward when robot is moving
    is_moving = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    
    return balance_reward * is_moving.float()

def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)
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
    return torch.square(curr_pos_w1-curr_pos_w2)

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
    return torch.square(curr_pos_w1+curr_pos_w2)


# def tracking_lin_vel_force_world_reward( 
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg, 
#     force_command_name: str, 
#     command_name: str, 
#     base_force_kds: torch.Tensor, 
#     tracking_sigma: float, 
#     lin_vel_x_clip: float, 
#     lin_vel_y_clip: float, 
#     ang_vel_yaw_clip: float, 
#     ) -> torch.Tensor: 
#     """ 
#     Reward for tracking linear velocity while accounting for external force. 
#     """ 
#     # --- robot --- 
#     asset: Articulation = env.scene[asset_cfg.name] 

#     # base linear velocity (world frame) 
#     # shape: (num_envs, 3) 
#     base_lin_vel_w = asset.data.root_lin_vel_w 

#     # base orientation 
#     base_quat_w = asset.data.root_quat_w 

#     # --- yaw-only quaternion --- 
#     from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz 
#     roll, pitch, yaw = euler_xyz_from_quat(base_quat_w) 
#     zeros = torch.zeros_like(yaw) 
#     base_yaw_quat = quat_from_euler_xyz(zeros, zeros, yaw) 

#     # --- external force command (base frame) --- 
#     # shape: (num_envs, 3) 
#     force_cmd_base = env.command_manager.get_command(force_command_name) 

#     # world → base(yaw) frame 
#     from isaaclab.utils.math import quat_rotate_inverse force_base_local = quat_rotate_inverse(base_yaw_quat, force_cmd_base) 

#     # --- velocity command --- 
#     # shape: (num_envs, 3) : [vx, vy, wz] 
#     vel_cmd = env.command_manager.get_command(command_name) 

#     # force → velocity offset 
#     base_lin_vel_offset = (force_base_local / base_force_kds)[:, :2] + vel_cmd[:, :2] 

#     # --- stop gating --- 
#     non_stop = ( (torch.abs(base_lin_vel_offset[:, 0]) > lin_vel_x_clip) 
#                 | (torch.abs(base_lin_vel_offset[:, 1]) > lin_vel_y_clip) 
#                 | (torch.abs(vel_cmd[:, 2]) > ang_vel_yaw_clip) 
#                 ) 
#     base_lin_vel_offset *= non_stop.unsqueeze(1) 

#     # --- tracking error --- 
#     lin_vel_error = torch.sum( 
#         torch.square(base_lin_vel_offset - base_lin_vel_w[:, :2]), dim=1 
#         ) 
    
#     return torch.exp(-lin_vel_error / tracking_sigma)

def tracking_lin_vel_force_reward( 
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    force_command_name: str, 
    vel_command_name: str, 
    damping: float, 
    sigma: float, 
    vel_clip: float = 0.01 
    # 정지 상태 판정 임계값 
    ) -> torch.Tensor: 
    asset: Articulation = env.scene[asset_cfg.name] 
    # 1. 실제 속도 (Base XY frame) 
    vel_actual = asset.data.root_lin_vel_b[:, :2] 
    # 2. 명령 속도 (v_cmd) 
    vel_cmd = env.command_manager.get_command(vel_command_name)[:, :2] 
    # 3. 외력 보상 (F_ext / D) 
    # UniformForceCommand에서 가져온 힘 (이미 Base Frame) 
    force_cmd = env.command_manager.get_command(force_command_name)[:, :2] 
    force_offset = force_cmd / damping 
    # 4. 최종 목표 속도 계산 
    target_vel = vel_cmd + force_offset #
    # 5. Stop Gating: 명령이 거의 0이면 타겟도 0으로 간주 
    is_moving = torch.norm(target_vel, dim=1) > vel_clip 
    target_vel *= is_moving.unsqueeze(1) 
    # 6. 오차 제곱 및 보상 계산 
    # 수식: exp(- ||v_act - v_target||^2 / sigma) 
    error_sq = torch.sum(torch.square(vel_actual - target_vel), dim=1) 
    return torch.exp(-error_sq / sigma)


def compliance_with_external_force_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    force_command_name: str,
    sigma: float,                 # (선택) 속도 크기 정규화 용도로 쓸 수 있음
    force_threshold: float = 20.0, # 이 이상일 때만 “순응” 보상 활성화
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names="torso_link"),
) -> torch.Tensor:
    """외력 방향에 순응해서 살짝 이동하도록 하는 보상 함수.

    - 외력이 충분히 클 때만 활성화됨.
    - 외력 방향과 실제 base 속도 방향이 같을수록 보상이 커짐.
    - 힘과 속도가 정확히 반대면(=힘에 저항) 보상이 0에 가까움.
    """

    asset: Articulation = env.scene[asset_cfg.name]

    # 1. 실제 base 선속도 (base frame XY)
    vel_actual = asset.data.root_lin_vel_b[:, :2]  # [N, 2]
    vel_norm = torch.norm(vel_actual, dim=1)       # [N]

    # 2. 외력: 커맨드 사용(권장) 또는 contact 센서
    if force_command_name is not None:
        forces_local_base = env.command_manager.get_command(force_command_name)  # [N, 3] base frame
    else:
        forces_local_base = torch.zeros((env.num_envs, 3), device=vel_actual.device)
        contact_sensor = None
        if hasattr(env.scene, "sensors"):
            contact_sensor = env.scene.sensors.get(sensor_cfg.name)
        if contact_sensor is not None:
            try:
                net_forces = contact_sensor.data.net_forces_w
                if net_forces is not None:
                    if net_forces.dim() == 4:
                        net = net_forces[:, -1, :, :]
                    else:
                        net = net_forces
                    try:
                        ids = sensor_cfg.body_ids
                    except Exception:
                        ids = None
                    if ids is None or len(ids) == 0:
                        base_forces_w = net[:, 0, :]
                    else:
                        base_forces_w = torch.sum(net[:, ids, :], dim=1)
                    base_yaw_q = yaw_quat(asset.data.root_quat_w)
                    forces_local_base = quat_apply_inverse(base_yaw_q, base_forces_w)
            except Exception:
                pass

    # 3. XY 평면에서의 외력 및 크기
    force_xy = forces_local_base[:, :2]            # [N, 2]
    force_norm = torch.norm(force_xy, dim=1)       # [N]

    # 4. 외력이 충분히 클 때만 순응 보상 활성화
    big_force = force_norm > force_threshold       # [N]

    # 외력이 거의 없으면 보상 0
    if not torch.any(big_force):
        return torch.zeros(env.num_envs, device=vel_actual.device)

    eps = 1e-6

    # 5. 외력 방향 단위벡터
    force_dir = torch.zeros_like(force_xy)         # [N, 2]
    force_dir[big_force] = (
        force_xy[big_force] /
        (force_norm[big_force].unsqueeze(1) + eps)
    )

    # 6. 속도를 외력 방향으로 project → 방향 정렬 정도 (코사인 유사도)
    # cos(theta) = (v · f) / (||v|| ||f||)
    cos_sim = torch.zeros(env.num_envs, device=vel_actual.device)
    denom = (vel_norm * force_norm + eps)
    cos_sim = torch.sum(vel_actual * force_xy, dim=1) / denom
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)      # 수치 안정화

    # 7. cos_sim ∈ [-1, 1] → [0, 1] 로 매핑
    # -1 : 완전 반대 (힘에 저항)  → 0
    #  0 : 직교                   → 0.5
    #  1 : 완전 같은 방향 (힘에 완전히 순응) → 1
    align_score = 0.5 * (cos_sim + 1.0)

    # 8. 너무 큰 속도는 살짝 억제 (힘에 “살짝” 끌려가도록)
    #    목표 속도 크기를 force_norm과 비례하도록 약하게 설정
    target_speed = 0.1 * torch.tanh(force_norm / (sigma + eps))  # [N]
    speed_error = (vel_norm - target_speed) ** 2                 # [N]
    speed_term = torch.exp(-speed_error / (sigma + eps))

    # 9. 최종 보상: 외력 방향 정렬 * 적당한 속도 크기, 외력이 큰 경우에만
    reward = align_score * speed_term
    reward = torch.where(big_force, reward, torch.zeros_like(reward))

    return reward

def shoulder_roll_limit(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*_shoulder_roll_joint"]),
) -> torch.Tensor:
    """
    Penalize if specified shoulder roll joints are out of custom limits.
    asset_cfg.joint_names로 관절을 지정할 수 있음.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # joint_deviation_l1 함수처럼 joint_ids를 사용
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    
    # 관절별로 패널티 계산
    penalties = torch.zeros(env.num_envs, device=asset.data.joint_pos.device)
    
    # joint_names가 있으면 이름으로 분기 처리
    if asset_cfg.joint_names:
        for i, name in enumerate(asset_cfg.joint_names):
            if i < joint_pos.shape[1]:  # 인덱스 범위 확인
                joint = joint_pos[:, i]
                # 왼쪽은 0.1 미만, 오른쪽은 -0.1 초과일 때 패널티 (이름에 따라 분기)
                if "left" in name:
                    penalties -= (joint < 0.1).float()
                elif "right" in name:
                    penalties -= (joint > -0.1).float()
    
    return penalties