# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import logging
import os
import time
import weakref
from pathlib import Path

import torch

import omni.physics.tensors.impl.api as physx
from isaacsim.core.simulation_manager import SimulationManager

import isaaclab.utils.math as math_utils
from isaaclab.utils.buffers import TimestampedBuffer

# import logger
logger = logging.getLogger(__name__)

import sfd_coreservice

_cudacri_dir = Path(__file__).resolve().parent

class ArticulationData:
    """Data container for an articulation.

    This class contains the data for an articulation in the simulation. The data includes the state of
    the root rigid body, the state of all the bodies in the articulation, and the joint state. The data is
    stored in the simulation world frame unless otherwise specified.

    An articulation is comprised of multiple rigid bodies or links. For a rigid body, there are two frames
    of reference that are used:

    - Actor frame: The frame of reference of the rigid body prim. This typically corresponds to the Xform prim
      with the rigid body schema.
    - Center of mass frame: The frame of reference of the center of mass of the rigid body.

    Depending on the settings, the two frames may not coincide with each other. In the robotics sense, the actor frame
    can be interpreted as the link frame.
    """

    def __init__(self, root_physx_view: physx.ArticulationView, device: str):
        """Initializes the articulation data.

        Args:
            root_physx_view: The root articulation view.
            device: The device used for processing.
        """
        # Set the parameters
        self.device = device
        # Set the root articulation view
        # note: this is stored as a weak reference to avoid circular references between the asset class
        #  and the data container. This is important to avoid memory leaks.
        self._root_physx_view: physx.ArticulationView = weakref.proxy(root_physx_view)

        # Set initial time stamp
        self._sim_timestamp = 0.0

        # obtain global simulation view
        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        gravity = self._physics_sim_view.get_gravity()
        # Convert to direction vector
        gravity_dir = torch.tensor((gravity[0], gravity[1], gravity[2]), device=self.device)
        gravity_dir = math_utils.normalize(gravity_dir.unsqueeze(0)).squeeze(0)

        # Initialize constants
        self.GRAVITY_VEC_W = gravity_dir.repeat(self._root_physx_view.count, 1)
        self.FORWARD_VEC_B = torch.tensor((1.0, 0.0, 0.0), device=self.device).repeat(self._root_physx_view.count, 1)

        # Initialize history for finite differencing
        self._previous_joint_vel = self._root_physx_view.get_dof_velocities().clone()

        # -- CRI (buffers before solver warm-up)
        self._CRI = TimestampedBuffer()
        self._CRI_float = TimestampedBuffer()
        self._cri_q_f64: torch.Tensor | None = None
        self._cri_qd_f64: torch.Tensor | None = None
        # First CRI eval at the current sim timestamp (survives mid-step joint writes / env reset).
        self._traj_cri_timestamp = -1.0
        self._traj_q: torch.Tensor | None = None
        self._traj_qd: torch.Tensor | None = None
        self._traj_cri: torch.Tensor | None = None
        self._cri_inference_time_s = 0.0
        self._cri_inference_count = 0
        self._cri_inference_time_total_s = 0.0
        self._cri_inference_time_min_s = float("inf")
        self._cri_inference_time_max_s = 0.0
        self._cri_inference_samples_s: list[float] = []
        self._cri_cuda_start_evt: torch.cuda.Event | None = None
        self._cri_cuda_end_evt: torch.cuda.Event | None = None
        # Mid-step env reset: keep pre-reset CRI for non-reset envs; refresh only dirty rows.
        self._cri_dirty: torch.Tensor | None = None
        self._cri_last_batch_rows: int = 0
        # Pose → CRI row from a full-N TRT solve (batch=1 ≠ batch=N numerically).
        self._cri_nbatch_row_cache: dict[tuple, torch.Tensor] = {}

        self.solver = sfd_coreservice.CoreService(str(_cudacri_dir), self._root_physx_view.count)
        self.solver.RunSolver_CUDA_LoadAnalysisForCRI(str(_cudacri_dir))
        self._warmup_cri_solver()
        
        # Initialize the lazy buffers.
        # -- link frame w.r.t. world frame
        self._root_link_pose_w = TimestampedBuffer()
        self._root_link_vel_w = TimestampedBuffer()
        self._body_link_pose_w = TimestampedBuffer()
        self._body_link_vel_w = TimestampedBuffer()
        # -- com frame w.r.t. link frame
        self._body_com_pose_b = TimestampedBuffer()
        # -- com frame w.r.t. world frame
        self._root_com_pose_w = TimestampedBuffer()
        self._root_com_vel_w = TimestampedBuffer()
        self._body_com_pose_w = TimestampedBuffer()
        self._body_com_vel_w = TimestampedBuffer()
        self._body_com_acc_w = TimestampedBuffer()
        # -- combined state (these are cached as they concatenate)
        self._root_state_w = TimestampedBuffer()
        self._root_link_state_w = TimestampedBuffer()
        self._root_com_state_w = TimestampedBuffer()
        self._body_state_w = TimestampedBuffer()
        self._body_link_state_w = TimestampedBuffer()
        self._body_com_state_w = TimestampedBuffer()
        # -- joint state
        self._joint_pos = TimestampedBuffer()
        self._joint_vel = TimestampedBuffer()
        self._joint_acc = TimestampedBuffer()
        self._body_incoming_joint_wrench_b = TimestampedBuffer()

    def update(self, dt: float):
        self._dt = dt
        # update the simulation timestamp
        self._sim_timestamp += dt
        # Trigger an update of the joint acceleration buffer at a higher frequency
        # since we do finite differencing.
        self.joint_acc

    ##
    # Names.
    ##

    body_names: list[str] = None
    """Body names in the order parsed by the simulation view."""

    joint_names: list[str] = None
    """Joint names in the order parsed by the simulation view."""

    fixed_tendon_names: list[str] = None
    """Fixed tendon names in the order parsed by the simulation view."""

    spatial_tendon_names: list[str] = None
    """Spatial tendon names in the order parsed by the simulation view."""

    ##
    # Defaults - Initial state.
    ##

    default_root_state: torch.Tensor = None
    """Default root state ``[pos, quat, lin_vel, ang_vel]`` in the local environment frame.
    Shape is (num_instances, 13).

    The position and quaternion are of the articulation root's actor frame. Meanwhile, the linear and angular
    velocities are of its center of mass frame.

    This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
    """

    default_joint_pos: torch.Tensor = None
    """Default joint positions of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
    """

    default_joint_vel: torch.Tensor = None
    """Default joint velocities of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
    """

    ##
    # Defaults - Physical properties.
    ##

    default_mass: torch.Tensor = None
    """Default mass for all the bodies in the articulation. Shape is (num_instances, num_bodies).

    This quantity is parsed from the USD schema at the time of initialization.
    """

    default_inertia: torch.Tensor = None
    """Default inertia for all the bodies in the articulation. Shape is (num_instances, num_bodies, 9).

    The inertia tensor should be given with respect to the center of mass, expressed in the articulation links'
    actor frame. The values are stored in the order
    :math:`[I_{xx}, I_{yx}, I_{zx}, I_{xy}, I_{yy}, I_{zy}, I_{xz}, I_{yz}, I_{zz}]`. However, due to the
    symmetry of inertia tensors, row- and column-major orders are equivalent.

    This quantity is parsed from the USD schema at the time of initialization.
    """

    default_joint_stiffness: torch.Tensor = None
    """Default joint stiffness of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the actuator model's :attr:`isaaclab.actuators.ActuatorBaseCfg.stiffness`
    parameter. If the parameter's value is None, the value parsed from the USD schema, at the time of initialization,
    is used.

    .. attention::
        The default stiffness is the value configured by the user or the value parsed from the USD schema.
        It should not be confused with :attr:`joint_stiffness`, which is the value set into the simulation.
    """

    default_joint_damping: torch.Tensor = None
    """Default joint damping of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the actuator model's :attr:`isaaclab.actuators.ActuatorBaseCfg.damping`
    parameter. If the parameter's value is None, the value parsed from the USD schema, at the time of initialization,
    is used.

    .. attention::
        The default stiffness is the value configured by the user or the value parsed from the USD schema.
        It should not be confused with :attr:`joint_damping`, which is the value set into the simulation.
    """

    default_joint_armature: torch.Tensor = None
    """Default joint armature of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the actuator model's :attr:`isaaclab.actuators.ActuatorBaseCfg.armature`
    parameter. If the parameter's value is None, the value parsed from the USD schema, at the time of initialization,
    is used.
    """

    default_joint_friction_coeff: torch.Tensor = None
    """Default joint static friction coefficient of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the actuator model's :attr:`isaaclab.actuators.ActuatorBaseCfg.friction`
    parameter. If the parameter's value is None, the value parsed from the USD schema, at the time of initialization,
    is used.

    Note:
        In Isaac Sim 4.5, this parameter is modeled as a coefficient. In Isaac Sim 5.0 and later,
        it is modeled as an effort (torque or force).
    """

    default_joint_dynamic_friction_coeff: torch.Tensor = None
    """Default joint dynamic friction coefficient of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the actuator model's
    :attr:`isaaclab.actuators.ActuatorBaseCfg.dynamic_friction` parameter. If the parameter's value is None,
    the value parsed from the USD schema, at the time of initialization, is used.

    Note:
        In Isaac Sim 4.5, this parameter is modeled as a coefficient. In Isaac Sim 5.0 and later,
        it is modeled as an effort (torque or force).
    """

    default_joint_viscous_friction_coeff: torch.Tensor = None
    """Default joint viscous friction coefficient of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the actuator model's
    :attr:`isaaclab.actuators.ActuatorBaseCfg.viscous_friction` parameter. If the parameter's value is None,
    the value parsed from the USD schema, at the time of initialization, is used.
    """

    default_joint_pos_limits: torch.Tensor = None
    """Default joint position limits of all joints. Shape is (num_instances, num_joints, 2).

    The limits are in the order :math:`[lower, upper]`. They are parsed from the USD schema at the
    time of initialization.
    """

    default_fixed_tendon_stiffness: torch.Tensor = None
    """Default tendon stiffness of all fixed tendons. Shape is (num_instances, num_fixed_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """

    default_fixed_tendon_damping: torch.Tensor = None
    """Default tendon damping of all fixed tendons. Shape is (num_instances, num_fixed_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """

    default_fixed_tendon_limit_stiffness: torch.Tensor = None
    """Default tendon limit stiffness of all fixed tendons. Shape is (num_instances, num_fixed_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """

    default_fixed_tendon_rest_length: torch.Tensor = None
    """Default tendon rest length of all fixed tendons. Shape is (num_instances, num_fixed_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """

    default_fixed_tendon_offset: torch.Tensor = None
    """Default tendon offset of all fixed tendons. Shape is (num_instances, num_fixed_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """

    default_fixed_tendon_pos_limits: torch.Tensor = None
    """Default tendon position limits of all fixed tendons. Shape is (num_instances, num_fixed_tendons, 2).

    The position limits are in the order :math:`[lower, upper]`. They are parsed from the USD schema at the time of
    initialization.
    """

    default_spatial_tendon_stiffness: torch.Tensor = None
    """Default tendon stiffness of all spatial tendons. Shape is (num_instances, num_spatial_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """

    default_spatial_tendon_damping: torch.Tensor = None
    """Default tendon damping of all spatial tendons. Shape is (num_instances, num_spatial_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """

    default_spatial_tendon_limit_stiffness: torch.Tensor = None
    """Default tendon limit stiffness of all spatial tendons. Shape is (num_instances, num_spatial_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """

    default_spatial_tendon_offset: torch.Tensor = None
    """Default tendon offset of all spatial tendons. Shape is (num_instances, num_spatial_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """

    ##
    # Joint commands -- Set into simulation.
    ##

    joint_pos_target: torch.Tensor = None
    """Joint position targets commanded by the user. Shape is (num_instances, num_joints).

    For an implicit actuator model, the targets are directly set into the simulation.
    For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
    which are then set into the simulation.
    """

    joint_vel_target: torch.Tensor = None
    """Joint velocity targets commanded by the user. Shape is (num_instances, num_joints).

    For an implicit actuator model, the targets are directly set into the simulation.
    For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
    which are then set into the simulation.
    """

    joint_effort_target: torch.Tensor = None
    """Joint effort targets commanded by the user. Shape is (num_instances, num_joints).

    For an implicit actuator model, the targets are directly set into the simulation.
    For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
    which are then set into the simulation.
    """

    ##
    # Joint commands -- Explicit actuators.
    ##

    computed_torque: torch.Tensor = None
    """Joint torques computed from the actuator model (before clipping). Shape is (num_instances, num_joints).

    This quantity is the raw torque output from the actuator mode, before any clipping is applied.
    It is exposed for users who want to inspect the computations inside the actuator model.
    For instance, to penalize the learning agent for a difference between the computed and applied torques.
    """

    applied_torque: torch.Tensor = None
    """Joint torques applied from the actuator model (after clipping). Shape is (num_instances, num_joints).

    These torques are set into the simulation, after clipping the :attr:`computed_torque` based on the
    actuator model.
    """

    ##
    # Joint properties.
    ##

    joint_stiffness: torch.Tensor = None
    """Joint stiffness provided to the simulation. Shape is (num_instances, num_joints).

    In the case of explicit actuators, the value for the corresponding joints is zero.
    """

    joint_damping: torch.Tensor = None
    """Joint damping provided to the simulation. Shape is (num_instances, num_joints)

    In the case of explicit actuators, the value for the corresponding joints is zero.
    """

    joint_armature: torch.Tensor = None
    """Joint armature provided to the simulation. Shape is (num_instances, num_joints)."""

    joint_friction_coeff: torch.Tensor = None
    """Joint static friction coefficient provided to the simulation. Shape is (num_instances, num_joints).

    Note: In Isaac Sim 4.5, this parameter is modeled as a coefficient. In Isaac Sim 5.0 and later,
    it is modeled as an effort (torque or force).
    """

    joint_dynamic_friction_coeff: torch.Tensor = None
    """Joint dynamic friction coefficient provided to the simulation. Shape is (num_instances, num_joints).

    Note: In Isaac Sim 4.5, this parameter is modeled as a coefficient. In Isaac Sim 5.0 and later,
    it is modeled as an effort (torque or force).
    """

    joint_viscous_friction_coeff: torch.Tensor = None
    """Joint viscous friction coefficient provided to the simulation. Shape is (num_instances, num_joints)."""

    joint_pos_limits: torch.Tensor = None
    """Joint position limits provided to the simulation. Shape is (num_instances, num_joints, 2).

    The limits are in the order :math:`[lower, upper]`.
    """

    joint_vel_limits: torch.Tensor = None
    """Joint maximum velocity provided to the simulation. Shape is (num_instances, num_joints)."""

    joint_effort_limits: torch.Tensor = None
    """Joint maximum effort provided to the simulation. Shape is (num_instances, num_joints)."""

    ##
    # Joint properties - Custom.
    ##

    soft_joint_pos_limits: torch.Tensor = None
    r"""Soft joint positions limits for all joints. Shape is (num_instances, num_joints, 2).

    The limits are in the order :math:`[lower, upper]`.The soft joint position limits are computed as
    a sub-region of the :attr:`joint_pos_limits` based on the
    :attr:`~isaaclab.assets.ArticulationCfg.soft_joint_pos_limit_factor` parameter.

    Consider the joint position limits :math:`[lower, upper]` and the soft joint position limits
    :math:`[soft_lower, soft_upper]`. The soft joint position limits are computed as:

    .. math::

        soft\_lower = (lower + upper) / 2 - factor * (upper - lower) / 2
        soft\_upper = (lower + upper) / 2 + factor * (upper - lower) / 2

    The soft joint position limits help specify a safety region around the joint limits. It isn't used by the
    simulation, but is useful for learning agents to prevent the joint positions from violating the limits.
    """

    soft_joint_vel_limits: torch.Tensor = None
    """Soft joint velocity limits for all joints. Shape is (num_instances, num_joints).

    These are obtained from the actuator model. It may differ from :attr:`joint_vel_limits` if the actuator model
    has a variable velocity limit model. For instance, in a variable gear ratio actuator model.
    """

    gear_ratio: torch.Tensor = None
    """Gear ratio for relating motor torques to applied Joint torques. Shape is (num_instances, num_joints)."""

    ##
    # Fixed tendon properties.
    ##

    fixed_tendon_stiffness: torch.Tensor = None
    """Fixed tendon stiffness provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""

    fixed_tendon_damping: torch.Tensor = None
    """Fixed tendon damping provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""

    fixed_tendon_limit_stiffness: torch.Tensor = None
    """Fixed tendon limit stiffness provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""

    fixed_tendon_rest_length: torch.Tensor = None
    """Fixed tendon rest length provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""

    fixed_tendon_offset: torch.Tensor = None
    """Fixed tendon offset provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""

    fixed_tendon_pos_limits: torch.Tensor = None
    """Fixed tendon position limits provided to the simulation. Shape is (num_instances, num_fixed_tendons, 2)."""

    ##
    # Spatial tendon properties.
    ##

    spatial_tendon_stiffness: torch.Tensor = None
    """Spatial tendon stiffness provided to the simulation. Shape is (num_instances, num_spatial_tendons)."""

    spatial_tendon_damping: torch.Tensor = None
    """Spatial tendon damping provided to the simulation. Shape is (num_instances, num_spatial_tendons)."""

    spatial_tendon_limit_stiffness: torch.Tensor = None
    """Spatial tendon limit stiffness provided to the simulation. Shape is (num_instances, num_spatial_tendons)."""

    spatial_tendon_offset: torch.Tensor = None
    """Spatial tendon offset provided to the simulation. Shape is (num_instances, num_spatial_tendons)."""

    ##
    # Root state properties.
    ##

    @property
    def root_link_pose_w(self) -> torch.Tensor:
        """Root link pose ``[pos, quat]`` in simulation world frame. Shape is (num_instances, 7).

        This quantity is the pose of the articulation root's actor frame relative to the world.
        The orientation is provided in (w, x, y, z) format.
        """
        if self._root_link_pose_w.timestamp < self._sim_timestamp:
            # read data from simulation
            pose = self._root_physx_view.get_root_transforms().clone()
            pose[:, 3:7] = math_utils.convert_quat(pose[:, 3:7], to="wxyz")
            # set the buffer data and timestamp
            self._root_link_pose_w.data = pose
            self._root_link_pose_w.timestamp = self._sim_timestamp

        return self._root_link_pose_w.data

    @property
    def root_link_vel_w(self) -> torch.Tensor:
        """Root link velocity ``[lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular velocities of the articulation root's actor frame
        relative to the world.
        """
        if self._root_link_vel_w.timestamp < self._sim_timestamp:
            # read the CoM velocity
            vel = self.root_com_vel_w.clone()
            # adjust linear velocity to link from center of mass
            vel[:, :3] += torch.linalg.cross(
                vel[:, 3:], math_utils.quat_apply(self.root_link_quat_w, -self.body_com_pos_b[:, 0]), dim=-1
            )
            # set the buffer data and timestamp
            self._root_link_vel_w.data = vel
            self._root_link_vel_w.timestamp = self._sim_timestamp

        return self._root_link_vel_w.data

    @property
    def root_com_pose_w(self) -> torch.Tensor:
        """Root center of mass pose ``[pos, quat]`` in simulation world frame. Shape is (num_instances, 7).

        This quantity is the pose of the articulation root's center of mass frame relative to the world.
        The orientation is provided in (w, x, y, z) format.
        """
        if self._root_com_pose_w.timestamp < self._sim_timestamp:
            # apply local transform to center of mass frame
            pos, quat = math_utils.combine_frame_transforms(
                self.root_link_pos_w, self.root_link_quat_w, self.body_com_pos_b[:, 0], self.body_com_quat_b[:, 0]
            )
            # set the buffer data and timestamp
            self._root_com_pose_w.data = torch.cat((pos, quat), dim=-1)
            self._root_com_pose_w.timestamp = self._sim_timestamp

        return self._root_com_pose_w.data

    @property
    def root_com_vel_w(self) -> torch.Tensor:
        """Root center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular velocities of the articulation root's center of mass frame
        relative to the world.
        """
        if self._root_com_vel_w.timestamp < self._sim_timestamp:
            self._root_com_vel_w.data = self._root_physx_view.get_root_velocities()
            self._root_com_vel_w.timestamp = self._sim_timestamp

        return self._root_com_vel_w.data

    @property
    def root_state_w(self):
        """Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13).

        The position and quaternion are of the articulation root's actor frame relative to the world. Meanwhile,
        the linear and angular velocities are of the articulation root's center of mass frame.
        """
        if self._root_state_w.timestamp < self._sim_timestamp:
            self._root_state_w.data = torch.cat((self.root_link_pose_w, self.root_com_vel_w), dim=-1)
            self._root_state_w.timestamp = self._sim_timestamp

        return self._root_state_w.data

    @property
    def root_link_state_w(self):
        """Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13).

        The position, quaternion, and linear/angular velocity are of the articulation root's actor frame relative to the
        world.
        """
        if self._root_link_state_w.timestamp < self._sim_timestamp:
            self._root_link_state_w.data = torch.cat((self.root_link_pose_w, self.root_link_vel_w), dim=-1)
            self._root_link_state_w.timestamp = self._sim_timestamp

        return self._root_link_state_w.data

    @property
    def root_com_state_w(self):
        """Root center of mass state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances, 13).

        The position, quaternion, and linear/angular velocity are of the articulation root link's center of mass frame
        relative to the world. Center of mass frame is assumed to be the same orientation as the link rather than the
        orientation of the principle inertia.
        """
        if self._root_com_state_w.timestamp < self._sim_timestamp:
            self._root_com_state_w.data = torch.cat((self.root_com_pose_w, self.root_com_vel_w), dim=-1)
            self._root_com_state_w.timestamp = self._sim_timestamp

        return self._root_com_state_w.data

    ##
    # Body state properties.
    ##

    @property
    def body_link_pose_w(self) -> torch.Tensor:
        """Body link pose ``[pos, quat]`` in simulation world frame.
        Shape is (num_instances, num_bodies, 7).

        This quantity is the pose of the articulation links' actor frame relative to the world.
        The orientation is provided in (w, x, y, z) format.
        """
        if self._body_link_pose_w.timestamp < self._sim_timestamp:
            # perform forward kinematics (shouldn't cause overhead if it happened already)
            self._physics_sim_view.update_articulations_kinematic()
            # read data from simulation
            poses = self._root_physx_view.get_link_transforms().clone()
            poses[..., 3:7] = math_utils.convert_quat(poses[..., 3:7], to="wxyz")
            # set the buffer data and timestamp
            self._body_link_pose_w.data = poses
            self._body_link_pose_w.timestamp = self._sim_timestamp

        return self._body_link_pose_w.data

    @property
    def body_link_vel_w(self) -> torch.Tensor:
        """Body link velocity ``[lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances, num_bodies, 6).

        This quantity contains the linear and angular velocities of the articulation links' actor frame
        relative to the world.
        """
        if self._body_link_vel_w.timestamp < self._sim_timestamp:
            # read data from simulation
            velocities = self.body_com_vel_w.clone()
            # adjust linear velocity to link from center of mass
            velocities[..., :3] += torch.linalg.cross(
                velocities[..., 3:], math_utils.quat_apply(self.body_link_quat_w, -self.body_com_pos_b), dim=-1
            )
            # set the buffer data and timestamp
            self._body_link_vel_w.data = velocities
            self._body_link_vel_w.timestamp = self._sim_timestamp

        return self._body_link_vel_w.data

    @property
    def body_com_pose_w(self) -> torch.Tensor:
        """Body center of mass pose ``[pos, quat]`` in simulation world frame.
        Shape is (num_instances, num_bodies, 7).

        This quantity is the pose of the center of mass frame of the articulation links relative to the world.
        The orientation is provided in (w, x, y, z) format.
        """
        if self._body_com_pose_w.timestamp < self._sim_timestamp:
            # apply local transform to center of mass frame
            pos, quat = math_utils.combine_frame_transforms(
                self.body_link_pos_w, self.body_link_quat_w, self.body_com_pos_b, self.body_com_quat_b
            )
            # set the buffer data and timestamp
            self._body_com_pose_w.data = torch.cat((pos, quat), dim=-1)
            self._body_com_pose_w.timestamp = self._sim_timestamp

        return self._body_com_pose_w.data

    @property
    def body_com_vel_w(self) -> torch.Tensor:
        """Body center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances, num_bodies, 6).

        This quantity contains the linear and angular velocities of the articulation links' center of mass frame
        relative to the world.
        """
        if self._body_com_vel_w.timestamp < self._sim_timestamp:
            self._body_com_vel_w.data = self._root_physx_view.get_link_velocities()
            self._body_com_vel_w.timestamp = self._sim_timestamp

        return self._body_com_vel_w.data

    @property
    def body_state_w(self):
        """State of all bodies `[pos, quat, lin_vel, ang_vel]` in simulation world frame.
        Shape is (num_instances, num_bodies, 13).

        The position and quaternion are of all the articulation links' actor frame. Meanwhile, the linear and angular
        velocities are of the articulation links's center of mass frame.
        """
        if self._body_state_w.timestamp < self._sim_timestamp:
            self._body_state_w.data = torch.cat((self.body_link_pose_w, self.body_com_vel_w), dim=-1)
            self._body_state_w.timestamp = self._sim_timestamp

        return self._body_state_w.data

    @property
    def body_link_state_w(self):
        """State of all bodies' link frame`[pos, quat, lin_vel, ang_vel]` in simulation world frame.
        Shape is (num_instances, num_bodies, 13).

        The position, quaternion, and linear/angular velocity are of the body's link frame relative to the world.
        """
        if self._body_link_state_w.timestamp < self._sim_timestamp:
            self._body_link_state_w.data = torch.cat((self.body_link_pose_w, self.body_link_vel_w), dim=-1)
            self._body_link_state_w.timestamp = self._sim_timestamp

        return self._body_link_state_w.data

    @property
    def body_com_state_w(self):
        """State of all bodies center of mass `[pos, quat, lin_vel, ang_vel]` in simulation world frame.
        Shape is (num_instances, num_bodies, 13).

        The position, quaternion, and linear/angular velocity are of the body's center of mass frame relative to the
        world. Center of mass frame is assumed to be the same orientation as the link rather than the orientation of the
        principle inertia.
        """
        if self._body_com_state_w.timestamp < self._sim_timestamp:
            self._body_com_state_w.data = torch.cat((self.body_com_pose_w, self.body_com_vel_w), dim=-1)
            self._body_com_state_w.timestamp = self._sim_timestamp

        return self._body_com_state_w.data

    @property
    def body_com_acc_w(self):
        """Acceleration of all bodies center of mass ``[lin_acc, ang_acc]``.
        Shape is (num_instances, num_bodies, 6).

        All values are relative to the world.
        """
        if self._body_com_acc_w.timestamp < self._sim_timestamp:
            # read data from simulation and set the buffer data and timestamp
            self._body_com_acc_w.data = self._root_physx_view.get_link_accelerations()
            self._body_com_acc_w.timestamp = self._sim_timestamp

        return self._body_com_acc_w.data

    @property
    def body_com_pose_b(self) -> torch.Tensor:
        """Center of mass pose ``[pos, quat]`` of all bodies in their respective body's link frames.
        Shape is (num_instances, 1, 7).

        This quantity is the pose of the center of mass frame of the rigid body relative to the body's link frame.
        The orientation is provided in (w, x, y, z) format.
        """
        if self._body_com_pose_b.timestamp < self._sim_timestamp:
            # read data from simulation
            pose = self._root_physx_view.get_coms().to(self.device)
            pose[..., 3:7] = math_utils.convert_quat(pose[..., 3:7], to="wxyz")
            # set the buffer data and timestamp
            self._body_com_pose_b.data = pose
            self._body_com_pose_b.timestamp = self._sim_timestamp

        return self._body_com_pose_b.data

    @property
    def body_incoming_joint_wrench_b(self) -> torch.Tensor:
        """Joint reaction wrench applied from body parent to child body in parent body frame.

        Shape is (num_instances, num_bodies, 6). All body reaction wrenches are provided including the root body to the
        world of an articulation.

        For more information on joint wrenches, please check the`PhysX documentation`_ and the underlying
        `PhysX Tensor API`_.

        .. _`PhysX documentation`: https://nvidia-omniverse.github.io/PhysX/physx/5.5.1/docs/Articulations.html#link-incoming-joint-force
        .. _`PhysX Tensor API`: https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.ArticulationView.get_link_incoming_joint_force
        """

        if self._body_incoming_joint_wrench_b.timestamp < self._sim_timestamp:
            self._body_incoming_joint_wrench_b.data = self._root_physx_view.get_link_incoming_joint_force()
            self._body_incoming_joint_wrench_b.time_stamp = self._sim_timestamp
        return self._body_incoming_joint_wrench_b.data

    ##
    # Joint state properties.
    ##

    @property
    def joint_pos(self):
        """Joint positions of all joints. Shape is (num_instances, num_joints)."""
        if self._joint_pos.timestamp < self._sim_timestamp:
            self._refresh_joint_kinematics()
        return self._joint_pos.data

    @property
    def joint_vel(self):
        """Joint velocities of all joints. Shape is (num_instances, num_joints)."""
        if self._joint_vel.timestamp < self._sim_timestamp:
            self._refresh_joint_kinematics()
        return self._joint_vel.data

    def _refresh_joint_kinematics(self) -> None:
        """Read DOF state from PhysX once; keep float64 cache for CRI hot path."""
        q = self._root_physx_view.get_dof_positions()
        qd = self._root_physx_view.get_dof_velocities()
        self._joint_pos.data = q
        self._joint_vel.data = qd
        self._joint_pos.timestamp = self._sim_timestamp
        self._joint_vel.timestamp = self._sim_timestamp
        # PhysX DOF 텐서와 storage 공유(alias)하면 GPU CRI 계산 중 in-place 갱신으로 0/깨진 값이 난다.
        if self._cri_q_f64 is None or self._cri_q_f64.shape != q.shape:
            self._cri_q_f64 = torch.empty(q.shape, device=q.device, dtype=torch.float64)
        self._cri_q_f64.copy_(q)
        if self._cri_qd_f64 is None or self._cri_qd_f64.shape != qd.shape:
            self._cri_qd_f64 = torch.empty(qd.shape, device=qd.device, dtype=torch.float64)
        self._cri_qd_f64.copy_(qd)

    def _cri_input_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._joint_pos.timestamp < self._sim_timestamp:
            self._refresh_joint_kinematics()
        if self._cri_q_f64 is None or self._cri_qd_f64 is None:
            raise RuntimeError("CRI input buffers are not initialized")
        return self._cri_q_f64, self._cri_qd_f64

    def _sync_cri_inputs_from_joint_buffers(self) -> None:
        """Copy current joint buffers into float64 CRI inputs after an in-place joint write."""
        q = self._joint_pos.data
        qd = self._joint_vel.data
        if q is None or qd is None:
            return
        if self._cri_q_f64 is None or self._cri_q_f64.shape != q.shape:
            self._cri_q_f64 = torch.empty(q.shape, device=q.device, dtype=torch.float64)
        self._cri_q_f64.copy_(q)
        if self._cri_qd_f64 is None or self._cri_qd_f64.shape != qd.shape:
            self._cri_qd_f64 = torch.empty(qd.shape, device=qd.device, dtype=torch.float64)
        self._cri_qd_f64.copy_(qd)

    @property
    def joint_acc(self):
        """Joint acceleration of all joints. Shape is (num_instances, num_joints)."""
        if self._joint_acc.timestamp < self._sim_timestamp:
            # note: we use finite differencing to compute acceleration
            time_elapsed = self._sim_timestamp - self._joint_acc.timestamp
            self._joint_acc.data = (self.joint_vel - self._previous_joint_vel) / time_elapsed
            self._joint_acc.timestamp = self._sim_timestamp
            # update the previous joint velocity
            self._previous_joint_vel[:] = self.joint_vel
        return self._joint_acc.data

    ##
    # Derived Properties.
    ##

    @property
    def projected_gravity_b(self):
        """Projection of the gravity direction on base frame. Shape is (num_instances, 3)."""
        return math_utils.quat_apply_inverse(self.root_link_quat_w, self.GRAVITY_VEC_W)

    @property
    def heading_w(self):
        """Yaw heading of the base frame (in radians). Shape is (num_instances,).

        Note:
            This quantity is computed by assuming that the forward-direction of the base
            frame is along x-direction, i.e. :math:`(1, 0, 0)`.
        """
        forward_w = math_utils.quat_apply(self.root_link_quat_w, self.FORWARD_VEC_B)
        return torch.atan2(forward_w[:, 1], forward_w[:, 0])

    @property
    def root_link_lin_vel_b(self) -> torch.Tensor:
        """Root link linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the articulation root's actor frame with respect to the
        its actor frame.
        """
        return math_utils.quat_apply_inverse(self.root_link_quat_w, self.root_link_lin_vel_w)

    @property
    def root_link_ang_vel_b(self) -> torch.Tensor:
        """Root link angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the articulation root's actor frame with respect to the
        its actor frame.
        """
        return math_utils.quat_apply_inverse(self.root_link_quat_w, self.root_link_ang_vel_w)

    @property
    def root_com_lin_vel_b(self) -> torch.Tensor:
        """Root center of mass linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the articulation root's center of mass frame with respect to the
        its actor frame.
        """
        return math_utils.quat_apply_inverse(self.root_link_quat_w, self.root_com_lin_vel_w)

    @property
    def root_com_ang_vel_b(self) -> torch.Tensor:
        """Root center of mass angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the articulation root's center of mass frame with respect to the
        its actor frame.
        """
        return math_utils.quat_apply_inverse(self.root_link_quat_w, self.root_com_ang_vel_w)

    ##
    # Sliced properties.
    ##

    @property
    def root_link_pos_w(self) -> torch.Tensor:
        """Root link position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        return self.root_link_pose_w[:, :3]

    @property
    def root_link_quat_w(self) -> torch.Tensor:
        """Root link orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the root rigid body.
        """
        return self.root_link_pose_w[:, 3:7]

    @property
    def root_link_lin_vel_w(self) -> torch.Tensor:
        """Root linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's actor frame relative to the world.
        """
        return self.root_link_vel_w[:, :3]

    @property
    def root_link_ang_vel_w(self) -> torch.Tensor:
        """Root link angular velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the actor frame of the root rigid body relative to the world.
        """
        return self.root_link_vel_w[:, 3:6]

    @property
    def root_com_pos_w(self) -> torch.Tensor:
        """Root center of mass position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        return self.root_com_pose_w[:, :3]

    @property
    def root_com_quat_w(self) -> torch.Tensor:
        """Root center of mass orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the root rigid body relative to the world.
        """
        return self.root_com_pose_w[:, 3:7]

    @property
    def root_com_lin_vel_w(self) -> torch.Tensor:
        """Root center of mass linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's center of mass frame relative to the world.
        """
        return self.root_com_vel_w[:, :3]

    @property
    def root_com_ang_vel_w(self) -> torch.Tensor:
        """Root center of mass angular velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the root rigid body's center of mass frame relative to the world.
        """
        return self.root_com_vel_w[:, 3:6]

    @property
    def body_link_pos_w(self) -> torch.Tensor:
        """Positions of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the position of the articulation bodies' actor frame relative to the world.
        """
        return self.body_link_pose_w[..., :3]

    @property
    def body_link_quat_w(self) -> torch.Tensor:
        """Orientation (w, x, y, z) of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 4).

        This quantity is the orientation of the articulation bodies' actor frame relative to the world.
        """
        return self.body_link_pose_w[..., 3:7]

    @property
    def body_link_lin_vel_w(self) -> torch.Tensor:
        """Linear velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the linear velocity of the articulation bodies' center of mass frame relative to the world.
        """
        return self.body_link_vel_w[..., :3]

    @property
    def body_link_ang_vel_w(self) -> torch.Tensor:
        """Angular velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the angular velocity of the articulation bodies' center of mass frame relative to the world.
        """
        return self.body_link_vel_w[..., 3:6]

    @property
    def body_com_pos_w(self) -> torch.Tensor:
        """Positions of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the position of the articulation bodies' actor frame.
        """
        return self.body_com_pose_w[..., :3]

    @property
    def body_com_quat_w(self) -> torch.Tensor:
        """Orientation (w, x, y, z) of the principle axis of inertia of all bodies in simulation world frame.
        Shape is (num_instances, num_bodies, 4).

        This quantity is the orientation of the articulation bodies' actor frame.
        """
        return self.body_com_pose_w[..., 3:7]

    @property
    def body_com_lin_vel_w(self) -> torch.Tensor:
        """Linear velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the linear velocity of the articulation bodies' center of mass frame.
        """
        return self.body_com_vel_w[..., :3]

    @property
    def body_com_ang_vel_w(self) -> torch.Tensor:
        """Angular velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the angular velocity of the articulation bodies' center of mass frame.
        """
        return self.body_com_vel_w[..., 3:6]

    @property
    def body_com_lin_acc_w(self) -> torch.Tensor:
        """Linear acceleration of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the linear acceleration of the articulation bodies' center of mass frame.
        """
        return self.body_com_acc_w[..., :3]

    @property
    def body_com_ang_acc_w(self) -> torch.Tensor:
        """Angular acceleration of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the angular acceleration of the articulation bodies' center of mass frame.
        """
        return self.body_com_acc_w[..., 3:6]

    @property
    def body_com_pos_b(self) -> torch.Tensor:
        """Center of mass position of all of the bodies in their respective link frames.
        Shape is (num_instances, num_bodies, 3).

        This quantity is the center of mass location relative to its body'slink frame.
        """
        return self.body_com_pose_b[..., :3]

    @property
    def body_com_quat_b(self) -> torch.Tensor:
        """Orientation (w, x, y, z) of the principle axis of inertia of all of the bodies in their
        respective link frames. Shape is (num_instances, num_bodies, 4).

        This quantity is the orientation of the principles axes of inertia relative to its body's link frame.
        """
        return self.body_com_pose_b[..., 3:7]

    def _cri_motion_tensors_f64(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Contiguous float64 q/qd for pybind (updated with joint_pos/joint_vel refresh)."""
        return self._cri_input_tensors()

    def warmup_cri_solver_rounds(self, rounds: int | None = None) -> None:
        """Extra GPU warm-up after env reset (tail-spike mitigation)."""
        if rounds is None:
            rounds = int(os.environ.get("SFD_ALLOC_WARMUP_ROUNDS", "15"))
        if rounds <= 0:
            return
        q_in, qd_in = self._cri_input_tensors()
        for _ in range(rounds):
            self.solver.RunSolver_CUDA_CRI_AtMotionState(q_in, qd_in)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        print(f"[CRI] warm-up complete: {rounds} rounds", flush=True)

    def _warmup_cri_solver(self) -> None:
        """Allocator/TRT tail-spike mitigation at articulation init."""
        rounds = int(os.environ.get("SFD_ALLOC_WARMUP_ROUNDS", "15"))
        if rounds <= 0:
            return
        q = self._root_physx_view.get_dof_positions()
        qd = self._root_physx_view.get_dof_velocities()
        if q.dtype != torch.float64 or not q.is_contiguous():
            q = q.to(dtype=torch.float64).contiguous()
        if qd.dtype != torch.float64 or not qd.is_contiguous():
            qd = qd.to(dtype=torch.float64).contiguous()
        self._cri_q_f64 = q
        self._cri_qd_f64 = qd
        for _ in range(rounds):
            self.solver.RunSolver_CUDA_CRI_AtMotionState(q, qd)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        print(f"[CRI] init warm-up: {rounds} rounds (SFD_ALLOC_WARMUP_ROUNDS)", flush=True)
    
    def _invalidate_cri_cache(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Invalidate CRI after joint writes / env reset.

        If a CRI result already exists for the current ``_sim_timestamp`` (typical path:
        termination ``CRI_OVF`` then reset), only ``env_ids`` are marked dirty so the next
        access refreshes those rows at the post-reset ``(q, qd)`` while non-reset envs keep
        the pre-reset CRI. That matches a full re-solve on the mixed batch (independent robots)
        without a second 4096-wide solve — and avoids the old bug where obs kept pre-reset CRI
        while ``qd`` was already 0.
        """
        has_fresh = self._CRI.data is not None and self._CRI.timestamp >= self._sim_timestamp
        if not has_fresh:
            self._CRI.timestamp = -1.0
            if self._cri_dirty is not None:
                self._cri_dirty.zero_()
            return

        n = self._root_physx_view.count
        if self._cri_dirty is None:
            self._cri_dirty = torch.zeros(n, device=self.device, dtype=torch.bool)

        if env_ids is None or env_ids == slice(None):
            self._cri_dirty[:] = True
            return

        if isinstance(env_ids, slice):
            self._cri_dirty[env_ids] = True
            return

        ids = env_ids.reshape(-1).to(device=self.device, dtype=torch.long)
        if ids.numel() == 0:
            return
        self._cri_dirty[ids] = True

    def _clear_cri_dirty(self) -> None:
        if self._cri_dirty is not None:
            self._cri_dirty.zero_()

    def _store_cri_traj_snapshot(self, q_in: torch.Tensor, qd_in: torch.Tensor, cri_float: torch.Tensor) -> None:
        """Keep the first (q, qd, CRI) at this sim timestamp for trajectory CSV export."""
        if self._traj_cri_timestamp >= self._sim_timestamp:
            return
        if self._traj_q is None or self._traj_q.shape != q_in.shape:
            self._traj_q = q_in.detach().to(dtype=torch.float32).clone()
            self._traj_qd = qd_in.detach().to(dtype=torch.float32).clone()
            self._traj_cri = cri_float.detach().clone()
        else:
            self._traj_q.copy_(q_in.detach().to(dtype=torch.float32))
            self._traj_qd.copy_(qd_in.detach().to(dtype=torch.float32))
            self._traj_cri.copy_(cri_float.detach())
        self._traj_cri_timestamp = self._sim_timestamp

    def get_cri_trajectory_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (q, qd, CRI) from the first CRI eval at the current physics step.

        Env reset may rewrite joint buffers after termination/reward CRI is computed. Trajectory
        export must use this snapshot so CSV rows stay consistent with the CRI that triggered
        OVF / logging, not the post-reset ``qd=0`` state.

        Hot-path CRI skips the snapshot unless ``SFD_CRI_TRAJ_SNAPSHOT=1``; this getter forces
        a snapshot so CSV export still works.
        """
        if self._traj_cri_timestamp < self._sim_timestamp or self._traj_cri is None:
            _ = self.CRI
            if self._traj_cri_timestamp < self._sim_timestamp or self._traj_cri is None:
                q_in, qd_in = self._cri_input_tensors()
                self._store_cri_traj_snapshot(q_in, qd_in, self._CRI_float.data)
        assert self._traj_q is not None and self._traj_qd is not None and self._traj_cri is not None
        return self._traj_q, self._traj_qd, self._traj_cri

    def _ensure_cri_cuda_events(self) -> None:
        if self._cri_cuda_start_evt is None:
            self._cri_cuda_start_evt = torch.cuda.Event(enable_timing=True)
            self._cri_cuda_end_evt = torch.cuda.Event(enable_timing=True)

    def _record_cri_inference_time(self, elapsed_s: float) -> None:
        self._cri_inference_time_s = elapsed_s
        self._cri_inference_count += 1
        self._cri_inference_time_total_s += elapsed_s
        self._cri_inference_time_min_s = min(self._cri_inference_time_min_s, elapsed_s)
        self._cri_inference_time_max_s = max(self._cri_inference_time_max_s, elapsed_s)
        self._cri_inference_samples_s.append(elapsed_s)
        # Print once per PPO collect (default 24 env steps) when SFD_CRI_TIMING=1.
        period = int(os.environ.get("SFD_CRI_TIMING_PRINT_EVERY", "24"))
        if period > 0 and self._cri_inference_count % period == 0:
            n = min(period, len(self._cri_inference_samples_s))
            window = self._cri_inference_samples_s[-n:]
            mean_ms = (sum(window) / n) * 1000.0
            last_ms = elapsed_s * 1000.0
            est_collect_s = mean_ms * 24.0 / 1000.0
            print(
                f"[CRI timing] n={self._cri_inference_count} "
                f"batch={self._cri_last_batch_rows} "
                f"last={last_ms:.2f}ms mean_last{n}={mean_ms:.2f}ms "
                f"est_24calls={est_collect_s:.3f}s",
                flush=True,
            )

    def _store_cri_output_buffers(self, cri_gpu: torch.Tensor | None) -> None:
        """Write solver output into preallocated CRI / float caches (no alloc on steady path)."""
        if cri_gpu is not None:
            if self._CRI.data is None or self._CRI.data.shape != cri_gpu.shape:
                self._CRI.data = torch.empty_like(cri_gpu)
            self._CRI.data.copy_(cri_gpu)
        else:
            self._CRI.data = cri_gpu
        if self._CRI.data is None:
            return
        if self._CRI_float.data is None or self._CRI_float.data.shape != self._CRI.data.shape:
            self._CRI_float.data = torch.clamp(self._CRI.data.float(), min=0.0, max=2.0)
        else:
            self._CRI_float.data.copy_(self._CRI.data)
            self._CRI_float.data.clamp_(min=0.0, max=2.0)

    def _invoke_cri_solver(self, q_in: torch.Tensor, qd_in: torch.Tensor) -> torch.Tensor | None:
        """Run CRI solver (+ optional timing / sync). Returns GPU CRI or None."""
        self._cri_last_batch_rows = int(q_in.shape[0])
        track_timing = os.environ.get("SFD_CRI_TIMING", "0") == "1"
        use_cuda = self.device.startswith("cuda")
        if track_timing and use_cuda and os.environ.get("SFD_CRI_TIMING_EVENTS", "0") == "1":
            self._ensure_cri_cuda_events()
            assert self._cri_cuda_start_evt is not None and self._cri_cuda_end_evt is not None
            self._cri_cuda_start_evt.record()
            cri_gpu = self.solver.RunSolver_CUDA_CRI_AtMotionState(q_in, qd_in)
            self._cri_cuda_end_evt.record()
            self._cri_cuda_end_evt.synchronize()
            self._record_cri_inference_time(self._cri_cuda_start_evt.elapsed_time(self._cri_cuda_end_evt) * 1e-3)
            return cri_gpu
        wall0 = time.perf_counter() if track_timing else 0.0
        cri_gpu = self.solver.RunSolver_CUDA_CRI_AtMotionState(q_in, qd_in)
        if use_cuda:
            torch.cuda.synchronize()
        if track_timing:
            self._record_cri_inference_time(time.perf_counter() - wall0)
        return cri_gpu

    def _index_copy_cri_rows(self, env_ids: torch.Tensor, cri_rows: torch.Tensor) -> None:
        """Scatter solved CRI rows into the full-env cache (raw + clamped float)."""
        assert self._CRI.data is not None
        self._CRI.data.index_copy_(0, env_ids, cri_rows)
        cri_f = torch.clamp(cri_rows.float(), min=0.0, max=2.0)
        if self._CRI_float.data is None or self._CRI_float.data.shape != self._CRI.data.shape:
            self._CRI_float.data = torch.clamp(self._CRI.data.float(), min=0.0, max=2.0)
        else:
            self._CRI_float.data.index_copy_(0, env_ids, cri_f)

    def _cri_pose_cache_key(self, q_row: torch.Tensor, qd_row: torch.Tensor) -> tuple:
        """Stable CPU key for a single-robot (q, qd) pose (micro-rad / micro-rad/s)."""
        q_i = torch.round(q_row.detach().reshape(-1).float() * 1e6).to(dtype=torch.int64).cpu().tolist()
        qd_i = torch.round(qd_row.detach().reshape(-1).float() * 1e6).to(dtype=torch.int64).cpu().tolist()
        return (tuple(q_i), tuple(qd_i))

    def _cri_row_from_full_n_batch(self, q_row: torch.Tensor, qd_row: torch.Tensor) -> torch.Tensor | None:
        """Return one CRI row solved at full env count (TRT batch = num_envs), with caching.

        TensorRT paths are batch-size sensitive: a batch=1 solve is not bit-identical to the
        corresponding row of a batch=num_envs solve. Cache entries are always produced with
        a full-N solve so post-reset obs match the previous double-full-solve semantics.
        """
        key = self._cri_pose_cache_key(q_row, qd_row)
        cached = self._cri_nbatch_row_cache.get(key)
        if cached is not None:
            self._cri_last_batch_rows = 0  # cache hit: no solver
            return cached

        n = self._root_physx_view.count
        q_fill = q_row.reshape(1, -1).expand(n, -1).contiguous()
        qd_fill = qd_row.reshape(1, -1).expand(n, -1).contiguous()
        cri_full = self._invoke_cri_solver(q_fill, qd_fill)
        if cri_full is None:
            return None
        row = cri_full[0].detach().clone()
        self._cri_nbatch_row_cache[key] = row
        return row

    def _refresh_dirty_cri_rows(self) -> None:
        """Recompute CRI only for envs marked dirty after a mid-step joint write / reset.

        Identical-pose dirty sets (P2P home reset) use a cached full-N TRT row then scatter.
        Heterogeneous dirty sets re-solve the full mixed ``(q, qd)`` batch (same as the old
        post-reset full recompute).
        """
        assert self._cri_dirty is not None and self._CRI.data is not None
        ids = self._cri_dirty.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() == 0:
            return
        q_in, qd_in = self._cri_input_tensors()
        q_d = q_in.index_select(0, ids)
        qd_d = qd_in.index_select(0, ids)
        if ids.numel() > 1 and torch.allclose(q_d, q_d[0:1]) and torch.allclose(qd_d, qd_d[0:1]):
            row = self._cri_row_from_full_n_batch(q_d[0], qd_d[0])
            if row is None:
                self._CRI.timestamp = -1.0
                self._clear_cri_dirty()
                return
            cri_rows = row.unsqueeze(0).expand(ids.numel(), -1).contiguous()
            self._index_copy_cri_rows(ids, cri_rows)
            self._clear_cri_dirty()
            return

        # Mixed / heterogeneous resets: match legacy full re-solve on current buffers.
        cri_gpu = self._invoke_cri_solver(q_in, qd_in)
        if cri_gpu is None:
            self._CRI.timestamp = -1.0
            self._clear_cri_dirty()
            return
        self._store_cri_output_buffers(cri_gpu)
        self._clear_cri_dirty()

    def _recompute_cri_full(self) -> None:
        """Full-batch CRI for a new physics timestamp."""
        q_in, qd_in = self._cri_input_tensors()
        cri_gpu = self._invoke_cri_solver(q_in, qd_in)
        self._store_cri_output_buffers(cri_gpu)
        if os.environ.get("SFD_CRI_TRAJ_SNAPSHOT", "0") == "1":
            self._store_cri_traj_snapshot(q_in, qd_in, self._CRI_float.data)
        self._CRI.timestamp = self._sim_timestamp
        self._clear_cri_dirty()

    def run_cri_at_motion_state_hot(self, *, record_timing: bool = False) -> torch.Tensor:
        """Native-like CRI call: solver + one ``cuda.synchronize`` (SFD_CoreService_Test hot-loop).

        Skips CUDA Event sandwich used by :attr:`CRI` when ``SFD_CRI_TIMING=1``. Always
        recomputes (ignores cache) so callers can pair with :meth:`_invalidate_cri_cache`
        for discard-flush patterns.
        """
        q_in, qd_in = self._cri_input_tensors()
        wall0 = time.perf_counter() if record_timing else 0.0
        cri_gpu = self.solver.RunSolver_CUDA_CRI_AtMotionState(q_in, qd_in)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        if record_timing:
            self._record_cri_inference_time(time.perf_counter() - wall0)
        self._store_cri_output_buffers(cri_gpu)
        if os.environ.get("SFD_CRI_TRAJ_SNAPSHOT", "0") == "1":
            self._store_cri_traj_snapshot(q_in, qd_in, self._CRI_float.data)
        self._CRI.timestamp = self._sim_timestamp
        self._clear_cri_dirty()
        return self._CRI_float.data

    def invalidate_cri_cache(self) -> None:
        """Public full-cache wipe for :meth:`_invalidate_cri_cache` (bench discard-flush)."""
        self._CRI.timestamp = -1.0
        self._clear_cri_dirty()

    @property
    def CRI(self):
        """Collision Risk Index from Safetics CRI solver. Shape is (num_instances, num_collision_points)."""

        if self._CRI.timestamp < self._sim_timestamp:
            self._recompute_cri_full()
        elif self._cri_dirty is not None and self._cri_dirty.any().item():
            # Same physics step after env reset: refresh only reset rows at post-reset (q, qd).
            self._refresh_dirty_cri_rows()
            # If refresh fell back to invalidation, recompute full once.
            if self._CRI.timestamp < self._sim_timestamp:
                self._recompute_cri_full()
        return self._CRI_float.data

    @property
    def cri_last_inference_time_s(self) -> float:
        """Wall-clock time of the most recent CRI solver call (seconds)."""
        return self._cri_inference_time_s

    def get_cri_inference_stats(self) -> dict[str, float | int]:
        """Return aggregated CRI solver inference timing statistics."""
        count = self._cri_inference_count
        out: dict[str, float | int] = {
            "count": count,
            "last_s": self._cri_inference_time_s,
            "mean_s": self._cri_inference_time_total_s / count if count else 0.0,
            "min_s": self._cri_inference_time_min_s if count else 0.0,
            "max_s": self._cri_inference_time_max_s if count else 0.0,
            "total_s": self._cri_inference_time_total_s,
        }
        if count and self._cri_inference_samples_s:
            import numpy as np

            arr = np.asarray(self._cri_inference_samples_s, dtype=np.float64)
            out["p95_s"] = float(np.percentile(arr, 95))
            out["p99_s"] = float(np.percentile(arr, 99))
        return out

    def cri_inference_samples_ms(self) -> list[float]:
        """Per-invocation CRI wall times in milliseconds (when SFD_CRI_TIMING=1)."""
        return [s * 1000.0 for s in self._cri_inference_samples_s]

    def reset_cri_inference_stats(self) -> None:
        """Reset accumulated CRI solver inference timing statistics."""
        self._cri_inference_time_s = 0.0
        self._cri_inference_count = 0
        self._cri_inference_time_total_s = 0.0
        self._cri_inference_time_min_s = float("inf")
        self._cri_inference_time_max_s = 0.0
        self._cri_inference_samples_s.clear()

    ##
    # Backward compatibility.
    ##

    @property
    def root_pose_w(self) -> torch.Tensor:
        """Same as :attr:`root_link_pose_w`."""
        return self.root_link_pose_w

    @property
    def root_pos_w(self) -> torch.Tensor:
        """Same as :attr:`root_link_pos_w`."""
        return self.root_link_pos_w

    @property
    def root_quat_w(self) -> torch.Tensor:
        """Same as :attr:`root_link_quat_w`."""
        return self.root_link_quat_w

    @property
    def root_vel_w(self) -> torch.Tensor:
        """Same as :attr:`root_com_vel_w`."""
        return self.root_com_vel_w

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        """Same as :attr:`root_com_lin_vel_w`."""
        return self.root_com_lin_vel_w

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        """Same as :attr:`root_com_ang_vel_w`."""
        return self.root_com_ang_vel_w

    @property
    def root_lin_vel_b(self) -> torch.Tensor:
        """Same as :attr:`root_com_lin_vel_b`."""
        return self.root_com_lin_vel_b

    @property
    def root_ang_vel_b(self) -> torch.Tensor:
        """Same as :attr:`root_com_ang_vel_b`."""
        return self.root_com_ang_vel_b

    @property
    def body_pose_w(self) -> torch.Tensor:
        """Same as :attr:`body_link_pose_w`."""
        return self.body_link_pose_w

    @property
    def body_pos_w(self) -> torch.Tensor:
        """Same as :attr:`body_link_pos_w`."""
        return self.body_link_pos_w

    @property
    def body_quat_w(self) -> torch.Tensor:
        """Same as :attr:`body_link_quat_w`."""
        return self.body_link_quat_w

    @property
    def body_vel_w(self) -> torch.Tensor:
        """Same as :attr:`body_com_vel_w`."""
        return self.body_com_vel_w

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """Same as :attr:`body_com_lin_vel_w`."""
        return self.body_com_lin_vel_w

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """Same as :attr:`body_com_ang_vel_w`."""
        return self.body_com_ang_vel_w

    @property
    def body_acc_w(self) -> torch.Tensor:
        """Same as :attr:`body_com_acc_w`."""
        return self.body_com_acc_w

    @property
    def body_lin_acc_w(self) -> torch.Tensor:
        """Same as :attr:`body_com_lin_acc_w`."""
        return self.body_com_lin_acc_w

    @property
    def body_ang_acc_w(self) -> torch.Tensor:
        """Same as :attr:`body_com_ang_acc_w`."""
        return self.body_com_ang_acc_w

    @property
    def com_pos_b(self) -> torch.Tensor:
        """Same as :attr:`body_com_pos_b`."""
        return self.body_com_pos_b

    @property
    def com_quat_b(self) -> torch.Tensor:
        """Same as :attr:`body_com_quat_b`."""
        return self.body_com_quat_b

    @property
    def joint_limits(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`joint_pos_limits` instead."""
        logger.warning(
            "The `joint_limits` property will be deprecated in a future release. Please use `joint_pos_limits` instead."
        )
        return self.joint_pos_limits

    @property
    def default_joint_limits(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`default_joint_pos_limits` instead."""
        logger.warning(
            "The `default_joint_limits` property will be deprecated in a future release. Please use"
            " `default_joint_pos_limits` instead."
        )
        return self.default_joint_pos_limits

    @property
    def joint_velocity_limits(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`joint_vel_limits` instead."""
        logger.warning(
            "The `joint_velocity_limits` property will be deprecated in a future release. Please use"
            " `joint_vel_limits` instead."
        )
        return self.joint_vel_limits

    @property
    def joint_friction(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`joint_friction_coeff` instead."""
        logger.warning(
            "The `joint_friction` property will be deprecated in a future release. Please use"
            " `joint_friction_coeff` instead."
        )
        return self.joint_friction_coeff

    @property
    def default_joint_friction(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`default_joint_friction_coeff` instead."""
        logger.warning(
            "The `default_joint_friction` property will be deprecated in a future release. Please use"
            " `default_joint_friction_coeff` instead."
        )
        return self.default_joint_friction_coeff

    @property
    def fixed_tendon_limit(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`fixed_tendon_pos_limits` instead."""
        logger.warning(
            "The `fixed_tendon_limit` property will be deprecated in a future release. Please use"
            " `fixed_tendon_pos_limits` instead."
        )
        return self.fixed_tendon_pos_limits

    @property
    def default_fixed_tendon_limit(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`default_fixed_tendon_pos_limits` instead."""
        logger.warning(
            "The `default_fixed_tendon_limit` property will be deprecated in a future release. Please use"
            " `default_fixed_tendon_pos_limits` instead."
        )
        return self.default_fixed_tendon_pos_limits
