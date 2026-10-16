# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play ``Isaac-Reach-UR10-Play-v0`` with Lula cubic-spline C-space trajectories (not an RL policy).

Exports the same CSV schema as :mod:`play_reach_csv`:

* ``env_<id>_traj.csv`` — ``global_step``, ``sim_time_s``, ``reach_event``, ``max_CRI``,
  ``cmd_{x,y,z,qw,qx,qy,qz}`` (ee_pose in robot base frame), ``q_*``, ``qd_*``, ``CRI_<i>``
  (q/qd/CRI from the first CRI eval at each env step — pre-reset if the env terminated)
* ``episode_reach.csv`` — per-attempt ``reached`` / ``outcome`` / command pose
* ``reach_summary.csv`` — aggregate reach rate

The cubic is logged until ``traj_duration`` (scaled by playback speed). A reach gate does not
end the log. The CRI rewind event is removed so the cubic plays through. CRI is still
computed and written to the trajectory CSV. ``reach_event`` remains a per-step column.

Prerequisites: Isaac Sim extensions ``isaacsim.robot_motion.lula`` and
``isaacsim.robot_motion.motion_generation``.

Example::

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_lula_reach_csv.py \\
        --task Isaac-Reach-UR10-Play-v0 --num_envs 4 \\
        --traj_duration 4.0 --traj_playback_speed 1.0 \\
        --export_csv_dir /home/safetics/Downloads/lula_reach \\
        --headless
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import IO, Any

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Lula C-spline reach play with trajectory CSV export.")
parser.add_argument("--video", action="store_true", default=False, help="Record one rgb_array video clip.")
parser.add_argument("--video_length", type=int, default=200, help="Video length in env steps when --video is set.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel envs (each gets its own Lula plan).")
parser.add_argument("--task", type=str, default="Isaac-Reach-UR10-Play-v0", help="Gym task id.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run close to real time if possible.")
parser.add_argument(
    "--traj_duration",
    type=float,
    default=4.0,
    help="Planned duration (seconds) for the cubic-spline segment from start config to IK goal.",
)
parser.add_argument(
    "--traj_playback_speed",
    type=float,
    default=1.0,
    help="Spline time advance per env step as a multiple of step_dt (>1 faster, <1 slower).",
)
parser.add_argument(
    "--export_csv_dir",
    type=str,
    default=None,
    help="Output directory for env_*_traj.csv (default: logs/lula_reach_csv/<task>/).",
)
parser.add_argument(
    "--export_csv_always",
    action="store_true",
    default=False,
    help="Keep logging after first strict reach (default: stop per env at reach).",
)
parser.add_argument(
    "--keep_resample_on_reach",
    action="store_true",
    default=False,
    help="Keep mid-episode target resample (default: disabled for P2P-style one-target attempts).",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Exit after this many env steps (0 = run until all envs finish one attempt).",
)
parser.add_argument(
    "--no_ovf",
    action="store_true",
    default=False,
    help="Disable OVF termination so attempts can run to timeout (PLAY defaults to CRI 0.96).",
)
parser.add_argument(
    "--ovf_threshold",
    type=float,
    default=0.96,
    help="CRI OVF termination threshold (default 0.96). Ignored if --no_ovf.",
)
parser.add_argument(
    "--gate_factor",
    type=float,
    default=1.0,
    help="Scale strict reach gates (distance, angle, twist) for eval. "
    "Use 2.0 for Jawale cooled gates (2 cm / 0.1 rad / twist < 0.02) from 1× cfg bases.",
)
parser.add_argument(
    "--keep_reach_success_term",
    action="store_true",
    default=False,
    help="Keep terminations.reach_success (success resets the env). "
    "Default disables it so finished envs are not reset mid-eval.",
)
parser.add_argument(
    "--ik_pos_tol",
    type=float,
    default=1.0e-3,
    help="Accept IK only if Isaac FK position error is below this [m].",
)
parser.add_argument(
    "--ik_ang_tol",
    type=float,
    default=5.0e-3,
    help="Accept IK only if Isaac FK orientation error is below this [rad].",
)
parser.add_argument(
    "--ik_retries",
    type=int,
    default=12,
    help="Extra Lula IK warm-starts (plus current q) before Isaac DLS refine.",
)
parser.add_argument(
    "--ik_refine_iters",
    type=int,
    default=40,
    help="Isaac-world DLS iterations to drive FK(q_goal) to the command pose.",
)
parser.add_argument(
    "--from_traj",
    nargs="+",
    default=None,
    help="Replay these env_*_traj.csv cubics at --traj_duration. Same q endpoints, new time scale.",
)
parser.add_argument(
    "--dump_reset_env",
    type=int,
    default=None,
    help="After reset, write this env's joint position and ee command, then exit.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Post-launch imports (Isaac Sim / Lula)
import carb  # noqa: E402
import gymnasium as gym  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension, get_extension_path_from_name  # noqa: E402

enable_extension("isaacsim.robot_motion.lula")
enable_extension("isaacsim.robot_motion.motion_generation")

import isaacsim.robot_motion.motion_generation.lula  # noqa: F401, E402
from isaacsim.robot_motion.motion_generation import (  # noqa: E402
    LulaCSpaceTrajectoryGenerator,
    LulaKinematicsSolver,
)

import isaaclab_tasks  # noqa: F401, E402
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    combine_frame_transforms,
    compute_pose_error,
    quat_error_magnitude,
    quat_mul,
)
from isaaclab.utils.seed import configure_seed  # noqa: E402
from isaaclab_tasks.manager_based.manipulation.reach.mdp.observations import command_origin_pose_w  # noqa: E402
from isaaclab_tasks.manager_based.manipulation.reach.mdp.rewards import _body_idx_single  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

# Shared CSV helpers (same schema as play_reach_csv.py)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import reach_traj_csv_utils as csv_utils  # noqa: E402


def _resolve_ur10_lula_asset_paths() -> tuple[str, str]:
    """Return (robot_description_yaml, urdf) for UR10 inside the motion_generation extension."""
    mg = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
    base = os.path.join(mg, "motion_policy_configs")
    candidates = [
        (
            os.path.join(base, "ur10", "rmpflow", "ur10_robot_description.yaml"),
            os.path.join(base, "ur10", "ur10_robot.urdf"),
        ),
        (
            os.path.join(base, "universal_robots", "ur10", "rmpflow", "ur10_robot_description.yaml"),
            os.path.join(base, "universal_robots", "ur10", "ur10_robot.urdf"),
        ),
    ]
    for desc, urdf in candidates:
        if os.path.isfile(desc) and os.path.isfile(urdf):
            return desc, urdf
    raise FileNotFoundError(
        "Could not find UR10 Lula robot description / URDF under motion_policy_configs. "
        f"Tried: {candidates}"
    )


def _joint_vec_in_names_order(joint_pos: torch.Tensor, joint_names: list[str], subset_names: list[str]) -> np.ndarray:
    name_to_i = {n: i for i, n in enumerate(joint_names)}
    out = np.zeros(len(subset_names), dtype=np.float64)
    for k, name in enumerate(subset_names):
        out[k] = float(joint_pos[name_to_i[name]].item())
    return out


def _normalize_quat_wxyz(q: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    v = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    v = v / n
    if v[0] < 0.0:
        v = -v
    return v


def _positions_to_manager_action(arm_term, q_full: torch.Tensor) -> torch.Tensor:
    """Map absolute joint positions to JointPositionAction raw actions."""
    joint_ids = arm_term._joint_ids
    q = q_full[:, joint_ids] if not isinstance(joint_ids, slice) else q_full
    scale = arm_term._scale
    offset = arm_term._offset
    if isinstance(scale, torch.Tensor):
        raw = (q - offset) / scale
    else:
        raw = (q - offset) / float(scale)
    return raw


def _ee_tracking(base_env: ManagerBasedRLEnv, command: torch.Tensor) -> tuple[torch.Tensor, ...]:
    asset_cfg = SceneEntityCfg("robot", body_names="ee_link")
    asset_cfg.resolve(base_env.scene)
    asset = base_env.scene[asset_cfg.name]
    bid = _body_idx_single(base_env, asset_cfg)
    origin_pos_w, origin_quat_w = command_origin_pose_w(base_env, "ee_pose", asset)
    des_pos_w, _ = combine_frame_transforms(origin_pos_w, origin_quat_w, command[:, :3])
    pos_err = torch.norm(asset.data.body_pos_w[:, bid] - des_pos_w, dim=1)
    des_quat_w = quat_mul(origin_quat_w, command[:, 3:7])
    ang_err = quat_error_magnitude(asset.data.body_quat_w[:, bid], des_quat_w)
    lin_vel = torch.norm(asset.data.body_lin_vel_w[:, bid, :], dim=-1)
    ang_vel = torch.norm(asset.data.body_ang_vel_w[:, bid, :], dim=-1)
    cri = torch.max(asset.data.CRI, dim=1).values
    return pos_err, ang_err, lin_vel, ang_vel, cri


def _attempt_extras(i: int, *bufs: torch.Tensor) -> dict[str, float]:
    keys = (
        "min_pos_err",
        "min_ang_err",
        "last_pos_err",
        "last_ang_err",
        "last_lin_vel",
        "last_ang_vel",
        "max_cri",
        "last_cri",
    )
    return {key: float(buf[i].item()) for key, buf in zip(keys, bufs)}


class LulaCubicSplineReachDriver:
    """C-space rest-to-rest cubic spline with Isaac-verified IK and exact q(t) playback."""

    def __init__(
        self,
        traj_duration: float,
        num_envs: int,
        playback_speed: float = 1.0,
        ik_pos_tol: float = 1.0e-3,
        ik_ang_tol: float = 5.0e-3,
        ik_retries: int = 12,
        ik_refine_iters: int = 40,
    ):
        desc, urdf = _resolve_ur10_lula_asset_paths()
        self._cgen = LulaCSpaceTrajectoryGenerator(robot_description_path=desc, urdf_path=urdf)
        self._ik = LulaKinematicsSolver(robot_description_path=desc, urdf_path=urdf)
        self._active_joint_names: list[str] = list(self._cgen.get_active_joints())
        lo, hi = self._cgen.get_c_space_position_limits()
        self._cspace_lo = np.asarray(lo, dtype=np.float64).reshape(-1)
        self._cspace_hi = np.asarray(hi, dtype=np.float64).reshape(-1)
        self._ee_frame = "ee_link"
        self._traj_duration = float(traj_duration)
        ps = float(playback_speed)
        if ps <= 0.0:
            raise ValueError(f"traj_playback_speed must be > 0, got {playback_speed}.")
        self._playback_speed = ps
        self._num_envs = int(num_envs)
        self._ik_pos_tol = float(ik_pos_tol)
        self._ik_ang_tol = float(ik_ang_tol)
        self._ik_retries = max(0, int(ik_retries))
        self._ik_refine_iters = max(1, int(ik_refine_iters))
        self._q_start = np.zeros((self._num_envs, len(self._active_joint_names)), dtype=np.float64)
        self._q_goal = np.zeros_like(self._q_start)
        self._plan_ok = np.zeros(self._num_envs, dtype=bool)
        self._t = np.zeros(self._num_envs, dtype=np.float64)
        self._last_cc: list[int | None] = [None] * self._num_envs
        self._fk_pos_err = np.full(self._num_envs, np.nan, dtype=np.float64)
        self._fk_ang_err = np.full(self._num_envs, np.nan, dtype=np.float64)
        self._robot_joint_ids: list[int] | None = None
        self.n_ik_ok = 0
        self.n_spline_ok = 0
        self.n_taskspace_fallback = 0
        self.n_plan_fail = 0

    @property
    def active_joint_names(self) -> list[str]:
        return self._active_joint_names

    def trajectory_active(self, env_idx: int) -> bool:
        return bool(self._plan_ok[env_idx])

    def _sync_lula_robot_base(self, robot, env_idx: int) -> None:
        pos = robot.data.root_pos_w[env_idx, :3].detach().cpu().numpy().astype(np.float64)
        quat = _normalize_quat_wxyz(robot.data.root_quat_w[env_idx].detach().cpu().numpy())
        self._ik.set_robot_base_pose(pos, quat)

    def _clamp_cspace(self, q: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(q, dtype=np.float64).reshape(-1), self._cspace_lo, self._cspace_hi)

    def _resolve_robot_joint_ids(self, robot) -> list[int]:
        if self._robot_joint_ids is None:
            ids, names = robot.find_joints(self._active_joint_names, preserve_order=True)
            if len(ids) != len(self._active_joint_names):
                raise RuntimeError(f"Joint map failed: {self._active_joint_names} -> {names}")
            self._robot_joint_ids = list(ids)
        return self._robot_joint_ids

    def _lula_ik_candidates(self, q_start: np.ndarray, pos_np: np.ndarray, quat_np: np.ndarray) -> list[np.ndarray]:
        warms = [q_start.copy()]
        rng = np.random.default_rng(0)
        for _ in range(self._ik_retries):
            warms.append(rng.uniform(self._cspace_lo, self._cspace_hi))
        out: list[np.ndarray] = []
        seen: set[tuple[float, ...]] = set()
        for warm in warms:
            q_goal, ok = self._ik.compute_inverse_kinematics(
                self._ee_frame,
                pos_np,
                quat_np,
                warm_start=warm,
            )
            if not ok:
                continue
            qg = self._clamp_cspace(np.asarray(q_goal, dtype=np.float64))
            key = tuple(np.round(qg, 6))
            if key in seen:
                continue
            seen.add(key)
            out.append(qg)
        return out

    def seed_lula_ik(self, env: ManagerBasedRLEnv, env_idx: int, robot) -> None:
        """Store q_start and a Lula IK seed. Isaac FK refine happens in ``finalize_plans``."""
        self._t[env_idx] = 0.0
        self._plan_ok[env_idx] = False
        self._sync_lula_robot_base(robot, env_idx)
        ee_term = env.command_manager.get_term("ee_pose")
        pose_w = ee_term.pose_command_w[env_idx]
        q_start = _joint_vec_in_names_order(
            robot.data.joint_pos[env_idx].cpu(), robot.joint_names, self._active_joint_names
        )
        self._q_start[env_idx] = q_start
        self._q_goal[env_idx] = q_start
        self._last_cc[env_idx] = int(ee_term.command_counter[env_idx].item())
        pos_np = pose_w[:3].detach().cpu().numpy()
        quat_np = _normalize_quat_wxyz(pose_w[3:7].detach().cpu().numpy())
        cands = self._lula_ik_candidates(q_start, pos_np, quat_np)
        if cands:
            self.n_ik_ok += 1
            self._q_goal[env_idx] = cands[0]

    def _refresh_kin(self, env: ManagerBasedRLEnv, robot) -> None:
        robot.write_data_to_sim()
        env.sim.forward()
        robot.update(0.0)

    def finalize_plans(self, env: ManagerBasedRLEnv, robot) -> None:
        """Isaac FK + DLS so FK(q_goal)=target, then accept rest-to-rest cubics only."""
        device = robot.data.joint_pos.device
        n = self._num_envs
        joint_ids = self._resolve_robot_joint_ids(robot)
        ee_idx = int(robot.find_bodies(self._ee_frame)[0][0])
        jac_body = ee_idx - 1 if robot.is_fixed_base else ee_idx
        q_full = robot.data.joint_pos.clone()
        q_start_full = q_full.clone()
        qd_zero = torch.zeros_like(q_full)
        q_act = torch.as_tensor(self._q_goal, device=device, dtype=q_full.dtype)
        q_full[:, joint_ids] = q_act
        robot.write_joint_state_to_sim(q_full, qd_zero)
        self._refresh_kin(env, robot)

        diff_ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            num_envs=n,
            device=str(device),
        )
        cmd = env.command_manager.get_term("ee_pose").pose_command_w[:, :7].clone()
        diff_ik.set_command(cmd)

        ok = torch.zeros(n, dtype=torch.bool, device=device)
        last_pos = torch.zeros(n, device=device)
        last_ang = torch.zeros(n, device=device)
        for _ in range(self._ik_refine_iters):
            ee_pos = robot.data.body_pos_w[:, ee_idx]
            ee_quat = robot.data.body_quat_w[:, ee_idx]
            pos_e, ang_e = compute_pose_error(ee_pos, ee_quat, cmd[:, :3], cmd[:, 3:7], rot_error_type="quat")
            last_pos = torch.norm(pos_e, dim=-1)
            last_ang = quat_error_magnitude(ee_quat, cmd[:, 3:7])
            ok = (last_pos <= self._ik_pos_tol) & (last_ang <= self._ik_ang_tol)
            if bool(ok.all().item()):
                break
            jac = robot.root_physx_view.get_jacobians()[:, jac_body, :, joint_ids]
            q_new = diff_ik.compute(ee_pos, ee_quat, jac, q_full[:, joint_ids])
            q_full[:, joint_ids] = q_new
            lo = torch.as_tensor(self._cspace_lo, device=device, dtype=q_full.dtype)
            hi = torch.as_tensor(self._cspace_hi, device=device, dtype=q_full.dtype)
            q_full[:, joint_ids] = torch.minimum(torch.maximum(q_full[:, joint_ids], lo), hi)
            robot.write_joint_state_to_sim(q_full, qd_zero)
            self._refresh_kin(env, robot)

        ee_pos = robot.data.body_pos_w[:, ee_idx]
        ee_quat = robot.data.body_quat_w[:, ee_idx]
        pos_e, _ = compute_pose_error(ee_pos, ee_quat, cmd[:, :3], cmd[:, 3:7], rot_error_type="quat")
        last_pos = torch.norm(pos_e, dim=-1)
        last_ang = quat_error_magnitude(ee_quat, cmd[:, 3:7])
        ok = (last_pos <= self._ik_pos_tol) & (last_ang <= self._ik_ang_tol)

        self._q_goal = q_full[:, joint_ids].detach().cpu().numpy().astype(np.float64)
        self._fk_pos_err = last_pos.detach().cpu().numpy()
        self._fk_ang_err = last_ang.detach().cpu().numpy()
        self._plan_ok = ok.detach().cpu().numpy().astype(bool)
        self.n_spline_ok = int(self._plan_ok.sum())
        self.n_plan_fail = int((~self._plan_ok).sum())

        robot.write_joint_state_to_sim(q_start_full, qd_zero)
        self._refresh_kin(env, robot)

    def write_plan_stats(self, path: str) -> None:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["env_idx", "plan_ok", "fk_pos_err", "fk_ang_err"],
            )
            writer.writeheader()
            for i in range(self._num_envs):
                writer.writerow(
                    {
                        "env_idx": i,
                        "plan_ok": int(self._plan_ok[i]),
                        "fk_pos_err": float(self._fk_pos_err[i]),
                        "fk_ang_err": float(self._fk_ang_err[i]),
                    }
                )

    def on_env_reset(self, env_ids: torch.Tensor) -> None:
        for e in env_ids.flatten().tolist():
            if 0 <= e < self._num_envs:
                self._last_cc[e] = None
                self._plan_ok[e] = False
                self._t[e] = 0.0

    def need_replan(self, env: ManagerBasedRLEnv, env_idx: int) -> bool:
        ee_term = env.command_manager.get_term("ee_pose")
        cc = int(ee_term.command_counter[env_idx].item())
        last = self._last_cc[env_idx]
        return last is None or cc != last

    def eval_cubic(self) -> tuple[np.ndarray, np.ndarray]:
        """Rest-to-rest cubic: q(0)=q_start, q(T)=q_goal, qd(0)=qd(T)=0."""
        t = np.minimum(self._t, self._traj_duration)
        s = np.divide(t, self._traj_duration, out=np.ones_like(t), where=self._traj_duration > 0)
        s = np.clip(s, 0.0, 1.0)
        blend = (3.0 * s * s - 2.0 * s * s * s)[:, None]
        dblend = (6.0 * s - 6.0 * s * s) / self._traj_duration
        dq = self._q_goal - self._q_start
        q = self._q_start + dq * blend
        qd = dq * dblend[:, None]
        done = self._t >= self._traj_duration - 1e-12
        q[done] = self._q_goal[done]
        qd[done] = 0.0
        q[~self._plan_ok] = self._q_start[~self._plan_ok]
        qd[~self._plan_ok] = 0.0
        return q, qd

    def apply_q_of_t(
        self, env: ManagerBasedRLEnv, robot, arm_term, *, advance: bool = True
    ) -> torch.Tensor:
        """Write q(t), qd(t) to sim (exact tracking) and return matching position actions."""
        q, qd = self.eval_cubic()
        if advance:
            self._t += float(env.step_dt) * self._playback_speed
        joint_ids = self._resolve_robot_joint_ids(robot)
        q_full = robot.data.joint_pos.clone()
        qd_full = torch.zeros_like(q_full)
        q_t = torch.as_tensor(q, device=q_full.device, dtype=q_full.dtype)
        qd_t = torch.as_tensor(qd, device=q_full.device, dtype=q_full.dtype)
        q_full[:, joint_ids] = q_t
        qd_full[:, joint_ids] = qd_t
        robot.write_joint_state_to_sim(q_full, qd_full)
        robot.set_joint_position_target(q_full)
        robot.set_joint_velocity_target(qd_full)
        return _positions_to_manager_action(arm_term, q_full)


def _load_cubic_replay(path: str) -> tuple[int, list[str], np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit rest-to-rest cubic endpoints from an exported ``env_<id>_traj.csv``."""
    base_name = os.path.basename(path)
    parts = base_name.split("_")
    if len(parts) < 3 or parts[0] != "env" or not parts[1].isdigit():
        raise ValueError(f"Expected env_<id>_traj.csv, got {path}.")
    env_id = int(parts[1])
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        raise ValueError(f"{path} has too few samples to fit a cubic.")
    q_cols = [key for key in rows[0] if key.startswith("q_") and not key.startswith("qd_")]
    t = np.array([float(row["sim_time_s"]) for row in rows], dtype=np.float64)
    q = np.array([[float(row[col]) for col in q_cols] for row in rows], dtype=np.float64)
    duration = float(t[-1])
    if duration <= 0.0:
        raise ValueError(f"{path} ends at sim_time_s={duration}.")
    blend = np.clip(t / duration, 0.0, 1.0)
    blend = 3.0 * blend * blend - 2.0 * blend * blend * blend
    q0 = np.zeros(q.shape[1], dtype=np.float64)
    q_goal = np.zeros(q.shape[1], dtype=np.float64)
    for joint_idx in range(q.shape[1]):
        design = np.stack([1.0 - blend, blend], axis=1)
        coef, *_ = np.linalg.lstsq(design, q[:, joint_idx], rcond=None)
        q0[joint_idx], q_goal[joint_idx] = coef
    cmd = np.array(
        [float(rows[0][key]) for key in ("cmd_x", "cmd_y", "cmd_z", "cmd_qw", "cmd_qx", "cmd_qy", "cmd_qz")],
        dtype=np.float64,
    )
    joint_names = [col[2:] for col in q_cols]
    return env_id, joint_names, q0, q_goal, cmd, duration


def main():
    replay = None
    if args_cli.from_traj:
        replay = [_load_cubic_replay(path) for path in args_cli.from_traj]
        args_cli.num_envs = len(replay)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        configure_seed(int(args_cli.seed), torch_deterministic=False)

    strict_reach_params = csv_utils.strict_reach_params_from_env_cfg(env_cfg)
    if strict_reach_params is None:
        raise ValueError(
            "Could not read strict reach_success_criteria params from env cfg "
            "(expected rewards.reach_success_bonus or events.resample_ee_pose_on_reach)."
        )
    gate_factor = float(args_cli.gate_factor)
    if gate_factor != 1.0:
        for key in ("max_distance", "max_angle_rad", "max_lin_vel", "max_ang_vel"):
            if key in strict_reach_params and strict_reach_params[key] is not None:
                strict_reach_params[key] = float(strict_reach_params[key]) * gate_factor

    disable_resample = not args_cli.keep_resample_on_reach
    csv_utils.configure_p2p_style_eval(
        env_cfg,
        disable_resample=disable_resample,
        freeze_finished_envs=not args_cli.keep_reach_success_term,
        disable_ovf=bool(args_cli.no_ovf),
        ovf_threshold=float(args_cli.ovf_threshold),
    )
    # Rewind teleports the arm to the spawn and the logged CRI becomes that pose.
    # Drop the event so q(t) stays on the cubic. CRI columns are unchanged.
    events_cfg = getattr(env_cfg, "events", None)
    if events_cfg is not None and getattr(events_cfg, "rewind_on_cri", None) is not None:
        events_cfg.rewind_on_cri = None
    env_cfg.actions.arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=False,
    )
    env_cfg.scene.robot.actuators["arm"] = ImplicitActuatorCfg(
        joint_names_expr=[".*"],
        velocity_limit=100.0,
        effort_limit=330.0,
        stiffness=1.0e8,
        damping=1.0e4,
    )

    export_csv_dir = args_cli.export_csv_dir or os.path.join(
        "logs", "lula_reach_csv", args_cli.task.replace(":", "_").replace("-", "_")
    )
    export_csv_dir = os.path.abspath(export_csv_dir)
    os.makedirs(export_csv_dir, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    base: ManagerBasedRLEnv = env.unwrapped
    if not isinstance(base, ManagerBasedRLEnv):
        raise RuntimeError(f"Expected ManagerBasedRLEnv, got {type(base)}")

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(export_csv_dir, "videos"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
        base = env.unwrapped

    csv_utils.apply_strict_reach_params_to_env(base, strict_reach_params)

    robot = base.scene["robot"]
    arm_term = base.action_manager.get_term("arm_action")
    num_envs = base.num_envs
    driver = LulaCubicSplineReachDriver(
        traj_duration=args_cli.traj_duration,
        num_envs=num_envs,
        playback_speed=args_cli.traj_playback_speed,
        ik_pos_tol=float(args_cli.ik_pos_tol),
        ik_ang_tol=float(args_cli.ik_ang_tol),
        ik_retries=int(args_cli.ik_retries),
        ik_refine_iters=int(args_cli.ik_refine_iters),
    )
    print(
        "[INFO] C-space cubic spline q(t): "
        f"traj_duration={args_cli.traj_duration}s, playback_speed={args_cli.traj_playback_speed}, "
        f"ik_pos_tol={args_cli.ik_pos_tol:g} m, ik_ang_tol={args_cli.ik_ang_tol:g} rad"
    )
    print(f"[INFO] Trajectory CSV output: {export_csv_dir}/env_*_traj.csv")
    print(
        "[INFO] Log the full cubic through traj_duration. "
        "Reach does not stop logging or reset the arm."
    )
    print(f"[INFO] Mid-episode target resample: {'enabled' if args_cli.keep_resample_on_reach else 'disabled'}")
    print(
        "[INFO] Strict reach thresholds: "
        f"max_distance={strict_reach_params['max_distance']}, "
        f"max_angle_rad={strict_reach_params['max_angle_rad']}, "
        f"max_lin_vel={strict_reach_params['max_lin_vel']}, "
        f"max_ang_vel={strict_reach_params.get('max_ang_vel', 'n/a')}, "
        f"max_lin_acc={strict_reach_params.get('max_lin_acc', float('inf'))}, "
        f"max_ang_acc={strict_reach_params.get('max_ang_acc', float('inf'))}, "
        f"gate_factor={gate_factor:g}"
    )
    if args_cli.no_ovf:
        print("[INFO] OVF termination: disabled (--no_ovf)")
    else:
        print(f"[INFO] OVF termination: enabled (threshold {float(args_cli.ovf_threshold):g})")
    print("[INFO] CRI rewind: disabled. The cubic plays through, and max_CRI / CRI_* stay in the CSV.")
    print(
        "[INFO] Disable reach_success reset: "
        f"{'no (--keep_reach_success_term)' if args_cli.keep_reach_success_term else 'yes'}"
    )

    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        _obs = reset_out[0]
    else:
        _obs = reset_out
    # pose_command_w can be zeros right after reset; refresh commands.
    base.command_manager.compute(0.0)

    if args_cli.dump_reset_env is not None:
        env_id = int(args_cli.dump_reset_env)
        ee_term = base.command_manager.get_term("ee_pose")
        names = list(robot.joint_names)
        q = robot.data.joint_pos[env_id].detach().cpu().tolist()
        cmd = ee_term.pose_command_b[env_id].detach().cpu().tolist()
        dump_path = os.path.join(export_csv_dir, f"reset_env_{env_id}.csv")
        with open(dump_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["name", "value"])
            for name, value in zip(names, q, strict=True):
                writer.writerow([f"q_{name}", value])
            for key, value in zip(
                ("cmd_x", "cmd_y", "cmd_z", "cmd_qw", "cmd_qx", "cmd_qy", "cmd_qz"), cmd, strict=True
            ):
                writer.writerow([key, value])
        print(f"[INFO] Dumped reset env {env_id} to {dump_path}")
        simulation_app.close()
        return

    replay_env_ids: list[int] | None = None
    if replay is None:
        t_plan = time.time()
        for e in range(num_envs):
            driver.seed_lula_ik(base, e, robot)
            if (e + 1) % 256 == 0 or (e + 1) == num_envs:
                print(f"[INFO] Lula IK seeds {e + 1}/{num_envs} (lula_ok={driver.n_ik_ok})")
        print("[INFO] Refining q_goal with Isaac FK / DLS ...")
        driver.finalize_plans(base, robot)
        driver.write_plan_stats(os.path.join(export_csv_dir, "plan_stats.csv"))
        print(
            f"[INFO] Plans: spline_ok={driver.n_spline_ok}, fail={driver.n_plan_fail}, "
            f"fk_pos p50={np.nanmedian(driver._fk_pos_err):.4f}, "
            f"fk_ang p50={np.nanmedian(driver._fk_ang_err):.4f}, "
            f"took {time.time() - t_plan:.1f}s"
        )
    else:
        replay_env_ids = []
        ee_term = base.command_manager.get_term("ee_pose")
        for slot, (env_id, joint_names, q0, q_goal, cmd, src_duration) in enumerate(replay):
            replay_env_ids.append(env_id)
            name_to_col = {name: col for col, name in enumerate(joint_names)}
            order = [name_to_col[name] for name in driver.active_joint_names]
            driver._q_start[slot] = q0[order]
            driver._q_goal[slot] = q_goal[order]
            driver._plan_ok[slot] = True
            driver._t[slot] = 0.0
            ee_term.pose_command_b[slot] = torch.as_tensor(
                cmd, device=ee_term.pose_command_b.device, dtype=ee_term.pose_command_b.dtype
            )
            print(
                f"[INFO] Replay env {env_id}: same cubic, {src_duration:.4f}s -> {args_cli.traj_duration:g}s "
                f"(time scale {src_duration / float(args_cli.traj_duration):.3f})."
            )
        driver.n_spline_ok = num_envs
        driver.n_plan_fail = 0
        driver._fk_pos_err[:] = 0.0
        driver._fk_ang_err[:] = 0.0
        driver.apply_q_of_t(base, robot, arm_term, advance=False)
        print(f"[INFO] Replaying {num_envs} stored cubics. IK is skipped.")

    dt = base.step_dt
    play_step = 0
    video_step = 0

    csv_files: dict[int, IO[str]] | None = None
    csv_writers: dict[int, csv.DictWriter] | None = None
    episode_reach_file: IO[str] | None = None
    episode_reach_writer: csv.DictWriter | None = None
    log_active: torch.Tensor | None = None
    attempt_active: torch.Tensor | None = None
    episode_ids: torch.Tensor | None = None
    total_episodes = 0
    total_reached_episodes = 0

    try:
        while simulation_app.is_running():
            loop_start = time.time()
            with torch.no_grad():
                reach_cmd_snapshot = (
                    base.command_manager.get_command(strict_reach_params["command_name"]).detach().clone()
                )

                action = driver.apply_q_of_t(base, robot, arm_term, advance=True)
                step_ret = env.step(action)
                # Snap again so logged (q, qd) is exactly the cubic, not a physics residual.
                driver.apply_q_of_t(base, robot, arm_term, advance=False)
                if len(step_ret) == 5:
                    _obs, _reward, reset_terminated, reset_time_outs, _extras = step_ret
                    episode_done = reset_terminated | reset_time_outs
                elif len(step_ret) == 4:
                    # Some wrappers collapse terminated/truncated.
                    _obs, _reward, episode_done, _extras = step_ret
                else:
                    raise RuntimeError(f"Unexpected env.step return length {len(step_ret)}.")

                play_step += 1
                base = env.unwrapped

                if csv_writers is None:
                    num_cri = int(robot.data.CRI.shape[1])
                    csv_files, csv_writers = csv_utils.open_traj_csv_writers(
                        export_csv_dir,
                        num_envs,
                        list(robot.joint_names),
                        num_cri,
                        log_actions=True,
                        action_dim=int(action.shape[-1]),
                        file_env_ids=replay_env_ids,
                    )
                    episode_reach_file, episode_reach_writer = csv_utils.open_episode_reach_csv(export_csv_dir)
                    log_active = torch.ones(num_envs, device=base.device, dtype=torch.bool)
                    attempt_active = torch.ones(num_envs, device=base.device, dtype=torch.bool)
                    episode_ids = torch.zeros(num_envs, device=base.device, dtype=torch.long)
                    min_pos_err = torch.full((num_envs,), float("inf"), device=base.device)
                    min_ang_err = torch.full((num_envs,), float("inf"), device=base.device)
                    max_cri = torch.zeros(num_envs, device=base.device)
                    ever_reached = torch.zeros(num_envs, device=base.device, dtype=torch.bool)
                    last_pos_err = torch.zeros(num_envs, device=base.device)
                    last_ang_err = torch.zeros(num_envs, device=base.device)
                    last_lin_vel = torch.zeros(num_envs, device=base.device)
                    last_ang_vel = torch.zeros(num_envs, device=base.device)
                    last_cri = torch.zeros(num_envs, device=base.device)

                assert (
                    csv_writers is not None
                    and log_active is not None
                    and attempt_active is not None
                    and episode_ids is not None
                    and episode_reach_writer is not None
                    and episode_reach_file is not None
                )

                reach_ev = csv_utils.resolve_reach_event(base, reach_cmd_snapshot, strict_reach_params)
                done_mask = episode_done.detach().bool().view(-1)
                pos_err, ang_err, lin_vel, ang_vel, cri = _ee_tracking(base, reach_cmd_snapshot)
                live = attempt_active
                min_pos_err[live] = torch.minimum(min_pos_err[live], pos_err[live])
                min_ang_err[live] = torch.minimum(min_ang_err[live], ang_err[live])
                max_cri[live] = torch.maximum(max_cri[live], cri[live])
                last_pos_err[live] = pos_err[live]
                last_ang_err[live] = ang_err[live]
                last_lin_vel[live] = lin_vel[live]
                last_ang_vel[live] = ang_vel[live]
                last_cri[live] = cri[live]

                csv_utils.append_traj_rows(
                    csv_writers,
                    robot,
                    play_step,
                    float(play_step) * dt,
                    log_active & attempt_active,
                    reach_ev,
                    command=reach_cmd_snapshot,
                    pos_err=pos_err,
                    ang_err=ang_err,
                    lin_spd=lin_vel,
                    ang_spd=ang_vel,
                    actions=action,
                )
                if csv_files is not None:
                    for handle in csv_files.values():
                        handle.flush()

                sim_time_s = float(play_step) * dt
                ever_reached |= attempt_active & reach_ev

                newly_fail = attempt_active & done_mask
                if bool(newly_fail.any().item()):
                    for env_idx in newly_fail.nonzero(as_tuple=False).view(-1).tolist():
                        outcome = csv_utils.classify_failure_outcome(base, int(env_idx))
                        logged_env = int(env_idx) if replay_env_ids is None else int(replay_env_ids[int(env_idx)])
                        csv_utils.record_attempt(
                            episode_reach_writer,
                            episode_reach_file,
                            logged_env,
                            int(episode_ids[env_idx].item()),
                            sim_time_s,
                            reached=False,
                            outcome=outcome,
                            command_pose=reach_cmd_snapshot[int(env_idx)],
                            extras=_attempt_extras(
                                int(env_idx),
                                min_pos_err,
                                min_ang_err,
                                last_pos_err,
                                last_ang_err,
                                last_lin_vel,
                                last_ang_vel,
                                max_cri,
                                last_cri,
                            ),
                        )
                    total_episodes += int(newly_fail.sum().item())
                    episode_ids[newly_fail] += 1
                    attempt_active &= ~newly_fail
                    log_active &= ~newly_fail

                    # Clear Lula caches for failed/reset envs (env already reset inside step).
                    fail_ids = newly_fail.nonzero(as_tuple=False).view(-1)
                    driver.on_env_reset(fail_ids)

                spline_done = float(np.max(driver._t)) >= float(args_cli.traj_duration) - 1e-8
                if spline_done and bool(attempt_active.any().item()):
                    finishing = attempt_active.clone()
                    for env_idx in finishing.nonzero(as_tuple=False).view(-1).tolist():
                        logged_env = int(env_idx) if replay_env_ids is None else int(replay_env_ids[int(env_idx)])
                        csv_utils.record_attempt(
                            episode_reach_writer,
                            episode_reach_file,
                            logged_env,
                            int(episode_ids[env_idx].item()),
                            sim_time_s,
                            reached=bool(ever_reached[env_idx].item()),
                            outcome="spline",
                            command_pose=reach_cmd_snapshot[int(env_idx)],
                            extras=_attempt_extras(
                                int(env_idx),
                                min_pos_err,
                                min_ang_err,
                                last_pos_err,
                                last_ang_err,
                                last_lin_vel,
                                last_ang_vel,
                                max_cri,
                                last_cri,
                            ),
                        )
                    total_episodes += int(finishing.sum().item())
                    total_reached_episodes += int(ever_reached[finishing].sum().item())
                    attempt_active &= ~finishing
                    print(
                        f"[INFO] Cubic logged through {sim_time_s:.4f}s "
                        f"(traj_duration={args_cli.traj_duration:g}s). "
                        "Reach did not end the trajectory."
                    )

                if not bool(attempt_active.any().item()):
                    pct = csv_utils.reach_percent(total_reached_episodes, total_episodes)
                    print(
                        f"[INFO] All envs finished 1 attempt; reach rate "
                        f"{total_reached_episodes}/{total_episodes} ({pct:.2f}%). Stopping."
                    )
                    break

            if args_cli.video:
                video_step += 1
                if video_step >= args_cli.video_length:
                    break

            if args_cli.max_steps > 0 and play_step >= args_cli.max_steps:
                print(f"[INFO] Reached max_steps={args_cli.max_steps}; stopping.")
                break

            if args_cli.real_time:
                sleep_time = dt - (time.time() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
    finally:
        if csv_files is not None:
            for handle in csv_files.values():
                handle.close()
            print(f"[INFO] Closed trajectory CSV files under {export_csv_dir}")
        if episode_reach_file is not None:
            episode_reach_file.close()
            pct = csv_utils.reach_percent(total_reached_episodes, total_episodes)
            episode_path = os.path.join(export_csv_dir, "episode_reach.csv")
            mean_lat = csv_utils.mean_success_latency_s_from_episode_csv(episode_path)
            summary_path = csv_utils.write_reach_summary_csv(
                export_csv_dir,
                total_reached_episodes,
                total_episodes,
                seed=args_cli.seed,
                mean_success_latency_s=mean_lat,
            )
            print(
                f"[INFO] Episode reach summary: {total_reached_episodes}/{total_episodes} "
                f"reached ({pct:.2f}%); failed={total_episodes - total_reached_episodes}"
            )
            print(f"[INFO] Per-episode CSV: {os.path.join(export_csv_dir, 'episode_reach.csv')}")
            print(f"[INFO] Aggregate CSV: {summary_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
