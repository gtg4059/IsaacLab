# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play Isaac-Reach-UR10-Play-v0 using Lula C-space trajectories (cubic spline), not an RL checkpoint.

This script builds configuration-space trajectories with
:class:`LulaCSpaceTrajectoryGenerator` (NVIDIA Isaac Sim motion generation / Lula stack),
using ``compute_timestamped_c_space_trajectory(..., interpolation_mode="cubic_spline")``
for cubic-spline interpolation between the current configuration and an IK solution for
the commanded end-effector pose.

Prerequisites: Isaac Sim with ``isaacsim.robot_motion.lula`` and
``isaacsim.robot_motion.motion_generation`` extensions (same as other Lula-based tools).

Example::

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_lula_UR.py \\
        --task Isaac-Reach-UR10-Play-v0 --num_envs 4 \\
        --traj_duration 5.0 --traj_playback_speed 1.5

``--traj_duration`` stretches the spline in time (larger ⇒ slower planned motion).
``--traj_playback_speed`` scales how fast we advance along that spline per env step.

Device selection uses the same ``--device`` flag as other Isaac Lab apps (registered by :class:`~isaaclab.app.AppLauncher`).
"""

from __future__ import annotations

import argparse
from typing import Any, Set
import os
import time

import numpy as np
import torch

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# App launch (must stay before other isaacsim / isaaclab imports)
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Reach UR10 play using Lula cubic-spline C-space trajectories.")
parser.add_argument("--video", action="store_true", default=False, help="Record one rgb_array video clip.")
parser.add_argument("--video_length", type=int, default=200, help="Video length in env steps when --video is set.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs (each gets its own Lula plan).")
parser.add_argument("--task", type=str, default="Isaac-Reach-UR10-Play-v0", help="Gym task id.")
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
    help=(
        "Trajectory time advance per env step, as a multiple of step_dt. "
        ">1.0 follows the spline faster (shorter wall-clock to finish); "
        "<1.0 slower. Does not change traj_duration; only how fast we move along the spline clock."
    ),
)
parser.add_argument(
    "--cri_stats_interval",
    type=int,
    default=100,
    help="Print CRI>1 rate every this many env steps (0 = only final summary on exit).",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Post-launch imports
# -----------------------------------------------------------------------------
import carb  # noqa: E402
import gymnasium as gym  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension, get_extension_path_from_name  # noqa: E402

enable_extension("isaacsim.robot_motion.lula")
enable_extension("isaacsim.robot_motion.motion_generation")

# Lula-backed generators (Isaac Sim ``motion_generation`` Python API; backed by ``...motion_generation.lula``).
import isaacsim.robot_motion.motion_generation.lula  # noqa: F401, E402
from isaacsim.robot_motion.motion_generation import (  # noqa: E402
    LulaCSpaceTrajectoryGenerator,
    LulaKinematicsSolver,
    LulaTaskSpaceTrajectoryGenerator,
)

import isaaclab_tasks  # noqa: F401, E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


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
    """Map a flat joint_pos tensor (full articulation order) to a numpy vector in subset_names order."""
    name_to_i = {n: i for i, n in enumerate(joint_names)}
    out = np.zeros(len(subset_names), dtype=np.float64)
    for k, name in enumerate(subset_names):
        out[k] = float(joint_pos[name_to_i[name]].item())
    return out


def _print_trajectory_timing_stats(driver: "LulaCubicSplineReachDriver", step_dt: float) -> None:
    """Print mean / variance of measured per-trajectory wall time and Lula domain span (seconds)."""
    sim = np.asarray(driver.traj_sim_duration_samples_s, dtype=np.float64)
    dom = np.asarray(driver.traj_lula_domain_duration_samples_s, dtype=np.float64)
    n = sim.shape[0]
    if n == 0:
        print("[INFO] 궤적 시간 통계: 완주로 기록된 궤적이 없습니다 (시뮬레이션을 더 오래 돌리거나 명령 재샘플을 기다리세요).")
        return
    mean_s = float(np.mean(sim))
    var_s = float(np.var(sim, ddof=1)) if n > 1 else 0.0
    mean_dom = float(np.mean(dom))
    var_dom = float(np.var(dom, ddof=1)) if n > 1 else 0.0
    print(
        f"[INFO] 궤적 시간 통계 (표본 수={n}, env step_dt={step_dt:.6g}s): "
        "‘측정’은 한 궤적이 Lula 종료 시각(end_time)에 도달할 때까지 소요된 시뮬레이션 시간, "
        "‘Lula 도메인’은 해당 궤적의 (end_time - start_time)입니다."
    )
    print(f"       측정 평균 = {mean_s:.6g}s, 표본 분산 = {var_s:.6g}s², 표본 표준편차 = {np.sqrt(var_s):.6g}s")
    print(f"       Lula 도메인 평균 = {mean_dom:.6g}s, 표본 분산 = {var_dom:.6g}s², 표본 표준편차 = {np.sqrt(var_dom):.6g}s")


def _print_cri_exceed_one_stats(
    label: str,
    total_steps: int,
    steps_any_env_max_cri_gt1: int,
    per_env_exceed_steps: list[int],
) -> None:
    """Print fraction of steps where max joint CRI exceeds 1.0 (same tensor as ``mdp.collision_risk_index``)."""
    if total_steps <= 0:
        print(f"[INFO] CRI 통계 ({label}): 아직 env step이 없습니다.")
        return
    p_any = 100.0 * steps_any_env_max_cri_gt1 / total_steps
    print(
        f"[INFO] CRI 통계 ({label}), 누적 env step={total_steps}: "
        f"임의 로봇에서 max(관절 CRI)>1 인 스텝 비율 = {p_any:.2f}%"
    )
    # for e, cnt in enumerate(per_env_exceed_steps):
    #     print(f"       env {e}: 해당 env에서 max(관절 CRI)>1 인 스텝 비율 = {100.0 * cnt / total_steps:.2f}%")


def _normalize_quat_wxyz(q: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Return a unit quaternion (w, x, y, z); prefer w >= 0 for stable Lula / SciPy conventions."""
    v = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    v = v / n
    if v[0] < 0.0:
        v = -v
    return v


def _velocities_to_manager_action(
    arm_term,
    joint_vel_active: np.ndarray,
    active_joint_names: list[str],
    device: str,
    env_idx: int = 0,
) -> torch.Tensor:
    """Build one row of actions for ``env_idx`` (full vector for the arm action term)."""
    name_to_lula = {n: i for i, n in enumerate(active_joint_names)}
    scale = float(arm_term._scale) if isinstance(arm_term._scale, (float, int)) else None
    if scale is None:
        raise TypeError("Expected scalar scale on JointVelocityAction for this script.")

    if isinstance(arm_term._offset, torch.Tensor):
        off_row = arm_term._offset[env_idx].cpu().numpy()
    else:
        off_row = np.full(arm_term.action_dim, float(arm_term._offset), dtype=np.float64)

    raw = np.zeros(arm_term.action_dim, dtype=np.float64)
    for i, jn in enumerate(arm_term._joint_names):
        li = name_to_lula[jn]
        raw[i] = (float(joint_vel_active[li]) - float(off_row[i])) / scale

    return torch.as_tensor(raw, device=device, dtype=torch.float32).unsqueeze(0)


class LulaCubicSplineReachDriver:
    """Plans cubic-spline C-space trajectories toward the current ee_pose command (per sub-environment)."""

    def __init__(self, traj_duration: float, num_envs: int, playback_speed: float = 1.0):
        desc, urdf = _resolve_ur10_lula_asset_paths()
        self._cgen = LulaCSpaceTrajectoryGenerator(robot_description_path=desc, urdf_path=urdf)
        self._ik = LulaKinematicsSolver(robot_description_path=desc, urdf_path=urdf)
        self._tsgen = LulaTaskSpaceTrajectoryGenerator(robot_description_path=desc, urdf_path=urdf)
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
        self._traj: list[Any | None] = [None] * self._num_envs
        self._t: list[float] = [0.0] * self._num_envs
        self._last_cc: list[int | None] = [None] * self._num_envs
        # Per-env: env-step index when the current trajectory began (first sample of this plan).
        self._traj_start_env_step: list[int | None] = [None] * self._num_envs
        self._duration_recorded: list[bool] = [False] * self._num_envs
        # After spline time passes end_time once, trigger a manual env reset (separate from RL terminations).
        self._traj_finish_reset_fired: list[bool] = [False] * self._num_envs
        # One entry each time an env's trajectory reaches end_time in simulation time.
        self.traj_sim_duration_samples_s: list[float] = []
        self.traj_lula_domain_duration_samples_s: list[float] = []

    def _sync_lula_robot_base(self, robot, env_idx: int) -> None:
        """Align Lula with the articulated robot root in the USD stage (world pose).

        Until :meth:`LulaKinematicsSolver.set_robot_base_pose` is called, Lula assumes the URDF base
        sits at the origin. Vectorized Isaac Lab scenes place each clone elsewhere, so IK and
        task-space path conversion fail without this update.
        """
        pos = robot.data.root_pos_w[env_idx, :3].detach().cpu().numpy().astype(np.float64)
        quat = _normalize_quat_wxyz(robot.data.root_quat_w[env_idx].detach().cpu().numpy())
        self._ik.set_robot_base_pose(pos, quat)
        # ``LulaTaskSpaceTrajectoryGenerator`` keeps its own ``LulaKinematicsSolver`` for TS→CS conversion.
        ts_ks = getattr(self._tsgen, "_kinematics_solver", None)
        if ts_ks is not None:
            ts_ks.set_robot_base_pose(pos, quat)
        c_ks = getattr(self._cgen, "_kinematics_solver", None)
        if c_ks is not None and c_ks is not ts_ks:
            c_ks.set_robot_base_pose(pos, quat)

    def _clamp_cspace(self, q: np.ndarray) -> np.ndarray:
        """Clamp configuration to Lula active-joint limits (reduces IK overshoot vs URDF limits)."""
        x = np.asarray(q, dtype=np.float64).reshape(-1)
        return np.clip(x, self._cspace_lo, self._cspace_hi)

    def _plan_cspace_spline(self, q_start: np.ndarray, q_goal: np.ndarray):
        """Try timestamped cubic → timestamped linear → time-optimal c-space trajectory."""
        qs = self._clamp_cspace(q_start)
        qg = self._clamp_cspace(q_goal)
        timestamps = np.array([0.0, self._traj_duration], dtype=np.float64)
        waypoints = np.stack([qs, qg], axis=0)

        traj = self._cgen.compute_timestamped_c_space_trajectory(
            waypoints, timestamps, interpolation_mode="cubic_spline"
        )
        if traj is not None:
            return traj
        traj = self._cgen.compute_timestamped_c_space_trajectory(
            waypoints, timestamps, interpolation_mode="linear"
        )
        if traj is not None:
            return traj
        return self._cgen.compute_c_space_trajectory(np.stack([qs, qg], axis=0))

    def _plan_taskspace_fallback(
        self,
        p0: np.ndarray,
        q0: np.ndarray,
        p1: np.ndarray,
        q1: np.ndarray,
    ):
        positions = np.stack([p0, p1], axis=0)
        orientations = np.stack([q0, q1], axis=0)
        return self._tsgen.compute_task_space_trajectory_from_points(positions, orientations, self._ee_frame)

    def replan(
        self,
        env: ManagerBasedRLEnv,
        env_idx: int,
        robot,
        arm_term,
        *,
        env_steps_completed_at_start: int = 0,
    ) -> None:
        self._sync_lula_robot_base(robot, env_idx)

        ee_term = env.command_manager.get_term("ee_pose")
        pose_w = ee_term.pose_command_w[env_idx]
        pos_t = pose_w[:3]
        quat_t = pose_w[3:7]

        joint_pos = robot.data.joint_pos[env_idx].cpu()
        q_start = _joint_vec_in_names_order(joint_pos, robot.joint_names, self._active_joint_names)

        warm = q_start.copy()
        pos_np = pos_t.detach().cpu().numpy()
        quat_np = _normalize_quat_wxyz(quat_t.detach().cpu().numpy())

        q_goal, ok = self._ik.compute_inverse_kinematics(
            self._ee_frame,
            pos_np,
            quat_np,
            warm_start=warm,
        )

        traj = None
        if ok:
            q_goal_arr = self._clamp_cspace(np.asarray(q_goal, dtype=np.float64))
            traj = self._plan_cspace_spline(q_start, q_goal_arr)

        if traj is None:
            body_idx = robot.find_bodies("ee_link")[0][0]
            ee = robot.data.body_state_w[env_idx, body_idx]
            p0 = ee[:3].detach().cpu().numpy()
            q0 = _normalize_quat_wxyz(ee[3:7].detach().cpu().numpy())
            traj = self._plan_taskspace_fallback(p0, q0, pos_np, quat_np)

        if traj is None:
            carb.log_warn(
                f"[play_lula_UR] Trajectory planning failed for env {env_idx}; holding zero motion until next command."
            )
            self._traj[env_idx] = None
            self._t[env_idx] = 0.0
            self._last_cc[env_idx] = int(ee_term.command_counter[env_idx].item())
            self._traj_start_env_step[env_idx] = None
            self._duration_recorded[env_idx] = False
            self._traj_finish_reset_fired[env_idx] = False
            return

        self._traj[env_idx] = traj
        self._t[env_idx] = float(traj.start_time)
        self._last_cc[env_idx] = int(ee_term.command_counter[env_idx].item())
        self._traj_start_env_step[env_idx] = int(env_steps_completed_at_start)
        self._duration_recorded[env_idx] = False
        self._traj_finish_reset_fired[env_idx] = False

    def on_env_reset(self, env_ids: torch.Tensor) -> None:
        """Clear cached plans for environments that were reset inside the vec env."""
        ids = env_ids.flatten().tolist()
        for e in ids:
            if 0 <= e < self._num_envs:
                self._last_cc[e] = None
                self._traj[e] = None
                self._t[e] = 0.0
                self._traj_start_env_step[e] = None
                self._duration_recorded[e] = False
                self._traj_finish_reset_fired[e] = False

    def need_replan(self, env: ManagerBasedRLEnv, env_idx: int) -> bool:
        """Replan only on a new pose command (command_counter change) or before the first plan."""
        ee_term = env.command_manager.get_term("ee_pose")
        cc = int(ee_term.command_counter[env_idx].item())
        last = self._last_cc[env_idx]
        return last is None or cc != last

    def action_from_trajectory(self, env: ManagerBasedRLEnv, robot, arm_term) -> torch.Tensor:
        rows: list[torch.Tensor] = []
        for e in range(self._num_envs):
            traj = self._traj[e]
            if traj is None:
                rows.append(
                    torch.zeros((1, env.action_manager.total_action_dim), device=env.device, dtype=torch.float32)
                )
                continue
            end_t = float(traj.end_time)
            t_eval = min(self._t[e], end_t)
            _, dq = traj.get_joint_targets(t_eval)
            self._t[e] += env.step_dt * self._playback_speed
            rows.append(
                _velocities_to_manager_action(
                    arm_term, np.asarray(dq), self._active_joint_names, env.device, env_idx=e
                )
            )
        return torch.cat(rows, dim=0)

    def post_step_trajectory(
        self,
        step_dt: float,
        env_steps_completed_now: int,
        skip_env_ids: Set[int] | None = None,
    ) -> list[int]:
        """Record timing for finished trajectories; return env indices that should be reset (playback complete).

        Envs listed in ``skip_env_ids`` are skipped (e.g. already reset inside ``env.step`` due to RL terminations).
        """
        skip = skip_env_ids or set()
        reset_ids: list[int] = []
        for e in range(self._num_envs):
            if e in skip:
                continue
            traj = self._traj[e]
            if traj is None:
                continue
            end_t = float(traj.end_time)
            if self._t[e] < end_t - 1e-10:
                continue
            if not self._traj_finish_reset_fired[e]:
                self._traj_finish_reset_fired[e] = True
                reset_ids.append(e)
            if not self._duration_recorded[e]:
                start_step = self._traj_start_env_step[e]
                if start_step is not None:
                    sim_elapsed_s = float(env_steps_completed_now - start_step) * float(step_dt)
                    lula_span_s = float(traj.end_time) - float(traj.start_time)
                    self.traj_sim_duration_samples_s.append(sim_elapsed_s)
                    self.traj_lula_domain_duration_samples_s.append(lula_span_s)
                    self._duration_recorded[e] = True
        return reset_ids


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # Unwrap gym wrappers (e.g. RecordVideo) down to the Isaac Lab manager env.
    base: ManagerBasedRLEnv = env.unwrapped
    if not isinstance(base, ManagerBasedRLEnv):
        raise RuntimeError(f"Expected ManagerBasedRLEnv, got {type(base)}")

    log_dir = os.path.join("logs", "lula_play", args_cli.task.replace("-", "_"))
    log_dir = os.path.abspath(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    robot = base.scene["robot"]
    arm_term = base.action_manager.get_term("arm_action")

    num_envs = base.num_envs
    driver = LulaCubicSplineReachDriver(
        traj_duration=args_cli.traj_duration,
        num_envs=num_envs,
        playback_speed=args_cli.traj_playback_speed,
    )
    print(
        "[INFO] Lula trajectory timing: "
        f"traj_duration={args_cli.traj_duration}s, traj_playback_speed={args_cli.traj_playback_speed}"
    )

    obs, _ = env.reset()
    # ``pose_command_w`` is filled in ``CommandTerm.compute`` via ``_update_metrics``; right after ``reset()`` it
    # can still be zeros, which breaks Lula IK (zero-norm quaternion). Advance commands with dt=0 to refresh.
    base.command_manager.compute(0.0)

    for e in range(num_envs):
        driver.replan(base, e, robot, arm_term, env_steps_completed_at_start=0)

    dt = base.step_dt
    video_steps = 0

    # Completed env.step count (used for CRI stats and trajectory elapsed-time samples).
    cri_total_steps = 0
    cri_steps_any_env_max_gt1 = 0
    cri_per_env_exceed_steps = [0] * num_envs

    try:
        while simulation_app.is_running():
            t0 = time.time()
            for e in range(num_envs):
                if driver.need_replan(base, e):
                    driver.replan(
                        base, e, robot, arm_term, env_steps_completed_at_start=cri_total_steps
                    )

            action = driver.action_from_trajectory(base, robot, arm_term)
            with torch.inference_mode():
                step_ret = env.step(action)
            # ManagerBasedRLEnv returns (obs, reward, reset_terminated, reset_time_outs, extras).
            if len(step_ret) == 5:
                _obs, _reward, reset_terminated, reset_time_outs, _extras = step_ret
                assert isinstance(reset_terminated, torch.Tensor) and isinstance(reset_time_outs, torch.Tensor)
                episode_done = reset_terminated | reset_time_outs
            else:
                raise RuntimeError(f"Unexpected env.step return length {len(step_ret)}.")

            cri_total_steps += 1
            # ``ArticulationData.CRI`` uses ``torch.clamp_`` in-place; buffers may be inference tensors after
            # ``env.step()``, so the read must occur under ``torch.inference_mode()`` (see PyTorch RFC #17).
            with torch.inference_mode():
                cri = robot.data.CRI
            max_per_env = cri.max(dim=-1).values
            exceed = max_per_env > 1.0
            if bool(exceed.any().item()):
                cri_steps_any_env_max_gt1 += 1
            exceed_cpu = exceed.cpu().tolist()
            for e in range(num_envs):
                if exceed_cpu[e]:
                    cri_per_env_exceed_steps[e] += 1

            ep_done_set: Set[int] = set()
            done_tensor: torch.Tensor | None = None
            if bool(episode_done.any()):
                done_tensor = episode_done.nonzero(as_tuple=False).reshape(-1)
                ep_done_set = {int(x) for x in done_tensor.tolist()}

            traj_reset_ids = driver.post_step_trajectory(
                dt, cri_total_steps, skip_env_ids=ep_done_set
            )

            interval = args_cli.cri_stats_interval
            if interval > 0 and cri_total_steps % interval == 0:
                _print_cri_exceed_one_stats(
                    f"중간 (step {cri_total_steps})",
                    cri_total_steps,
                    cri_steps_any_env_max_gt1,
                    cri_per_env_exceed_steps,
                )

            if args_cli.video:
                video_steps += 1
                if video_steps >= args_cli.video_length:
                    break

            if traj_reset_ids:
                ids_tensor = torch.tensor(traj_reset_ids, device=base.device, dtype=torch.int64)
                driver.on_env_reset(ids_tensor)
                # After ``env.step()`` under ``inference_mode``, articulation buffers can be inference tensors;
                # ``reset`` → event terms → ``write_joint_*_to_sim`` use in-place writes and must run in this context.
                with torch.inference_mode():
                    base.reset(env_ids=ids_tensor)
                    base.command_manager.compute(0.0)
                for e in traj_reset_ids:
                    driver.replan(
                        base, int(e), robot, arm_term, env_steps_completed_at_start=cri_total_steps
                    )

            if done_tensor is not None:
                driver.on_env_reset(done_tensor)
                for e in done_tensor.tolist():
                    driver.replan(
                        base, int(e), robot, arm_term, env_steps_completed_at_start=cri_total_steps
                    )

            if args_cli.real_time:
                sleep_time = dt - (time.time() - t0)
                if sleep_time > 0:
                    time.sleep(sleep_time)
    finally:
        _print_cri_exceed_one_stats(
            "종료 시 누적",
            cri_total_steps,
            cri_steps_any_env_max_gt1,
            cri_per_env_exceed_steps,
        )
        _print_trajectory_timing_stats(driver, dt)
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
