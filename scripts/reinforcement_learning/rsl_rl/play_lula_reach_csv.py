# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play ``Isaac-Reach-UR10-Play-v0`` with Lula cubic-spline C-space trajectories (not an RL policy).

Exports the same CSV schema as :mod:`play_reach_csv`:

* ``env_<id>_traj.csv`` — ``global_step``, ``sim_time_s``, ``reach_event``, ``max_CRI``, ``q_*``, ``qd_*``, ``CRI_<i>``
  (q/qd/CRI from the first CRI eval at each env step — pre-reset if the env terminated)
* ``episode_reach.csv`` — per-attempt ``reached`` / ``outcome``
* ``reach_summary.csv`` — aggregate reach rate

Evaluation is P2P-style one-shot (strict thresholds, one target, log until reach or episode failure).

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
    LulaTaskSpaceTrajectoryGenerator,
)

import isaaclab_tasks  # noqa: F401, E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
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


def _velocities_to_manager_action(
    arm_term,
    joint_vel_active: np.ndarray,
    active_joint_names: list[str],
    device: str,
    env_idx: int = 0,
) -> torch.Tensor:
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
    """Plans cubic-spline C-space trajectories toward the current ee_pose command."""

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
        self._hold_zero: list[bool] = [False] * self._num_envs

    @property
    def active_joint_names(self) -> list[str]:
        return self._active_joint_names

    def trajectory_active(self, env_idx: int) -> bool:
        return self._traj[env_idx] is not None

    def _sync_lula_robot_base(self, robot, env_idx: int) -> None:
        pos = robot.data.root_pos_w[env_idx, :3].detach().cpu().numpy().astype(np.float64)
        quat = _normalize_quat_wxyz(robot.data.root_quat_w[env_idx].detach().cpu().numpy())
        self._ik.set_robot_base_pose(pos, quat)
        ts_ks = getattr(self._tsgen, "_kinematics_solver", None)
        if ts_ks is not None:
            ts_ks.set_robot_base_pose(pos, quat)
        c_ks = getattr(self._cgen, "_kinematics_solver", None)
        if c_ks is not None and c_ks is not ts_ks:
            c_ks.set_robot_base_pose(pos, quat)

    def _clamp_cspace(self, q: np.ndarray) -> np.ndarray:
        x = np.asarray(q, dtype=np.float64).reshape(-1)
        return np.clip(x, self._cspace_lo, self._cspace_hi)

    def _plan_cspace_spline(self, q_start: np.ndarray, q_goal: np.ndarray):
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

    def _plan_taskspace_fallback(self, p0, q0, p1, q1):
        positions = np.stack([p0, p1], axis=0)
        orientations = np.stack([q0, q1], axis=0)
        return self._tsgen.compute_task_space_trajectory_from_points(positions, orientations, self._ee_frame)

    def replan(self, env: ManagerBasedRLEnv, env_idx: int, robot, arm_term) -> None:
        self._hold_zero[env_idx] = False
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
                f"[play_lula_reach_csv] Trajectory planning failed for env {env_idx}; holding zero motion."
            )
            self._traj[env_idx] = None
            self._t[env_idx] = 0.0
            self._last_cc[env_idx] = int(ee_term.command_counter[env_idx].item())
            self._hold_zero[env_idx] = True
            return

        self._traj[env_idx] = traj
        self._t[env_idx] = float(traj.start_time)
        self._last_cc[env_idx] = int(ee_term.command_counter[env_idx].item())

    def on_env_reset(self, env_ids: torch.Tensor) -> None:
        for e in env_ids.flatten().tolist():
            if 0 <= e < self._num_envs:
                self._last_cc[e] = None
                self._traj[e] = None
                self._t[e] = 0.0
                self._hold_zero[e] = False

    def need_replan(self, env: ManagerBasedRLEnv, env_idx: int) -> bool:
        ee_term = env.command_manager.get_term("ee_pose")
        cc = int(ee_term.command_counter[env_idx].item())
        last = self._last_cc[env_idx]
        return last is None or cc != last

    def action_from_trajectory(self, env: ManagerBasedRLEnv, robot, arm_term) -> torch.Tensor:
        rows: list[torch.Tensor] = []
        for e in range(self._num_envs):
            traj = self._traj[e]
            if traj is None or self._hold_zero[e]:
                rows.append(
                    torch.zeros((1, env.action_manager.total_action_dim), device=env.device, dtype=torch.float32)
                )
                continue
            end_t = float(traj.end_time)
            t_eval = min(self._t[e], end_t)
            _, dq = traj.get_joint_targets(t_eval)
            self._t[e] += env.step_dt * self._playback_speed
            # One-shot eval: after spline ends, hold zero until reach / timeout (do not auto-reset).
            if self._t[e] >= end_t - 1e-10:
                self._hold_zero[e] = True
            rows.append(
                _velocities_to_manager_action(
                    arm_term, np.asarray(dq), self._active_joint_names, env.device, env_idx=e
                )
            )
        return torch.cat(rows, dim=0)


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed

    strict_reach_params = csv_utils.strict_reach_params_from_env_cfg(env_cfg)
    if strict_reach_params is None:
        raise ValueError(
            "Could not read strict reach_success_criteria params from env cfg "
            "(expected rewards.reach_success_bonus or events.resample_ee_pose_on_reach)."
        )

    disable_resample = not args_cli.keep_resample_on_reach
    csv_utils.configure_p2p_style_eval(env_cfg, disable_resample=disable_resample)

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
    )
    print(
        "[INFO] Lula C-spline: "
        f"traj_duration={args_cli.traj_duration}s, traj_playback_speed={args_cli.traj_playback_speed}"
    )
    print(f"[INFO] Trajectory CSV output: {export_csv_dir}/env_*_traj.csv")
    print(
        "[INFO] P2P-style one-shot eval: log until first strict reach; "
        "no reach before episode end => failure."
    )
    print(f"[INFO] Mid-episode target resample: {'enabled' if args_cli.keep_resample_on_reach else 'disabled'}")
    print(
        "[INFO] Strict reach thresholds: "
        f"max_distance={strict_reach_params['max_distance']}, "
        f"max_angle_rad={strict_reach_params['max_angle_rad']}, "
        f"max_lin_vel={strict_reach_params['max_lin_vel']}, "
        f"max_ang_vel={strict_reach_params['max_ang_vel']}, "
        f"max_lin_acc={strict_reach_params['max_lin_acc']}, "
        f"max_ang_acc={strict_reach_params['max_ang_acc']}"
    )

    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        _obs = reset_out[0]
    else:
        _obs = reset_out
    # pose_command_w can be zeros right after reset; refresh commands.
    base.command_manager.compute(0.0)

    for e in range(num_envs):
        driver.replan(base, e, robot, arm_term)

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
            with torch.inference_mode():
                # Replan only for still-active attempts when command counter changes.
                for e in range(num_envs):
                    if attempt_active is not None and not bool(attempt_active[e].item()):
                        continue
                    if driver.need_replan(base, e):
                        driver.replan(base, e, robot, arm_term)

                reach_cmd_snapshot = (
                    base.command_manager.get_command(strict_reach_params["command_name"]).detach().clone()
                )

                action = driver.action_from_trajectory(base, robot, arm_term)
                step_ret = env.step(action)
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
                        export_csv_dir, num_envs, list(robot.joint_names), num_cri
                    )
                    episode_reach_file, episode_reach_writer = csv_utils.open_episode_reach_csv(export_csv_dir)
                    log_active = torch.ones(num_envs, device=base.device, dtype=torch.bool)
                    attempt_active = torch.ones(num_envs, device=base.device, dtype=torch.bool)
                    episode_ids = torch.zeros(num_envs, device=base.device, dtype=torch.long)

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

                csv_utils.append_traj_rows(
                    csv_writers,
                    robot,
                    play_step,
                    float(play_step) * dt,
                    log_active & attempt_active,
                    reach_ev,
                )
                if csv_files is not None:
                    for handle in csv_files.values():
                        handle.flush()

                sim_time_s = float(play_step) * dt
                newly_success = attempt_active & reach_ev
                if bool(newly_success.any().item()):
                    for env_idx in newly_success.nonzero(as_tuple=False).view(-1).tolist():
                        csv_utils.record_attempt(
                            episode_reach_writer,
                            episode_reach_file,
                            int(env_idx),
                            int(episode_ids[env_idx].item()),
                            sim_time_s,
                            reached=True,
                            outcome="success",
                        )
                    total_episodes += int(newly_success.sum().item())
                    total_reached_episodes += int(newly_success.sum().item())
                    episode_ids[newly_success] += 1
                    attempt_active &= ~newly_success
                    if not args_cli.export_csv_always:
                        log_active &= ~newly_success

                newly_fail = attempt_active & done_mask
                if bool(newly_fail.any().item()):
                    for env_idx in newly_fail.nonzero(as_tuple=False).view(-1).tolist():
                        outcome = csv_utils.classify_failure_outcome(base, int(env_idx))
                        csv_utils.record_attempt(
                            episode_reach_writer,
                            episode_reach_file,
                            int(env_idx),
                            int(episode_ids[env_idx].item()),
                            sim_time_s,
                            reached=False,
                            outcome=outcome,
                        )
                    total_episodes += int(newly_fail.sum().item())
                    episode_ids[newly_fail] += 1
                    attempt_active &= ~newly_fail
                    log_active &= ~newly_fail

                    # Clear Lula caches for failed/reset envs (env already reset inside step).
                    fail_ids = newly_fail.nonzero(as_tuple=False).view(-1)
                    driver.on_env_reset(fail_ids)

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
            summary_path = csv_utils.write_reach_summary_csv(
                export_csv_dir, total_reached_episodes, total_episodes
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
