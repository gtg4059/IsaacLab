# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a reach-task RSL-RL checkpoint and export per-env trajectory CSV (q, qd, CRI, reach_event, cmd pose).

Evaluation protocol matches ``Isaac-Reach-UR10-P2P-Play-v0`` semantics:

* one target per env attempt (in-episode ``resample_ee_pose_on_reach`` disabled)
* strict final reach thresholds (curriculum easing disabled)
* trajectory logged only until first strict reach (inclusive), or until episode failure
* ``terminations.reach_success`` is disabled by default so success does not reset mid-eval
  (zip / ``09dca287``-compatible); policy keeps running on finished envs unless ``--freeze_actions``
* attempt outcome: ``success`` on reach; ``fail_timeout`` / ``fail_ovf`` / ``fail_other`` if the
  episode ends without reach (same idea as P2P: reach terminates success, timeout is failure)
* stops after every env finishes exactly one attempt

Example (one target per env, strict cfg gates, clean observations). Prints reach rate
and success-time mean / median / quartiles, and writes ``reach_summary.csv``::

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_reach_csv.py \\
        --task Isaac-Reach-UR10-v0 \\
        --num_envs 2048 \\
        --checkpoint logs/rsl_rl/reach_ur10/<run>/model_*.pt \\
        --export_csv_dir logs/rsl_rl/reach_ur10/<run>/eval_<iter> \\
        --seed 42 \\
        --no_traj_csv \\
        --no_print_targets \\
        --headless

Pass ``--enable_corruption`` to keep the training observation noise. Without it, play forces
``observations.policy.enable_corruption = False``.

Multi-seed eval (same checkpoint, different env RNG seeds; Isaac Sim started once)::

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_reach_csv.py \\
        --task Isaac-Reach-UR10-Play-v0 \\
        --num_envs 2000 \\
        --checkpoint /path/to/model.pt \\
        --seeds 0,1,2,3,4 \\
        --no_traj_csv \\
        --export_csv_dir /path/to/out \\
        --headless

Writes ``seed_<s>/episode_reach.csv``, ``seed_<s>/reach_summary.csv``, and
``multi_seed_summary.csv`` (mean ± std of reach rate across seeds).

For a P2P-trained policy use ``--task Isaac-Reach-UR10-P2P-Play-v0`` (env already ends on reach).

CSV columns: ``global_step``, ``sim_time_s``, ``reach_event``, ``max_CRI``,
``cmd_{x,y,z,qw,qx,qy,qz}`` (ee_pose in robot base frame), ``q_*``, ``qd_*``, ``CRI_<i>``.
``episode_reach.csv`` also stores the same command pose per attempt.

Trajectory rows use the first CRI evaluation at each env step (pre-reset motion state), so OVF
termination does not pair post-reset ``qd=0`` with the pre-reset CRI that caused the reset.
Also writes ``episode_reach.csv`` and ``reach_summary.csv``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import IO, Any

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play reach RSL-RL checkpoint and export trajectory CSV.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Single environment seed (ignored if --seeds is set).")
parser.add_argument(
    "--seeds",
    type=str,
    default=None,
    help="Comma-separated eval seeds for multi-seed acquisition (e.g. 0,1,2,3,4). "
    "Runs each seed sequentially in one process; writes seed_<s>/ plus multi_seed_summary.csv.",
)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--stochastic",
    action="store_true",
    default=False,
    help="Sample actions from the policy Gaussian (training noise). Default plays the mean.",
)
parser.add_argument(
    "--export_csv_dir",
    type=str,
    default=None,
    help="Output directory for env_*_traj.csv (default: <checkpoint_dir>/joint_trajectory).",
)
parser.add_argument(
    "--no_traj_csv",
    action="store_true",
    default=False,
    help="Skip writing env_*_traj.csv (still write episode_reach / summaries). Recommended with --seeds.",
)
parser.add_argument(
    "--max_cri_trace",
    action="store_true",
    default=False,
    help="Write max_cri_trace.csv (env_idx, sim_time_s, max_cri) for each attempt step.",
)
parser.add_argument(
    "--export_csv_always",
    action="store_true",
    default=False,
    help="Keep logging after first strict reach (default: stop per env at reach, P2P-style).",
)
parser.add_argument(
    "--keep_resample_on_reach",
    action="store_true",
    default=False,
    help="Keep mid-episode target resample (default: disabled for P2P-style one-target attempts).",
)
parser.add_argument(
    "--print_targets",
    action="store_true",
    default=True,
    help="Print ee_pose targets at attempt start (robot base frame).",
)
parser.add_argument(
    "--no_print_targets",
    action="store_true",
    default=False,
    help="Disable --print_targets.",
)
parser.add_argument(
    "--print_targets_limit",
    type=int,
    default=32,
    help="Max envs to print target poses for (default: 32).",
)
parser.add_argument(
    "--keep_reach_success_term",
    action="store_true",
    default=False,
    help="Keep terminations.reach_success (success resets the env). "
    "Default disables it so finished envs are not reset mid-eval (zip-compatible).",
)
parser.add_argument(
    "--freeze_actions",
    action="store_true",
    default=False,
    help="Zero actions + stall timeout on finished envs. Default keeps policy running "
    "(zip / 09dca287-compatible). Implies reach_success term is off unless "
    "--keep_reach_success_term is set.",
)
parser.add_argument(
    "--deterministic_sim",
    action="store_true",
    default=False,
    help="Opt-in PhysX enhanced determinism (changes dynamics vs training; can lower reach rate).",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Exit after this many env steps (0 = run until all envs finish one attempt).",
)
parser.add_argument(
    "--dump_reset_env",
    type=int,
    default=None,
    help="After the eval reset, write this env's joint position and ee command, then exit.",
)
parser.add_argument(
    "--task_traj_dir",
    type=str,
    default=None,
    help="Directory of env_<id>_traj.csv. After reset, place each env at that file's "
    "rest-to-rest cubic start and set ee_pose to the logged command.",
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
    "--enable_corruption",
    action="store_true",
    default=False,
    help="Apply policy observation noise (PLAY is clean by default).",
)
parser.add_argument(
    "--obs_pose_noise",
    type=float,
    default=None,
    help="Half-range uniform noise on ee_pose_error (metres and rad). "
    "Implies --enable_corruption. Default keeps cfg ±0.002 if corruption is on.",
)
parser.add_argument(
    "--obs_cri_noise",
    type=float,
    default=None,
    help="Half-range uniform noise on CRI. Implies --enable_corruption. "
    "Default keeps cfg ±0.001 if corruption is on.",
)
parser.add_argument(
    "--obs_joint_pos_noise",
    type=float,
    default=None,
    help="Half-range uniform noise on joint_pos (rad). Implies --enable_corruption. "
    "Pose and CRI stay clean unless their flags or --enable_corruption are set.",
)
parser.add_argument(
    "--obs_joint_vel_noise",
    type=float,
    default=None,
    help="Half-range uniform noise on joint_vel (rad/s). Implies --enable_corruption. "
    "Pose and CRI stay clean unless their flags or --enable_corruption are set.",
)
parser.add_argument(
    "--hold_square_max",
    type=int,
    default=None,
    help="Override reach_success_bonus hold_square_max (e.g. 1 = first in-gate step is success). "
    "Default keeps the env cfg value.",
)
parser.add_argument(
    "--gate_factor",
    type=float,
    default=1.0,
    help="Scale strict reach gates (distance, angle, twist) for eval. "
    "Use 2.0 for Jawale cooled gates (2 cm / 0.1 rad / twist < 0.02) from 1× cfg bases.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata

import gymnasium as gym
import numpy as np
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnv,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul
from isaaclab.utils.seed import configure_seed
from isaaclab_tasks.manager_based.manipulation.reach.mdp.observations import command_origin_pose_w
from isaaclab_tasks.manager_based.manipulation.reach.mdp.rewards import _body_idx_single


def _configure_eval_rng(seed: int) -> None:
    """Re-pin RNGs for play eval (no torch.use_deterministic_algorithms — breaks CircularBuffer)."""
    configure_seed(seed, torch_deterministic=False)

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import reach_traj_csv_utils as csv_utils

installed_version = metadata.version("rsl-rl-lib")


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


def _parse_eval_seeds(default_seed: int) -> list[int]:
    if args_cli.seeds is not None:
        parts = [p.strip() for p in args_cli.seeds.split(",") if p.strip() != ""]
        if not parts:
            raise ValueError("--seeds must contain at least one integer (e.g. 0,1,2).")
        return [int(p) for p in parts]
    if args_cli.seed is not None:
        return [args_cli.seed]
    return [default_seed]


def _load_fixed_task(path: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Joint names, rest-to-rest cubic start, and the logged base-frame ee command."""
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        raise ValueError(f"{path} has too few samples to fit a cubic.")
    q_cols = [key for key in rows[0] if key.startswith("q_") and not key.startswith("qd_")]
    t = np.array([float(row["sim_time_s"]) for row in rows], dtype=np.float64)
    q = np.array([[float(row[col]) for col in q_cols] for row in rows], dtype=np.float64)
    duration = float(t[-1])
    blend = np.clip(t / duration, 0.0, 1.0)
    blend = 3.0 * blend * blend - 2.0 * blend * blend * blend
    q0 = np.zeros(q.shape[1], dtype=np.float64)
    design_const = 1.0 - blend
    for joint_idx in range(q.shape[1]):
        design = np.stack([design_const, blend], axis=1)
        coef, *_ = np.linalg.lstsq(design, q[:, joint_idx], rcond=None)
        q0[joint_idx] = coef[0]
    cmd = np.array(
        [float(rows[0][key]) for key in ("cmd_x", "cmd_y", "cmd_z", "cmd_qw", "cmd_qx", "cmd_qy", "cmd_qz")],
        dtype=np.float64,
    )
    return [col[2:] for col in q_cols], q0, cmd


def _apply_fixed_tasks(base_env: ManagerBasedRLEnv, traj_dir: str) -> None:
    """Overwrite the random reset with the stored cubic starts and ee commands."""
    robot = base_env.scene["robot"]
    names = list(robot.joint_names)
    q0 = torch.zeros(base_env.num_envs, len(names), device=base_env.device)
    cmd = torch.zeros(base_env.num_envs, 7, device=base_env.device)
    for env_idx in range(base_env.num_envs):
        path = os.path.join(traj_dir, f"env_{env_idx}_traj.csv")
        file_names, q_np, cmd_np = _load_fixed_task(path)
        if file_names != names:
            raise ValueError(f"{path} joint order {file_names} != robot {names}.")
        q0[env_idx] = torch.as_tensor(q_np, device=base_env.device, dtype=q0.dtype)
        cmd[env_idx] = torch.as_tensor(cmd_np, device=base_env.device, dtype=cmd.dtype)
    qd = torch.zeros_like(q0)
    robot.write_joint_state_to_sim(q0, qd)
    robot.set_joint_position_target(q0)
    robot.set_joint_velocity_target(qd)
    robot.write_data_to_sim()
    base_env.sim.forward()
    robot.update(0.0)
    ee_term = base_env.command_manager.get_term("ee_pose")
    ee_term.pose_command_b[:] = cmd
    ee_term.time_left[:] = 1.0e6
    ee_term._update_metrics()
    print(
        f"[INFO] Fixed tasks from {traj_dir}: env 0 q0 {q0[0].detach().cpu().tolist()} "
        f"cmd {cmd[0, :3].detach().cpu().tolist()}",
        flush=True,
    )


def _run_one_shot_eval(
    *,
    env: RslRlVecEnvWrapper,
    base_env: ManagerBasedRLEnv,
    policy: Any,
    policy_nn: Any,
    strict_reach_params: dict[str, Any],
    export_csv_dir: str,
    resume_path: str,
    eval_seed: int,
    write_traj: bool,
    record_video: bool,
) -> dict[str, Any]:
    """Run P2P-style one-shot eval for the current reset state; write CSVs under export_csv_dir."""
    # Re-pin RNGs at the start of every seed so multi-seed / repeat runs are reproducible.
    _configure_eval_rng(eval_seed)
    if args_cli.task_traj_dir:
        _apply_fixed_tasks(base_env, args_cli.task_traj_dir)

    dt = base_env.step_dt
    obs = env.get_observations()

    play_step = 0
    video_step = 0
    csv_files: dict[int, IO[str]] | None = None
    csv_writers: dict[int, csv.DictWriter] | None = None
    episode_reach_file: IO[str] | None = None
    episode_reach_writer: csv.DictWriter | None = None
    cri_trace_file: IO[str] | None = None
    cri_trace_writer: csv.DictWriter | None = None
    log_active: torch.Tensor | None = None
    attempt_active: torch.Tensor | None = None
    episode_ids: torch.Tensor | None = None
    printed_targets = False
    total_episodes = 0
    total_reached_episodes = 0

    try:
        while simulation_app.is_running():
            loop_start = time.time()
            # Use no_grad (not inference_mode): Isaac Lab buffers are mutated in-place on reset,
            # and inference tensors from a prior seed block multi-seed reset().
            with torch.no_grad():
                reach_cmd_snapshot = (
                    base_env.command_manager.get_command(strict_reach_params["command_name"]).detach().clone()
                )
                if args_cli.print_targets and not args_cli.no_print_targets and not printed_targets:
                    cmd_name = strict_reach_params["command_name"]
                    csv_utils.print_command_targets_from_env(
                        base_env,
                        cmd_name,
                        label=f"seed={eval_seed}",
                        max_rows=args_cli.print_targets_limit,
                    )
                    printed_targets = True

                actions = policy(obs, stochastic_output=bool(args_cli.stochastic))
                # Optional: hold finished envs still. Default keeps policy on (zip-compatible).
                if attempt_active is not None and args_cli.freeze_actions:
                    actions = csv_utils.freeze_finished_env_controls(base_env, actions, attempt_active)
                obs, _, dones, _ = env.step(actions)

                if version.parse(installed_version) >= version.parse("4.0.0"):
                    policy.reset(dones)
                elif policy_nn is not None:
                    policy_nn.reset(dones)

                play_step += 1

                if episode_reach_writer is None:
                    if write_traj:
                        robot = base_env.scene["robot"]
                        num_cri = int(robot.data.CRI.shape[1])
                        csv_files, csv_writers = csv_utils.open_traj_csv_writers(
                            export_csv_dir,
                            base_env.num_envs,
                            list(robot.joint_names),
                            num_cri,
                            log_actions=True,
                            action_dim=int(actions.shape[-1]),
                        )
                    episode_reach_file, episode_reach_writer = csv_utils.open_episode_reach_csv(export_csv_dir)
                    if args_cli.max_cri_trace:
                        cri_trace_path = os.path.join(export_csv_dir, "max_cri_trace.csv")
                        cri_trace_file = open(cri_trace_path, "w", newline="")
                        cri_trace_writer = csv.DictWriter(
                            cri_trace_file, fieldnames=["env_idx", "sim_time_s", "max_cri"]
                        )
                        cri_trace_writer.writeheader()
                    log_active = torch.ones(base_env.num_envs, device=base_env.device, dtype=torch.bool)
                    attempt_active = torch.ones(base_env.num_envs, device=base_env.device, dtype=torch.bool)
                    episode_ids = torch.zeros(base_env.num_envs, device=base_env.device, dtype=torch.long)
                    min_pos_err = torch.full((base_env.num_envs,), float("inf"), device=base_env.device)
                    min_ang_err = torch.full((base_env.num_envs,), float("inf"), device=base_env.device)
                    max_cri = torch.zeros(base_env.num_envs, device=base_env.device)
                    n_pure = torch.zeros(base_env.num_envs, device=base_env.device)
                    last_pos_err = torch.zeros(base_env.num_envs, device=base_env.device)
                    last_ang_err = torch.zeros(base_env.num_envs, device=base_env.device)
                    last_lin_vel = torch.zeros(base_env.num_envs, device=base_env.device)
                    last_ang_vel = torch.zeros(base_env.num_envs, device=base_env.device)
                    last_cri = torch.zeros(base_env.num_envs, device=base_env.device)

                assert (
                    attempt_active is not None
                    and log_active is not None
                    and episode_ids is not None
                    and episode_reach_writer is not None
                    and episode_reach_file is not None
                )

                reach_ev = csv_utils.resolve_reach_event(base_env, reach_cmd_snapshot, strict_reach_params)
                done_mask = dones.detach().bool().view(-1)
                pos_err, ang_err, lin_vel, ang_vel, cri = _ee_tracking(base_env, reach_cmd_snapshot)
                live = attempt_active
                min_pos_err[live] = torch.minimum(min_pos_err[live], pos_err[live])
                min_ang_err[live] = torch.minimum(min_ang_err[live], ang_err[live])
                max_cri[live] = torch.maximum(max_cri[live], cri[live])
                last_pos_err[live] = pos_err[live]
                last_ang_err[live] = ang_err[live]
                last_lin_vel[live] = lin_vel[live]
                last_ang_vel[live] = ang_vel[live]
                last_cri[live] = cri[live]
                n_pure[live] += (cri[live] < float(args_cli.ovf_threshold)).to(n_pure.dtype)

                if cri_trace_writer is not None and bool(live.any().item()):
                    live_idx = live.nonzero(as_tuple=False).view(-1)
                    cri_now = cri[live_idx].detach().cpu().tolist()
                    t_now = float(play_step) * dt
                    cri_trace_writer.writerows(
                        {"env_idx": int(i), "sim_time_s": t_now, "max_cri": float(v)}
                        for i, v in zip(live_idx.tolist(), cri_now, strict=True)
                    )

                if write_traj and csv_writers is not None:
                    csv_utils.append_traj_rows(
                        csv_writers,
                        base_env.scene["robot"],
                        play_step,
                        float(play_step) * dt,
                        log_active & attempt_active,
                        reach_ev,
                        command=reach_cmd_snapshot,
                        pos_err=pos_err,
                        ang_err=ang_err,
                        lin_spd=lin_vel,
                        ang_spd=ang_vel,
                        actions=actions,
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
                            command_pose=reach_cmd_snapshot[int(env_idx)],
                            extras={
                                **_attempt_extras(
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
                                "pure_duration_s": float(n_pure[int(env_idx)].item()) * float(dt),
                            },
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
                        outcome = csv_utils.classify_failure_outcome(base_env, int(env_idx))
                        csv_utils.record_attempt(
                            episode_reach_writer,
                            episode_reach_file,
                            int(env_idx),
                            int(episode_ids[env_idx].item()),
                            sim_time_s,
                            reached=False,
                            outcome=outcome,
                            command_pose=reach_cmd_snapshot[int(env_idx)],
                            extras={
                                **_attempt_extras(
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
                                "pure_duration_s": float(n_pure[int(env_idx)].item()) * float(dt),
                            },
                        )
                    total_episodes += int(newly_fail.sum().item())
                    episode_ids[newly_fail] += 1
                    attempt_active &= ~newly_fail
                    log_active &= ~newly_fail

                if not bool(attempt_active.any().item()):
                    pct = csv_utils.reach_percent(total_reached_episodes, total_episodes)
                    print(
                        f"[INFO] seed={eval_seed}: all envs finished 1 attempt; reach rate "
                        f"{total_reached_episodes}/{total_episodes} ({pct:.2f}%)."
                    )
                    break

            if record_video:
                video_step += 1
                if video_step >= args_cli.video_length:
                    break

            if args_cli.max_steps > 0 and play_step >= args_cli.max_steps:
                print(f"[INFO] seed={eval_seed}: reached max_steps={args_cli.max_steps}; stopping.")
                break

            if args_cli.real_time:
                sleep_time = dt - (time.time() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
    finally:
        if csv_files is not None:
            for handle in csv_files.values():
                handle.close()
            print(f"[INFO] Closed trajectory CSV files under {os.path.abspath(export_csv_dir)}")
        if cri_trace_file is not None:
            cri_trace_file.close()
        if episode_reach_file is not None:
            episode_reach_file.close()

    episode_path = os.path.join(export_csv_dir, "episode_reach.csv")
    latency_stats = csv_utils.success_latency_stats_from_episode_csv(episode_path)
    mean_lat = latency_stats.get("mean_success_latency_s")
    mean_lat_f = None if mean_lat in (None, "") else float(mean_lat)
    summary_path = csv_utils.write_reach_summary_csv(
        export_csv_dir,
        total_reached_episodes,
        total_episodes,
        seed=eval_seed,
        checkpoint=resume_path,
        mean_success_latency_s=mean_lat_f,
        latency_stats=latency_stats,
    )
    pct = csv_utils.reach_percent(total_reached_episodes, total_episodes)
    print(
        f"[INFO] seed={eval_seed}: reach rate {total_reached_episodes}/{total_episodes} ({pct:.2f}%). "
        f"fail_timeout={latency_stats['n_fail_timeout']} "
        f"fail_ovf={latency_stats['n_fail_ovf']} "
        f"fail_other={latency_stats['n_fail_other']}"
    )
    if mean_lat_f is None:
        print("[INFO] Success latency: no successful attempts.")
    else:
        print(
            "[INFO] Success latency (s): "
            f"mean={mean_lat_f:.3f} "
            f"median={float(latency_stats['median_success_latency_s']):.3f} "
            f"q1={float(latency_stats['q1_success_latency_s']):.3f} "
            f"q3={float(latency_stats['q3_success_latency_s']):.3f} "
            f"min={float(latency_stats['min_success_latency_s']):.3f} "
            f"max={float(latency_stats['max_success_latency_s']):.3f}"
        )
    print(f"[INFO] Per-episode CSV: {os.path.abspath(episode_path)}")
    print(f"[INFO] Aggregate CSV: {summary_path}")

    return {
        "seed": eval_seed,
        "checkpoint": resume_path,
        "total_episodes": total_episodes,
        "reached_episodes": total_reached_episodes,
        "failed_episodes": total_episodes - total_reached_episodes,
        "reach_percent": round(pct, 4),
        "mean_success_latency_s": "" if mean_lat_f is None else round(mean_lat_f, 6),
    }


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent and write trajectory CSV under P2P-style one-shot eval."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    eval_seeds = _parse_eval_seeds(agent_cfg.seed)
    multi_seed = args_cli.seeds is not None
    # First seed applied at env construction; later seeds via reset(seed=...).
    env_cfg.seed = eval_seeds[0]
    agent_cfg.seed = eval_seeds[0]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    export_csv_root = args_cli.export_csv_dir or os.path.join(log_dir, "joint_trajectory")
    env_cfg.log_dir = log_dir

    if args_cli.hold_square_max is not None:
        bonus = getattr(getattr(env_cfg, "rewards", None), "reach_success_bonus", None)
        if bonus is None or getattr(bonus, "params", None) is None:
            raise ValueError("--hold_square_max requires rewards.reach_success_bonus.params")
        bonus.params["hold_square_max"] = int(args_cli.hold_square_max)
        # hold_steps==0 means use hold_square_max; keep that so 1-hold is first in-gate step.
        if int(bonus.params.get("hold_steps", 0) or 0) > 0:
            bonus.params["hold_steps"] = int(args_cli.hold_square_max)

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
    disable_reach_success_term = not args_cli.keep_reach_success_term
    touch_pose_cri = bool(args_cli.enable_corruption) or args_cli.obs_pose_noise is not None or (
        args_cli.obs_cri_noise is not None
    )
    touch_joints = args_cli.obs_joint_pos_noise is not None or args_cli.obs_joint_vel_noise is not None
    enable_corruption = touch_pose_cri or touch_joints
    policy_obs = getattr(getattr(env_cfg, "observations", None), "policy", None)
    if enable_corruption:
        if policy_obs is None:
            raise ValueError("--enable_corruption requires observations.policy")
        policy_obs.enable_corruption = True
        from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

        if touch_pose_cri:
            pose_n = 0.002 if args_cli.obs_pose_noise is None else float(args_cli.obs_pose_noise)
            cri_n = 0.001 if args_cli.obs_cri_noise is None else float(args_cli.obs_cri_noise)
            policy_obs.ee_pose_error.noise = Unoise(n_min=-pose_n, n_max=pose_n)
            policy_obs.CRI.noise = Unoise(n_min=-cri_n, n_max=cri_n)
        else:
            # Joint-only sweep must not turn on the training pose/CRI noise.
            policy_obs.ee_pose_error.noise = None
            policy_obs.CRI.noise = None
        if args_cli.obs_joint_pos_noise is not None:
            jpos_n = float(args_cli.obs_joint_pos_noise)
            policy_obs.joint_pos.noise = Unoise(n_min=-jpos_n, n_max=jpos_n)
        if args_cli.obs_joint_vel_noise is not None:
            jvel_n = float(args_cli.obs_joint_vel_noise)
            policy_obs.joint_vel.noise = Unoise(n_min=-jvel_n, n_max=jvel_n)
    elif policy_obs is not None and hasattr(policy_obs, "enable_corruption"):
        # Training cfg leaves corruption on. Play scoring is clean unless requested.
        policy_obs.enable_corruption = False
    csv_utils.configure_p2p_style_eval(
        env_cfg,
        disable_resample=disable_resample,
        freeze_finished_envs=disable_reach_success_term,
        deterministic_sim=bool(args_cli.deterministic_sim),
        disable_ovf=bool(args_cli.no_ovf),
        ovf_threshold=float(args_cli.ovf_threshold),
    )
    # Seed before env construction (PhysX / scene spawn paths also read global RNGs).
    _configure_eval_rng(eval_seeds[0])

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play_reach_csv"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if args_cli.dump_reset_env is not None:
        base_env = env.unwrapped
        env_id = int(args_cli.dump_reset_env)
        robot = base_env.scene["robot"]
        ee_term = base_env.command_manager.get_term("ee_pose")
        names = list(robot.joint_names)
        q = robot.data.joint_pos[env_id].detach().cpu().tolist()
        cmd = ee_term.pose_command_b[env_id].detach().cpu().tolist()
        os.makedirs(export_csv_root, exist_ok=True)
        dump_path = os.path.join(export_csv_root, f"reset_env_{env_id}.csv")
        with open(dump_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["name", "value"])
            for name, value in zip(names, q, strict=True):
                writer.writerow([f"q_{name}", value])
            for key, value in zip(
                ("cmd_x", "cmd_y", "cmd_z", "cmd_qw", "cmd_qx", "cmd_qy", "cmd_qz"), cmd, strict=True
            ):
                writer.writerow([key, value])
            handle.flush()
        print(f"[INFO] Dumped reset env {env_id} to {dump_path}", flush=True)
        os._exit(0)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    policy_nn = None
    if version.parse(installed_version) < version.parse("4.0.0"):
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

    base_env = env.unwrapped
    if not isinstance(base_env, ManagerBasedRLEnv):
        raise TypeError(
            f"Trajectory CSV export requires ManagerBasedRLEnv; got {type(base_env).__name__}. "
            "Use a reach Play task (e.g. Isaac-Reach-UR10-Play-v0 or Isaac-Reach-UR10-P2P-Play-v0)."
        )
    csv_utils.apply_strict_reach_params_to_env(base_env, strict_reach_params)

    has_reach_term = "reach_success" in base_env.termination_manager.active_terms
    write_traj = not args_cli.no_traj_csv
    print(f"[INFO] Eval seeds: {eval_seeds}" + (" (multi-seed mode)" if multi_seed else ""))
    print(f"[INFO] Trajectory CSV: {'enabled' if write_traj else 'disabled (--no_traj_csv)'}")
    print(f"[INFO] CSV root: {os.path.abspath(export_csv_root)}")
    print(
        "[INFO] P2P-style one-shot eval: log until first strict reach; "
        "no reach before episode end => failure."
    )
    print(f"[INFO] Env has reach_success termination: {has_reach_term}")
    print(
        "[INFO] Disable reach_success reset: "
        f"{'yes (zip-compatible; policy keeps running after success)' if disable_reach_success_term else 'no (--keep_reach_success_term)'}"
    )
    print(
        "[INFO] Freeze finished actions: "
        f"{'enabled (--freeze_actions)' if args_cli.freeze_actions else 'disabled (default)'}"
    )
    print(
        "[INFO] PhysX deterministic_sim: "
        f"{'enabled (--deterministic_sim)' if args_cli.deterministic_sim else 'disabled (default)'}"
    )
    print(f"[INFO] Mid-episode target resample: {'enabled' if args_cli.keep_resample_on_reach else 'disabled'}")
    if args_cli.no_ovf:
        ovf_msg = "disabled (--no_ovf)"
    else:
        ovf_msg = f"enabled (threshold {float(args_cli.ovf_threshold):g})"
    print(f"[INFO] OVF termination: {ovf_msg}")
    if enable_corruption:
        if touch_pose_cri:
            pose_n = 0.002 if args_cli.obs_pose_noise is None else float(args_cli.obs_pose_noise)
            cri_n = 0.001 if args_cli.obs_cri_noise is None else float(args_cli.obs_cri_noise)
            pose_msg = f"ee_pose_error ±{pose_n:g}, CRI ±{cri_n:g}"
        else:
            pose_msg = "ee_pose_error off, CRI off"
        jpos_msg = "off" if args_cli.obs_joint_pos_noise is None else f"±{float(args_cli.obs_joint_pos_noise):g}"
        jvel_msg = "off" if args_cli.obs_joint_vel_noise is None else f"±{float(args_cli.obs_joint_vel_noise):g}"
        print(f"[INFO] Observation noise: {pose_msg}, joint_pos {jpos_msg}, joint_vel {jvel_msg}")
    else:
        print("[INFO] Observation noise: off (PLAY clean)")
    print(f"[INFO] Policy actions: {'sampled Gaussian' if args_cli.stochastic else 'mean'}")
    print(
        "[INFO] Strict reach thresholds: "
        f"max_distance={strict_reach_params['max_distance']}, "
        f"max_angle_rad={strict_reach_params['max_angle_rad']}, "
        f"max_lin_vel={strict_reach_params['max_lin_vel']}, "
        f"max_ang_vel={strict_reach_params.get('max_ang_vel', 'n/a')}, "
        f"max_lin_acc={strict_reach_params.get('max_lin_acc', float('inf'))}, "
        f"max_ang_acc={strict_reach_params.get('max_ang_acc', float('inf'))}, "
        f"hold_square_max={strict_reach_params.get('hold_square_max', 'cfg')}"
    )
    if args_cli.export_csv_always:
        print("[INFO] export_csv_always: keep traj rows after reach (outcome still one-shot).")
    if args_cli.max_steps > 0:
        print(f"[INFO] Will exit after {args_cli.max_steps} env steps per seed.")

    seed_rows: list[dict[str, Any]] = []
    for seed_idx, eval_seed in enumerate(eval_seeds):
        if multi_seed:
            seed_dir = os.path.join(export_csv_root, f"seed_{eval_seed}")
        else:
            seed_dir = export_csv_root

        if seed_idx > 0:
            print(f"[INFO] Resetting env with seed={eval_seed} ...")
            with torch.no_grad():
                base_env.reset(seed=eval_seed)
                # Clear any recurrent / distributional policy state between seeds.
                if version.parse(installed_version) >= version.parse("4.0.0"):
                    zeros = torch.ones(base_env.num_envs, device=base_env.device, dtype=torch.long)
                    policy.reset(zeros)
                elif policy_nn is not None:
                    zeros = torch.ones(base_env.num_envs, device=base_env.device, dtype=torch.long)
                    policy_nn.reset(zeros)

        record_video = bool(args_cli.video) and seed_idx == 0
        row = _run_one_shot_eval(
            env=env,
            base_env=base_env,
            policy=policy,
            policy_nn=policy_nn,
            strict_reach_params=strict_reach_params,
            export_csv_dir=seed_dir,
            resume_path=resume_path,
            eval_seed=eval_seed,
            write_traj=write_traj,
            record_video=record_video,
        )
        seed_rows.append(row)

        if not simulation_app.is_running():
            break

    if multi_seed and len(seed_rows) == len(eval_seeds):
        multi_path = csv_utils.write_multi_seed_summary_csv(export_csv_root, seed_rows)
        rates = [float(r["reach_percent"]) for r in seed_rows]
        mean_r = sum(rates) / len(rates)
        if len(rates) >= 2:
            std_r = (sum((x - mean_r) ** 2 for x in rates) / (len(rates) - 1)) ** 0.5
        else:
            std_r = 0.0
        print(
            f"[INFO] Multi-seed reach rate: {mean_r:.4f}% ± {std_r:.4f}% "
            f"(n={len(rates)} seeds)"
        )
        print(f"[INFO] Multi-seed summary: {multi_path}")
    elif multi_seed and seed_rows:
        print(
            f"[WARN] Multi-seed run incomplete: {len(seed_rows)}/{len(eval_seeds)} seeds finished; "
            "skipping multi_seed_summary.csv"
        )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
