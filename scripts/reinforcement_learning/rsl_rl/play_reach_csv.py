# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a reach-task RSL-RL checkpoint and export per-env trajectory CSV (q, qd, CRI, reach_event).

Example::

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_reach_csv.py \\
        --task Isaac-Reach-UR10-Play-v0 \\
        --num_envs 4 \\
        --checkpoint logs/rsl_rl/reach_ur10/2026-07-09_19-01-03/model_322200.pt \\
        --export_csv_dir logs/rsl_rl/reach_ur10/2026-07-09_19-01-03/joint_trajectory \\
        --headless --max_steps 5000

CSV columns: ``global_step``, ``sim_time_s``, ``reach_event``, ``max_CRI``, ``q_*``, ``qd_*``, ``CRI_<i>``
(``CRI_*`` are collision-point indices from ``robot.data.CRI``, not joint names).

Reach stop uses **final** thresholds from the env cfg (e.g. ``max_distance=0.03``), not curriculum-relaxed
runtime values. Each env stops logging after its first strict ``reach_success_criteria`` (inclusive row).
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
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--export_csv_dir",
    type=str,
    default=None,
    help="Output directory for env_*_traj.csv (default: <checkpoint_dir>/joint_trajectory).",
)
parser.add_argument(
    "--export_csv_always",
    action="store_true",
    default=False,
    help="Keep logging after each env's first strict reach_success_criteria (default: stop per env at reach).",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Exit after this many env steps (0 = run until the app closes). Useful with --headless.",
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
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.reach.mdp.rewards import reach_success_criteria
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

installed_version = metadata.version("rsl-rl-lib")


def _strict_reach_params_from_env_cfg(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> dict[str, Any] | None:
    """Final reach thresholds from env cfg **before** the sim starts.

    Curriculum terms mutate runtime reward/event params to relaxed values at env init; capture the
    design-time finals here so CSV stop / ``reach_event`` use strict criteria.
    """
    rewards_cfg = getattr(env_cfg, "rewards", None)
    if rewards_cfg is not None:
        term = getattr(rewards_cfg, "reach_success_bonus", None)
        if term is not None and getattr(term, "params", None):
            return dict(term.params)
    events_cfg = getattr(env_cfg, "events", None)
    if events_cfg is not None:
        term = getattr(events_cfg, "resample_ee_pose_on_reach", None)
        if term is not None and getattr(term, "params", None):
            return dict(term.params)
    return None


def _open_traj_csv_writers(
    csv_dir: str,
    num_envs: int,
    joint_names: list[str],
    num_cri_points: int,
) -> tuple[dict[int, IO[str]], dict[int, csv.DictWriter]]:
    """Open one CSV per sub-environment."""
    os.makedirs(csv_dir, exist_ok=True)
    fieldnames = (
        ["global_step", "sim_time_s", "reach_event", "max_CRI"]
        + [f"q_{name}" for name in joint_names]
        + [f"qd_{name}" for name in joint_names]
        + [f"CRI_{i}" for i in range(num_cri_points)]
    )
    files: dict[int, IO[str]] = {}
    writers: dict[int, csv.DictWriter] = {}
    for env_idx in range(num_envs):
        path = os.path.join(csv_dir, f"env_{env_idx}_traj.csv")
        handle = open(path, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        handle.flush()
        files[env_idx] = handle
        writers[env_idx] = writer
    return files, writers


def _append_traj_rows(
    writers: dict[int, csv.DictWriter],
    robot,
    global_step: int,
    sim_time_s: float,
    env_log_mask: torch.Tensor,
    reach_event: torch.Tensor | None,
) -> None:
    """Append one row per env from ``robot`` buffers (call after ``env.step``)."""
    joint_names = robot.joint_names
    with torch.inference_mode():
        joint_pos = robot.data.joint_pos.detach().cpu().numpy()
        joint_vel = robot.data.joint_vel.detach().cpu().numpy()
        cri = robot.data.CRI.detach().cpu().numpy()
        if reach_event is not None:
            reach_np = reach_event.detach().cpu().numpy()
        else:
            reach_np = None

    mask = env_log_mask.detach().bool().cpu()
    num_cri = int(cri.shape[1]) if cri.ndim == 2 else 0

    for env_idx, writer in writers.items():
        if env_idx >= int(mask.shape[0]) or not bool(mask[env_idx].item()):
            continue
        row: dict[str, Any] = {
            "global_step": global_step,
            "sim_time_s": sim_time_s,
            "reach_event": int(bool(reach_np[env_idx])) if reach_np is not None else 0,
            "max_CRI": float(cri[env_idx].max()) if num_cri > 0 else float("nan"),
        }
        for joint_idx, name in enumerate(joint_names):
            row[f"q_{name}"] = float(joint_pos[env_idx, joint_idx])
            row[f"qd_{name}"] = float(joint_vel[env_idx, joint_idx])
        for cri_idx in range(num_cri):
            row[f"CRI_{cri_idx}"] = float(cri[env_idx, cri_idx])
        writer.writerow(row)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent and write trajectory CSV."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.seed = agent_cfg.seed
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
    export_csv_dir = args_cli.export_csv_dir or os.path.join(log_dir, "joint_trajectory")
    env_cfg.log_dir = log_dir

    strict_reach_params = _strict_reach_params_from_env_cfg(env_cfg)
    if strict_reach_params is None:
        raise ValueError(
            "Could not read strict reach_success_criteria params from env cfg "
            "(expected rewards.reach_success_bonus or events.resample_ee_pose_on_reach)."
        )

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
            "Use a reach Play task (e.g. Isaac-Reach-UR10-Play-v0)."
        )

    dt = base_env.step_dt
    obs = env.get_observations()

    play_step = 0
    video_step = 0
    csv_files: dict[int, IO[str]] | None = None
    csv_writers: dict[int, csv.DictWriter] | None = None
    log_active: torch.Tensor | None = None

    print(f"[INFO] Trajectory CSV output: {os.path.abspath(export_csv_dir)}/env_*_traj.csv")
    print(
        "[INFO] Strict reach thresholds (curriculum ignored for CSV): "
        f"max_distance={strict_reach_params['max_distance']}, "
        f"max_angle_rad={strict_reach_params['max_angle_rad']}, "
        f"max_lin_vel={strict_reach_params['max_lin_vel']}, "
        f"max_ang_vel={strict_reach_params['max_ang_vel']}, "
        f"max_lin_acc={strict_reach_params['max_lin_acc']}, "
        f"max_ang_acc={strict_reach_params['max_ang_acc']}"
    )
    if args_cli.export_csv_always:
        print("[INFO] export_csv_always: keep logging after strict reach (reach_event still uses strict criteria).")
    else:
        print("[INFO] Each env CSV ends at the first step where strict reach_success_criteria is true.")
    if args_cli.max_steps > 0:
        print(f"[INFO] Will exit after {args_cli.max_steps} env steps.")

    try:
        while simulation_app.is_running():
            loop_start = time.time()
            with torch.inference_mode():
                reach_cmd_snapshot = (
                    base_env.command_manager.get_command(strict_reach_params["command_name"]).detach().clone()
                )

                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)

                if version.parse(installed_version) >= version.parse("4.0.0"):
                    policy.reset(dones)
                elif policy_nn is not None:
                    policy_nn.reset(dones)

                base_env = env.unwrapped
                play_step += 1

                if csv_writers is None:
                    robot = base_env.scene["robot"]
                    num_cri = int(robot.data.CRI.shape[1])
                    csv_files, csv_writers = _open_traj_csv_writers(
                        export_csv_dir, base_env.num_envs, list(robot.joint_names), num_cri
                    )
                    log_active = torch.ones(base_env.num_envs, device=base_env.device, dtype=torch.bool)

                reach_ev = reach_success_criteria(base_env, command_b=reach_cmd_snapshot, **strict_reach_params)

                assert csv_writers is not None and log_active is not None
                _append_traj_rows(
                    csv_writers,
                    base_env.scene["robot"],
                    play_step,
                    float(play_step) * dt,
                    log_active,
                    reach_ev,
                )
                if csv_files is not None:
                    for handle in csv_files.values():
                        handle.flush()

                if not args_cli.export_csv_always and bool(log_active.any().item()):
                    newly_done = log_active & reach_ev
                    if bool(newly_done.any().item()):
                        log_active &= ~reach_ev
                        for env_idx in newly_done.nonzero(as_tuple=False).view(-1).tolist():
                            print(
                                f"[INFO] env {env_idx}: strict reach_success_criteria satisfied — "
                                "CSV logging stopped for this env."
                            )
                    if not bool(log_active.any().item()):
                        print("[INFO] All envs reached strict criteria; stopping simulation.")
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
            print(f"[INFO] Closed trajectory CSV files under {os.path.abspath(export_csv_dir)}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
