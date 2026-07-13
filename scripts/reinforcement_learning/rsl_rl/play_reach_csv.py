# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a reach-task RSL-RL checkpoint and export per-env trajectory CSV (q, qd, CRI, reach_event).

Evaluation protocol matches ``Isaac-Reach-UR10-P2P-Play-v0`` semantics:

* one target per env attempt (in-episode ``resample_ee_pose_on_reach`` disabled)
* strict final reach thresholds (curriculum easing disabled)
* trajectory logged only until first strict reach (inclusive), or until episode failure
* attempt outcome: ``success`` on reach; ``fail_timeout`` / ``fail_ovf`` / ``fail_other`` if the
  episode ends without reach (same idea as P2P: reach terminates success, timeout is failure)
* stops after every env finishes exactly one attempt

Example (continuous-reach policy, P2P-style one-shot scoring)::

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_reach_csv.py \\
        --task Isaac-Reach-UR10-Play-v0 \\
        --num_envs 4 \\
        --checkpoint logs/rsl_rl/reach_ur10/<run>/model_*.pt \\
        --export_csv_dir /path/to/out \\
        --headless

For a P2P-trained policy use ``--task Isaac-Reach-UR10-P2P-Play-v0`` (env already ends on reach).

CSV columns: ``global_step``, ``sim_time_s``, ``reach_event``, ``max_CRI``, ``q_*``, ``qd_*``, ``CRI_<i>``
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
    help="Keep logging after first strict reach (default: stop per env at reach, P2P-style).",
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
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import reach_traj_csv_utils as csv_utils

installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent and write trajectory CSV under P2P-style one-shot eval."""
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

    strict_reach_params = csv_utils.strict_reach_params_from_env_cfg(env_cfg)
    if strict_reach_params is None:
        raise ValueError(
            "Could not read strict reach_success_criteria params from env cfg "
            "(expected rewards.reach_success_bonus or events.resample_ee_pose_on_reach)."
        )

    disable_resample = not args_cli.keep_resample_on_reach
    csv_utils.configure_p2p_style_eval(env_cfg, disable_resample=disable_resample)

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
            "Use a reach Play task (e.g. Isaac-Reach-UR10-Play-v0 or Isaac-Reach-UR10-P2P-Play-v0)."
        )
    csv_utils.apply_strict_reach_params_to_env(base_env, strict_reach_params)

    dt = base_env.step_dt
    obs = env.get_observations()

    play_step = 0
    video_step = 0
    csv_files: dict[int, IO[str]] | None = None
    csv_writers: dict[int, csv.DictWriter] | None = None
    episode_reach_file: IO[str] | None = None
    episode_reach_writer: csv.DictWriter | None = None
    # Per-env: still logging / still scoring this one-shot attempt (P2P-style).
    log_active: torch.Tensor | None = None
    attempt_active: torch.Tensor | None = None
    episode_ids: torch.Tensor | None = None
    total_episodes = 0
    total_reached_episodes = 0

    has_reach_term = "reach_success" in base_env.termination_manager.active_terms
    print(f"[INFO] Trajectory CSV output: {os.path.abspath(export_csv_dir)}/env_*_traj.csv")
    print(f"[INFO] Episode reach summary: {os.path.abspath(export_csv_dir)}/episode_reach.csv")
    print(
        "[INFO] P2P-style one-shot eval: log until first strict reach; "
        "no reach before episode end => failure."
    )
    print(f"[INFO] Env has reach_success termination: {has_reach_term}")
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
    if args_cli.export_csv_always:
        print("[INFO] export_csv_always: keep traj rows after reach (outcome still one-shot).")
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
                    csv_files, csv_writers = csv_utils.open_traj_csv_writers(
                        export_csv_dir, base_env.num_envs, list(robot.joint_names), num_cri
                    )
                    episode_reach_file, episode_reach_writer = csv_utils.open_episode_reach_csv(export_csv_dir)
                    log_active = torch.ones(base_env.num_envs, device=base_env.device, dtype=torch.bool)
                    attempt_active = torch.ones(base_env.num_envs, device=base_env.device, dtype=torch.bool)
                    episode_ids = torch.zeros(base_env.num_envs, device=base_env.device, dtype=torch.long)

                assert (
                    csv_writers is not None
                    and log_active is not None
                    and attempt_active is not None
                    and episode_ids is not None
                    and episode_reach_writer is not None
                    and episode_reach_file is not None
                )

                reach_ev = csv_utils.resolve_reach_event(base_env, reach_cmd_snapshot, strict_reach_params)
                done_mask = dones.detach().bool().view(-1)

                # Traj rows only while attempt/log active (includes the reach success row).
                csv_utils.append_traj_rows(
                    csv_writers,
                    base_env.scene["robot"],
                    play_step,
                    float(play_step) * dt,
                    log_active & attempt_active,
                    reach_ev,
                )
                if csv_files is not None:
                    for handle in csv_files.values():
                        handle.flush()

                # 1) Success: first strict reach ends the attempt (P2P reach_success equivalent).
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

                # 2) Failure: episode ended without prior success (timeout / OVF / other).
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
                        )
                    total_episodes += int(newly_fail.sum().item())
                    episode_ids[newly_fail] += 1
                    attempt_active &= ~newly_fail
                    log_active &= ~newly_fail

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
            print(f"[INFO] Closed trajectory CSV files under {os.path.abspath(export_csv_dir)}")
        if episode_reach_file is not None:
            episode_reach_file.close()
            pct = csv_utils.reach_percent(total_reached_episodes, total_episodes)
            summary_path = csv_utils.write_reach_summary_csv(export_csv_dir, total_reached_episodes, total_episodes)
            print(
                f"[INFO] Episode reach summary: {total_reached_episodes}/{total_episodes} "
                f"reached ({pct:.2f}%); failed={total_episodes - total_reached_episodes}"
            )
            print(f"[INFO] Per-episode CSV: {os.path.abspath(export_csv_dir)}/episode_reach.csv")
            print(f"[INFO] Aggregate CSV: {summary_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
