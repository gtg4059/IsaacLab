# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark policy forward pass and CRI+policy pipeline latency for a single robot.

Launch Isaac Sim Simulator first.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from collections.abc import Callable

from isaaclab.app import AppLauncher

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
import scripts.reinforcement_learning.rsl_rl.cli_args as cli_args  # isort: skip

parser = argparse.ArgumentParser(
    description="Benchmark CRI solver and policy forward-pass latency for a single robot."
)
parser.add_argument("--task", type=str, default="Isaac-Reach-UR10-Play-v0", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments (use 1 for single-robot).")
parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations before timing.")
parser.add_argument("--repeats", type=int, default=200, help="Timed iterations per benchmark.")
parser.add_argument(
    "--skip-checkpoint",
    action="store_true",
    default=False,
    help="Skip loading a checkpoint and benchmark with randomly initialized policy weights.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

installed_version = metadata.version("rsl-rl-lib")


def invalidate_cri_cache(env) -> None:
    """Force the CRI solver to recompute on the next access."""
    robot = env.scene["robot"]
    robot.data._CRI.timestamp = -1.0


def gpu_time_ms(fn: Callable[[], None], device: str, warmup: int, repeats: int) -> list[float]:
    """Measure GPU kernel time in milliseconds using CUDA events."""
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    for _ in range(warmup):
        fn()
        if use_cuda:
            torch.cuda.synchronize(device)

    timings: list[float] = []
    for _ in range(repeats):
        if use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize(device)
            timings.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter_ns()
            fn()
            timings.append((time.perf_counter_ns() - t0) / 1e6)
    return timings


def summarize_ms(samples: list[float]) -> dict[str, float]:
    """Return basic latency statistics in milliseconds."""
    sorted_samples = sorted(samples)
    p95_index = max(0, min(len(sorted_samples) - 1, int(0.95 * len(sorted_samples)) - 1))
    return {
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "std": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        "min": min(samples),
        "max": max(samples),
        "p95": sorted_samples[p95_index],
    }


def print_stats(title: str, samples: list[float]) -> dict[str, float]:
    """Print and return latency statistics."""
    stats = summarize_ms(samples)
    print(
        f"{title:28s}  "
        f"mean={stats['mean']:7.3f} ms  "
        f"median={stats['median']:7.3f} ms  "
        f"p95={stats['p95']:7.3f} ms  "
        f"min={stats['min']:7.3f} ms  "
        f"max={stats['max']:7.3f} ms"
    )
    return stats


def measure_cri(base_env) -> None:
    invalidate_cri_cache(base_env)
    _ = base_env.scene["robot"].data.CRI


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Benchmark CRI and policy latency for one robot."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base_env = env.unwrapped
    device = base_env.device

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    if not args_cli.skip_checkpoint:
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        if args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)
        print(f"[INFO] Loading checkpoint: {resume_path}")
        runner.load(resume_path)
    else:
        print("[INFO] Skipping checkpoint load; using randomly initialized policy weights.")

    policy = runner.get_inference_policy(device=device)
    obs_manager = base_env.observation_manager

    # Prime buffers with one sim step so joint states and observation history are valid.
    with torch.inference_mode():
        obs = env.get_observations()
        actions = policy(obs)
        env.step(actions)

    policy_obs = obs["policy"]
    policy_obs_td = obs  # rsl-rl inference policy expects a TensorDict, not a raw tensor.
    print(f"[INFO] Device: {device}")
    print(f"[INFO] num_envs: {base_env.num_envs}")
    print(f"[INFO] policy obs shape: {tuple(policy_obs.shape)}")
    print(f"[INFO] action dim: {env.num_actions}")
    print(f"[INFO] warmup={args_cli.warmup}, repeats={args_cli.repeats}")
    print("-" * 88)

    cri_timings = gpu_time_ms(
        lambda: measure_cri(base_env),
        device=device,
        warmup=args_cli.warmup,
        repeats=args_cli.repeats,
    )
    cri_stats = print_stats("CRI solver", cri_timings)

    policy_timings = gpu_time_ms(
        lambda: policy(policy_obs_td), device=device, warmup=args_cli.warmup, repeats=args_cli.repeats
    )
    policy_stats = print_stats("Policy forward", policy_timings)

    def pipeline_cri_policy() -> None:
        measure_cri(base_env)
        policy(policy_obs_td)

    pipeline_timings = gpu_time_ms(
        pipeline_cri_policy, device=device, warmup=args_cli.warmup, repeats=args_cli.repeats
    )
    pipeline_stats = print_stats("Pipeline (CRI + policy)", pipeline_timings)

    obs_timings = gpu_time_ms(
        lambda: (invalidate_cri_cache(base_env), obs_manager.compute_group("policy"))[1],
        device=device,
        warmup=args_cli.warmup,
        repeats=args_cli.repeats,
    )
    obs_stats = print_stats("Policy obs (incl. CRI)", obs_timings)

    def pipeline_obs_policy() -> None:
        invalidate_cri_cache(base_env)
        obs_tensor = obs_manager.compute_group("policy")
        policy(TensorDict({"policy": obs_tensor}, batch_size=[base_env.num_envs]))

    full_pipeline_timings = gpu_time_ms(
        pipeline_obs_policy, device=device, warmup=args_cli.warmup, repeats=args_cli.repeats
    )
    full_pipeline_stats = print_stats("Pipeline (obs + policy)", full_pipeline_timings)

    print("-" * 88)
    print(
        "[SUMMARY] single-robot inference latency\n"
        f"  CRI only              : {cri_stats['mean']:.3f} ms (median {cri_stats['median']:.3f} ms)\n"
        f"  Policy forward only   : {policy_stats['mean']:.3f} ms (median {policy_stats['median']:.3f} ms)\n"
        f"  CRI + policy          : {pipeline_stats['mean']:.3f} ms (median {pipeline_stats['median']:.3f} ms)\n"
        f"  Obs (incl. CRI)       : {obs_stats['mean']:.3f} ms (median {obs_stats['median']:.3f} ms)\n"
        f"  Obs + policy          : {full_pipeline_stats['mean']:.3f} ms (median {full_pipeline_stats['median']:.3f} ms)\n"
        f"  Sum check (CRI+policy): {cri_stats['mean'] + policy_stats['mean']:.3f} ms"
    )
    print(
        "[NOTE] 'CRI + policy' uses a fixed observation tensor for the policy input to isolate CRI cost.\n"
        "       'Obs + policy' is the deployment-realistic path: build the full policy observation, then infer."
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
