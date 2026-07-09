# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark CRI (Collision Risk Index) solver inference time in simulation.

For CRI + policy (action) latency, use benchmark_cri_policy_inference.py instead.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

os.environ.setdefault("SFD_CRI_TIMING", "1")

parser = argparse.ArgumentParser(description="Benchmark CRI solver inference time.")
parser.add_argument("--task", type=str, default="Isaac-Reach-UR10-v0", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel environments.")
parser.add_argument("--warmup_steps", type=int, default=20, help="Warmup steps before measurement.")
parser.add_argument("--benchmark_steps", type=int, default=200, help="Measured steps after warmup.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def _format_stats(stats: dict[str, float | int], num_envs: int) -> str:
    count = int(stats["count"])
    mean_ms = float(stats["mean_s"]) * 1000.0
    min_ms = float(stats["min_s"]) * 1000.0
    max_ms = float(stats["max_s"]) * 1000.0
    last_ms = float(stats["last_s"]) * 1000.0
    per_env_us = (float(stats["mean_s"]) / num_envs) * 1e6 if num_envs else 0.0
    return (
        f"count={count}, last={last_ms:.3f} ms, mean={mean_ms:.3f} ms, "
        f"min={min_ms:.3f} ms, max={max_ms:.3f} ms, per_env={per_env_us:.2f} us"
    )


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    robot = env.unwrapped.scene["robot"]

    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Device: {env.unwrapped.device}")
    print(f"[INFO] Num envs: {env.unwrapped.num_envs}")
    print(f"[INFO] Warmup steps: {args_cli.warmup_steps}, benchmark steps: {args_cli.benchmark_steps}")

    env.reset()
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

    with torch.inference_mode():
        for _ in range(args_cli.warmup_steps):
            env.step(actions)

        robot.data.reset_cri_inference_stats()

        for step_idx in range(args_cli.benchmark_steps):
            env.step(actions)
            if not simulation_app.is_running():
                break

    stats = robot.data.get_cri_inference_stats()
    num_envs = env.unwrapped.num_envs
    print("[RESULT] CRI inference timing:")
    print(f"  {_format_stats(stats, num_envs)}")
    if stats["count"]:
        throughput = num_envs / float(stats["mean_s"])
        print(f"  throughput: {throughput:,.0f} env-steps/s (CRI solver only)")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
