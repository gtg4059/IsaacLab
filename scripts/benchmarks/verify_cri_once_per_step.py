#!/usr/bin/env python3
"""Verify CRI solver invocations per env step for Isaac-Reach-UR10-v0."""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("SFD_CRI_TIMING", "1")
os.environ.setdefault("SFD_CRI_TIMING_PRINT_EVERY", "24")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify CRI is solved once (or twice) per env step.")
parser.add_argument("--num_steps", type=int, default=48)
parser.add_argument("--task", type=str, default="Isaac-Reach-UR10-v0")
parser.add_argument("--num_envs", type=int, default=512)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    # One-step episodes: every env step triggers timeout reset (matches train's per-step dirty CRI).
    env_cfg.episode_length_s = 1.0e-6
    env = gym.make(args_cli.task, cfg=env_cfg)

    base = env.unwrapped
    robot = base.scene["robot"]
    data = robot.data

    env.reset()
    data.reset_cri_inference_stats()

    action_dim = env.action_space.shape[-1]
    t0 = time.perf_counter()
    for _ in range(args_cli.num_steps):
        actions = torch.zeros((base.num_envs, action_dim), device=base.device)
        env.step(actions)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    n = args_cli.num_steps
    solves = data._cri_inference_count
    print("\n========== CRI once-per-step verification ==========")
    print(f"env_steps={n}  wall={elapsed:.3f}s  mean_step={elapsed / n * 1000:.1f}ms")
    print(f"solver_calls={solves}  calls_per_step={solves / max(n, 1):.3f}")
    print(
        f"paths: full={data._cri_path_full} dirty_full={data._cri_path_dirty_full} "
        f"dirty_ident_solve={data._cri_path_dirty_identical_solve} dirty_cache={data._cri_path_dirty_cache} "
        f"dirty_skip={data._cri_path_dirty_skip}"
    )
    cri_ovf_rew = getattr(base.cfg.rewards, "CRI_OVF", None)
    ovf_term = getattr(base.cfg.terminations, "OVF", None)
    print(f"note: CRI_OVF reward active={cri_ovf_rew is not None}")
    print(f"note: OVF termination active={ovf_term is not None}")
    ratio = solves / max(n, 1)
    if ratio >= 1.5:
        print("RESULT: DOUBLE (or more) CRI solves per step — matches ~1.2s collect regression.")
    elif ratio <= 1.15:
        print("RESULT: ~1 CRI solve per step — matches ~0.8s collect target.")
    else:
        print("RESULT: mixed / partial double-solve.")
    print("====================================================\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
