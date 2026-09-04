# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Short headless run of Isaac-Reach-UR10-CRI-F-Play-v0 to verify the CBF filter."""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify CBF CRI filter on CRI-F Play.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Reach-UR10-CRI-F-Play-v0", help="Name of the task.")
parser.add_argument("--steps", type=int, default=40, help="Control steps after reset.")
parser.add_argument(
    "--trace",
    type=str,
    default="/tmp/cbf_cri_trace.json",
    help="Write per-step CRI raw/filtered series to this JSON path.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def _max_cri(cri: torch.Tensor) -> float:
    return float(cri.amax().item()) if cri.numel() > 0 else float("nan")


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    arm = unwrapped.action_manager.get_term("arm_action")

    print(f"[VERIFY] Gym observation space: {env.observation_space}")
    print(f"[VERIFY] Gym action space: {env.action_space}")
    print(
        f"[VERIFY] filter_enabled={arm.cfg.filter_enabled} cri_limit={arm.cfg.cri_limit} "
        f"cbf_alpha={arm.cfg.cbf_alpha} approach={robot.data.cri_filter_approach_limit}",
        flush=True,
    )

    obs_names = list(unwrapped.observation_manager.active_terms["policy"])
    if "cri_scale" in obs_names:
        raise RuntimeError(f"policy still exposes cri_scale: {obs_names}")
    print(f"[VERIFY] policy terms: {obs_names}", flush=True)

    env.reset()
    cri0 = _max_cri(robot.data.cri_filter_pre)
    print(f"[VERIFY] reset CRI={cri0:.6f} (expect 0)", flush=True)

    max_cri_pre = cri0
    max_cri_out = 0.0
    max_delta = 0.0
    saw_correction = False
    last_alpha = float(robot.data.cri_filter_cbf_alpha)
    last_approach = float(robot.data.cri_filter_approach_limit)
    series: list[dict[str, float]] = [
        {"step": 0.0, "cri_pre": cri0, "cri_out": 0.0, "delta": 0.0},
    ]
    os.environ["SFD_CRI_TIMING"] = "1"
    os.environ["SFD_CRI_TIMING_PRINT_EVERY"] = "0"
    warmup = min(8, max(0, int(args_cli.steps) // 5))
    filter_ms: list[float] = []
    atmotion_ms: list[float] = []
    filter_off_ms: list[float] = []
    solver = robot.data.solver

    def _sync_ms(fn) -> float:
        if unwrapped.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if unwrapped.device.startswith("cuda"):
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0

    for step in range(int(args_cli.steps)):
        with torch.inference_mode():
            # Large random velocities so cri_pre can cross the 0.96 hard limit.
            actions = 6 * torch.rand(env.action_space.shape, device=unwrapped.device) - 3
            env.step(actions)

        cri_pre = _max_cri(robot.data.cri_filter_pre)
        cri_out = _max_cri(robot.data.cri_filter_out)
        delta = float(robot.data.cri_filter_delta.amax().item())
        max_cri_pre = max(max_cri_pre, cri_pre)
        max_cri_out = max(max_cri_out, cri_out)
        max_delta = max(max_delta, delta)
        last_alpha = float(robot.data.cri_filter_cbf_alpha)
        last_approach = float(robot.data.cri_filter_approach_limit)
        series.append({"step": float(step + 1), "cri_pre": cri_pre, "cri_out": cri_out, "delta": delta})
        if delta > 1e-6:
            saw_correction = True
        print(
            f"[VERIFY] step={step+1} cri_pre={cri_pre:.4f} cri_out={cri_out:.4f} "
            f"delta={delta:.6f} alpha={last_alpha:.4f} approach={last_approach:.4f}",
            flush=True,
        )
        if step + 1 <= warmup:
            continue
        q_in, _ = robot.data._cri_input_tensors()
        qd_rl = robot.data.cri_filter_qd_rl.detach().clone()
        q_bench = q_in.detach().clone()
        with torch.inference_mode():
            filter_ms.append(_sync_ms(lambda: solver.run_cri_filter(q_bench, qd_rl)))
            atmotion_ms.append(
                _sync_ms(lambda: solver.RunSolver_CUDA_CRI_AtMotionState(q_bench, qd_rl))
            )
            filter_off_ms.append(
                _sync_ms(lambda: solver.run_cri_filter(q_bench, qd_rl, enabled=False))
            )

    solves = float(robot.data.cri_filter_solves_per_step)
    print(
        f"[VERIFY] done steps={args_cli.steps} max_cri_pre={max_cri_pre:.4f} max_cri_out={max_cri_out:.4f} "
        f"max_delta={max_delta:.6f} solves/step={solves:.3f} alpha={last_alpha:.4f} "
        f"approach={last_approach:.4f} saw_correction={saw_correction}",
        flush=True,
    )

    failures = []
    if cri0 > 1e-6:
        failures.append(f"reset CRI={cri0} expected 0")
    if abs(last_alpha - 0.02) > 1e-6:
        failures.append(f"cbf_alpha={last_alpha} expected 0.02")
    if abs(last_approach - 0.96 * (1.0 - 0.02)) > 1e-4:
        failures.append(f"approach_limit={last_approach} expected {0.96 * 0.98}")
    if max_cri_out > 0.96 + 1e-6:
        failures.append(f"max_cri_out={max_cri_out} exceeded hard clamp 0.96")
    if solves < 0.99:
        failures.append(f"solves_per_step={solves} expected ~1")
    if failures:
        raise RuntimeError("CBF filter verification failed: " + "; ".join(failures))
    print("[VERIFY] CBF filter library path OK", flush=True)

    def _summarize(samples: list[float]) -> dict[str, float]:
        if not samples:
            return {}
        ordered = sorted(samples)
        n = len(ordered)
        p = lambda q: ordered[min(n - 1, int(round((q / 100.0) * (n - 1))))]
        return {
            "n": float(n),
            "min_ms": ordered[0],
            "p50_ms": p(50),
            "mean_ms": sum(ordered) / n,
            "p95_ms": p(95),
            "max_ms": ordered[-1],
        }

    timing = {
        "warmup_discarded": warmup,
        "filter_on": _summarize(filter_ms),
        "filter_off": _summarize(filter_off_ms),
        "atmotion": _summarize(atmotion_ms),
        "note": "GPU synchronize before/after each library call. batch=1 UR10.",
    }
    print("[VERIFY] library wall times (ms, GPU-sync):", flush=True)
    for name, stats in (
        ("filter_on  run_cri_filter", timing["filter_on"]),
        ("filter_off run_cri_filter", timing["filter_off"]),
        ("AtMotionState (no filter)", timing["atmotion"]),
    ):
        if not stats:
            continue
        print(
            f"  {name}: n={int(stats['n'])} min={stats['min_ms']:.2f} "
            f"p50={stats['p50_ms']:.2f} mean={stats['mean_ms']:.2f} "
            f"p95={stats['p95_ms']:.2f} max={stats['max_ms']:.2f}",
            flush=True,
        )
    payload = {
        "task": args_cli.task,
        "steps": int(args_cli.steps),
        "cri_limit": float(arm.cfg.cri_limit),
        "cbf_alpha": last_alpha,
        "approach_limit": last_approach,
        "max_cri_pre": max_cri_pre,
        "max_cri_out": max_cri_out,
        "max_delta": max_delta,
        "timing": timing,
        "series": series,
    }
    with open(args_cli.trace, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[VERIFY] wrote trace {args_cli.trace}", flush=True)
    env.close()


if __name__ == "__main__":
    os.environ.setdefault("SFD_ALLOC_WARMUP_ROUNDS", "3")
    main()
    simulation_app.close()
