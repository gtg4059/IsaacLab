# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark CRI solver and policy inference latency (single- or multi-env).

Measures:
  - CRI solver output
  - Policy forward (action output)
  - Policy observation build (includes CRI)
  - End-to-end inference pipelines

Example (50Hz deploy path only — default):
    ./isaaclab.sh -p scripts/benchmarks/benchmark_cri_policy_inference.py \\
        --task Isaac-Reach-UR10-Play-v0 \\
        --checkpoint logs/rsl_rl/reach_ur10/.../model_0.pt \\
        --num_envs 1 --headless --warmup 100 --repeats 1000

All isolated phases (CRI only, policy only, …): add ``--full-bench``.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
import time

# Tail-latency env: set before AppLauncher / first CUDA alloc when possible.
if os.environ.get("SFD_SPIKE_MITIGATION", "1") not in ("0", "false", "FALSE"):
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.9",
    )
    os.environ.setdefault("CUDA_MODULE_LOADING", "EAGER")
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "32")
    os.environ.setdefault("SAFETICS_USE_TENSOR_CALC_LOOP", "1")

import numpy as np
from isaaclab.app import AppLauncher

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../reinforcement_learning/rsl_rl"))
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Benchmark CRI + policy inference latency.")
parser.add_argument("--task", type=str, default="Isaac-Reach-UR10-Play-v0", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--warmup", type=int, default=50, help="Warmup env steps before measurement.")
parser.add_argument("--cri-warmup", type=int, default=None, help="Extra CRI-only warm-up rounds after env warmup.")
parser.add_argument("--repeats", type=int, default=200, help="Number of timed inference repeats.")
parser.add_argument("--budget-ms", type=float, default=20.0, help="50Hz hard budget in ms for pass-rate report.")
parser.add_argument(
    "--burn-in",
    type=int,
    default=None,
    help="Per-phase samples excluded from steady-state stats (default: SFD_BENCH_BURN_IN or 25).",
)
parser.add_argument(
    "--full-bench",
    action="store_true",
    default=False,
    help="Run all isolated phases. Default: deploy (obs+policy) only.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
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
from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from tensordict import TensorDict

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401

installed_version = metadata.version("rsl-rl-lib")


def _sync_device(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _drain_gpu(device: str) -> None:
    """Drain PhysX + PyTorch queues before timed sections (reduces tail spikes)."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        # Second sync: occasional driver latency shows up on first wait only.
        torch.cuda.synchronize()


def _advance_physics(env: RslRlVecEnvWrapper, actions: torch.Tensor) -> None:
    """Advance simulation to the post-physics state (before observation compute)."""
    base = env.unwrapped
    base.action_manager.process_action(actions.to(base.device))
    is_rendering = base.sim.has_gui() or base.sim.has_rtx_sensors()
    for _ in range(base.cfg.decimation):
        base._sim_step_counter += 1
        base.action_manager.apply_action()
        base.scene.write_data_to_sim()
        base.sim.step(render=False)
        if base._sim_step_counter % base.cfg.sim.render_interval == 0 and is_rendering:
            base.sim.render()
        base.scene.update(dt=base.physics_dt)
    base.command_manager.compute(dt=base.step_dt)
    if "interval" in base.event_manager.available_modes:
        base.event_manager.apply(mode="interval", dt=base.step_dt)


class _LatencyRecorder:
    """Collects wall-clock latency samples in milliseconds."""

    def __init__(self) -> None:
        self.samples: list[float] = []
        self.indexed: list[tuple[int, float]] = []

    def record(self, elapsed_s: float, index: int = -1) -> None:
        ms = elapsed_s * 1000.0
        self.samples.append(ms)
        self.indexed.append((index, ms))

    def summary(self) -> dict[str, float]:
        if not self.samples:
            return {"mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0, "pass_pct": 100.0}
        arr = np.asarray(self.samples, dtype=np.float64)
        budget = float(os.environ.get("SFD_BENCH_BUDGET_MS", "20"))
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "pass_pct": float(100.0 * np.mean(arr <= budget)),
            "violations": int(np.sum(arr > budget)),
        }

    def outliers(self, budget_ms: float, top_n: int = 8) -> list[tuple[int, float]]:
        bad = [(i, ms) for i, ms in self.indexed if ms > budget_ms]
        bad.sort(key=lambda x: x[1], reverse=True)
        return bad[:top_n]

    def summary_after(self, skip_first: int) -> dict[str, float]:
        if skip_first <= 0 or skip_first >= len(self.samples):
            return self.summary()
        trimmed = _LatencyRecorder()
        for idx, ms in self.indexed[skip_first:]:
            trimmed.samples.append(ms)
            trimmed.indexed.append((idx, ms))
        return trimmed.summary()

    def outliers_after(self, skip_first: int, budget_ms: float, top_n: int = 8) -> list[tuple[int, float]]:
        bad = [(i, ms) for i, ms in self.indexed[skip_first:] if ms > budget_ms]
        bad.sort(key=lambda x: x[1], reverse=True)
        return bad[:top_n]


def _print_outliers(label: str, recorder: _LatencyRecorder, budget_ms: float, top_n: int = 8) -> None:
    hits = recorder.outliers(budget_ms, top_n)
    if not hits:
        return
    print(f"[outlier] {label} violations={recorder.summary().get('violations', len(hits))} top:")
    for idx, ms in hits:
        print(f"  iter={idx} wall_ms={ms:.3f} (+{ms - budget_ms:.3f} vs {budget_ms:.1f}ms budget)")


def _phase_warmup(
    robot,
    env,
    device: str,
    actions: torch.Tensor,
    rounds: int,
    *,
    include_obs: bool = False,
    policy=None,
    fixed_obs=None,
) -> None:
    if rounds <= 0:
        return
    for _ in range(rounds):
        _advance_physics(env, actions)
        _drain_gpu(device)
        _ = robot.data.CRI
        if include_obs:
            env.unwrapped.observation_manager.compute(update_history=True)
        if policy is not None and fixed_obs is not None:
            policy(fixed_obs)
    _drain_gpu(device)


def _format_row(name: str, stats: dict[str, float]) -> str:
    return (
        f"{name:<28}  mean={stats['mean']:7.3f} ms  median={stats['median']:7.3f} ms  "
        f"p95={stats['p95']:7.3f} ms  p99={stats.get('p99', stats['p95']):7.3f} ms  "
        f"min={stats['min']:7.3f} ms  max={stats['max']:7.3f} ms  pass={stats.get('pass_pct', 0):5.1f}%"
    )


def _bench_cri_only(robot, env, device: str, actions: torch.Tensor, repeats: int, recorder: _LatencyRecorder) -> None:
    for i in range(repeats):
        _advance_physics(env, actions)
        _drain_gpu(device)
        t0 = time.perf_counter()
        _ = robot.data.CRI
        _sync_device(device)
        recorder.record(time.perf_counter() - t0, i)


def _bench_policy_only(device: str, policy, fixed_obs, repeats: int, recorder: _LatencyRecorder) -> None:
    for i in range(repeats):
        _sync_device(device)
        t0 = time.perf_counter()
        policy(fixed_obs)
        _sync_device(device)
        recorder.record(time.perf_counter() - t0, i)


def _bench_cri_policy(
    robot,
    env,
    device: str,
    actions: torch.Tensor,
    policy,
    fixed_obs,
    repeats: int,
    recorder: _LatencyRecorder,
    cri_part: _LatencyRecorder | None = None,
) -> None:
    for i in range(repeats):
        _advance_physics(env, actions)
        _drain_gpu(device)
        t0 = time.perf_counter()
        _ = robot.data.CRI
        t1 = time.perf_counter()
        policy(fixed_obs)
        _sync_device(device)
        if cri_part is not None:
            cri_part.record(t1 - t0, i)
        recorder.record(time.perf_counter() - t0, i)


def _bench_obs_only(env, device: str, actions: torch.Tensor, repeats: int, recorder: _LatencyRecorder) -> None:
    for i in range(repeats):
        _advance_physics(env, actions)
        _drain_gpu(device)
        t0 = time.perf_counter()
        env.unwrapped.observation_manager.compute(update_history=True)
        _sync_device(device)
        recorder.record(time.perf_counter() - t0, i)


def _bench_obs_policy(
    env,
    device: str,
    actions: torch.Tensor,
    policy,
    repeats: int,
    recorder: _LatencyRecorder | None,
    *,
    start_index: int = 0,
    obs_td: TensorDict | None = None,
) -> tuple[torch.Tensor, TensorDict | None]:
    last_actions = actions
    for i in range(repeats):
        _advance_physics(env, last_actions)
        _drain_gpu(device)
        t0 = time.perf_counter()
        obs_dict = env.unwrapped.observation_manager.compute(update_history=True)
        if obs_td is None:
            obs_td = TensorDict(obs_dict, batch_size=[env.unwrapped.num_envs])
        else:
            obs_td.update(obs_dict)
        last_actions = policy(obs_td)
        _sync_device(device)
        if recorder is not None:
            recorder.record(time.perf_counter() - t0, start_index + i)
    return last_actions, obs_td


def _warmup_deploy(
    env, device: str, actions: torch.Tensor, policy, rounds: int, obs_td: TensorDict | None = None
) -> tuple[torch.Tensor, TensorDict | None]:
    """Full obs+policy path warm-up (matches deploy control loop)."""
    if rounds <= 0:
        return actions, obs_td
    return _bench_obs_policy(env, device, actions, policy, rounds, None, obs_td=obs_td)


def _print_deploy_report(
    deploy_steady: dict[str, float],
    *,
    budget_ms: float,
    repeats: int,
    burn_in: int,
    recorder: _LatencyRecorder,
    deploy_all: dict[str, float] | None = None,
) -> None:
    steady_n = max(0, repeats - burn_in)
    print("-" * 88)
    print(_format_row("Deploy steady-state", deploy_steady))
    if deploy_all is not None and burn_in > 0:
        print(_format_row(f"Deploy all samples (incl. burn-in {burn_in})", deploy_all))
    print("-" * 88)
    print("[SUMMARY] 50Hz deploy (obs incl. CRI → policy → actions)")
    print(f"  steady mean   : {deploy_steady['mean']:.3f} ms (median {deploy_steady['median']:.3f} ms)")
    print(f"  steady p95    : {deploy_steady['p95']:.3f} ms")
    print(f"  steady p99    : {deploy_steady['p99']:.3f} ms")
    print(f"  steady min/max: {deploy_steady['min']:.3f} / {deploy_steady['max']:.3f} ms")
    print(
        f"  50Hz pass     : {deploy_steady.get('pass_pct', 0):.1f}% @ {budget_ms}ms "
        f"({deploy_steady.get('violations', 0)} violations / {steady_n} steady samples)"
    )
    if burn_in > 0:
        print(f"  burn-in       : first {burn_in} measured samples excluded (cold-start guard)")
    if deploy_all is not None and deploy_all["max"] > deploy_steady["max"] + 0.5:
        print(
            f"  all-sample max: {deploy_all['max']:.3f} ms "
            f"(includes burn-in; not used for 50Hz verdict)"
        )
    print("[NOTE] physics step before obs is excluded from the timed window.")
    hits = recorder.outliers_after(burn_in, budget_ms)
    if hits:
        print(f"[outlier] deploy steady violations={deploy_steady.get('violations', len(hits))} top:")
        for idx, ms in hits:
            print(f"  iter={idx} wall_ms={ms:.3f} (+{ms - budget_ms:.3f} vs {budget_ms:.1f}ms budget)")
    if deploy_steady["max"] > budget_ms:
        print(
            f"[HINT] steady max {deploy_steady['max']:.2f}ms > {budget_ms}ms — "
            "try: echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"
        )
    elif deploy_steady.get("pass_pct", 0) >= 100.0:
        print(f"[OK] deploy steady-state meets {budget_ms}ms for all {steady_n} samples.")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "policy"):
        env_cfg.observations.policy.enable_corruption = False

    log_root_path = f"logs/rsl_rl/{agent_cfg.experiment_name}"
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    device = env.unwrapped.device
    robot = env.unwrapped.scene["robot"]

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    print(f"[INFO] Loading checkpoint: {resume_path}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=device)

    obs = env.get_observations()
    policy_obs_shape = tuple(obs["policy"].shape)
    action_dim = env.num_actions
    fixed_obs = obs.clone()

    print(f"[INFO] Device: {device}")
    print(f"[INFO] num_envs: {env.unwrapped.num_envs}")
    print(f"[INFO] policy obs shape: {policy_obs_shape}")
    print(f"[INFO] action dim: {action_dim}")
    print(f"[INFO] warmup={args_cli.warmup}, repeats={args_cli.repeats}")
    os.environ["SFD_BENCH_BUDGET_MS"] = str(args_cli.budget_ms)
    spike_on = os.environ.get("SFD_SPIKE_MITIGATION", "1") not in ("0", "false", "FALSE")
    alloc_warmup = os.environ.get("SFD_ALLOC_WARMUP_ROUNDS", "15")
    cri_warmup = (
        args_cli.cri_warmup
        if args_cli.cri_warmup is not None
        else int(os.environ.get("SFD_BENCH_CRI_WARMUP", str(max(50, args_cli.repeats // 20))))
    )
    phase_warmup = int(os.environ.get("SFD_BENCH_PHASE_WARMUP", str(max(20, args_cli.repeats // 50))))
    full_bench = args_cli.full_bench or os.environ.get("SFD_BENCH_FULL", "0") in ("1", "true", "TRUE")
    burn_in = (
        args_cli.burn_in
        if args_cli.burn_in is not None
        else int(os.environ.get("SFD_BENCH_BURN_IN", "50" if not full_bench else "25"))
    )
    print(
        f"[INFO] mode={'full-bench' if full_bench else 'deploy-only'} "
        f"spike_mitigation={spike_on} SFD_ALLOC_WARMUP_ROUNDS={alloc_warmup} "
        f"deploy_warmup={cri_warmup} phase_warmup={phase_warmup} burn_in={burn_in} "
        f"budget_ms={args_cli.budget_ms}"
    )
    if alloc_warmup == "15" and os.environ.get("SFD_ALLOC_WARMUP_ROUNDS") is None:
        print("[INFO] tip: prefix command with SFD_ALLOC_WARMUP_ROUNDS=10 (same line as ./isaaclab.sh)")

    last_actions = torch.zeros(env.num_envs, action_dim, device=device)

    with torch.inference_mode():
        for _ in range(args_cli.warmup):
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            last_actions = actions
            if not simulation_app.is_running():
                break

        fixed_obs = obs.clone()
        # Keep last_actions from env warmup (do not zero — avoids cold physics on deploy path).

        obs_policy_rec = _LatencyRecorder()
        deploy_obs_td: TensorDict | None = None

        if full_bench:
            print(f"[INFO] post-env CRI warm-up: {cri_warmup} physics+CRI steps")
            _bench_cri_only(robot, env, device, last_actions, cri_warmup, _LatencyRecorder())
            _sync_device(device)

            cri_rec = _LatencyRecorder()
            policy_rec = _LatencyRecorder()
            cri_policy_rec = _LatencyRecorder()
            cri_policy_cri_part = _LatencyRecorder()
            obs_rec = _LatencyRecorder()

            def _run_phase(label: str, fn, *, warmup_mode: str = "cri") -> None:
                print(f"[INFO] measuring {label}...")
                if phase_warmup > 0:
                    print(f"[INFO]   phase warm-up ({warmup_mode}): {phase_warmup} steps")
                    if warmup_mode == "cri":
                        _phase_warmup(robot, env, device, last_actions, phase_warmup)
                    elif warmup_mode == "cri_policy":
                        _phase_warmup(
                            robot, env, device, last_actions, phase_warmup, policy=policy, fixed_obs=fixed_obs
                        )
                    elif warmup_mode == "obs":
                        obs_warm = max(phase_warmup, burn_in + 10)
                        print(f"[INFO]   phase warm-up (obs): {obs_warm} steps")
                        _phase_warmup(robot, env, device, last_actions, obs_warm, include_obs=True)
                    elif warmup_mode == "deploy":
                        deploy_warm = max(phase_warmup, burn_in + 10)
                        print(f"[INFO]   phase warm-up (deploy): {deploy_warm} steps")
                        last_actions, deploy_obs_td = _warmup_deploy(
                            env, device, last_actions, policy, deploy_warm, deploy_obs_td
                        )
                fn()
                _drain_gpu(device)

            _run_phase(
                "CRI only (isolated loop)",
                lambda: _bench_cri_only(robot, env, device, last_actions, args_cli.repeats, cri_rec),
                warmup_mode="cri",
            )
            _run_phase(
                "policy only",
                lambda: _bench_policy_only(device, policy, fixed_obs, args_cli.repeats, policy_rec),
                warmup_mode="cri",
            )
            _run_phase(
                "CRI + policy",
                lambda: _bench_cri_policy(
                    robot, env, device, last_actions, policy, fixed_obs,
                    args_cli.repeats, cri_policy_rec, cri_policy_cri_part,
                ),
                warmup_mode="cri_policy",
            )
            _run_phase(
                "policy obs (incl. CRI)",
                lambda: _bench_obs_only(env, device, last_actions, args_cli.repeats, obs_rec),
                warmup_mode="obs",
            )
            _run_phase(
                "obs + policy (deploy)",
                lambda: _bench_obs_policy(
                    env, device, last_actions, policy, args_cli.repeats, obs_policy_rec, obs_td=deploy_obs_td
                ),
                warmup_mode="deploy",
            )
        else:
            total_deploy_warm = cri_warmup + max(phase_warmup, burn_in + 10)
            print(f"[INFO] deploy warm-up: {total_deploy_warm} obs+policy steps (continuous, obs_td retained)")
            last_actions, deploy_obs_td = _warmup_deploy(
                env, device, last_actions, policy, total_deploy_warm, deploy_obs_td
            )
            _drain_gpu(device)
            print(f"[INFO] measuring deploy x{args_cli.repeats} (burn-in={burn_in}, steady verdict)...")
            _bench_obs_policy(
                env, device, last_actions, policy, args_cli.repeats, obs_policy_rec, obs_td=deploy_obs_td
            )
            _drain_gpu(device)

    deploy_all = obs_policy_rec.summary()
    deploy_steady = obs_policy_rec.summary_after(burn_in)

    if full_bench:
        cri_stats = cri_rec.summary()
        policy_stats = policy_rec.summary()
        cri_policy_stats = cri_policy_rec.summary()
        obs_stats = obs_rec.summary()
        cri_steady = cri_rec.summary_after(burn_in)
        obs_steady = obs_rec.summary_after(burn_in)

        print("-" * 88)
        print(_format_row("CRI solver", cri_stats))
        if burn_in > 0:
            print(_format_row(f"CRI solver (steady +{burn_in})", cri_steady))
        print(_format_row("Policy forward (actions)", policy_stats))
        print(_format_row("Pipeline (CRI + policy)", cri_policy_stats))
        print(_format_row("Policy obs (incl. CRI)", obs_stats))
        if burn_in > 0:
            print(_format_row(f"Policy obs (steady +{burn_in})", obs_steady))
        _print_deploy_report(
            deploy_steady,
            budget_ms=args_cli.budget_ms,
            repeats=args_cli.repeats,
            burn_in=burn_in,
            recorder=obs_policy_rec,
            deploy_all=deploy_all,
        )
        _print_outliers("CRI only", cri_rec, args_cli.budget_ms)
        _print_outliers("CRI+policy (total)", cri_policy_rec, args_cli.budget_ms)
        if cri_policy_cri_part.samples:
            cp = cri_policy_cri_part.summary()
            print(f"[outlier] CRI+policy CRI part: max={cp['max']:.3f}ms p99={cp['p99']:.3f}ms")
        _print_outliers("policy obs", obs_rec, args_cli.budget_ms)
    else:
        _print_deploy_report(
            deploy_steady,
            budget_ms=args_cli.budget_ms,
            repeats=args_cli.repeats,
            burn_in=burn_in,
            recorder=obs_policy_rec,
            deploy_all=deploy_all,
        )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
