# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark 50Hz control latency (isolated deploy / real-robot / sim reference).

Default ``--mode isolated-deploy`` matches ``benchmark_cri_policy_inference.py``
deploy-only (50Hz verdict path)::

    advance_physics(last_actions)          # untimed
    t0 = now()
    obs = observation_manager.compute()    # incl. CRI
    actions = policy(obs_td)
    record(now() - t0)

``--mode deploy``: real-robot tick with explicit joint read.
``--mode sim``: full env.step() (PhysX included, reference only).

Policy loading supports RSL-RL checkpoints (``model_*.pt``) and exported TorchScript
(``exported/policy.pt``). JIT is auto-detected when the path contains ``/exported/``,
or force with ``--policy-jit``.

Example (isolated deploy, RSL-RL checkpoint):
    SFD_SPIKE_MITIGATION=1 SFD_CRI_TIMING=1 SFD_ALLOC_WARMUP_ROUNDS=15 \\
    ./isaaclab.sh -p scripts/benchmarks/benchmark_env_step_realtime.py \\
        --mode isolated-deploy \\
        --task Isaac-Reach-UR10-Play-v0 \\
        --checkpoint logs/rsl_rl/reach_ur10/2026-07-06_17-36-07/model_0.pt \\
        --num_envs 1 --headless --warmup 50 --steps 1000 --burn-in 50

Example (isolated deploy, exported JIT):
    SFD_SPIKE_MITIGATION=1 SFD_CRI_TIMING=1 SFD_ALLOC_WARMUP_ROUNDS=15 \\
    ./isaaclab.sh -p scripts/benchmarks/benchmark_env_step_realtime.py \\
        --mode isolated-deploy \\
        --task Isaac-Reach-UR10-Play-v0 \\
        --checkpoint logs/rsl_rl/reach_ur10/2026-07-06_17-36-07/exported/policy.pt \\
        --num_envs 1 --headless --warmup 50 --steps 1000 --burn-in 50
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
import time

os.environ.setdefault("SFD_CRI_TIMING", "1")

if os.environ.get("SFD_SPIKE_MITIGATION", "1") not in ("0", "false", "FALSE"):
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.95",
    )
    os.environ.setdefault("CUDA_MODULE_LOADING", "EAGER")
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "32")
    os.environ.setdefault("SAFETICS_USE_TENSOR_CALC_LOOP", "1")

import numpy as np
from isaaclab.app import AppLauncher

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../reinforcement_learning/rsl_rl"))
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Benchmark isolated deploy / real-robot / sim latency.")
parser.add_argument(
    "--mode",
    type=str,
    default="isolated-deploy",
    choices=("isolated-deploy", "deploy", "sim"),
    help="isolated-deploy=benchmark_cri_policy_inference deploy-only. "
    "deploy=real-robot joint-read path. sim=full env.step().",
)
parser.add_argument("--task", type=str, default="Isaac-Reach-UR10-Play-v0", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--warmup", type=int, default=50, help="Initial env.step() warm-up before deploy path.")
parser.add_argument("--steps", type=int, default=1000, help="Measured deploy samples after warm-up.")
parser.add_argument("--budget-ms", type=float, default=20.0, help="50Hz hard budget in ms for pass-rate report.")
parser.add_argument(
    "--burn-in",
    type=int,
    default=None,
    help="Measured samples excluded from steady-state stats (default: SFD_BENCH_BURN_IN or 50).",
)
parser.add_argument(
    "--cri-warmup",
    type=int,
    default=None,
    help="Extra CRI/deploy-path warm-up (default: SFD_BENCH_CRI_WARMUP or max(50, steps/20)).",
)
parser.add_argument(
    "--phase-warmup",
    type=int,
    default=None,
    help="Per-phase warm-up before measurement (default: SFD_BENCH_PHASE_WARMUP or max(20, steps/50)).",
)
parser.add_argument(
    "--no-policy",
    action="store_true",
    default=False,
    help="Use zero actions instead of loading a policy checkpoint.",
)
parser.add_argument(
    "--policy-jit",
    action="store_true",
    default=False,
    help="Load --checkpoint as TorchScript (torch.jit.load). Auto-detected for paths under exported/.",
)
parser.add_argument(
    "--policy-checkpoint",
    action="store_true",
    default=False,
    help="Force RSL-RL checkpoint loading even if the path looks like an exported JIT file.",
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


class _JitPolicyWrapper:
    """TorchScript policy with the same call signature as RSL-RL inference policies."""

    def __init__(self, module: torch.jit.ScriptModule, device: str) -> None:
        self.module = module.to(device).eval()
        self._device = device

    def __call__(self, obs) -> torch.Tensor:
        if isinstance(obs, TensorDict):
            obs_tensor = obs["policy"]
        elif isinstance(obs, dict):
            obs_tensor = obs["policy"]
        else:
            obs_tensor = obs
        return self.module(obs_tensor)

    def reset(self, dones) -> None:
        del dones


def _is_jit_policy(resume_path: str) -> bool:
    if args_cli.policy_checkpoint:
        return False
    if args_cli.policy_jit:
        return True
    norm = os.path.normpath(resume_path).replace("\\", "/").lower()
    return "/exported/" in norm or norm.endswith("exported/policy.pt")


def _load_inference_policy(
    env: RslRlVecEnvWrapper,
    agent_cfg: RslRlBaseRunnerCfg,
    resume_path: str,
    device: str,
):
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)

    if _is_jit_policy(resume_path):
        print(f"[INFO] Loading TorchScript JIT policy: {resume_path}")
        jit_module = torch.jit.load(resume_path, map_location=device)
        return _JitPolicyWrapper(jit_module, device), "torchscript-jit", resume_path

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    print(f"[INFO] Loading RSL-RL checkpoint: {resume_path}")
    runner.load(resume_path)
    return runner.get_inference_policy(device=device), "rsl-rl-checkpoint", resume_path


def _sync_device(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _drain_gpu(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.synchronize()


def _advance_physics(env: RslRlVecEnvWrapper, actions: torch.Tensor) -> None:
    """Advance sim between 50Hz ticks (untimed — real robot motion is off-RT path)."""
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


def _format_row(name: str, stats: dict[str, float]) -> str:
    return (
        f"{name:<32}  mean={stats['mean']:7.3f} ms  median={stats['median']:7.3f} ms  "
        f"p95={stats['p95']:7.3f} ms  p99={stats.get('p99', stats['p95']):7.3f} ms  "
        f"min={stats['min']:7.3f} ms  max={stats['max']:7.3f} ms  pass={stats.get('pass_pct', 0):5.1f}%"
    )


def _print_deploy_verdict(
    label: str,
    steady: dict[str, float],
    *,
    budget_ms: float,
    steps: int,
    burn_in: int,
    recorder: _LatencyRecorder,
    all_stats: dict[str, float] | None = None,
    verdict_note: str = "50Hz verdict",
) -> None:
    steady_n = max(0, steps - burn_in)
    print("-" * 92)
    print(_format_row(f"{label} steady", steady))
    if all_stats is not None and burn_in > 0:
        print(_format_row(f"{label} all (burn-in {burn_in})", all_stats))
    print("-" * 92)
    print(f"[SUMMARY] {label} @ {budget_ms}ms ({verdict_note})")
    print(f"  steady mean   : {steady['mean']:.3f} ms (median {steady['median']:.3f} ms)")
    print(f"  steady p95/p99: {steady['p95']:.3f} / {steady['p99']:.3f} ms")
    print(f"  steady min/max: {steady['min']:.3f} / {steady['max']:.3f} ms")
    print(
        f"  50Hz pass     : {steady.get('pass_pct', 0):.1f}% "
        f"({steady.get('violations', 0)} violations / {steady_n} steady samples)"
    )
    if burn_in > 0:
        print(f"  burn-in       : first {burn_in} measured samples excluded")
    hits = recorder.outliers_after(burn_in, budget_ms)
    if hits:
        print(f"[outlier] {label} steady violations top:")
        for idx, ms in hits:
            print(f"  iter={idx} wall_ms={ms:.3f} (+{ms - budget_ms:.3f} vs {budget_ms:.1f}ms budget)")
    elif steady.get("pass_pct", 0) >= 100.0:
        print(f"[OK] {label} meets {budget_ms}ms for all {steady_n} steady samples.")


def _bench_obs_policy(
    env: RslRlVecEnvWrapper,
    robot,
    device: str,
    actions: torch.Tensor,
    policy,
    steps: int,
    deploy_rec: _LatencyRecorder,
    cri_rec: _LatencyRecorder,
    *,
    use_policy: bool,
    zero_actions: torch.Tensor,
    obs_td: TensorDict | None = None,
) -> tuple[torch.Tensor, TensorDict | None]:
    """Isolated deploy path — identical to benchmark_cri_policy_inference._bench_obs_policy."""
    last_actions = actions
    for i in range(steps):
        _advance_physics(env, last_actions)
        _drain_gpu(device)
        t0 = time.perf_counter()
        obs_dict = env.unwrapped.observation_manager.compute(update_history=True)
        if obs_td is None:
            obs_td = TensorDict(obs_dict, batch_size=[env.unwrapped.num_envs])
        else:
            obs_td.update(obs_dict)
        if use_policy:
            last_actions = policy(obs_td)
        else:
            last_actions = zero_actions
        _sync_device(device)
        deploy_rec.record(time.perf_counter() - t0, i)
        cri_rec.record(robot.data.cri_last_inference_time_s, i)
        if not simulation_app.is_running():
            break
    return last_actions, obs_td


def _bench_isolated_deploy(
    env: RslRlVecEnvWrapper,
    robot,
    device: str,
    actions: torch.Tensor,
    policy,
    *,
    steps: int,
    burn_in: int,
    cri_warmup: int,
    phase_warmup: int,
    use_policy: bool,
    zero_actions: torch.Tensor,
) -> tuple[_LatencyRecorder, _LatencyRecorder, torch.Tensor]:
    deploy_rec = _LatencyRecorder()
    cri_rec = _LatencyRecorder()
    obs_td: TensorDict | None = None
    last_actions = actions

    total_warm = cri_warmup + max(phase_warmup, burn_in + 10)
    print(f"[INFO] isolated deploy warm-up: {total_warm} obs+policy (continuous, obs_td retained)...")
    last_actions, obs_td = _bench_obs_policy(
        env, robot, device, last_actions, policy, total_warm, _LatencyRecorder(), _LatencyRecorder(),
        use_policy=use_policy, zero_actions=zero_actions, obs_td=obs_td,
    )
    _drain_gpu(device)
    robot.data.reset_cri_inference_stats()

    print(f"[INFO] measuring isolated deploy x{steps} (burn-in={burn_in}, steady verdict)...")
    last_actions, obs_td = _bench_obs_policy(
        env, robot, device, last_actions, policy, steps, deploy_rec, cri_rec,
        use_policy=use_policy, zero_actions=zero_actions, obs_td=obs_td,
    )
    _drain_gpu(device)
    return deploy_rec, cri_rec, last_actions


def _bench_deploy_tick(
    env: RslRlVecEnvWrapper,
    robot,
    device: str,
    actions: torch.Tensor,
    policy,
    *,
    steps: int,
    burn_in: int,
    cri_warmup: int,
    use_policy: bool,
    zero_actions: torch.Tensor,
) -> tuple[
    _LatencyRecorder,
    _LatencyRecorder,
    _LatencyRecorder,
    _LatencyRecorder,
    _LatencyRecorder,
    torch.Tensor,
]:
    """One real-robot 50Hz tick: untimed physics, timed state+obs+policy."""
    tick_rec = _LatencyRecorder()
    state_rec = _LatencyRecorder()
    obs_policy_rec = _LatencyRecorder()
    policy_rec = _LatencyRecorder()
    cri_rec = _LatencyRecorder()
    last_actions = actions
    obs_td: TensorDict | None = None

    total_warm = cri_warmup + max(10, burn_in + 10)
    print(f"[INFO] deploy path warm-up: {total_warm} ticks (physics untimed, obs_td retained)...")
    for _ in range(total_warm):
        _advance_physics(env, last_actions)
        _drain_gpu(device)
        obs_dict = env.unwrapped.observation_manager.compute(update_history=True)
        if obs_td is None:
            obs_td = TensorDict(obs_dict, batch_size=[env.unwrapped.num_envs])
        else:
            obs_td.update(obs_dict)
        if use_policy:
            last_actions = policy(obs_td)
        else:
            last_actions = zero_actions

    _drain_gpu(device)
    robot.data.reset_cri_inference_stats()

    print(f"[INFO] measuring {steps} real-robot deploy ticks (PhysX excluded from timer)...")
    for tick_idx in range(steps):
        _advance_physics(env, last_actions)
        _drain_gpu(device)

        _sync_device(device)
        t0 = time.perf_counter()
        _ = robot.data.joint_pos
        _ = robot.data.joint_vel
        t1 = time.perf_counter()

        obs_dict = env.unwrapped.observation_manager.compute(update_history=True)
        if obs_td is None:
            obs_td = TensorDict(obs_dict, batch_size=[env.unwrapped.num_envs])
        else:
            obs_td.update(obs_dict)
        t2 = time.perf_counter()

        if use_policy:
            last_actions = policy(obs_td)
        else:
            last_actions = zero_actions
        _sync_device(device)
        t3 = time.perf_counter()

        state_rec.record(t1 - t0, tick_idx)
        obs_policy_rec.record(t3 - t1, tick_idx)
        policy_rec.record(t3 - t2, tick_idx)
        tick_rec.record(t3 - t0, tick_idx)
        cri_rec.record(robot.data.cri_last_inference_time_s, tick_idx)

        if not simulation_app.is_running():
            break

    return tick_rec, state_rec, obs_policy_rec, policy_rec, cri_rec, last_actions


def _bench_sim_step(
    env: RslRlVecEnvWrapper,
    robot,
    device: str,
    policy,
    obs,
    *,
    steps: int,
    use_policy: bool,
    zero_actions: torch.Tensor,
) -> tuple[_LatencyRecorder, _LatencyRecorder, _LatencyRecorder, _LatencyRecorder]:
    tick_rec = _LatencyRecorder()
    step_rec = _LatencyRecorder()
    policy_rec = _LatencyRecorder()
    cri_rec = _LatencyRecorder()

    print(f"[INFO] measuring {steps} full env.step() iterations (sim reference)...")
    for step_idx in range(steps):
        _sync_device(device)
        t0 = time.perf_counter()
        if use_policy:
            actions = policy(obs)
        else:
            actions = zero_actions
        t1 = time.perf_counter()
        obs, _, dones, _ = env.step(actions)
        if use_policy and version.parse(installed_version) >= version.parse("4.0.0"):
            policy.reset(dones)
        _sync_device(device)
        t2 = time.perf_counter()

        tick_rec.record(t2 - t0, step_idx)
        policy_rec.record(t1 - t0, step_idx)
        step_rec.record(t2 - t1, step_idx)
        cri_rec.record(robot.data.cri_last_inference_time_s, step_idx)

        if not simulation_app.is_running():
            break

    return tick_rec, step_rec, policy_rec, cri_rec


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "policy"):
        env_cfg.observations.policy.enable_corruption = False

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    device = env.unwrapped.device
    robot = env.unwrapped.scene["robot"]

    policy = None
    policy_backend = "none"
    if args_cli.no_policy:
        print("[INFO] --no-policy: using zero actions (no checkpoint)")
    else:
        log_root_path = f"logs/rsl_rl/{agent_cfg.experiment_name}"
        if args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        policy, policy_backend, resume_path = _load_inference_policy(env, agent_cfg, resume_path, device)

    obs = env.get_observations()
    action_dim = env.num_actions
    zero_actions = torch.zeros(env.num_envs, action_dim, device=device)
    use_policy = policy is not None

    burn_in = (
        args_cli.burn_in
        if args_cli.burn_in is not None
        else int(os.environ.get("SFD_BENCH_BURN_IN", "50"))
    )
    cri_warmup = (
        args_cli.cri_warmup
        if args_cli.cri_warmup is not None
        else int(os.environ.get("SFD_BENCH_CRI_WARMUP", str(max(50, args_cli.steps // 20))))
    )
    phase_warmup = (
        args_cli.phase_warmup
        if args_cli.phase_warmup is not None
        else int(os.environ.get("SFD_BENCH_PHASE_WARMUP", str(max(20, args_cli.steps // 50))))
    )
    os.environ["SFD_BENCH_BUDGET_MS"] = str(args_cli.budget_ms)
    cri_timing = os.environ.get("SFD_CRI_TIMING", "0") == "1"

    print(f"[INFO] mode={args_cli.mode}")
    print(f"[INFO] Device: {device}, num_envs: {env.unwrapped.num_envs}")
    if use_policy:
        print(f"[INFO] policy backend: {policy_backend}")
        if policy_backend == "torchscript-jit":
            print(f"[INFO] policy path: {resume_path}")
    print(f"[INFO] policy obs shape: {tuple(obs['policy'].shape)}, action dim: {action_dim}")
    print(
        f"[INFO] warmup={args_cli.warmup}, steps={args_cli.steps}, burn_in={burn_in}, "
        f"cri_warmup={cri_warmup}, phase_warmup={phase_warmup}, "
        f"budget_ms={args_cli.budget_ms}, SFD_CRI_TIMING={int(cri_timing)}"
    )
    if not cri_timing:
        print("[WARN] SFD_CRI_TIMING=0 — export SFD_CRI_TIMING=1 for per-tick CRI timing")

    last_actions = torch.zeros(env.num_envs, action_dim, device=device)

    with torch.inference_mode():
        print(f"[INFO] initial env warm-up: {args_cli.warmup} env.step()...")
        for _ in range(args_cli.warmup):
            if use_policy:
                actions = policy(obs)
            else:
                actions = zero_actions
            obs, _, dones, _ = env.step(actions)
            if use_policy and version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            last_actions = actions
            if not simulation_app.is_running():
                break

        if args_cli.mode == "isolated-deploy":
            deploy_rec, cri_rec, last_actions = _bench_isolated_deploy(
                env,
                robot,
                device,
                last_actions,
                policy,
                steps=args_cli.steps,
                burn_in=burn_in,
                cri_warmup=cri_warmup,
                phase_warmup=phase_warmup,
                use_policy=use_policy,
                zero_actions=zero_actions,
            )
            deploy_steady = deploy_rec.summary_after(burn_in)
            deploy_all = deploy_rec.summary()
            cri_steady = cri_rec.summary_after(burn_in)
            cri_stats = robot.data.get_cri_inference_stats()

            print()
            print("[RESULT] Isolated deploy (obs incl. CRI → policy → actions)")
            _print_deploy_verdict(
                "Deploy obs+policy",
                deploy_steady,
                budget_ms=args_cli.budget_ms,
                steps=args_cli.steps,
                burn_in=burn_in,
                recorder=deploy_rec,
                all_stats=deploy_all,
                verdict_note="isolated deploy 50Hz verdict",
            )
            print()
            print(_format_row("CRI (AtMotionState)", cri_steady))
            if cri_stats.get("count"):
                print(
                    f"[CRI stats] count={cri_stats['count']} "
                    f"mean={float(cri_stats['mean_s']) * 1000:.3f} ms "
                    f"max={float(cri_stats['max_s']) * 1000:.3f} ms"
                    + (
                        f" p99={float(cri_stats['p99_s']) * 1000:.3f} ms"
                        if "p99_s" in cri_stats
                        else ""
                    )
                )
            nonzero_cri = sum(1 for ms in cri_rec.samples if ms > 0.01)
            print(f"[CHECK] CRI non-zero samples: {nonzero_cri}/{len(cri_rec.samples)}")
            print("[NOTE] Physics before obs is excluded from timed window (same as benchmark_cri_policy_inference.py).")

        elif args_cli.mode == "deploy":
            tick_rec, state_rec, obs_policy_rec, policy_rec, cri_rec, last_actions = _bench_deploy_tick(
                env,
                robot,
                device,
                last_actions,
                policy,
                steps=args_cli.steps,
                burn_in=burn_in,
                cri_warmup=cri_warmup,
                use_policy=use_policy,
                zero_actions=zero_actions,
            )

            deploy_steady = tick_rec.summary_after(burn_in)
            deploy_all = tick_rec.summary()
            state_steady = state_rec.summary_after(burn_in)
            obs_steady = obs_policy_rec.summary_after(burn_in)
            policy_steady = policy_rec.summary_after(burn_in)
            cri_steady = cri_rec.summary_after(burn_in)
            cri_stats = robot.data.get_cri_inference_stats()

            print()
            print("[RESULT] Real-robot deploy path (PhysX excluded, 50Hz control tick)")
            _print_deploy_verdict(
                "Deploy tick (state+obs+policy)",
                deploy_steady,
                budget_ms=args_cli.budget_ms,
                steps=args_cli.steps,
                burn_in=burn_in,
                recorder=tick_rec,
                all_stats=deploy_all,
                verdict_note="real-robot tick (joint read included)",
            )
            print()
            print(_format_row("Joint state read", state_steady))
            print(_format_row("Obs build + policy", obs_steady))
            print(_format_row("Policy forward only", policy_steady))
            print(_format_row("CRI (AtMotionState)", cri_steady))
            if cri_stats.get("count"):
                print(
                    f"[CRI stats] count={cri_stats['count']} "
                    f"mean={float(cri_stats['mean_s']) * 1000:.3f} ms "
                    f"max={float(cri_stats['max_s']) * 1000:.3f} ms"
                    + (
                        f" p99={float(cri_stats['p99_s']) * 1000:.3f} ms"
                        if "p99_s" in cri_stats
                        else ""
                    )
                )
            nonzero_cri = sum(1 for ms in cri_rec.samples if ms > 0.01)
            print(f"[CHECK] CRI non-zero ticks: {nonzero_cri}/{len(cri_rec.samples)}")
            print(
                "[NOTE] Physics/sim.step runs **between** ticks (untimed), like real robot motion. "
                "Use --mode sim only to compare Isaac overhead."
            )

        else:
            _sync_device(device)
            robot.data.reset_cri_inference_stats()
            tick_rec, step_rec, policy_rec, cri_rec = _bench_sim_step(
                env, robot, device, policy, obs, steps=args_cli.steps, use_policy=use_policy, zero_actions=zero_actions
            )

            print()
            print("[RESULT] Isaac sim reference (env.step includes PhysX — NOT deploy verdict)")
            _print_deploy_verdict(
                "Sim tick (policy+env.step)",
                tick_rec.summary_after(burn_in),
                budget_ms=args_cli.budget_ms,
                steps=args_cli.steps,
                burn_in=burn_in,
                recorder=tick_rec,
                all_stats=tick_rec.summary(),
                verdict_note="sim reference only",
            )
            print()
            print(_format_row("env.step() (PhysX+obs)", step_rec.summary_after(burn_in)))
            print(_format_row("Policy forward", policy_rec.summary_after(burn_in)))
            print(_format_row("CRI (AtMotionState)", cri_rec.summary_after(burn_in)))

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
