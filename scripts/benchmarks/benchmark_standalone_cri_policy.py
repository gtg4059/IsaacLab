#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone 50Hz CRI + policy latency (no Isaac Sim / Kit).

Real-robot deploy verdict path — same hot-loop style as ``SFD_CoreService_Test``::

    CudaSync
    t0 = now()
    cri = RunSolver_CUDA_CRI_AtMotionState(q, qd)
    actions = jit_policy(obs)          # obs may embed latest CRI
    CudaSync
    record(now() - t0)

Does **not** launch Omniverse. Uses bundled ``sfd_coreservice`` under
``isaaclab/assets/articulation`` + an exported TorchScript ``policy.pt``.

Example::

    SFD_LOCK_GPU_CLOCK=1 SFD_SPIKE_MITIGATION=1 SFD_SCHED_FIFO=1 \\
    SFD_CPU_AFFINITY=2,3 \\
    python scripts/benchmarks/benchmark_standalone_cri_policy.py \\
        --policy logs/rsl_rl/reach_ur10/2026-07-13_19-00-20/exported/policy.pt \\
        --warmup 50 --steps 4000 --burn-in 50 --rt
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Bootstrap sfd_coreservice without importing isaaclab.assets.articulation
# (that package pulls Omniverse / PhysX).
# ---------------------------------------------------------------------------
_ART_DIR = Path(__file__).resolve().parents[2] / "source/isaaclab/isaaclab/assets/articulation"
_LIB_DIR = _ART_DIR / "lib"
if not _LIB_DIR.is_dir():
    raise SystemExit(f"[ERROR] CUDACRI lib not found: {_LIB_DIR}")

for _p in (_ART_DIR, _LIB_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_torch_lib = Path(torch.__file__).resolve().parent / "lib"
os.environ["LD_LIBRARY_PATH"] = ":".join(
    p for p in (str(_LIB_DIR), str(_torch_lib), os.environ.get("LD_LIBRARY_PATH", "")) if p
)

from sfd_setup import (  # noqa: E402
    apply_process_rt_hardening,
    apply_realtime_host_tuning,
    apply_spike_mitigation_env_early,
    print_gpu_runtime_snapshot,
    restore_realtime_host,
)
from cri_realtime_monitor import (  # noqa: E402
    CriRealtimeMonitor,
    RealTimeBudgetConfig,
    format_outlier_lines,
    format_realtime_budget_report,
)

import sfd_coreservice  # noqa: E402

# Obs layout for CRI reach policy (history_length=5), matches Play env:
# ee_pose_error(30) + CRI(40) + joint_pos(30) + joint_vel(30) + pose_command(35) + actions(30) = 195
_OBS_DIM = 195
_CRI_HIST_DIM = 40  # 8 collision points × 5 history
_CRI_SLICE = slice(30, 70)  # after ee_pose_error


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no", "")


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone CRI+policy 50Hz latency (no Isaac Kit).")
    p.add_argument(
        "--policy",
        type=str,
        default="logs/rsl_rl/reach_ur10/2026-07-13_19-00-20/exported/policy.pt",
        help="TorchScript policy.pt path.",
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num-joints", type=int, default=6, help="q/qd width (UR10=6; native often pads to 8).")
    p.add_argument("--obs-dim", type=int, default=_OBS_DIM)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--steps", type=int, default=2000, help="Measured hot-loop samples.")
    p.add_argument("--burn-in", type=int, default=50, help="Exclude first N measured samples from steady stats.")
    p.add_argument("--budget-ms", type=float, default=20.0)
    p.add_argument(
        "--embed-cri",
        action="store_true",
        default=True,
        help="Copy latest CRI into obs[30:70] each tick (default on).",
    )
    p.add_argument("--no-embed-cri", action="store_false", dest="embed_cri")
    p.add_argument(
        "--moving",
        action="store_true",
        default=False,
        help="Animate q/qd slightly each tick (default: fixed pose like native hot-loop).",
    )
    p.add_argument(
        "--no-host-tune",
        action="store_true",
        default=False,
        help="Skip sfd_setup realtime host tuning (GPU clock / swap).",
    )
    p.add_argument(
        "--rt",
        action="store_true",
        default=False,
        help="Enable process RT hardening: mlockall + SCHED_FIFO (needs privileges). "
        "Also set SFD_CPU_AFFINITY=... to pin CPUs.",
    )
    p.add_argument(
        "--no-freeze-gc",
        action="store_true",
        default=False,
        help="Keep Python GC enabled during measurement (default: freeze).",
    )
    return p.parse_args()


def _make_fixed_pose(num_joints: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """UR10-ish ready pose (rad); matches a typical mid-workspace configuration."""
    q = torch.zeros(1, num_joints, dtype=torch.float64, device=device)
    # shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3 (+ pad)
    seed = [-0.5, -1.2, 1.4, -1.5, 1.57, 0.0]
    for i in range(min(num_joints, len(seed))):
        q[0, i] = seed[i]
    qd = torch.zeros_like(q)
    if num_joints >= 6:
        # Small nonzero velocity so CRI is non-trivial (native verifies moving CRI).
        qd[0, :6] = torch.tensor([0.05, -0.08, 0.06, -0.04, 0.03, 0.02], dtype=torch.float64, device=device)
    return q.contiguous(), qd.contiguous()


def _pack_cri_into_obs(obs: torch.Tensor, cri: torch.Tensor) -> None:
    """Write current CRI into the history-flattened CRI slice (repeat across history)."""
    cri_f = cri.detach().to(dtype=obs.dtype).reshape(obs.shape[0], -1)
    n = min(cri_f.shape[-1], _CRI_HIST_DIM)
    # Fill history slots with the same current CRI (hot-loop has no real history).
    block = cri_f[:, :n]
    reps = _CRI_HIST_DIM // n
    rem = _CRI_HIST_DIM % n
    parts = [block] * reps
    if rem:
        parts.append(block[:, :rem])
    obs[:, _CRI_SLICE] = torch.cat(parts, dim=-1)


def _summarize(samples: list[float], budget_ms: float) -> dict[str, float]:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0, "pass_pct": 100.0, "violations": 0}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "pass_pct": float(100.0 * np.mean(arr <= budget_ms)),
        "violations": int(np.sum(arr > budget_ms)),
    }


def _print_row(name: str, stats: dict[str, float]) -> None:
    print(
        f"{name:<32}  mean={stats['mean']:7.3f} ms  median={stats['median']:7.3f} ms  "
        f"p95={stats['p95']:7.3f} ms  p99={stats['p99']:7.3f} ms  "
        f"min={stats['min']:7.3f} ms  max={stats['max']:7.3f} ms  pass={stats['pass_pct']:5.1f}%"
    )


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        print("[ERROR] CUDA required for standalone CRI+policy benchmark.", file=sys.stderr)
        return 1

    apply_spike_mitigation_env_early()
    print_gpu_runtime_snapshot()
    if args.rt:
        os.environ.setdefault("SFD_PROCESS_RT", "1")
        os.environ.setdefault("SFD_MLOCKALL", "1")
        os.environ.setdefault("SFD_SCHED_FIFO", "1")
    if not args.no_host_tune:
        apply_realtime_host_tuning()

    device = args.device
    policy_path = Path(args.policy).expanduser().resolve()
    if not policy_path.is_file():
        print(f"[ERROR] policy not found: {policy_path}", file=sys.stderr)
        return 1

    print(f"[INFO] standalone CRI+policy (no Isaac Kit)")
    print(f"[INFO] analysis dir: {_ART_DIR}")
    print(f"[INFO] policy: {policy_path}")
    print(
        f"[INFO] warmup={args.warmup}, steps={args.steps}, burn_in={args.burn_in}, "
        f"budget_ms={args.budget_ms}, joints={args.num_joints}, embed_cri={int(args.embed_cri)}, "
        f"rt={int(args.rt)}, freeze_gc={int(not args.no_freeze_gc)}"
    )

    # --- CRI solver ---
    solver = sfd_coreservice.CoreService(str(_ART_DIR), 1)
    solver.RunSolver_CUDA_LoadAnalysisForCRI(str(_ART_DIR))
    q, qd = _make_fixed_pose(args.num_joints, device)

    # --- JIT policy ---
    policy = torch.jit.load(str(policy_path), map_location=device)
    policy.eval()
    obs = torch.zeros(1, args.obs_dim, dtype=torch.float32, device=device)
    # Seed with mild noise so EmpiricalNormalization path (if any) is stable.
    obs.uniform_(-0.01, 0.01)

    # --- Warm-up (allocator / TRT / JIT) ---
    print(f"[INFO] warm-up: {args.warmup} CRI+policy ticks...")
    with torch.inference_mode():
        for i in range(args.warmup):
            if args.moving:
                q = q + 1e-4 * torch.sin(torch.tensor(i * 0.1, device=device, dtype=q.dtype))
            cri = solver.RunSolver_CUDA_CRI_AtMotionState(q, qd)
            if args.embed_cri:
                _pack_cri_into_obs(obs, cri)
            _ = policy(obs)
            _sync()

    _sync()
    # Lock pages / raise RT priority after working set is resident.
    apply_process_rt_hardening()
    if _env_flag("SFD_SPIKE_MITIGATION", "1"):
        _sync()

    # --- Measure ---
    tick_ms: list[float] = []
    cri_ms: list[float] = []
    pol_ms: list[float] = []
    monitor = CriRealtimeMonitor(RealTimeBudgetConfig(control_hz=int(round(1000.0 / args.budget_ms))))

    freeze_gc = not args.no_freeze_gc
    gc_was_enabled = gc.isenabled()
    if freeze_gc:
        gc.collect()
        gc.disable()
        print("[INFO] Python GC frozen for measurement", flush=True)

    print(f"[INFO] measuring standalone hot-loop x{args.steps}...")
    try:
        with torch.inference_mode():
            for i in range(args.steps):
                if args.moving:
                    q = q + 1e-4 * torch.sin(torch.tensor(i * 0.1, device=device, dtype=q.dtype))

                _sync()
                t0 = time.perf_counter()

                t_cri0 = time.perf_counter()
                cri = solver.RunSolver_CUDA_CRI_AtMotionState(q, qd)
                _sync()
                t_cri1 = time.perf_counter()

                if args.embed_cri:
                    _pack_cri_into_obs(obs, cri)

                t_pol0 = time.perf_counter()
                _ = policy(obs)
                _sync()
                t_end = time.perf_counter()

                wall = (t_end - t0) * 1000.0
                tick_ms.append(wall)
                cri_ms.append((t_cri1 - t_cri0) * 1000.0)
                pol_ms.append((t_end - t_pol0) * 1000.0)
                monitor.record(wall)
    finally:
        if freeze_gc and gc_was_enabled:
            gc.enable()

    burn = max(0, min(args.burn_in, len(tick_ms) - 1))
    tick_steady = _summarize(tick_ms[burn:], args.budget_ms)
    cri_steady = _summarize(cri_ms[burn:], args.budget_ms)
    pol_steady = _summarize(pol_ms[burn:], args.budget_ms)
    tick_all = _summarize(tick_ms, args.budget_ms)

    print()
    print("[RESULT] Standalone deploy (CRI → optional obs embed → JIT policy)")
    print("-" * 92)
    _print_row("Deploy tick (CRI+policy) steady", tick_steady)
    if burn > 0:
        _print_row(f"Deploy tick all (burn-in {burn})", tick_all)
    print("-" * 92)
    print(f"[SUMMARY] Deploy tick (CRI+policy) @ {args.budget_ms}ms (standalone 50Hz verdict, no Kit)")
    print(f"  steady mean   : {tick_steady['mean']:.3f} ms (median {tick_steady['median']:.3f} ms)")
    print(f"  steady p95/p99: {tick_steady['p95']:.3f} / {tick_steady['p99']:.3f} ms")
    print(f"  steady min/max: {tick_steady['min']:.3f} / {tick_steady['max']:.3f} ms")
    steady_n = max(0, len(tick_ms) - burn)
    print(
        f"  50Hz pass     : {tick_steady['pass_pct']:.1f}% "
        f"({tick_steady['violations']} violations / {steady_n} steady samples)"
    )
    if burn > 0:
        print(f"  burn-in       : first {burn} measured samples excluded")

    bad = [(i, ms) for i, ms in enumerate(tick_ms[burn:], start=burn) if ms > args.budget_ms]
    bad.sort(key=lambda x: x[1], reverse=True)
    if bad:
        print("[outlier] Deploy tick steady violations top:")
        for idx, ms in bad[:8]:
            print(f"  iter={idx} wall_ms={ms:.3f} (+{ms - args.budget_ms:.3f} vs {args.budget_ms:.1f}ms budget)")
    else:
        print(f"[OK] Deploy tick meets {args.budget_ms}ms for all {steady_n} steady samples.")

    print()
    _print_row("CRI (AtMotionState)", cri_steady)
    _print_row("JIT policy", pol_steady)
    nonzero = sum(1 for ms in cri_ms if ms > 0.01)
    print(f"[CHECK] CRI timed samples: {nonzero}/{len(cri_ms)}")

    report = monitor.build_report("standalone CRI+policy hot-loop")
    # Recompute steady-only monitor-style report from burned samples
    print()
    print(format_realtime_budget_report(report))
    print(
        format_outlier_lines(
            report.hard_outliers,
            budget_ms=report.hard_budget_ms,
            title="standalone hard budget violations (all samples)",
        )
    )
    grade = report.grade
    if tick_steady["pass_pct"] >= 100.0:
        print("[realtime] verdict: standalone CRI+policy meets 50Hz target (no Isaac Kit).")
    else:
        print(
            f"[realtime] verdict: steady pass={tick_steady['pass_pct']:.1f}% "
            f"(grade_all={grade}); Kit-free path still has {tick_steady['violations']} tail violations."
        )

    restore_realtime_host()
    return 0 if tick_steady["pass_pct"] >= 100.0 else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        restore_realtime_host()
        raise
