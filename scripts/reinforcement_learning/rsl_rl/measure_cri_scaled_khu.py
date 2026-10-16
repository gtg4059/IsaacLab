# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""CRI along Case_7 c-splines after one global velocity scale.

The logged KHU_Recommend CRI is left untouched. This replays the same joint
path with qd multiplied by --vel_scale, which shortens every duration by that
factor and matches the mean learning duration when the scale is
mean(T_spline)/mean(T_learn).
"""

from __future__ import annotations

import argparse
import csv
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Scaled-speed CRI for KHU_Recommend splines.")
parser.add_argument("--task", type=str, default="Isaac-Reach-UR10-v0")
parser.add_argument("--num_envs", type=int, default=8192)
parser.add_argument("--case_root", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--vel_scale", type=float, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

POS_COLS = [f"Position_{i}" for i in range(2, 8)]
VEL_COLS = [f"Velocity_{i}" for i in range(2, 8)]


def _load(case_root: str, env_count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    series: list[tuple[list[list[float]], list[list[float]], list[float]]] = []
    max_len = 0
    for env_idx in range(env_count):
        path = os.path.join(case_root, f"Path_{env_idx}", "KHU_Recommend", "KHU_Collision_Risk_Analyze_Result.csv")
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        q = [[float(row[col]) for col in POS_COLS] for row in rows]
        qd = [[float(row[col]) for col in VEL_COLS] for row in rows]
        t = [float(row["Time"]) for row in rows]
        series.append((q, qd, t))
        max_len = max(max_len, len(rows))
    n_j = 6
    q_buf = torch.zeros(env_count, max_len, n_j)
    qd_buf = torch.zeros(env_count, max_len, n_j)
    t_buf = torch.zeros(env_count, max_len)
    lengths = torch.zeros(env_count, dtype=torch.long)
    for env_idx, (q, qd, t) in enumerate(series):
        t_len = len(q)
        lengths[env_idx] = t_len
        if t_len == 0:
            continue
        q_buf[env_idx, :t_len] = torch.tensor(q)
        qd_buf[env_idx, :t_len] = torch.tensor(qd)
        t_buf[env_idx, :t_len] = torch.tensor(t)
    return q_buf, qd_buf, t_buf, lengths


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    env.reset()
    robot = base.scene["robot"]
    q_buf, qd_buf, t_buf, lengths = _load(args_cli.case_root, base.num_envs)
    device = base.device
    q_buf = q_buf.to(device=device, dtype=torch.float64)
    qd_buf = qd_buf.to(device=device, dtype=torch.float64)
    t_buf = t_buf.to(device)
    lengths = lengths.to(device)
    scale = float(args_cli.vel_scale)
    n_env = base.num_envs
    cri_sum = torch.zeros(n_env, device=device)
    cri_max = torch.zeros(n_env, device=device)
    cri_count = torch.zeros(n_env, device=device)
    hist = torch.zeros(20, device=device)
    phase_sum = torch.zeros(10, device=device)
    phase_n = torch.zeros(10, device=device)
    t0 = t_buf[:, 0]
    t_last = t_buf.gather(1, (lengths - 1).clamp_min(0).unsqueeze(1)).squeeze(1)
    span = (t_last - t0).clamp_min(1.0e-6)
    solver = robot.data.solver
    max_len = int(q_buf.shape[1])
    printed = False
    for step in range(max_len):
        active = lengths > step
        if not bool(active.any().item()):
            break
        q = q_buf[:, step].contiguous()
        qd = (qd_buf[:, step] * scale).contiguous()
        cri = solver.RunSolver_CUDA_CRI_AtMotionState(q, qd)
        if cri is None:
            raise RuntimeError("CRI solver returned None")
        if not printed:
            print(f"[INFO] CRI shape {tuple(cri.shape)} dtype {cri.dtype}", flush=True)
            printed = True
        step_max = cri.amax(dim=-1) if cri.ndim == 2 else cri
        step_max = step_max.clamp(0.0, 2.0)
        cri_sum[active] += step_max[active]
        cri_max[active] = torch.maximum(cri_max[active], step_max[active])
        cri_count[active] += 1
        hist += torch.histc(step_max[active].float(), bins=20, min=0.0, max=2.0)
        phase = ((t_buf[:, step] - t0) / span * 10).long().clamp(0, 9)
        for bin_id in range(10):
            sel = active & (phase == bin_id)
            if bool(sel.any().item()):
                phase_sum[bin_id] += step_max[sel].sum()
                phase_n[bin_id] += sel.sum()
        if step % 40 == 0:
            live = int(active.sum().item())
            sample = float(step_max[active][0].item()) if live else 0.0
            print(f"[INFO] step {step} live {live} sample maxCRI {sample:.4f}", flush=True)
    os.makedirs(args_cli.out_dir, exist_ok=True)
    mean = (cri_sum / cri_count.clamp_min(1)).detach().cpu()
    peak = cri_max.detach().cpu()
    csv_path = os.path.join(args_cli.out_dir, "spline_cri_same_mean_duration.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path_idx", "cri_mean", "cri_max", "n_steps", "vel_scale"])
        for path_idx, (cri_mean, cri_peak, n_steps) in enumerate(
            zip(mean.tolist(), peak.tolist(), cri_count.detach().cpu().tolist(), strict=True)
        ):
            writer.writerow([path_idx, cri_mean, cri_peak, int(n_steps), scale])
    phase_mean = (phase_sum / phase_n.clamp_min(1)).detach().cpu().tolist()
    payload = {
        "vel_scale": scale,
        "hist_edges": [i / 10 for i in range(21)],
        "hist_counts": hist.detach().cpu().to(dtype=torch.int64).tolist(),
        "phase_mean_cri": phase_mean,
        "n_steps": int(cri_count.sum().item()),
    }
    json_path = os.path.join(args_cli.out_dir, "spline_cri_same_mean_duration.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[INFO] Wrote {csv_path}", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
