# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Inverse-dynamics ITS for Case_7 KHU_Recommend c-splines.

Writes the logged-duration ITS and a second ITS whose joint velocity is scaled
by --vel_scale for every path, so the mean duration can be matched to learning.
τ = M qdd + C(q, qd) + g. Coriolis scales with velocity squared.
"""

from __future__ import annotations

import argparse
import csv
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="ITS from KHU_Recommend c-spline CSVs.")
parser.add_argument("--task", type=str, default="Isaac-Reach-UR10-v0")
parser.add_argument("--num_envs", type=int, default=8192)
parser.add_argument("--case_root", type=str, required=True)
parser.add_argument("--its_csv", type=str, required=True)
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
    dt_buf = torch.zeros(env_count, max_len - 1)
    lengths = torch.zeros(env_count, dtype=torch.long)
    for env_idx, (q, qd, t) in enumerate(series):
        t_len = len(q)
        lengths[env_idx] = t_len
        if t_len == 0:
            continue
        q_buf[env_idx, :t_len] = torch.tensor(q)
        qd_buf[env_idx, :t_len] = torch.tensor(qd)
        if t_len >= 2:
            dt_buf[env_idx, : t_len - 1] = torch.tensor(t[1:]) - torch.tensor(t[:-1])
    return q_buf, qd_buf, dt_buf, lengths


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    env.reset()
    robot = base.scene["robot"]
    print(f"[INFO] joints {list(robot.joint_names)}", flush=True)
    q_buf, qd_buf, dt_buf, lengths = _load(args_cli.case_root, base.num_envs)
    q_buf = q_buf.to(base.device)
    qd_buf = qd_buf.to(base.device)
    dt_buf = dt_buf.to(base.device)
    lengths = lengths.to(base.device)
    view = robot.root_physx_view
    its = torch.zeros(base.num_envs, device=base.device)
    its_scaled = torch.zeros(base.num_envs, device=base.device)
    counted = torch.zeros(base.num_envs, dtype=torch.long, device=base.device)
    scale = float(args_cli.vel_scale)
    scale_sq = scale * scale
    max_len = int(q_buf.shape[1])
    checked = False
    for step in range(max_len - 1):
        active = (lengths > (step + 1)) & (dt_buf[:, step] > 1.0e-6)
        if not bool(active.any().item()):
            continue
        q = q_buf[:, step]
        qd = qd_buf[:, step]
        dt = dt_buf[:, step].clamp_min(1.0e-6)
        qdd = (qd_buf[:, step + 1] - qd) / dt.unsqueeze(-1)
        robot.write_joint_state_to_sim(q, qd)
        base.sim.physics_sim_view.update_articulations_kinematic()
        mass = view.get_generalized_mass_matrices()
        coriolis = view.get_coriolis_and_centrifugal_compensation_forces()
        gravity = view.get_gravity_compensation_forces()
        n_j = q.shape[1]
        if not checked:
            print(
                f"[INFO] M {tuple(mass.shape)} C {tuple(coriolis.shape)} g {tuple(gravity.shape)}",
                flush=True,
            )
            checked = True
        inertial = torch.bmm(mass[:, :n_j, :n_j], qdd.unsqueeze(-1)).squeeze(-1) + coriolis[:, :n_j]
        tau = inertial + gravity[:, :n_j]
        tau_scaled = inertial * scale_sq + gravity[:, :n_j]
        dt_active = dt_buf[:, step]
        its[active] += tau[active].square().sum(dim=-1) * dt_active[active]
        its_scaled[active] += tau_scaled[active].square().sum(dim=-1) * (dt_active[active] / scale)
        counted[active] += 1
        if step % 40 == 0:
            live = int(active.sum().item())
            sample = float(tau[active][0].norm().item()) if live else 0.0
            print(f"[INFO] step {step} live {live} sample ||tau|| {sample:.3f} Nm", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.its_csv)), exist_ok=True)
    its_cpu = its.detach().cpu().tolist()
    scaled_cpu = its_scaled.detach().cpu().tolist()
    count_cpu = counted.detach().cpu().tolist()
    with open(args_cli.its_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path_idx", "its_nm2_s", "its_same_mean_duration_nm2_s", "n_intervals", "vel_scale"])
        for path_idx, (native, scaled, n_int) in enumerate(zip(its_cpu, scaled_cpu, count_cpu, strict=True)):
            writer.writerow([path_idx, native, scaled, n_int, scale])
    print(f"[INFO] Wrote {args_cli.its_csv}", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
