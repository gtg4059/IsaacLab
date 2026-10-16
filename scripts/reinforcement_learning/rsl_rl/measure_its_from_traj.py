# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Integral of squared inverse-dynamics torque along logged reach trajectories.

τ = M(q) qdd + C(q, qd) + g(q), with qdd = (qd_{t+1} - qd_t) / dt.
ITS = Σ ||τ||^2 dt. The implicit drive gain is not used.
"""

from __future__ import annotations

import argparse
import csv
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="ITS from logged joint trajectories.")
parser.add_argument("--task", type=str, default="Isaac-Reach-UR10-v0")
parser.add_argument("--num_envs", type=int, default=8192)
parser.add_argument("--traj_dir", type=str, required=True)
parser.add_argument("--its_csv", type=str, required=True)
parser.add_argument("--dt", type=float, default=1.0 / 15.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def _load_columns(traj_dir: str, env_count: int, joint_names: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_cols = [f"q_{name}" for name in joint_names]
    qd_cols = [f"qd_{name}" for name in joint_names]
    series: list[tuple[list[list[float]], list[list[float]]]] = []
    max_len = 0
    for env_idx in range(env_count):
        path = os.path.join(traj_dir, f"env_{env_idx}_traj.csv")
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        q = [[float(row[col]) for col in q_cols] for row in rows]
        qd = [[float(row[col]) for col in qd_cols] for row in rows]
        series.append((q, qd))
        max_len = max(max_len, len(rows))
    n_j = len(joint_names)
    q_buf = torch.zeros(env_count, max_len, n_j)
    qd_buf = torch.zeros(env_count, max_len, n_j)
    lengths = torch.zeros(env_count, dtype=torch.long)
    for env_idx, (q, qd) in enumerate(series):
        t_len = len(q)
        lengths[env_idx] = t_len
        if t_len == 0:
            continue
        q_buf[env_idx, :t_len] = torch.tensor(q)
        qd_buf[env_idx, :t_len] = torch.tensor(qd)
    return q_buf, qd_buf, lengths


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    env.reset()
    robot = base.scene["robot"]
    joint_names = list(robot.joint_names)
    print(f"[INFO] Loading trajectories for joints {joint_names}", flush=True)
    q_buf, qd_buf, lengths = _load_columns(args_cli.traj_dir, base.num_envs, joint_names)
    q_buf = q_buf.to(base.device)
    qd_buf = qd_buf.to(base.device)
    lengths = lengths.to(base.device)
    view = robot.root_physx_view
    its = torch.zeros(base.num_envs, device=base.device)
    counted = torch.zeros(base.num_envs, dtype=torch.long, device=base.device)
    dt = float(args_cli.dt)
    max_len = int(q_buf.shape[1])
    checked = False
    for step in range(max_len - 1):
        active = lengths > (step + 1)
        if not bool(active.any().item()):
            break
        q = q_buf[:, step]
        qd = qd_buf[:, step]
        qdd = (qd_buf[:, step + 1] - qd) / dt
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
        tau = torch.bmm(mass[:, :n_j, :n_j], qdd.unsqueeze(-1)).squeeze(-1)
        tau = tau + coriolis[:, :n_j] + gravity[:, :n_j]
        its[active] += tau[active].square().sum(dim=-1) * dt
        counted[active] += 1
        if step % 50 == 0:
            live = int(active.sum().item())
            sample = float(tau[active][0].norm().item()) if live else 0.0
            print(f"[INFO] step {step} live {live} sample ||tau|| {sample:.3f} Nm", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.its_csv)), exist_ok=True)
    its_cpu = its.detach().cpu().tolist()
    count_cpu = counted.detach().cpu().tolist()
    with open(args_cli.its_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["env_idx", "its_nm2_s", "n_intervals"])
        for env_idx, (value, n_int) in enumerate(zip(its_cpu, count_cpu, strict=True)):
            writer.writerow([env_idx, value, n_int])
    print(f"[INFO] Wrote {args_cli.its_csv}", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
