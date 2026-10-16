# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""ITS along a packed no-OVF trajectory. τ = M qdd + C + g, qdd = Δqd / dt."""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="ITS from a packed q/qd npz.")
parser.add_argument("--task", type=str, default="Isaac-Reach-UR10-v0")
parser.add_argument("--num_envs", type=int, default=8192)
parser.add_argument("--packed_npz", type=str, required=True)
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


def main() -> None:
    packed = np.load(args_cli.packed_npz)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    env.reset()
    robot = base.scene["robot"]
    view = robot.root_physx_view
    q_buf = torch.as_tensor(packed["q"], device=base.device)
    qd_buf = torch.as_tensor(packed["qd"], device=base.device)
    lengths = torch.as_tensor(packed["lengths"], device=base.device)
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
            print(f"[INFO] M {tuple(mass.shape)} C {tuple(coriolis.shape)} g {tuple(gravity.shape)}", flush=True)
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
    with open(args_cli.its_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["env_idx", "its_nm2_s", "n_intervals"])
        for env_idx, (value, n_int) in enumerate(zip(its.detach().cpu().tolist(), counted.detach().cpu().tolist(), strict=True)):
            writer.writerow([env_idx, value, n_int])
    print(f"[INFO] Wrote {args_cli.its_csv}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
