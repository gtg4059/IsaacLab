# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Play a reach checkpoint and report old vs new hold-gate constraint rates."""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Measure hold-gate constraint rates on a reach checkpoint.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--task", type=str, default="Isaac-Reach-UR10-Play-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--steps", type=int, default=480, help="Env steps to roll out (one 32 s episode).")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = False
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata
import os

from packaging import version

installed_version = metadata.version("rsl-rl-lib")
_rsl_ge_4 = version.parse(installed_version) >= version.parse("4.0.0")

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg, handle_deprecated_rsl_rl_checkpoint
from isaaclab_tasks.manager_based.manipulation.reach.mdp.observations import command_origin_pose_w
from isaaclab_tasks.manager_based.manipulation.reach.mdp.rewards import _body_idx_single
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401


def _ee_errors(env):
    asset_cfg = SceneEntityCfg("robot", body_names="ee_link")
    asset_cfg.resolve(env.scene)
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("ee_pose")
    bid = _body_idx_single(env, asset_cfg)
    origin_pos_w, origin_quat_w = command_origin_pose_w(env, "ee_pose", asset)
    des_pos_w, _ = combine_frame_transforms(origin_pos_w, origin_quat_w, command[:, :3])
    distance = torch.norm(asset.data.body_pos_w[:, bid] - des_pos_w, dim=1)
    des_quat_w = quat_mul(origin_quat_w, command[:, 3:7])
    angle = quat_error_magnitude(asset.data.body_quat_w[:, bid], des_quat_w)
    lin_vel = torch.norm(asset.data.body_lin_vel_w[:, bid, :], dim=-1)
    ang_vel = torch.norm(asset.data.body_ang_vel_w[:, bid, :], dim=-1)
    return distance, angle, lin_vel, ang_vel


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # One attempt per env: do not reset on 4-hold success.
    if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "reach_success"):
        env_cfg.terminations.reach_success = None
    resume_path = retrieve_file_path(args_cli.checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print(f"[INFO] checkpoint={resume_path}  steps={args_cli.steps}  num_envs={args_cli.num_envs}")

    base = env.unwrapped
    obs = env.get_observations()
    keys = (
        "dist_lt_2cm", "dist_lt_3cm", "dist_lt_5cm", "angle_lt_0.1",
        "lin_lt_0.01", "lin_lt_0.02", "ang_lt_0.02", "ang_lt_0.1",
        "old_gate", "new_gate", "new_pos_ori", "new_pos_ori_lin",
    )
    sums = {k: 0.0 for k in keys}
    dist_sum = ang_sum = lin_sum = avel_sum = 0.0
    dist_min = torch.full((args_cli.num_envs,), 1e9, device=env.unwrapped.device)
    ever_old = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=env.unwrapped.device)
    ever_new = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=env.unwrapped.device)
    hold_new = torch.zeros(args_cli.num_envs, dtype=torch.long, device=env.unwrapped.device)
    max_hold_new = torch.zeros(args_cli.num_envs, dtype=torch.long, device=env.unwrapped.device)
    first_new = torch.full((args_cli.num_envs,), -1, dtype=torch.long, device=env.unwrapped.device)
    n = 0

    with torch.inference_mode():
        for _ in range(int(args_cli.steps)):
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if _rsl_ge_4:
                policy.reset(dones)
            distance, angle, lin_vel, ang_vel = _ee_errors(base)
            dist_min = torch.minimum(dist_min, distance)
            dist_ok_2 = distance <= 0.02
            dist_ok_3 = distance <= 0.03
            ang_ok = angle <= 0.1
            lin_01 = lin_vel <= 0.01
            lin_02 = lin_vel <= 0.02
            avel_002 = ang_vel <= 0.02
            avel_01 = ang_vel <= 0.1
            old = dist_ok_3 & ang_ok & lin_01 & avel_01
            new = dist_ok_2 & ang_ok & lin_02 & avel_002
            ever_old |= old
            ever_new |= new
            hold_new = torch.where(new, hold_new + 1, torch.zeros_like(hold_new))
            max_hold_new = torch.maximum(max_hold_new, hold_new)
            first_new = torch.where((first_new < 0) & new, torch.full_like(first_new, n), first_new)
            n += 1
            sums["dist_lt_2cm"] += float(dist_ok_2.float().mean())
            sums["dist_lt_3cm"] += float(dist_ok_3.float().mean())
            sums["dist_lt_5cm"] += float((distance <= 0.05).float().mean())
            sums["angle_lt_0.1"] += float(ang_ok.float().mean())
            sums["lin_lt_0.01"] += float(lin_01.float().mean())
            sums["lin_lt_0.02"] += float(lin_02.float().mean())
            sums["ang_lt_0.02"] += float(avel_002.float().mean())
            sums["ang_lt_0.1"] += float(avel_01.float().mean())
            sums["old_gate"] += float(old.float().mean())
            sums["new_gate"] += float(new.float().mean())
            sums["new_pos_ori"] += float((dist_ok_2 & ang_ok).float().mean())
            sums["new_pos_ori_lin"] += float((dist_ok_2 & ang_ok & lin_02).float().mean())
            dist_sum += float(distance.mean())
            ang_sum += float(angle.mean())
            lin_sum += float(lin_vel.mean())
            avel_sum += float(ang_vel.mean())

    print("\n=== mean over rollout (fraction of env-steps) ===")
    for k, v in sums.items():
        print(f"  {k:18s} {v / n:.4f}")
    print("\n=== mean EE state ===")
    print(f"  distance_m   {dist_sum / n:.4f}")
    print(f"  angle_rad    {ang_sum / n:.4f}")
    print(f"  lin_vel      {lin_sum / n:.4f}")
    print(f"  ang_vel      {avel_sum / n:.4f}")
    print("\n=== per-env best distance ===")
    dm = dist_min.detach().cpu()
    print(f"  mean_min {float(dm.mean()):.4f}  p10 {float(torch.quantile(dm, 0.1)):.4f}  "
          f"p50 {float(torch.quantile(dm, 0.5)):.4f}  p90 {float(torch.quantile(dm, 0.9)):.4f}")
    print(f"  frac min<2cm {float((dm <= 0.02).float().mean()):.4f}  "
          f"min<3cm {float((dm <= 0.03).float().mean()):.4f}  "
          f"min<5cm {float((dm <= 0.05).float().mean()):.4f}")
    print("\n=== episode ever-in-gate (1-step stay = success) ===")
    print(f"  old 3cm/0.01/0.1  {float(ever_old.float().mean()):.4f}")
    print(f"  new 2cm/0.02/0.02 {float(ever_new.float().mean()):.4f}")
    mh = max_hold_new.detach().cpu().float()
    print("\n=== consecutive new-gate hold (max streak / env) ===")
    print(f"  ever >=1  {float((mh >= 1).float().mean()):.4f}")
    print(f"  ever >=2  {float((mh >= 2).float().mean()):.4f}")
    print(f"  ever >=4  {float((mh >= 4).float().mean()):.4f}")
    print(f"  ever >=8  {float((mh >= 8).float().mean()):.4f}")
    print(f"  max_hold mean {float(mh.mean()):.2f}  p50 {float(torch.quantile(mh, 0.5)):.1f}  "
          f"p90 {float(torch.quantile(mh, 0.9)):.1f}")
    hit = first_new[first_new >= 0].detach().cpu().float()
    if hit.numel() > 0:
        dt = float(base.step_dt)
        print("\n=== first 1-hold time among successes ===")
        print(f"  n={int(hit.numel())}  mean {(float(hit.mean()) + 1) * dt:.2f}s  "
              f"p50 {(float(torch.quantile(hit, 0.5)) + 1) * dt:.2f}s  "
              f"p90 {(float(torch.quantile(hit, 0.9)) + 1) * dt:.2f}s")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
