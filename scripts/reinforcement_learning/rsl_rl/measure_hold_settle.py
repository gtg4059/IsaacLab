# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure ReachSuccessCriteria hold_count / gate rates for a checkpoint."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Measure hold_steps settle depth for a checkpoint.")
parser.add_argument("--video", dest="video", action="store_true")
parser.add_argument("--no_video", dest="video", action="store_false")
parser.set_defaults(video=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--task", type=str, default="Isaac-Reach-UR10-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--use_pretrained_checkpoint", action="store_true")
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument("--steps", type=int, default=1440, help="Env steps to roll out (720 = 1 episode).")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

import os

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.reach.mdp.observations import command_origin_pose_w
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


def _gate_masks(env, params: dict) -> dict[str, torch.Tensor]:
    asset = env.scene[params["asset_cfg"].name]
    command = env.command_manager.get_command(params["command_name"])
    body_ids = params["asset_cfg"].body_ids
    bid = int(body_ids[0]) if isinstance(body_ids, list) else int(body_ids)
    origin_pos_w, origin_quat_w = command_origin_pose_w(env, params["command_name"], asset)
    des_pos_w, _ = combine_frame_transforms(origin_pos_w, origin_quat_w, command[:, :3])
    distance = torch.norm(asset.data.body_pos_w[:, bid] - des_pos_w, dim=1)
    des_quat_w = quat_mul(origin_quat_w, command[:, 3:7])
    quat_err = quat_error_magnitude(asset.data.body_quat_w[:, bid], des_quat_w)
    lin_spd = torch.norm(asset.data.body_lin_vel_w[:, bid, :], dim=-1)
    ang_spd = torch.norm(asset.data.body_ang_vel_w[:, bid, :], dim=-1)
    lin_acc = torch.norm(asset.data.body_lin_acc_w[:, bid, :], dim=-1)
    ang_acc = torch.norm(asset.data.body_ang_acc_w[:, bid, :], dim=-1)
    pose_ok = (distance <= params["max_distance"]) & (quat_err <= params["max_angle_rad"])
    vel_ok = (lin_spd <= params["max_lin_vel"]) & (ang_spd <= params["max_ang_vel"])
    acc_ok = (lin_acc <= params["max_lin_acc"]) & (ang_acc <= params["max_ang_acc"])
    return {
        "pose_ok": pose_ok,
        "vel_ok": vel_ok,
        "acc_ok": acc_ok,
        "instant": pose_ok & vel_ok & acc_ok,
        "pos_ok": distance <= params["max_distance"],
        "ori_ok": quat_err <= params["max_angle_rad"],
        "lin_vel_ok": lin_spd <= params["max_lin_vel"],
        "ang_vel_ok": ang_spd <= params["max_ang_vel"],
        "lin_acc_ok": lin_acc <= params["max_lin_acc"],
        "ang_acc_ok": ang_acc <= params["max_ang_acc"],
        "distance": distance,
        "quat_err": quat_err,
        "lin_spd": lin_spd,
        "ang_spd": ang_spd,
        "lin_acc": lin_acc,
        "ang_acc": ang_acc,
    }


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Pre-trained checkpoint unavailable.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base = env.unwrapped
    term_cfg = base.reward_manager.get_term_cfg("reach_success_bonus")
    criteria = term_cfg.func
    params = term_cfg.params
    required = int(params.get("hold_steps", 0))
    hist_size = max(required + 8, 16)
    device = base.device
    hold_hist = torch.zeros(hist_size, dtype=torch.long, device=device)
    episode_peak = torch.zeros(base.num_envs, dtype=torch.long, device=device)
    episode_peaks: list[int] = []
    gate_hits = {k: 0 for k in (
        "pose_ok", "vel_ok", "acc_ok", "instant", "pos_ok", "ori_ok",
        "lin_vel_ok", "ang_vel_ok", "lin_acc_ok", "ang_acc_ok",
    )}
    gate_n = 0
    metric_sum = {k: 0.0 for k in ("distance", "quat_err", "lin_spd", "ang_spd", "lin_acc", "ang_acc")}
    metric_min = {k: float("inf") for k in metric_sum}
    global_max_hold = 0
    success_steps = 0

    orig = criteria.compute_success

    def _hooked(*args, **kwargs):
        nonlocal global_max_hold, success_steps, gate_n
        prev_step = criteria._updated_step
        success = orig(*args, **kwargs)
        if criteria._updated_step == prev_step:
            return success
        hold = criteria._hold_count
        global_max_hold = max(global_max_hold, int(hold.max().item()))
        success_steps += int(success.sum().item())
        clipped = torch.clamp(hold, max=hist_size - 1)
        hold_hist.index_add_(0, clipped, torch.ones_like(clipped))
        episode_peak.copy_(torch.maximum(episode_peak, hold))
        gates = _gate_masks(base, params)
        gate_n += hold.numel()
        for key in gate_hits:
            gate_hits[key] += int(gates[key].sum().item())
        for key in metric_sum:
            metric_sum[key] += float(gates[key].sum().item())
            metric_min[key] = min(metric_min[key], float(gates[key].min().item()))
        return success

    criteria.compute_success = _hooked

    obs = env.get_observations()
    steps = max(int(args_cli.steps), 1)
    print(
        f"[HOLD] required={required} num_envs={base.num_envs} steps={steps} "
        f"episode_length={base.max_episode_length}",
        flush=True,
    )
    for step in range(steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            done = dones.bool() if dones.dtype != torch.bool else dones
            if done.any():
                episode_peaks.extend(episode_peak[done].tolist())
                episode_peak[done] = 0
        if (step + 1) % 240 == 0 or step + 1 == steps:
            print(
                f"[HOLD] step={step + 1}/{steps} max_so_far={global_max_hold} "
                f"instant_frac={gate_hits['instant'] / max(gate_n, 1):.4f}",
                flush=True,
            )

    if episode_peak.numel():
        episode_peaks.extend(episode_peak.tolist())

    hist_cpu = hold_hist.cpu().tolist()
    print("========== hold_steps settle ==========")
    print(f"checkpoint: {resume_path}")
    print(f"required hold_steps: {required}")
    print(f"global max hold_count: {global_max_hold}")
    print(f"success steps (hold>=required): {success_steps}")
    print("hold_count histogram (env-steps):")
    for i, count in enumerate(hist_cpu):
        if count:
            print(f"  hold={i}: {count}")
    if episode_peaks:
        peak_t = torch.tensor(episode_peaks, dtype=torch.long)
        print(
            f"episode peak hold: n={peak_t.numel()} max={int(peak_t.max())} "
            f"mean={float(peak_t.float().mean()):.3f} p50={int(peak_t.median())} "
            f"p95={int(torch.quantile(peak_t.float(), 0.95))}"
        )
        for k in range(1, required + 1):
            frac = float((peak_t >= k).float().mean())
            print(f"  episodes with peak>={k}: {frac:.4f}")
    print("gate rates (env-steps):")
    for key in (
        "pos_ok", "ori_ok", "pose_ok", "lin_vel_ok", "ang_vel_ok", "vel_ok",
        "lin_acc_ok", "ang_acc_ok", "acc_ok", "instant",
    ):
        print(f"  {key}: {gate_hits[key] / max(gate_n, 1):.4f}")
    print("metric mean / min:")
    for key in metric_sum:
        print(f"  {key}: mean={metric_sum[key] / max(gate_n, 1):.5f} min={metric_min[key]:.5f}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
