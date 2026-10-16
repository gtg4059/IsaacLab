# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Replay the three slowest model_40000 successes and write one rgb video each.

``--mode rollout`` repeats the seed-42, 2048-env eval and saves q(t) for the
requested env indices. ``--mode video`` plays those logs on one robot.
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Record videos of the slow model_40000 reaches.")
parser.add_argument("--mode", choices=("rollout", "video"), required=True)
parser.add_argument("--task", type=str, default="Isaac-Reach-UR10-v0")
parser.add_argument("--agent", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--env_indices",
    type=str,
    default="806,503,1349",
    help="Comma-separated env indices from the seed-42 eval.",
)
parser.add_argument(
    "--out_dir",
    type=str,
    default="logs/rsl_rl/reach_ur10/2026-09-26_20-59-56/eval_40000/slow_videos",
)
parser.add_argument("--hold_s", type=float, default=1.5)
parser.add_argument("--fps", type=int, default=15)
parser.add_argument("--disable_fabric", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.mode == "video":
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from packaging import version  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul  # noqa: E402
from isaaclab.utils.seed import configure_seed  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.manager_based.manipulation.reach.mdp.observations import command_origin_pose_w  # noqa: E402
from isaaclab_tasks.manager_based.manipulation.reach.mdp.rewards import _body_idx_single  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg  # noqa: E402

import importlib.metadata as metadata  # noqa: E402

import reach_traj_csv_utils as csv_utils  # noqa: E402

Q_COLS = [
    "q_shoulder_pan_joint",
    "q_shoulder_lift_joint",
    "q_elbow_joint",
    "q_wrist_1_joint",
    "q_wrist_2_joint",
    "q_wrist_3_joint",
]
JOINT_NAMES = [c[2:] for c in Q_COLS]


def _env_indices() -> list[int]:
    return [int(p.strip()) for p in args_cli.env_indices.split(",") if p.strip()]


def _load_policy(env, checkpoint: str, device: str):
    agent_cfg = load_cfg_from_registry(args_cli.task.split(":")[-1], "rsl_rl_cfg_entry_point")
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.seed = int(args_cli.seed)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(retrieve_file_path(checkpoint))
    policy = runner.get_inference_policy(device=device)
    return policy, metadata.version("rsl-rl-lib")


def _tracking(base: ManagerBasedRLEnv, command: torch.Tensor):
    asset_cfg = SceneEntityCfg("robot", body_names="ee_link")
    asset_cfg.resolve(base.scene)
    asset = base.scene[asset_cfg.name]
    bid = _body_idx_single(base, asset_cfg)
    origin_pos_w, origin_quat_w = command_origin_pose_w(base, "ee_pose", asset)
    des_pos_w, _ = combine_frame_transforms(origin_pos_w, origin_quat_w, command[:, :3])
    pos_err = torch.norm(asset.data.body_pos_w[:, bid] - des_pos_w, dim=1)
    des_quat_w = quat_mul(origin_quat_w, command[:, 3:7])
    ang_err = quat_error_magnitude(asset.data.body_quat_w[:, bid], des_quat_w)
    cri, _ = torch.max(asset.data.CRI, dim=1)
    return pos_err, ang_err, cri


def rollout() -> None:
    indices = _env_indices()
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=int(args_cli.num_envs),
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = int(args_cli.seed)
    strict = csv_utils.strict_reach_params_from_env_cfg(env_cfg)
    if strict is None:
        raise RuntimeError("reach_success_bonus params missing")
    csv_utils.configure_p2p_style_eval(
        env_cfg,
        disable_resample=True,
        freeze_finished_envs=True,
        ovf_threshold=0.96,
    )
    configure_seed(int(args_cli.seed), torch_deterministic=False)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=None)
    checkpoint = args_cli.checkpoint
    if checkpoint is None:
        raise ValueError("--checkpoint is required for rollout")
    policy, rsl_version = _load_policy(env, checkpoint, env.unwrapped.device)
    base: ManagerBasedRLEnv = env.unwrapped
    csv_utils.apply_strict_reach_params_to_env(base, strict)
    # Match play_reach_csv: re-seed after reset so observation noise starts from the same stream.
    configure_seed(int(args_cli.seed), torch_deterministic=False)

    robot = base.scene["robot"]
    name_to_col = {n: i for i, n in enumerate(robot.joint_names)}
    q_ids = [name_to_col[n] for n in JOINT_NAMES]
    idx_t = torch.tensor(indices, device=base.device, dtype=torch.long)
    logs = {i: [] for i in indices}
    done = {i: False for i in indices}
    cmds = {}

    obs = env.get_observations()
    step = 0
    max_steps = int(round(float(base.max_episode_length)))
    while simulation_app.is_running() and step < max_steps and not all(done.values()):
        with torch.no_grad():
            command = base.command_manager.get_command("ee_pose").detach().clone()
            if step == 0:
                for i in indices:
                    cmds[i] = command[i].detach().cpu().numpy().copy()
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if version.parse(rsl_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            step += 1
            pos_err, ang_err, cri = _tracking(base, command)
            reached_mask = csv_utils.resolve_reach_event(base, command, strict)
            q = robot.data.joint_pos[:, q_ids].detach().cpu().numpy()
            t = step * float(base.step_dt)
            for i in indices:
                if done[i]:
                    continue
                logs[i].append((t, q[i].copy(), float(pos_err[i]), float(ang_err[i]), float(cri[i])))
                reached = bool(reached_mask[i].item())
                if reached or bool(dones[i].item()):
                    done[i] = True
                    outcome = "success" if reached else "stopped"
                    print(
                        f"[INFO] env {i} {outcome} at {t:.3f}s  "
                        f"pos {float(pos_err[i]):.3f} ang {float(ang_err[i]):.3f} cri {float(cri[i]):.3f}"
                    )

    out_dir = os.path.abspath(args_cli.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    for i in indices:
        path = os.path.join(out_dir, f"env_{i}_traj.csv")
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sim_time_s", *Q_COLS, "pos_err", "ang_err", "max_cri_step"])
            writer.writeheader()
            for t, q, pos, ang, cri in logs[i]:
                row = {"sim_time_s": t, "pos_err": pos, "ang_err": ang, "max_cri_step": cri}
                for name, value in zip(Q_COLS, q):
                    row[name] = float(value)
                writer.writerow(row)
        cmd_path = os.path.join(out_dir, f"env_{i}_cmd.csv")
        with open(cmd_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["x", "y", "z", "qw", "qx", "qy", "qz"])
            writer.writerow([float(v) for v in cmds[i]])
        print(f"[INFO] wrote {path}  steps={len(logs[i])}  cmd={np.array2string(cmds[i], precision=4)}")
    env.close()


def _load_traj(path: str) -> tuple[np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    t = np.array([float(r["sim_time_s"]) for r in rows], dtype=np.float64)
    q = np.array([[float(r[c]) for c in Q_COLS] for r in rows], dtype=np.float64)
    return t, q


def _load_cmd(path: str) -> list[float]:
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    r = rows[0]
    return [float(r[k]) for k in ("x", "y", "z", "qw", "qx", "qy", "qz")]


def _resample(t: np.ndarray, q: np.ndarray, dt: float, hold_s: float) -> np.ndarray:
    t = t - t[0]
    grid = np.arange(0.0, float(t[-1]) + 0.5 * dt, dt)
    out = np.stack([np.interp(grid, t, q[:, j]) for j in range(6)], axis=1)
    n_hold = max(0, int(round(hold_s / dt)))
    if n_hold:
        out = np.vstack([out, np.repeat(out[-1][None], n_hold, axis=0)])
    return out


def _write_q(robot, q6: torch.Tensor) -> torch.Tensor:
    names = list(robot.joint_names)
    ids = [names.index(n) for n in JOINT_NAMES]
    q_full = robot.data.joint_pos.clone()
    qd_full = torch.zeros_like(q_full)
    q_full[:, ids] = q6
    robot.write_joint_state_to_sim(q_full, qd_full)
    robot.set_joint_position_target(q_full)
    robot.set_joint_velocity_target(qd_full)
    return q_full


def _positions_to_action(arm_term, q_full: torch.Tensor) -> torch.Tensor:
    joint_ids = arm_term._joint_ids
    q = q_full[:, joint_ids] if not isinstance(joint_ids, slice) else q_full
    scale = arm_term._scale
    offset = arm_term._offset
    if isinstance(scale, torch.Tensor):
        return (q - offset) / scale
    return (q - offset) / float(scale)


def _set_command(base: ManagerBasedRLEnv, cmd7: list[float]) -> None:
    term = base.command_manager.get_term("ee_pose")
    val = torch.tensor(cmd7, device=term.pose_command_b.device, dtype=term.pose_command_b.dtype).view(1, 7)
    term.pose_command_b[:] = val
    term.time_left[:] = 1.0e6


def video() -> None:
    import imageio.v2 as imageio

    out_dir = os.path.abspath(args_cli.out_dir)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    csv_utils.configure_p2p_style_eval(env_cfg, disable_resample=True, freeze_finished_envs=True, disable_ovf=True)
    env_cfg.episode_length_s = 40.0
    env_cfg.actions.arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=False,
    )
    env_cfg.scene.robot.actuators["arm"] = ImplicitActuatorCfg(
        joint_names_expr=[".*"],
        velocity_limit=100.0,
        effort_limit=330.0,
        stiffness=1.0e8,
        damping=1.0e4,
    )
    env_cfg.viewer.eye = (2.8, -2.4, 1.7)
    env_cfg.viewer.lookat = (0.25, 0.05, 0.35)
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    if hasattr(env_cfg.commands, "ee_pose"):
        env_cfg.commands.ee_pose.debug_vis = True

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    base: ManagerBasedRLEnv = env.unwrapped
    robot = base.scene["robot"]
    arm_term = base.action_manager.get_term("arm_action")
    dt = float(base.step_dt)

    for env_i in _env_indices():
        traj_path = os.path.join(out_dir, f"env_{env_i}_traj.csv")
        cmd = _load_cmd(os.path.join(out_dir, f"env_{env_i}_cmd.csv"))
        t_raw, q_raw = _load_traj(traj_path)
        q_play = _resample(t_raw, q_raw, dt, float(args_cli.hold_s))
        env.reset()
        frames = []
        for i in range(int(q_play.shape[0])):
            if not simulation_app.is_running():
                break
            q6 = torch.as_tensor(q_play[i], device=robot.data.joint_pos.device, dtype=robot.data.joint_pos.dtype).view(1, 6)
            _set_command(base, cmd)
            q_full = _write_q(robot, q6)
            action = _positions_to_action(arm_term, q_full)
            env.step(action)
            frame = base.render()
            if frame is not None and frame.size > 0 and int(frame.max()) > 0:
                frames.append(np.ascontiguousarray(frame[:, :, :3]))
        duration = float(t_raw[-1] - t_raw[0] + dt)
        video_path = os.path.join(out_dir, f"env_{env_i}_{duration:.1f}s.mp4")
        imageio.mimsave(video_path, frames, fps=int(args_cli.fps), macro_block_size=1)
        print(f"[INFO] video {video_path}  frames={len(frames)}  motion={duration:.2f}s")
    env.close()


if __name__ == "__main__":
    if args_cli.mode == "rollout":
        rollout()
    else:
        video()
    simulation_app.close()
