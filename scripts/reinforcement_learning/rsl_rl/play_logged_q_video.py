# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Play a logged 6-DoF joint trajectory and record one rgb video."""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a logged q(t) file and record a video.")
parser.add_argument("--task", type=str, default="Isaac-Reach-UR10-Play-v0")
parser.add_argument("--traj_file", type=str, required=True, help="RL env_*_traj.csv or Case_5 ST_MotionInfo.txt")
parser.add_argument("--video_dir", type=str, required=True)
parser.add_argument("--name_prefix", type=str, default="logged_q")
parser.add_argument("--hold_s", type=float, default=1.5, help="Hold the last pose after the logged motion.")
parser.add_argument("--fps", type=int, default=15)
parser.add_argument("--cmd_xyzquat", type=float, nargs=7, default=None, help="Optional ee_pose cmd x y z qw qx qy qz.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import reach_traj_csv_utils as csv_utils  # noqa: E402

Q_COLS = [
    "q_shoulder_pan_joint",
    "q_shoulder_lift_joint",
    "q_elbow_joint",
    "q_wrist_1_joint",
    "q_wrist_2_joint",
    "q_wrist_3_joint",
]


def load_traj(path: str) -> tuple[np.ndarray, np.ndarray]:
    path = os.path.abspath(path)
    if path.endswith(".csv"):
        rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
        t = np.array([float(r["sim_time_s"]) for r in rows], dtype=np.float64)
        q = np.array([[float(r[c]) for c in Q_COLS] for r in rows], dtype=np.float64)
        return t, q
    arr = np.loadtxt(path)
    return arr[:, 0].astype(np.float64), arr[:, 1:7].astype(np.float64)


def resample_q(t: np.ndarray, q: np.ndarray, dt: float, hold_s: float) -> np.ndarray:
    t = t - t[0]
    t_end = float(t[-1])
    grid = np.arange(0.0, t_end + 0.5 * dt, dt)
    out = np.stack([np.interp(grid, t, q[:, j]) for j in range(6)], axis=1)
    n_hold = max(0, int(round(hold_s / dt)))
    if n_hold:
        out = np.vstack([out, np.repeat(out[-1][None], n_hold, axis=0)])
    return out


def positions_to_action(arm_term, q_full: torch.Tensor) -> torch.Tensor:
    joint_ids = arm_term._joint_ids
    q = q_full[:, joint_ids] if not isinstance(joint_ids, slice) else q_full
    scale = arm_term._scale
    offset = arm_term._offset
    if isinstance(scale, torch.Tensor):
        return (q - offset) / scale
    return (q - offset) / float(scale)


def write_q(robot, q6: torch.Tensor) -> torch.Tensor:
    names = list(robot.joint_names)
    ids = [names.index(c[2:] if c.startswith("q_") else c) for c in Q_COLS]
    q_full = robot.data.joint_pos.clone()
    qd_full = torch.zeros_like(q_full)
    q_full[:, ids] = q6
    robot.write_joint_state_to_sim(q_full, qd_full)
    robot.set_joint_position_target(q_full)
    robot.set_joint_velocity_target(qd_full)
    return q_full


def set_ee_command(base: ManagerBasedRLEnv, cmd7: list[float]) -> None:
    term = base.command_manager.get_term("ee_pose")
    val = torch.tensor(cmd7, device=term.pose_command_b.device, dtype=term.pose_command_b.dtype).view(1, 7)
    term.pose_command_b[:] = val


def main():
    t_raw, q_raw = load_traj(args_cli.traj_file)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    csv_utils.configure_p2p_style_eval(
        env_cfg,
        disable_resample=True,
        freeze_finished_envs=True,
        disable_ovf=True,
    )
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
    if hasattr(env_cfg.commands, "ee_pose"):
        env_cfg.commands.ee_pose.debug_vis = True

    os.makedirs(args_cli.video_dir, exist_ok=True)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    base: ManagerBasedRLEnv = env.unwrapped
    q_play = resample_q(t_raw, q_raw, float(base.step_dt), args_cli.hold_s)
    video_length = int(q_play.shape[0])
    video_kwargs = {
        "video_folder": os.path.abspath(args_cli.video_dir),
        "step_trigger": lambda step: step == 0,
        "video_length": video_length,
        "name_prefix": args_cli.name_prefix,
        "disable_logger": True,
        "fps": int(args_cli.fps),
    }
    print("[INFO] Recording video")
    print_dict(video_kwargs, nesting=4)
    print(f"[INFO] traj {args_cli.traj_file}  samples={len(t_raw)} play_steps={video_length} T≈{video_length * base.step_dt:.2f}s")
    env = gym.wrappers.RecordVideo(env, **video_kwargs)
    base = env.unwrapped

    robot = base.scene["robot"]
    arm_term = base.action_manager.get_term("arm_action")
    env.reset()
    if args_cli.cmd_xyzquat is not None:
        set_ee_command(base, args_cli.cmd_xyzquat)

    for i in range(video_length):
        q6 = torch.as_tensor(q_play[i], device=robot.data.joint_pos.device, dtype=robot.data.joint_pos.dtype).view(1, 6)
        if args_cli.cmd_xyzquat is not None:
            set_ee_command(base, args_cli.cmd_xyzquat)
        q_full = write_q(robot, q6)
        action = positions_to_action(arm_term, q_full)
        env.step(action)

    env.close()
    print(f"[INFO] Video written under {os.path.abspath(args_cli.video_dir)}")


if __name__ == "__main__":
    main()
    simulation_app.close()
