# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for reach trajectory CSV export (RL play and Lula C-spline)."""

from __future__ import annotations

import csv
import os
from typing import IO, Any

import torch

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnv, ManagerBasedRLEnvCfg

from isaaclab_tasks.manager_based.manipulation.reach.mdp.rewards import reach_success_criteria

_REACH_CRITERIA_PARAM_KEYS = (
    "max_distance",
    "max_angle_rad",
    "max_lin_vel",
    "max_ang_vel",
    "max_lin_acc",
    "max_ang_acc",
)


def strict_reach_params_from_env_cfg(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
) -> dict[str, Any] | None:
    """Final reach thresholds from env cfg before sim start (before curriculum mutates them)."""
    rewards_cfg = getattr(env_cfg, "rewards", None)
    if rewards_cfg is not None:
        term = getattr(rewards_cfg, "reach_success_bonus", None)
        if term is not None and getattr(term, "params", None):
            return dict(term.params)
    events_cfg = getattr(env_cfg, "events", None)
    if events_cfg is not None:
        term = getattr(events_cfg, "resample_ee_pose_on_reach", None)
        if term is not None and getattr(term, "params", None):
            return dict(term.params)
    return None


def configure_p2p_style_eval(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    *,
    disable_resample: bool,
) -> None:
    """Align play eval with P2P-Play: strict thresholds, optional one target per attempt."""
    curriculum_cfg = getattr(env_cfg, "curriculum", None)
    if curriculum_cfg is not None:
        if hasattr(curriculum_cfg, "reach_success_criteria"):
            curriculum_cfg.reach_success_criteria = None
        # Prevent play-time OVF threshold from starting at 2.0 via curriculum schedule.
        if hasattr(curriculum_cfg, "cri_ovf_term_threshold"):
            curriculum_cfg.cri_ovf_term_threshold = None
        if hasattr(curriculum_cfg, "cri_ovf_reward_weight"):
            curriculum_cfg.cri_ovf_reward_weight = None

    terminations_cfg = getattr(env_cfg, "terminations", None)
    if terminations_cfg is not None and hasattr(terminations_cfg, "OVF"):
        ovf = terminations_cfg.OVF
        if ovf is not None and getattr(ovf, "params", None) is not None:
            ovf.params["threshold"] = 0.96

    rewards_cfg = getattr(env_cfg, "rewards", None)
    if rewards_cfg is not None and hasattr(rewards_cfg, "CRI_OVF"):
        cri_ovf = rewards_cfg.CRI_OVF
        if cri_ovf is not None and getattr(cri_ovf, "params", None) is not None:
            cri_ovf.params["threshold"] = 0.96

    if disable_resample:
        events_cfg = getattr(env_cfg, "events", None)
        if events_cfg is not None and hasattr(events_cfg, "resample_ee_pose_on_reach"):
            events_cfg.resample_ee_pose_on_reach = None

def apply_strict_reach_params_to_env(env: ManagerBasedRLEnv, strict_params: dict[str, Any]) -> None:
    """Force reward/event reach thresholds to the captured strict finals."""
    criteria = {key: strict_params[key] for key in _REACH_CRITERIA_PARAM_KEYS if key in strict_params}
    if not criteria:
        return

    if "reach_success_bonus" in env.reward_manager.active_terms:
        reward_cfg = env.reward_manager.get_term_cfg("reach_success_bonus")
        reward_cfg.params.update(criteria)
        env.reward_manager.set_term_cfg("reach_success_bonus", reward_cfg)

    event_names = [name for names in env.event_manager.active_terms.values() for name in names]
    if "resample_ee_pose_on_reach" in event_names:
        event_cfg = env.event_manager.get_term_cfg("resample_ee_pose_on_reach")
        event_cfg.params.update(criteria)
        env.event_manager.set_term_cfg("resample_ee_pose_on_reach", event_cfg)


def open_traj_csv_writers(
    csv_dir: str,
    num_envs: int,
    joint_names: list[str],
    num_cri_points: int,
) -> tuple[dict[int, IO[str]], dict[int, csv.DictWriter]]:
    os.makedirs(csv_dir, exist_ok=True)
    fieldnames = (
        ["global_step", "sim_time_s", "reach_event", "max_CRI"]
        + [f"q_{name}" for name in joint_names]
        + [f"qd_{name}" for name in joint_names]
        + [f"CRI_{i}" for i in range(num_cri_points)]
    )
    files: dict[int, IO[str]] = {}
    writers: dict[int, csv.DictWriter] = {}
    for env_idx in range(num_envs):
        path = os.path.join(csv_dir, f"env_{env_idx}_traj.csv")
        handle = open(path, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        handle.flush()
        files[env_idx] = handle
        writers[env_idx] = writer
    return files, writers


def open_episode_reach_csv(csv_dir: str) -> tuple[IO[str], csv.DictWriter]:
    os.makedirs(csv_dir, exist_ok=True)
    path = os.path.join(csv_dir, "episode_reach.csv")
    handle = open(path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        handle,
        fieldnames=["env_idx", "episode_id", "ended_at_s", "reached", "outcome"],
    )
    writer.writeheader()
    handle.flush()
    return handle, writer


def reach_percent(reached: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return 100.0 * float(reached) / float(total)


def write_reach_summary_csv(csv_dir: str, reached: int, total: int) -> str:
    os.makedirs(csv_dir, exist_ok=True)
    path = os.path.join(csv_dir, "reach_summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["total_episodes", "reached_episodes", "failed_episodes", "reach_percent"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "total_episodes": total,
                "reached_episodes": reached,
                "failed_episodes": total - reached,
                "reach_percent": round(reach_percent(reached, total), 4),
            }
        )
    return os.path.abspath(path)


def append_traj_rows(
    writers: dict[int, csv.DictWriter],
    robot,
    global_step: int,
    sim_time_s: float,
    env_log_mask: torch.Tensor,
    reach_event: torch.Tensor,
) -> None:
    """Append one CSV row per active env using the pre-reset CRI motion snapshot.

    ``env.step`` may reset terminated envs before the play loop logs. Reading live
    ``joint_pos`` / ``joint_vel`` then pairs post-reset ``qd=0`` with a stale CRI.
    ``get_cri_trajectory_state`` keeps the (q, qd, CRI) from the first CRI eval at
    this physics step (termination/reward), which is the state that must be exported.
    """
    joint_names = robot.joint_names
    with torch.inference_mode():
        joint_pos, joint_vel, cri = robot.data.get_cri_trajectory_state()
        joint_pos = joint_pos.detach().cpu().numpy()
        joint_vel = joint_vel.detach().cpu().numpy()
        cri = cri.detach().cpu().numpy()
        reach_np = reach_event.detach().cpu().numpy()

    mask = env_log_mask.detach().bool().cpu()
    num_cri = int(cri.shape[1]) if cri.ndim == 2 else 0
    num_joints = min(len(joint_names), int(joint_pos.shape[1]) if joint_pos.ndim == 2 else 0)

    for env_idx, writer in writers.items():
        if env_idx >= int(mask.shape[0]) or not bool(mask[env_idx].item()):
            continue
        row: dict[str, Any] = {
            "global_step": global_step,
            "sim_time_s": sim_time_s,
            "reach_event": int(bool(reach_np[env_idx])),
            "max_CRI": float(cri[env_idx].max()) if num_cri > 0 else float("nan"),
        }
        for joint_idx in range(num_joints):
            name = joint_names[joint_idx]
            row[f"q_{name}"] = float(joint_pos[env_idx, joint_idx])
            row[f"qd_{name}"] = float(joint_vel[env_idx, joint_idx])
        for cri_idx in range(num_cri):
            row[f"CRI_{cri_idx}"] = float(cri[env_idx, cri_idx])
        writer.writerow(row)


def classify_failure_outcome(base: ManagerBasedRLEnv, env_idx: int) -> str:
    tm = base.termination_manager
    if "time_out" in tm.active_terms and bool(tm.get_term("time_out")[env_idx].item()):
        return "fail_timeout"
    if "OVF" in tm.active_terms and bool(tm.get_term("OVF")[env_idx].item()):
        return "fail_ovf"
    return "fail_other"


def record_attempt(
    writer: csv.DictWriter,
    handle: IO[str],
    env_idx: int,
    episode_id: int,
    ended_at_s: float,
    reached: bool,
    outcome: str,
) -> None:
    writer.writerow(
        {
            "env_idx": env_idx,
            "episode_id": episode_id,
            "ended_at_s": ended_at_s,
            "reached": int(reached),
            "outcome": outcome,
        }
    )
    handle.flush()


def resolve_reach_event(
    base: ManagerBasedRLEnv,
    reach_cmd_snapshot: torch.Tensor,
    strict_reach_params: dict[str, Any],
) -> torch.Tensor:
    """Prefer pre-reset ``reach_success`` termination (P2P); else strict criteria."""
    if "reach_success" in base.termination_manager.active_terms:
        return base.termination_manager.get_term("reach_success")
    return reach_success_criteria(base, command_b=reach_cmd_snapshot, **strict_reach_params)
