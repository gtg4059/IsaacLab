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
_REACH_CRITERIA_CALL_KEYS = ("command_name", "asset_cfg", *_REACH_CRITERIA_PARAM_KEYS)

# ee_pose command in robot base frame: (x, y, z, qw, qx, qy, qz)
_CMD_POSE_FIELDS = ("cmd_x", "cmd_y", "cmd_z", "cmd_qw", "cmd_qx", "cmd_qy", "cmd_qz")


def print_command_targets(
    command: torch.Tensor,
    *,
    label: str = "TARGET",
    max_rows: int = 32,
    pose_w: torch.Tensor | None = None,
) -> None:
    """Print ee_pose targets. ``command`` is (N, 7) in the robot base / origin frame."""
    cmd = command.detach().cpu()
    n = int(cmd.shape[0])
    world = None if pose_w is None else pose_w.detach().cpu()
    print(
        f"[INFO] {label}: ee_pose in robot base frame (x, y, z, r=hypot(x,y))  n={n}"
        + ("  + world xyz" if world is not None else "")
    )
    hdr = f"{'env':>5} {'x':>8} {'y':>8} {'z':>8} {'r':>8}"
    if world is not None:
        hdr += f" {'xw':>8} {'yw':>8} {'zw':>8}"
    print(f"[INFO] {label}: {hdr}")
    rows = min(n, max(0, int(max_rows)))
    for i in range(rows):
        x, y, z = float(cmd[i, 0]), float(cmd[i, 1]), float(cmd[i, 2])
        r = (x * x + y * y) ** 0.5
        line = f"{i:5d} {x:8.4f} {y:8.4f} {z:8.4f} {r:8.4f}"
        if world is not None:
            line += f" {float(world[i, 0]):8.4f} {float(world[i, 1]):8.4f} {float(world[i, 2]):8.4f}"
        print(f"[INFO] {label}: {line}")
    if n > rows:
        print(f"[INFO] {label}: ... {n - rows} more envs (full poses in episode_reach.csv / traj cmd_*)")


def print_command_targets_from_env(env, command_name: str, *, label: str = "TARGET", max_rows: int = 32) -> None:
    """Print current command term pose (base + world if the term has ``pose_command_w``)."""
    term = env.command_manager.get_term(command_name)
    pose_w = getattr(term, "pose_command_w", None)
    print_command_targets(term.command, label=label, max_rows=max_rows, pose_w=pose_w)


def command_pose_row(command_7: Any | None) -> dict[str, Any]:
    """Serialize a 7D pose command (base frame) for CSV columns."""
    if command_7 is None:
        return {key: "" for key in _CMD_POSE_FIELDS}
    vals = command_7.detach().cpu().reshape(-1).tolist() if hasattr(command_7, "detach") else list(command_7)
    if len(vals) < 7:
        return {key: "" for key in _CMD_POSE_FIELDS}
    return {
        "cmd_x": float(vals[0]),
        "cmd_y": float(vals[1]),
        "cmd_z": float(vals[2]),
        "cmd_qw": float(vals[3]),
        "cmd_qx": float(vals[4]),
        "cmd_qy": float(vals[5]),
        "cmd_qz": float(vals[6]),
    }


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
    freeze_finished_envs: bool = True,
    deterministic_sim: bool = False,
    disable_ovf: bool = False,
) -> None:
    """Align play eval with P2P-Play: strict thresholds, optional one target per attempt.

    When ``freeze_finished_envs`` is True, disable ``terminations.reach_success`` so a success
    does not reset that env mid-eval. Play still scores success via strict criteria; by default
    the policy keeps acting on finished envs (zip-compatible). Optional ``--freeze_actions``
    zeros those actions separately.

    ``deterministic_sim`` (PhysX enhanced determinism etc.) is opt-in: it changes dynamics vs
    training and can drop reach rate / reintroduce spurious OVF.
    """
    curriculum_cfg = getattr(env_cfg, "curriculum", None)
    if curriculum_cfg is not None:
        if hasattr(curriculum_cfg, "reach_success_criteria"):
            curriculum_cfg.reach_success_criteria = None
        # Prevent play-time OVF threshold from starting at 2.0 via curriculum schedule.
        if hasattr(curriculum_cfg, "cri_ovf_term_threshold"):
            curriculum_cfg.cri_ovf_term_threshold = None
        if hasattr(curriculum_cfg, "cri_ovf_reward_weight"):
            curriculum_cfg.cri_ovf_reward_weight = None
        if hasattr(curriculum_cfg, "ee_pose_pos_r"):
            pos_r_term = curriculum_cfg.ee_pose_pos_r
            final_pos_r = None
            if pos_r_term is not None:
                final_pos_r = getattr(pos_r_term, "params", {}).get("modify_params", {}).get("final_range")
            curriculum_cfg.ee_pose_pos_r = None
            ranges = getattr(getattr(getattr(env_cfg, "commands", None), "ee_pose", None), "ranges", None)
            if ranges is not None and final_pos_r is not None:
                ranges.pos_r = tuple(final_pos_r)

    terminations_cfg = getattr(env_cfg, "terminations", None)
    if terminations_cfg is not None and hasattr(terminations_cfg, "OVF"):
        if disable_ovf:
            # Timeout geography: do not end the episode on CRI (PLAY/curriculum still force 0.96).
            terminations_cfg.OVF = None
        else:
            ovf = terminations_cfg.OVF
            if ovf is not None and getattr(ovf, "params", None) is not None:
                ovf.params["threshold"] = 0.96

    if freeze_finished_envs and terminations_cfg is not None and hasattr(terminations_cfg, "reach_success"):
        # Success is scored in play_reach_csv; keep the robot pose (no mid-eval reset).
        terminations_cfg.reach_success = None

    rewards_cfg = getattr(env_cfg, "rewards", None)
    if rewards_cfg is not None and hasattr(rewards_cfg, "CRI_OVF"):
        cri_ovf = rewards_cfg.CRI_OVF
        if cri_ovf is not None and getattr(cri_ovf, "params", None) is not None:
            cri_ovf.params.pop("threshold", None)
            cri_ovf.params["limit"] = 0.96

    if disable_resample:
        events_cfg = getattr(env_cfg, "events", None)
        if events_cfg is not None and hasattr(events_cfg, "resample_ee_pose_on_reach"):
            events_cfg.resample_ee_pose_on_reach = None

    if deterministic_sim:
        sim_cfg = getattr(env_cfg, "sim", None)
        physx = getattr(sim_cfg, "physx", None) if sim_cfg is not None else None
        if physx is not None:
            if hasattr(physx, "enable_enhanced_determinism"):
                physx.enable_enhanced_determinism = True
            if hasattr(physx, "enable_external_forces_every_iteration"):
                # Reduces noisy articulation velocities that seed late trajectory divergence.
                physx.enable_external_forces_every_iteration = True
            if hasattr(physx, "min_velocity_iteration_count"):
                physx.min_velocity_iteration_count = max(int(physx.min_velocity_iteration_count), 1)


def freeze_finished_env_controls(
    base_env: ManagerBasedRLEnv,
    actions: torch.Tensor,
    attempt_active: torch.Tensor,
) -> torch.Tensor:
    """Hold finished envs still: zero actions and stall episode length (no timeout reset).

    Do not teleport finished robots to ``default_joint_pos``: mass pose jumps couple into the
    shared CRI TensorRT batch and cause spurious ``fail_ovf`` on still-active envs.
    """
    frozen = ~attempt_active
    if not bool(frozen.any().item()):
        return actions
    actions = actions.clone()
    actions[frozen] = 0.0
    # ``env.step`` increments ``episode_length_buf`` then checks timeout; keep frozen envs at 0
    # beforehand so they never reach ``max_episode_length``.
    base_env.episode_length_buf[frozen] = 0
    return actions


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
        + list(_CMD_POSE_FIELDS)
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
        fieldnames=["env_idx", "episode_id", "ended_at_s", "reached", "outcome", *_CMD_POSE_FIELDS],
    )
    writer.writeheader()
    handle.flush()
    return handle, writer


def reach_percent(reached: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return 100.0 * float(reached) / float(total)


def write_reach_summary_csv(
    csv_dir: str,
    reached: int,
    total: int,
    *,
    seed: int | None = None,
    checkpoint: str | None = None,
    mean_success_latency_s: float | None = None,
) -> str:
    os.makedirs(csv_dir, exist_ok=True)
    path = os.path.join(csv_dir, "reach_summary.csv")
    fieldnames = [
        "seed",
        "checkpoint",
        "total_episodes",
        "reached_episodes",
        "failed_episodes",
        "reach_percent",
        "mean_success_latency_s",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "seed": "" if seed is None else seed,
                "checkpoint": checkpoint or "",
                "total_episodes": total,
                "reached_episodes": reached,
                "failed_episodes": total - reached,
                "reach_percent": round(reach_percent(reached, total), 4),
                "mean_success_latency_s": (
                    "" if mean_success_latency_s is None else round(mean_success_latency_s, 6)
                ),
            }
        )
    return os.path.abspath(path)


def mean_success_latency_s_from_episode_csv(episode_csv_path: str) -> float | None:
    """Mean ``ended_at_s`` over successful attempts in ``episode_reach.csv``."""
    if not os.path.isfile(episode_csv_path):
        return None
    latencies: list[float] = []
    with open(episode_csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if int(row.get("reached", 0)) != 1:
                continue
            latencies.append(float(row["ended_at_s"]))
    if not latencies:
        return None
    return float(sum(latencies) / len(latencies))


def write_multi_seed_summary_csv(csv_dir: str, seed_rows: list[dict[str, Any]]) -> str:
    """Write per-seed rows plus mean/std of reach_percent across seeds."""
    os.makedirs(csv_dir, exist_ok=True)
    path = os.path.join(csv_dir, "multi_seed_summary.csv")
    fieldnames = [
        "seed",
        "checkpoint",
        "total_episodes",
        "reached_episodes",
        "failed_episodes",
        "reach_percent",
        "mean_success_latency_s",
    ]
    rates = [float(row["reach_percent"]) for row in seed_rows]
    n = len(rates)
    mean_rate = sum(rates) / n if n else 0.0
    if n >= 2:
        var = sum((r - mean_rate) ** 2 for r in rates) / (n - 1)
        std_rate = var**0.5
    else:
        std_rate = 0.0

    latency_vals = [
        float(row["mean_success_latency_s"])
        for row in seed_rows
        if row.get("mean_success_latency_s") not in (None, "")
    ]
    mean_lat = sum(latency_vals) / len(latency_vals) if latency_vals else None

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames
            + ["n_seeds", "mean_reach_percent", "std_reach_percent", "mean_success_latency_s_across_seeds"],
        )
        writer.writeheader()
        for row in seed_rows:
            out = {key: row.get(key, "") for key in fieldnames}
            out["n_seeds"] = n
            out["mean_reach_percent"] = round(mean_rate, 4)
            out["std_reach_percent"] = round(std_rate, 4)
            out["mean_success_latency_s_across_seeds"] = (
                "" if mean_lat is None else round(mean_lat, 6)
            )
            writer.writerow(out)
    return os.path.abspath(path)


def append_traj_rows(
    writers: dict[int, csv.DictWriter],
    robot,
    global_step: int,
    sim_time_s: float,
    env_log_mask: torch.Tensor,
    reach_event: torch.Tensor,
    command: torch.Tensor | None = None,
) -> None:
    """Append one CSV row per active env using the pre-reset CRI motion snapshot.

    ``env.step`` may reset terminated envs before the play loop logs. Reading live
    ``joint_pos`` / ``joint_vel`` then pairs post-reset ``qd=0`` with a stale CRI.
    ``get_cri_trajectory_state`` keeps the (q, qd, CRI) from the first CRI eval at
    this physics step (termination/reward), which is the state that must be exported.

    ``command`` should be the pre-step ``ee_pose`` snapshot (base frame, 7D). Live
    command after ``step`` can already be the next target on reset envs.
    """
    joint_names = robot.joint_names
    # Prefer no_grad over inference_mode so sim buffers stay mutable across multi-seed resets.
    with torch.no_grad():
        joint_pos, joint_vel, cri = robot.data.get_cri_trajectory_state()
        joint_pos = joint_pos.detach().cpu().numpy()
        joint_vel = joint_vel.detach().cpu().numpy()
        cri = cri.detach().cpu().numpy()
        reach_np = reach_event.detach().cpu().numpy()
        cmd_np = None if command is None else command.detach().cpu().numpy()

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
        cmd_vec = cmd_np[env_idx] if cmd_np is not None and env_idx < int(cmd_np.shape[0]) else None
        row.update(command_pose_row(cmd_vec))
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
    command_pose: Any | None = None,
) -> None:
    row: dict[str, Any] = {
        "env_idx": env_idx,
        "episode_id": episode_id,
        "ended_at_s": ended_at_s,
        "reached": int(reached),
        "outcome": outcome,
    }
    row.update(command_pose_row(command_pose))
    writer.writerow(row)
    handle.flush()


def resolve_reach_event(
    base: ManagerBasedRLEnv,
    reach_cmd_snapshot: torch.Tensor,
    strict_reach_params: dict[str, Any],
) -> torch.Tensor:
    """Prefer ``reach_success`` termination when active; else evaluate strict criteria.

    One-shot play eval typically disables ``terminations.reach_success`` (freeze finished envs)
    and relies on the criteria path so success does not reset mid-batch.
    """
    if "reach_success" in base.termination_manager.active_terms:
        return base.termination_manager.get_term("reach_success")
    criteria = {key: strict_reach_params[key] for key in _REACH_CRITERIA_CALL_KEYS if key in strict_reach_params}
    return reach_success_criteria(base, command_b=reach_cmd_snapshot, **criteria)
