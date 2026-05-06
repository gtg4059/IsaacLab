# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--termination_stats_interval",
    type=int,
    default=500,
    help="Print episode termination stats (time_out / reach / OVF) every this many env steps. 0 disables periodic prints; final summary still prints on exit.",
)
parser.add_argument(
    "--reach_time_boxplot_out_dir",
    type=str,
    default=None,
    help="Directory for reach trajectory time boxplot PNGs. Default: <checkpoint_dir>/play_reach_time_boxplots",
)
parser.add_argument(
    "--reach_time_boxplot_max_stored",
    type=int,
    default=1_000_000,
    help="Max reach-time samples kept for the boxplot (FIFO ring). 0 = unlimited (can use large RAM on long runs).",
)
parser.add_argument(
    "--joint_traj_csv",
    action="store_true",
    default=False,
    help="Enable: per-env CSV (global_step, sim_time_s, q_*, qd_*, CRI_*). Default: stop after the first "
    "``reach_success_criteria`` hit in any env (thresholds from env cfg). See --joint_traj_csv_always.",
)
parser.add_argument(
    "--joint_traj_csv_always",
    action="store_true",
    default=False,
    help="With --joint_traj_csv: keep recording for the full play session. If unset, logging stops after the "
    "first reach (same test as reward ``reach_success_bonus`` / event ``resample_ee_pose_on_reach`` params).",
)
parser.add_argument(
    "--joint_traj_csv_dir",
    type=str,
    default=None,
    help="If --joint_traj_csv: output directory for q/qd CSVs (default: <checkpoint_dir>/joint_trajectory).",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import gymnasium as gym
import os
import time
from collections import deque
from typing import Any, IO

import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, ManagerBasedRLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.reach.mdp.rewards import reach_success_criteria
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# PLACEHOLDER: Extension template (do not remove this comment)


def _joint_csv_reach_params_from_env_cfg(base: ManagerBasedRLEnv) -> dict[str, Any] | None:
    """Build kwargs for :func:`reach_success_criteria` from task cfg (reward term, else resample event)."""
    rewards_cfg = getattr(base.cfg, "rewards", None)
    if rewards_cfg is not None:
        term = getattr(rewards_cfg, "reach_success_bonus", None)
        if term is not None and getattr(term, "params", None):
            return dict(term.params)
    events_cfg = getattr(base.cfg, "events", None)
    if events_cfg is not None:
        term = getattr(events_cfg, "resample_ee_pose_on_reach", None)
        if term is not None and getattr(term, "params", None):
            return dict(term.params)
    return None


def _open_joint_traj_csv_writers_rl(
    csv_dir: str,
    num_envs: int,
    joint_names: list[str],
) -> tuple[dict[int, IO[str]], dict[int, csv.DictWriter]]:
    """One CSV per env: global_step, sim_time_s, q_*, qd_*, CRI_* (CRI from ``robot.data.CRI``)."""
    os.makedirs(csv_dir, exist_ok=True)
    fieldnames = (
        ["global_step", "sim_time_s"]
        + [f"q_{j}" for j in joint_names]
        + [f"qd_{j}" for j in joint_names]
        + [f"CRI_{j}" for j in joint_names]
    )
    files: dict[int, IO[str]] = {}
    writers: dict[int, csv.DictWriter] = {}
    for e in range(num_envs):
        path = os.path.join(csv_dir, f"env_{e}_joint_q_qd.csv")
        f = open(path, "w", newline="", encoding="utf-8")
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        f.flush()
        files[e] = f
        writers[e] = w
    return files, writers


def _append_joint_traj_rows_rl(
    writers: dict[int, csv.DictWriter],
    robot,
    global_step: int,
    sim_time_s: float,
    env_log_mask: torch.Tensor,
) -> None:
    """Log joint_pos, joint_vel, and CRI from ``robot`` (post-``env.step``); skip envs with mask False."""
    joint_names = robot.joint_names
    with torch.inference_mode():
        jp = robot.data.joint_pos.detach().cpu().numpy()
        jv = robot.data.joint_vel.detach().cpu().numpy()
        cri = robot.data.CRI.detach().cpu().numpy()
    n_j = len(joint_names)
    n_cri = int(cri.shape[1]) if cri.ndim == 2 else 0
    if n_cri != n_j:
        # CRI is defined per articulation joint in the same dof order as joint_pos for this asset.
        n_use = min(n_j, n_cri) if n_cri > 0 else 0
    else:
        n_use = n_j
    mask = env_log_mask.detach().bool().cpu()
    for e, w in writers.items():
        if e >= int(mask.shape[0]) or not bool(mask[e].item()):
            continue
        row: dict[str, Any] = {"global_step": global_step, "sim_time_s": sim_time_s}
        for k, name in enumerate(joint_names):
            row[f"q_{name}"] = float(jp[e, k])
            row[f"qd_{name}"] = float(jv[e, k])
            if n_use > 0 and k < n_use:
                row[f"CRI_{name}"] = float(cri[e, k])
            else:
                row[f"CRI_{name}"] = float("nan")
        w.writerow(row)


def _count_episode_terminations_legacy(
    reset_time_outs: torch.Tensor | None, reset_terminated: torch.Tensor | None
) -> tuple[int, int]:
    """Fallback: timeout vs other (no per-term split). Timeout wins when both apply."""
    if reset_time_outs is None or reset_terminated is None:
        return 0, 0
    trunc = reset_time_outs.bool()
    term = reset_terminated.bool()
    n_timeout = int(trunc.sum().item())
    n_non_timeout = int((term & ~trunc).sum().item())
    return n_timeout, n_non_timeout


def _reach_termination_mask_exclusive_timeout(base: ManagerBasedRLEnv) -> torch.Tensor | None:
    """Mask of envs whose episode ended this step as **reach**, excluding simultaneous **time_out**.

    Matches the ``reach`` bucket in :func:`_count_episode_terminations_reach_env` (``reach & ~timeout`` on ``done``).
    Returns ``None`` if the reach MDP termination terms are unavailable.
    """
    tm = getattr(base, "termination_manager", None)
    if tm is None:
        return None
    required = ("time_out", "reach", "OVF")
    if not all(name in tm.active_terms for name in required):
        return None
    done = base.reset_buf.bool()
    timeout = tm.get_term("time_out") & done
    reach = tm.get_term("reach") & done
    return reach & ~timeout


def _count_episode_terminations_reach_env(base: ManagerBasedRLEnv) -> tuple[int, int, int] | None:
    """Count ended episodes this step as time_out vs reach vs OVF (mutually exclusive).

    Uses :meth:`TerminationManager.get_term` for ``time_out``, ``reach``, and ``OVF``.
    If several fire on the same env, priority is **time_out > reach > OVF** so shares sum to 100%.
    """
    tm = getattr(base, "termination_manager", None)
    if tm is None:
        return None
    required = ("time_out", "reach", "OVF")
    if not all(name in tm.active_terms for name in required):
        return None
    done = base.reset_buf.bool()
    if not done.any():
        return 0, 0, 0
    timeout = tm.get_term("time_out") & done
    reach = tm.get_term("reach") & done
    ovf = tm.get_term("OVF") & done
    n_time = int(timeout.sum().item())
    reach_excl = reach & ~timeout
    n_reach = int(reach_excl.sum().item())
    n_ovf = int((ovf & ~timeout & ~reach).sum().item())
    return n_time, n_reach, n_ovf


def _save_reach_trajectory_boxplot(
    times_s: list[float] | deque[float], out_png: str, title: str, ylabel: str = "Time to reach (s)"
) -> bool:
    """Save a boxplot of per-episode reach trajectory times. Uses non-interactive Agg backend."""
    data = list(times_s)
    if len(data) < 1:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"[WARN] matplotlib not available; skipping boxplot ({e}).")
        return False

    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.boxplot(data, vert=True, showfliers=True)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def _print_reach_mean_and_boxplot(
    label: str,
    sum_seconds: float,
    n_reach_episodes: int,
    times_for_plot: list[float] | deque[float],
    out_png: str,
    plot_title: str,
    max_stored: int,
) -> None:
    """Print cumulative mean reach time and write a boxplot PNG from stored samples."""
    if n_reach_episodes <= 0:
        print(f"[INFO] {label}: reach-terminated trajectory stats N/A (no reach ends yet).")
        return
    mean_s = sum_seconds / float(n_reach_episodes)
    n_plot = len(times_for_plot)
    cap_note = ""
    if max_stored > 0 and n_plot < n_reach_episodes:
        cap_note = f" (boxplot uses last {n_plot} samples; cumulative n={n_reach_episodes})"
    elif max_stored == 0:
        cap_note = f" (boxplot uses all n={n_reach_episodes} samples)"
    else:
        cap_note = f" (boxplot uses n={n_plot} samples)"
    print(
        f"[INFO] {label}: mean time to reach = {mean_s:.6f} s (cumulative episodes n={n_reach_episodes}){cap_note}"
    )
    if _save_reach_trajectory_boxplot(times_for_plot, out_png, plot_title):
        print(f"[INFO] {label}: saved boxplot -> {os.path.abspath(out_png)}")
    else:
        print(f"[INFO] {label}: boxplot not written (no samples or matplotlib error).")


def _print_three_way_termination_stats(label: str, n_time: int, n_reach: int, n_ovf: int) -> None:
    """Print counts and percentages for mutually exclusive time_out / reach / OVF buckets."""
    total = n_time + n_reach + n_ovf
    if total <= 0:
        print(f"[INFO] {label}: no episodes ended yet.")
        return
    p_time = 100.0 * n_time / total
    p_reach = 100.0 * n_reach / total
    p_ovf = 100.0 * n_ovf / total
    print(
        f"[INFO] {label} ({total} ended episodes): "
        f"time_out={n_time} ({p_time:.1f}%), "
        f"reach={n_reach} ({p_reach:.1f}%), "
        f"OVF={n_ovf} ({p_ovf:.1f}%)"
    )


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    reach_boxplot_dir = args_cli.reach_time_boxplot_out_dir or os.path.join(log_dir, "play_reach_time_boxplots")
    joint_traj_csv_dir = args_cli.joint_traj_csv_dir or os.path.join(log_dir, "joint_trajectory")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    dt = env.unwrapped.step_dt

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    total_time_out = 0
    total_reach = 0
    total_ovf = 0
    total_timeout_legacy = 0
    total_non_timeout_legacy = 0
    use_three_way_stats: bool | None = None
    play_step = 0
    steps_since_reset: torch.Tensor | None = None
    reach_traj_time_sum_s = 0.0
    reach_traj_time_n = 0
    reach_plot_max = args_cli.reach_time_boxplot_max_stored
    reach_traj_times_plot: list[float] | deque[float] = (
        [] if reach_plot_max == 0 else deque(maxlen=reach_plot_max)
    )
    joint_csv_files: dict[int, IO[str]] | None = None
    joint_csv_writers: dict[int, csv.DictWriter] | None = None
    joint_traj_log_active: torch.Tensor | None = None
    joint_reach_params: dict[str, Any] | None = None
    joint_csv_first_reach_notice_printed = False
    print(
        "[INFO] Tracking episode ends: time_out vs reach vs OVF (mutually exclusive; "
        "priority if multiple: time_out > reach > OVF). Counters accumulate until you close the app."
    )
    print(
        "[INFO] Also recording per-episode time-to-reach (seconds) for **reach** ends (exclusive of time_out); "
        f"mean is cumulative, boxplot uses stored samples (max_stored={reach_plot_max or 'unlimited'}). "
        f"Boxplot directory: {os.path.abspath(reach_boxplot_dir)}"
    )
    if args_cli.joint_traj_csv:
        print(
            f"[INFO] Joint q/qd/CRI CSV: enabled → {os.path.abspath(joint_traj_csv_dir)}/env_*_joint_q_qd.csv "
            "(default: stop after first ``reach_success_criteria`` in any env; use --joint_traj_csv_always for full run)"
        )
    else:
        print("[INFO] Joint q/qd/CRI CSV: disabled (use --joint_traj_csv to record per-env trajectories)")
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _extras = env.step(actions)
            base = env.unwrapped
            reach_termination_excl: torch.Tensor | None = None
            if use_three_way_stats is None and isinstance(base, ManagerBasedRLEnv):
                sample = _count_episode_terminations_reach_env(base)
                use_three_way_stats = sample is not None
                if not use_three_way_stats:
                    print(
                        "[INFO] Termination terms time_out/reach/OVF not all present; "
                        "falling back to time_out vs other only."
                    )
            if use_three_way_stats:
                if steps_since_reset is None:
                    steps_since_reset = torch.zeros(base.num_envs, device=base.device, dtype=torch.long)
                steps_since_reset += 1
                triple = _count_episode_terminations_reach_env(base)
                assert triple is not None
                total_time_out += triple[0]
                total_reach += triple[1]
                total_ovf += triple[2]
                reach_mask = _reach_termination_mask_exclusive_timeout(base)
                assert reach_mask is not None
                reach_termination_excl = reach_mask
                if reach_mask.any():
                    times_s = steps_since_reset[reach_mask].to(dtype=torch.float32) * dt
                    reach_traj_time_sum_s += float(times_s.sum().item())
                    reach_traj_time_n += int(reach_mask.sum().item())
                    vals = [float(v) for v in times_s.detach().cpu().tolist()]
                    reach_traj_times_plot.extend(vals)
                steps_since_reset[base.reset_buf.bool()] = 0
            else:
                t0, t1 = _count_episode_terminations_legacy(
                    getattr(base, "reset_time_outs", None),
                    getattr(base, "reset_terminated", None),
                )
                total_timeout_legacy += t0
                total_non_timeout_legacy += t1
            if args_cli.joint_traj_csv and joint_csv_writers is None and isinstance(base, ManagerBasedRLEnv):
                robot = base.scene["robot"]
                joint_csv_files, joint_csv_writers = _open_joint_traj_csv_writers_rl(
                    joint_traj_csv_dir, base.num_envs, list(robot.joint_names)
                )
                joint_traj_log_active = torch.ones(base.num_envs, device=base.device, dtype=torch.bool)
                joint_reach_params = _joint_csv_reach_params_from_env_cfg(base)
                print(f"[INFO] Writing joint q/qd/CRI to {os.path.abspath(joint_traj_csv_dir)}/env_*_joint_q_qd.csv")
                if not args_cli.joint_traj_csv_always:
                    if joint_reach_params is None:
                        print(
                            "[WARN] joint_traj_csv: no ``reach_success_bonus`` / ``resample_ee_pose_on_reach`` "
                            "params in env cfg — cannot detect first reach; logging all steps."
                        )
                    else:
                        print(
                            "[INFO] joint_traj_csv: will stop **all** env CSVs after the first step where "
                            "``reach_success_criteria`` is true for any env (params from env cfg)."
                        )
                else:
                    print("[INFO] joint_traj_csv: --joint_traj_csv_always — logging all control steps for all envs.")
            play_step += 1
            if (
                joint_csv_writers is not None
                and joint_csv_files is not None
                and joint_traj_log_active is not None
                and isinstance(base, ManagerBasedRLEnv)
            ):
                # Append using mask before this step’s reach/unmark update (includes reach-termination row).
                _append_joint_traj_rows_rl(
                    joint_csv_writers,
                    base.scene["robot"],
                    play_step,
                    float(play_step) * dt,
                    joint_traj_log_active,
                )
                for f in joint_csv_files.values():
                    f.flush()
            if args_cli.joint_traj_csv and joint_traj_log_active is not None and isinstance(base, ManagerBasedRLEnv):
                if (
                    not args_cli.joint_traj_csv_always
                    and joint_reach_params is not None
                    and bool(joint_traj_log_active.any().item())
                ):
                    crit = reach_success_criteria(base, **joint_reach_params)
                    if bool(crit.any().item()):
                        joint_traj_log_active[:] = False
                        if not joint_csv_first_reach_notice_printed:
                            print(
                                "[INFO] joint_traj_csv: first ``reach_success_criteria`` satisfied — "
                                "no further rows will be written (close files on exit as usual)."
                            )
                            joint_csv_first_reach_notice_printed = True
                elif args_cli.joint_traj_csv_always:
                    rmask = reach_termination_excl
                    if rmask is not None and use_three_way_stats:
                        reset_ = base.reset_buf.bool()
                        joint_traj_log_active[reset_ & ~rmask] = True
                        joint_traj_log_active[rmask] = False
            interval = args_cli.termination_stats_interval
            if interval > 0 and play_step % interval == 0:
                if use_three_way_stats:
                    _print_three_way_termination_stats(
                        "Termination stats (cumulative)", total_time_out, total_reach, total_ovf
                    )
                    _print_reach_mean_and_boxplot(
                        "Reach trajectory time (cumulative)",
                        reach_traj_time_sum_s,
                        reach_traj_time_n,
                        reach_traj_times_plot,
                        os.path.join(reach_boxplot_dir, "reach_time_boxplot_latest.png"),
                        plot_title=f"Reach trajectory time (s) — env step {play_step}",
                        max_stored=reach_plot_max,
                    )
                else:
                    tot = total_timeout_legacy + total_non_timeout_legacy
                    if tot > 0:
                        print(
                            f"[INFO] Termination stats (cumulative over {tot} ended episodes): "
                            f"time_out={total_timeout_legacy} ({100.0 * total_timeout_legacy / tot:.1f}%), "
                            f"other={total_non_timeout_legacy} ({100.0 * total_non_timeout_legacy / tot:.1f}%)"
                        )
                    else:
                        print("[INFO] Termination stats (cumulative): no episodes ended yet.")
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if use_three_way_stats:
        _print_three_way_termination_stats(
            "Final termination breakdown (cumulative)", total_time_out, total_reach, total_ovf
        )
        _print_reach_mean_and_boxplot(
            "Final reach trajectory time (cumulative)",
            reach_traj_time_sum_s,
            reach_traj_time_n,
            reach_traj_times_plot,
            os.path.join(reach_boxplot_dir, "reach_time_boxplot_final.png"),
            plot_title="Reach trajectory time (s) — final cumulative",
            max_stored=reach_plot_max,
        )
    else:
        tot = total_timeout_legacy + total_non_timeout_legacy
        if tot > 0:
            print(
                f"[INFO] Final termination breakdown over {tot} ended episodes: "
                f"time_out={total_timeout_legacy} ({100.0 * total_timeout_legacy / tot:.1f}%), "
                f"other={total_non_timeout_legacy} ({100.0 * total_non_timeout_legacy / tot:.1f}%)"
            )
        else:
            print("[INFO] No episode ended during this play session (no termination stats).")

    if joint_csv_files is not None:
        for f in joint_csv_files.values():
            f.close()
        print(f"[INFO] Closed joint trajectory CSV file handles under {os.path.abspath(joint_traj_csv_dir)}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
