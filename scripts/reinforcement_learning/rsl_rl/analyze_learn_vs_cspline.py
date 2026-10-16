# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Unpaired secondary-metric comparison: model_188000 reaches vs Case_7 c-splines.

Path index is not the same start/goal as the learning env index. The reported
difference is mean(learn) - mean(spline). A second clock scales every spline
velocity so the mean spline duration equals the mean learning duration.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

Q_COLS = [
    "q_shoulder_pan_joint",
    "q_shoulder_lift_joint",
    "q_elbow_joint",
    "q_wrist_1_joint",
    "q_wrist_2_joint",
    "q_wrist_3_joint",
]
QD_COLS = [name.replace("q_", "qd_", 1) for name in Q_COLS]
A_COLS = [f"a_{i}" for i in range(6)]
POS_COLS = [f"Position_{i}" for i in range(2, 8)]
VEL_COLS = [f"Velocity_{i}" for i in range(2, 8)]
CRI_EDGES = np.linspace(0.0, 2.0, 21)
COARSE_EDGES = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0])


def _stats(values: np.ndarray) -> dict[str, float | int]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "mean": None, "median": None, "p95": None, "std": None, "min": None, "max": None}
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _diff(learn: np.ndarray, spline: np.ndarray, rng: np.random.Generator, n_boot: int = 2000) -> dict[str, float | int]:
    learn = learn[np.isfinite(learn)]
    spline = spline[np.isfinite(spline)]
    mean_diff = float(learn.mean() - spline.mean())
    se = float(np.sqrt(learn.var(ddof=1) / learn.size + spline.var(ddof=1) / spline.size))
    med_diff = float(np.median(learn) - np.median(spline))
    boot_mean = np.empty(n_boot)
    boot_med = np.empty(n_boot)
    for i in range(n_boot):
        a = rng.choice(learn, size=learn.size, replace=True)
        b = rng.choice(spline, size=spline.size, replace=True)
        boot_mean[i] = a.mean() - b.mean()
        boot_med[i] = np.median(a) - np.median(b)
    return {
        "n_learn": int(learn.size),
        "n_spline": int(spline.size),
        "mean_learn": float(learn.mean()),
        "mean_spline": float(spline.mean()),
        "mean_diff_learn_minus_spline": mean_diff,
        "se_mean_diff": se,
        "ci95_low": float(np.percentile(boot_mean, 2.5)),
        "ci95_high": float(np.percentile(boot_mean, 97.5)),
        "median_learn": float(np.median(learn)),
        "median_spline": float(np.median(spline)),
        "median_diff_learn_minus_spline": med_diff,
        "median_diff_ci95_low": float(np.percentile(boot_med, 2.5)),
        "median_diff_ci95_high": float(np.percentile(boot_med, 97.5)),
    }


def _hist_share(counts: np.ndarray, edges: np.ndarray) -> list[float]:
    total = float(counts.sum())
    if total <= 0:
        return [0.0] * (len(edges) - 1)
    shares = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        left = int(np.searchsorted(CRI_EDGES, lo, side="left"))
        right = int(np.searchsorted(CRI_EDGES, hi, side="left"))
        shares.append(100.0 * float(counts[left:right].sum()) / total)
    return shares


def _kinematic(t: np.ndarray, qd: np.ndarray, min_dt: float) -> tuple[float, float]:
    dt = np.diff(t)
    keep = dt >= min_dt
    if keep.sum() < 2:
        return float("nan"), float("nan")
    qdd = (qd[1:] - qd[:-1])[keep] / dt[keep, None]
    dt_k = dt[keep]
    # Jerk uses consecutive kept intervals. If a short tail was dropped, the
    # last two kept samples are still neighbors on the 0.1 s grid.
    jerk_dt = dt_k[1:]
    jerk = np.linalg.norm(qdd[1:] - qdd[:-1], axis=1) / jerk_dt
    accel = np.linalg.norm(qdd, axis=1)
    return float(np.mean(jerk)), float(np.mean(accel))


def _phase_bin(t: np.ndarray) -> np.ndarray:
    span = float(t[-1] - t[0])
    if span <= 0:
        return np.zeros(t.shape[0], dtype=np.int64)
    phase = np.clip((t - t[0]) / span, 0.0, 0.999999)
    return np.minimum((phase * 10).astype(np.int64), 9)


def _one_learn(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    t = np.array([float(row["sim_time_s"]) for row in rows], dtype=np.float64)
    qd = np.array([[float(row[col]) for col in QD_COLS] for row in rows], dtype=np.float64)
    cri = np.array([float(row["max_CRI"]) for row in rows], dtype=np.float64)
    act = np.array([[float(row[col]) for col in A_COLS] for row in rows], dtype=np.float64)
    jerk, accel = _kinematic(t, qd, min_dt=1e-6)
    if act.shape[0] >= 2:
        smooth = float(np.mean(np.linalg.norm(np.diff(act, axis=0), axis=1)))
    else:
        smooth = float("nan")
    hist = np.histogram(cri, bins=CRI_EDGES)[0].astype(np.int64)
    phase = _phase_bin(t)
    phase_sum = np.zeros(10, dtype=np.float64)
    phase_n = np.zeros(10, dtype=np.int64)
    for bin_id in range(10):
        mask = phase == bin_id
        phase_sum[bin_id] = float(cri[mask].sum()) if mask.any() else 0.0
        phase_n[bin_id] = int(mask.sum())
    summary = np.array(
        [jerk, accel, smooth, float(np.mean(cri)), float(np.max(cri)), float(t[-1]), float(t.shape[0])],
        dtype=np.float64,
    )
    return summary, hist, np.concatenate([phase_sum, phase_n.astype(np.float64)])


def _one_spline(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    t = np.array([float(row["Time"]) for row in rows], dtype=np.float64)
    qd = np.array([[float(row[col]) for col in VEL_COLS] for row in rows], dtype=np.float64)
    cri = np.array([float(row["MaxCRI"]) for row in rows], dtype=np.float64)
    jerk, accel = _kinematic(t, qd, min_dt=0.05)
    hist = np.histogram(cri, bins=CRI_EDGES)[0].astype(np.int64)
    phase = _phase_bin(t)
    phase_sum = np.zeros(10, dtype=np.float64)
    phase_n = np.zeros(10, dtype=np.int64)
    for bin_id in range(10):
        mask = phase == bin_id
        phase_sum[bin_id] = float(cri[mask].sum()) if mask.any() else 0.0
        phase_n[bin_id] = int(mask.sum())
    summary = np.array(
        [jerk, accel, float(np.mean(cri)), float(np.max(cri)), float(t[-1]), float(t.shape[0])],
        dtype=np.float64,
    )
    return summary, hist, np.concatenate([phase_sum, phase_n.astype(np.float64)])


def _pool_map(fn, paths: list[str], workers: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    summaries = []
    hists = []
    phases = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for summary, hist, phase in pool.map(fn, paths, chunksize=32):
            summaries.append(summary)
            hists.append(hist)
            phases.append(phase)
    return np.stack(summaries), np.stack(hists), np.stack(phases)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--learn_dir",
        default="logs/rsl_rl/reach_ur10/2026-10-14_02-20-35/eval_188000_pure",
    )
    parser.add_argument(
        "--case_root",
        default="/home/safetics/Downloads/Case_7/KHU_Example_Output",
    )
    parser.add_argument(
        "--out_dir",
        default="logs/rsl_rl/reach_ur10/2026-10-14_02-20-35/eval_188000_vs_case7_spline",
    )
    parser.add_argument("--num", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    learn_paths = [os.path.join(args.learn_dir, f"env_{i}_traj.csv") for i in range(args.num)]
    spline_paths = [
        os.path.join(args.case_root, f"Path_{i}", "KHU_Recommend", "KHU_Collision_Risk_Analyze_Result.csv")
        for i in range(args.num)
    ]
    learn, learn_hist, learn_phase = _pool_map(_one_learn, learn_paths, args.workers)
    spline, spline_hist, spline_phase = _pool_map(_one_spline, spline_paths, args.workers)

    reach = list(csv.DictReader(open(os.path.join(args.learn_dir, "episode_reach.csv"), newline="", encoding="utf-8")))
    clean = np.zeros(args.num, dtype=bool)
    for row in reach:
        env_idx = int(row["env_idx"])
        clean[env_idx] = row["reached"] == "1" and float(row["max_cri"]) < 0.96
    spline_ok = spline[:, 3] < 0.96

    # Columns: learn jerk, accel, action_smooth, cri_mean, cri_max, T, n
    # spline jerk, accel, cri_mean, cri_max, T, n
    vel_scale = float(spline[:, 4].mean() / learn[:, 5].mean())
    spline_jerk_same = spline[:, 0] * vel_scale**3
    spline_accel_same = spline[:, 1] * vel_scale**2

    rng = np.random.default_rng(0)
    groups = {
        "all": (np.ones(args.num, dtype=bool), np.ones(args.num, dtype=bool)),
        "clean_vs_under096": (clean, spline_ok),
    }
    metrics = {
        "path_jerk": (learn[:, 0], spline[:, 0], spline_jerk_same),
        "kinematic_smoothness": (learn[:, 1], spline[:, 1], spline_accel_same),
        "step_cri_mean": (learn[:, 3], spline[:, 2], None),
        "episode_max_cri": (learn[:, 4], spline[:, 3], None),
        "duration_s": (learn[:, 5], spline[:, 4], spline[:, 4] / vel_scale),
    }
    comparison = {}
    for group_name, (learn_mask, spline_mask) in groups.items():
        comparison[group_name] = {}
        for metric_name, (learn_v, spline_v, spline_same) in metrics.items():
            comparison[group_name][metric_name] = {
                "logged": _diff(learn_v[learn_mask], spline_v[spline_mask], rng),
            }
            if spline_same is not None:
                comparison[group_name][metric_name]["same_mean_duration"] = _diff(
                    learn_v[learn_mask], spline_same[spline_mask], rng
                )

    def _phase_mean(phase: np.ndarray, mask: np.ndarray) -> list[float]:
        summed = phase[mask, :10].sum(axis=0)
        counted = phase[mask, 10:].sum(axis=0)
        return [float(summed[i] / counted[i]) if counted[i] else None for i in range(10)]

    payload = {
        "pairing": "unpaired population. Case_7 Path_i is not the learning env_i start/goal.",
        "vel_scale_to_match_mean_duration": vel_scale,
        "definitions": {
            "path_jerk": "mean ||qdd_{k+1}-qdd_k||_2 / dt, qdd=(qd_{k+1}-qd_k)/dt",
            "kinematic_smoothness": "mean ||qdd||_2 in rad/s^2. Policy-action smoothness is learning-only.",
            "same_mean_duration": "spline velocity multiplied by mean(T_spline)/mean(T_learn); jerk scales by that cubed",
            "cri": "logged MaxCRI / max_CRI. Same-duration CRI is not this file.",
        },
        "duration": {
            "learn": _stats(learn[:, 5]),
            "spline": _stats(spline[:, 4]),
        },
        "action_smoothness_learn_only": {
            "all": _stats(learn[:, 2]),
            "clean": _stats(learn[clean, 2]),
        },
        "comparison": comparison,
        "cri_hist_share_pct": {
            "bins": ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0", "1.0-2.0"],
            "learn_all": _hist_share(learn_hist.sum(axis=0), COARSE_EDGES),
            "learn_clean": _hist_share(learn_hist[clean].sum(axis=0), COARSE_EDGES),
            "spline_all": _hist_share(spline_hist.sum(axis=0), COARSE_EDGES),
            "spline_under096": _hist_share(spline_hist[spline_ok].sum(axis=0), COARSE_EDGES),
        },
        "cri_hist_counts": {
            "edges": CRI_EDGES.tolist(),
            "learn_all": learn_hist.sum(axis=0).astype(int).tolist(),
            "learn_clean": learn_hist[clean].sum(axis=0).astype(int).tolist(),
            "spline_all": spline_hist.sum(axis=0).astype(int).tolist(),
            "spline_under096": spline_hist[spline_ok].sum(axis=0).astype(int).tolist(),
        },
        "phase_mean_cri": {
            "learn_all": _phase_mean(learn_phase, np.ones(args.num, dtype=bool)),
            "learn_clean": _phase_mean(learn_phase, clean),
            "spline_all": _phase_mean(spline_phase, np.ones(args.num, dtype=bool)),
            "spline_under096": _phase_mean(spline_phase, spline_ok),
        },
        "counts": {
            "learn_clean": int(clean.sum()),
            "spline_under096": int(spline_ok.sum()),
        },
    }
    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "kinematic_compare.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    np.savez(
        os.path.join(args.out_dir, "per_traj.npz"),
        learn=learn,
        spline=spline,
        clean=clean,
        spline_under096=spline_ok,
        vel_scale=np.array([vel_scale]),
    )
    print(f"[INFO] vel_scale {vel_scale:.6f}", flush=True)
    print(f"[INFO] duration learn mean {learn[:, 5].mean():.3f} spline {spline[:, 4].mean():.3f}", flush=True)
    logged = comparison["all"]["path_jerk"]["logged"]
    print(
        f"[INFO] jerk mean learn {logged['mean_learn']:.3f} spline {logged['mean_spline']:.3f} "
        f"diff {logged['mean_diff_learn_minus_spline']:.3f}",
        flush=True,
    )
    print(f"[INFO] Wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
