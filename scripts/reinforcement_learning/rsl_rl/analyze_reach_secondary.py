#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Post-process play_reach_csv outputs for Concern #10 secondary metrics.

Metrics (see Response-to-Reviewer_plan_v2.md §4.2):
  - path jerk: episode mean of ||Δqdd_t||_2 with qdd_t = (qd_{t+1}-qd_t)/dt
  - smoothness: episode mean of ||Δa_t||_2 (policy actions logged in traj CSV)
  - energy proxy: Σ_t ||qdd_t||_2^2 * dt  (integrated squared joint acceleration; not electrical)
  - CRI: per-episode peak max_CRI (median / p95 / max across pool)

Reads ``episode_reach.csv`` and ``env_<idx>_traj.csv`` under one export directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    n = len(sorted_vals)
    i = min(n - 1, max(0, int(p * (n - 1))))
    return sorted_vals[i]


def _pool_stats(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {}
    s = sorted(vals)
    return {
        "n": len(vals),
        "mean": statistics.mean(s),
        "median": statistics.median(s),
        "p95": _percentile(s, 0.95),
        "max": s[-1],
        "min": s[0],
    }


def _read_traj_metrics(traj_path: Path, dt: float) -> dict[str, float] | None:
    with traj_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 3:
        return None

    qd_cols = [c for c in rows[0].keys() if c.startswith("qd_")]
    a_cols = sorted(c for c in rows[0].keys() if c.startswith("a_"))
    if not qd_cols:
        return None

    qd = [[float(r[c]) for c in qd_cols] for r in rows]
    cri_peaks = [float(r["max_CRI"]) for r in rows if r.get("max_CRI", "") != ""]
    step_cri = cri_peaks  # max_CRI per step

    qdd_norms: list[float] = []
    jerk_norms: list[float] = []
    for t in range(len(qd) - 1):
        qdd = [(qd[t + 1][j] - qd[t][j]) / dt for j in range(len(qd_cols))]
        qdd_norms.append(math.sqrt(sum(x * x for x in qdd)))
    for t in range(len(qdd_norms) - 1):
        delta = qdd_norms[t + 1] - qdd_norms[t]
        jerk_norms.append(abs(delta) / dt)

    act_smooth: list[float] = []
    if a_cols and len(a_cols) > 0:
        acts = [[float(r[c]) for c in a_cols] for r in rows]
        for t in range(len(acts) - 1):
            da = [acts[t + 1][j] - acts[t][j] for j in range(len(a_cols))]
            act_smooth.append(math.sqrt(sum(x * x for x in da)))

    energy = sum(n * n for n in qdd_norms) * dt
    return {
        "path_jerk_mean": statistics.mean(jerk_norms) if jerk_norms else float("nan"),
        "smoothness_mean_da": statistics.mean(act_smooth) if act_smooth else float("nan"),
        "energy_proxy_qdd": energy,
        "peak_cri": max(cri_peaks) if cri_peaks else float("nan"),
        "median_step_cri": statistics.median(step_cri) if step_cri else float("nan"),
        "n_steps": len(rows),
    }


def _gate_fail_reason(last_pos: float, last_ang: float, last_lv: float, last_av: float) -> str:
    parts: list[str] = []
    if last_pos > 0.02:
        parts.append(f"pos>{last_pos*100:.2f}cm")
    if last_ang > 0.1:
        parts.append(f"ang>{last_ang:.3f}rad")
    twist = math.hypot(last_lv, last_av)
    if twist > 0.02:
        parts.append(f"twist>{twist:.4f}")
    return ", ".join(parts) if parts else "gate_ok"


def analyze(export_dir: Path, dt: float) -> dict[str, Any]:
    ep_path = export_dir / "episode_reach.csv"
    if not ep_path.is_file():
        raise FileNotFoundError(ep_path)

    episodes = list(csv.DictReader(ep_path.open(newline="", encoding="utf-8")))
    traj_dir = export_dir

    per_ep: list[dict[str, Any]] = []
    for ep in episodes:
        env_idx = int(ep["env_idx"])
        traj = traj_dir / f"env_{env_idx}_traj.csv"
        traj_m = _read_traj_metrics(traj, dt) if traj.is_file() else None
        row = {
            "env_idx": env_idx,
            "reached": int(ep["reached"]),
            "outcome": ep["outcome"],
            "ended_at_s": float(ep["ended_at_s"]),
            "min_pos_err": float(ep["min_pos_err"]),
            "min_ang_err": float(ep["min_ang_err"]),
            "last_pos_err": float(ep["last_pos_err"]),
            "last_ang_err": float(ep["last_ang_err"]),
            "max_cri": float(ep["max_cri"]),
        }
        if traj_m:
            row.update(traj_m)
        if int(ep["reached"]) == 0:
            row["fail_gates"] = _gate_fail_reason(
                row["last_pos_err"],
                row["last_ang_err"],
                float(ep["last_lin_vel"]),
                float(ep["last_ang_vel"]),
            )
        per_ep.append(row)

    succ = [r for r in per_ep if r["reached"] == 1]
    fail = [r for r in per_ep if r["reached"] == 0]
    all_traj = [r for r in per_ep if "path_jerk_mean" in r and not math.isnan(r["path_jerk_mean"])]

    def collect(key: str, subset: list[dict]) -> dict[str, float]:
        vals = [float(r[key]) for r in subset if key in r and not math.isnan(float(r[key]))]
        return _pool_stats(vals)

    report: dict[str, Any] = {
        "export_dir": str(export_dir.resolve()),
        "n_episodes": len(episodes),
        "n_success": len(succ),
        "n_fail": len(fail),
        "reach_percent": 100.0 * len(succ) / len(episodes) if episodes else 0.0,
        "dt_s": dt,
        "metrics_all_with_traj": {
            "path_jerk_mean": collect("path_jerk_mean", all_traj),
            "smoothness_mean_da": collect("smoothness_mean_da", all_traj),
            "energy_proxy_qdd": collect("energy_proxy_qdd", all_traj),
            "peak_cri": collect("peak_cri", all_traj),
            "median_step_cri": collect("median_step_cri", all_traj),
        },
        "metrics_success_only": {
            "path_jerk_mean": collect("path_jerk_mean", [r for r in succ if "path_jerk_mean" in r]),
            "smoothness_mean_da": collect("smoothness_mean_da", [r for r in succ if "smoothness_mean_da" in r]),
            "energy_proxy_qdd": collect("energy_proxy_qdd", [r for r in succ if "energy_proxy_qdd" in r]),
            "peak_cri": collect("peak_cri", [r for r in succ if "peak_cri" in r]),
        },
        "failures": sorted(
            [
                {
                    "env_idx": r["env_idx"],
                    "outcome": r["outcome"],
                    "ended_at_s": r["ended_at_s"],
                    "min_pos_cm": round(r["min_pos_err"] * 100, 3),
                    "min_ang_rad": round(r["min_ang_err"], 4),
                    "last_pos_cm": round(r["last_pos_err"] * 100, 3),
                    "last_ang_rad": round(r["last_ang_err"], 4),
                    "max_cri": round(r["max_cri"], 4),
                    "fail_gates": r.get("fail_gates", ""),
                }
                for r in fail
            ],
            key=lambda x: (x["min_pos_cm"] + x["min_ang_rad"] * 10),
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Secondary reach metrics from play_reach_csv export.")
    parser.add_argument("export_dir", type=Path, help="Directory with episode_reach.csv and env_*_traj.csv")
    parser.add_argument(
        "--dt",
        type=float,
        default=4.0 / 60.0,
        help="Control step dt in seconds (default: decimation=4, sim.dt=1/60).",
    )
    parser.add_argument("--json_out", type=Path, default=None, help="Write JSON report path.")
    args = parser.parse_args()

    report = analyze(args.export_dir, args.dt)
    text = json.dumps(report, indent=2)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
        print(f"[INFO] Wrote {args.json_out}")


if __name__ == "__main__":
    main()
