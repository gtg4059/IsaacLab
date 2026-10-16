#!/usr/bin/env python3
"""Aggregate eval_28000_sensitivity Play CSVs for tab:sensitivity."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

CRI_TH = 0.96

LEVELS = [
    ("noise_0_clean", "0× (clean PLAY)", 0.0, 0.0),
    ("noise_2mm", "1×", 0.002, 0.001),
    ("noise_4mm", "2×", 0.004, 0.002),
    ("noise_10mm", "5×", 0.010, 0.005),
    ("noise_20mm", "10×", 0.020, 0.010),
]


def load_episodes(level_dir: Path) -> list[dict]:
    rows: list[dict] = []
    if not level_dir.is_dir():
        return rows
    # flat or seed_*/episode_reach.csv
    flat = level_dir / "episode_reach.csv"
    if flat.is_file():
        rows.extend(csv.DictReader(flat.open(newline="", encoding="utf-8")))
    for ep in sorted(level_dir.glob("seed_*/episode_reach.csv")):
        seed = int(ep.parent.name.split("_")[1])
        for r in csv.DictReader(ep.open(newline="", encoding="utf-8")):
            rows.append({**r, "pool_seed": seed})
    return rows


def summarize(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    n = len(rows)
    reached = sum(int(r["reached"]) for r in rows)
    safe = sum(int(r["reached"]) and float(r["max_cri"]) < CRI_TH for r in rows)
    peak_over = sum(float(r["max_cri"]) >= CRI_TH for r in rows)
    ovf = sum(r["outcome"] == "fail_ovf" for r in rows)
    timeout = sum(r["outcome"] == "fail_timeout" for r in rows)
    by_seed: dict[int, list[dict]] = {}
    for r in rows:
        by_seed.setdefault(int(r.get("pool_seed", -1)), []).append(r)
    seed_rates = []
    for s, sub in by_seed.items():
        if s < 0:
            continue
        seed_rates.append(100.0 * sum(int(x["reached"]) for x in sub) / len(sub))
    return {
        "n_episodes": n,
        "reach_percent": round(100.0 * reached / n, 4),
        "safe_reach_percent": round(100.0 * safe / n, 4),
        "peak_cri_ge_th_episode_percent": round(100.0 * peak_over / n, 4),
        "fail_ovf": ovf,
        "fail_timeout": timeout,
        "n_eval_seeds": len(seed_rates),
        "reach_percent_mean_across_seeds": round(statistics.mean(seed_rates), 4) if seed_rates else None,
        "reach_percent_std_across_seeds": round(statistics.pstdev(seed_rates), 4) if len(seed_rates) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sensitivity_root", type=Path)
    parser.add_argument("--json_out", type=Path, default=None)
    args = parser.parse_args()

    report: list[dict] = []
    for tag, label, pose_n, cri_n in LEVELS:
        rows = load_episodes(args.sensitivity_root / tag)
        stats = summarize(rows)
        entry = {
            "dir": tag,
            "level": label,
            "ee_pose_error_half_range": pose_n,
            "cri_half_range": cri_n,
            "present": stats is not None,
        }
        if stats:
            entry.update(stats)
        report.append(entry)

    text = json.dumps(report, indent=2)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
        print(f"[INFO] Wrote {args.json_out}")


if __name__ == "__main__":
    main()
