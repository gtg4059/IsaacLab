#!/usr/bin/env bash
# Observation-noise sensitivity for Concern #5 (model_28000, jawale 2× gates).
set -euo pipefail
PY="${ISAACLAB_PYTHON:-/home/safetics/miniconda3/envs/env_isaaclab/bin/python}"
ROOT="/home/safetics/IsaacLab"
CKPT="${ROOT}/logs/rsl_rl/reach_ur10/2026-09-17_18-45-40/model_28000.pt"
OUT="${ROOT}/logs/rsl_rl/reach_ur10/2026-09-17_18-45-40/eval_28000_sensitivity"
LOG="${OUT}/sweep.log"
SEEDS="${SWEEP_SEEDS:-42,43,44,45}"
NUM_ENVS="${SWEEP_NUM_ENVS:-2000}"

run_level() {
  local tag="$1"
  shift
  echo "========== START ${tag} $(date -Iseconds) ==========" | tee -a "${LOG}"
  cd "${ROOT}"
  TERM=xterm "${PY}" scripts/reinforcement_learning/rsl_rl/play_reach_csv.py \
    --task Isaac-Reach-UR10-Play-v0 \
    --num_envs "${NUM_ENVS}" \
    --checkpoint "${CKPT}" \
    --seeds "${SEEDS}" \
    --gate_factor 2.0 \
    --no_traj_csv \
    --no_print_targets \
    --headless \
    --export_csv_dir "${OUT}/${tag}" \
    "$@" 2>&1 | tee -a "${LOG}"
  echo "========== DONE ${tag} $(date -Iseconds) ==========" | tee -a "${LOG}"
}

mkdir -p "${OUT}"

if [[ "${1:-all}" == "all" ]]; then
  run_level noise_0_clean
  run_level noise_10mm --enable_corruption --obs_pose_noise 0.01 --obs_cri_noise 0.005
  run_level noise_20mm --enable_corruption --obs_pose_noise 0.02 --obs_cri_noise 0.01
  echo "========== ALL DONE $(date -Iseconds) ==========" | tee -a "${LOG}"
else
  run_level "$@"
fi
