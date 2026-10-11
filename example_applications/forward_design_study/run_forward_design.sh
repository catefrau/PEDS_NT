#!/bin/bash
set -euo pipefail

# Unified launcher for `forward_design_study`.
# Usage:
#   bash run_forward_design.sh screen
#   bash run_forward_design.sh verify
#   bash run_forward_design.sh diffusion
#   bash run_forward_design.sh report
#   bash run_forward_design.sh plot
#   bash run_forward_design.sh all
#
# Notes:
# - `screen` uses `peds_ensemble_screen.py`
# - `verify` uses `openmc_verify_topk.py`
# - `diffusion` uses `diffusion_only_recheck.py`
# - `report` uses `build_report.py`
# - `plot` writes a parity figure for the active run directory.
#   The archived ens5k figure stays in `publication_plots/`.

ACTION="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"

LATEST_RUN_FILE="${SCRIPT_DIR}/LATEST_RUN.txt"
if [[ -n "${KEFF_SCAN_OUT_DIR:-}" ]]; then
  RUN_DIR="${KEFF_SCAN_OUT_DIR}"
elif [[ -f "${LATEST_RUN_FILE}" ]]; then
  RUN_DIR="$(cat "${LATEST_RUN_FILE}")"
else
  RUN_DIR=""
fi

run_screen() {
  conda activate jax-env
  cd "${SCRIPT_DIR}"
  python peds_ensemble_screen.py
  RUN_DIR="$(cat "${LATEST_RUN_FILE}")"
  export KEFF_SCAN_OUT_DIR="${RUN_DIR}"
}

run_verify() {
  if [[ -z "${RUN_DIR}" ]]; then
    echo "No run directory set. Run 'screen' first or export KEFF_SCAN_OUT_DIR." >&2
    exit 1
  fi
  conda activate mc-env
  cd "${SCRIPT_DIR}"
  KEFF_SCAN_OUT_DIR="${RUN_DIR}" python openmc_verify_topk.py
}

run_diffusion() {
  if [[ -z "${RUN_DIR}" ]]; then
    echo "No run directory set. Run 'screen' first or export KEFF_SCAN_OUT_DIR." >&2
    exit 1
  fi
  conda activate jax-env
  cd "${SCRIPT_DIR}"
  python diffusion_only_recheck.py \
    --feasible-csv "${RUN_DIR}/lhs_peds_feasible_ranked.csv" \
    --hf-comparison-csv "${RUN_DIR}/lhs_peds_hf_comparison.csv" \
    --out-dir "${RUN_DIR}/diffusion_only_check"
}

run_report() {
  if [[ -z "${RUN_DIR}" ]]; then
    echo "No run directory set. Run 'screen' first or export KEFF_SCAN_OUT_DIR." >&2
    exit 1
  fi
  conda activate jax-env
  cd "${SCRIPT_DIR}"
  KEFF_SCAN_OUT_DIR="${RUN_DIR}" python build_report.py
}

run_plot() {
  if [[ -z "${RUN_DIR}" ]]; then
    RUN_DIR="${SCRIPT_DIR}/results/ens5k_kefffirst_08092342"
  fi
  if [[ ! -f "${RUN_DIR}/lhs_peds_hf_comparison.csv" ]]; then
    echo "Missing ${RUN_DIR}/lhs_peds_hf_comparison.csv" >&2
    exit 1
  fi
  conda activate jax-env
  cd "${SCRIPT_DIR}"
  if [[ "${RUN_DIR}" == *"/ens5k_kefffirst_08092342" ]]; then
    OUT_PNG="${SCRIPT_DIR}/publication_plots/peds_hf_parity_ens5k_kefffirst.png"
    EXPECT_N=30
  else
    OUT_PNG="${RUN_DIR}/peds_hf_parity.png"
    EXPECT_N=0
  fi
  python publication_plots/plot_peds_hf_parity.py \
    --peds-csv "${RUN_DIR}/lhs_peds_hf_comparison.csv" \
    --out "${OUT_PNG}" \
    --expect-n "${EXPECT_N}"
}

case "${ACTION}" in
  screen) run_screen ;;
  verify) run_verify ;;
  diffusion) run_diffusion ;;
  report) run_report ;;
  plot) run_plot ;;
  all)
    run_screen
    run_verify
    run_report
    run_diffusion
    run_plot
    ;;
  *)
    echo "Unknown action: ${ACTION}" >&2
    echo "Use one of: screen | verify | diffusion | report | plot | all" >&2
    exit 1
    ;;
esac

echo "Done: ${ACTION}"
