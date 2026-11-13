#!/bin/bash
#SBATCH --job-name=study_analysis
#SBATCH --account=fc_neutronix
#SBATCH --partition=savio3
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
# Submit from utils/:
#   cd utils && sbatch run_analysis.sh
#SBATCH --output=slurm_logs/study_analysis_%j.out
#SBATCH --error=slurm_logs/study_analysis_%j.err

set -euo pipefail

# Slurm copies this script to a spool directory on compute nodes, so do NOT
# resolve paths from BASH_SOURCE. Use SLURM_SUBMIT_DIR (directory where sbatch
# was run) or fall back to a fixed repo path.
# -----------------------------------------------------------------------------

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/global/home/users/caterinafrau/PEDS_NT/utils}"
REPO_ROOT="$(cd "$SUBMIT_DIR/.." && pwd)"
MODELS_DIR="$REPO_ROOT/models"
CONFIG_RUN_DIR="$REPO_ROOT/config_and_run"
SLURM_LOG_DIR="$SUBMIT_DIR/slurm_logs"
ANALYSIS_PY="$REPO_ROOT/utils/analysis/run_study_analysis.py"

mkdir -p "$SLURM_LOG_DIR"
cd "$CONFIG_RUN_DIR"

echo "Repo root:      $REPO_ROOT"
echo "Working dir:    $(pwd)"
echo "Slurm stdout:   $SLURM_LOG_DIR/study_analysis_${SLURM_JOB_ID:-local}.out"
echo "Slurm stderr:   $SLURM_LOG_DIR/study_analysis_${SLURM_JOB_ID:-local}.err"
echo "Started at:     $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate jax-env

export OMP_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# ============================== CONFIG (EDIT) ==============================
# Parent folder containing your study: RUNS/RESULTS (under config_and_run/) or an absolute path.
STUDY_PARENT_FOLDER="$REPO_ROOT/example_applications/main_results"

# Study folder name, e.g. complete_strat or PEDS.
STUDY_FOLDER="PEDS"

# Run used for XS-stats inputs (must contain XS/test_final_xs.csv and
# XS/logratio_saturation.csv under <parent>/<study>/<run>/).
XS_SOURCE_RUN="train_1000_seed_0"

# Error metric mode:
#   "false"  → |Δρ| = |(k_pred - k_ref) / (k_pred·k_ref)| × 10^5 pcm  (reactivity, default)
#   "true"   → |Δk| = |k_pred - k_ref| × 10^5 pcm  (linear in keff, mode-insensitive)
USE_DELTA_K="true"

# Tasks: "all" or comma-separated, e.g. test-metrics,param-error
TASKS="all"
# ===========================================================================

ARGS=(
  --study-parent-folder "$STUDY_PARENT_FOLDER"
  --study-folder "$STUDY_FOLDER"
  --xs-source-run "$XS_SOURCE_RUN"
  --tasks "$TASKS"
)

if [[ "$USE_DELTA_K" == "true" ]]; then
  ARGS+=(--use-delta-k)
fi

echo
echo "Running study analysis with:"
echo "  STUDY_PARENT_FOLDER=$STUDY_PARENT_FOLDER"
echo "  STUDY_FOLDER=$STUDY_FOLDER"
echo "  XS_SOURCE_RUN=$XS_SOURCE_RUN"
echo "  USE_DELTA_K=$USE_DELTA_K"
echo "  TASKS=$TASKS"
echo

python "$ANALYSIS_PY" "${ARGS[@]}"

echo
echo "Finished at: $(date)"
