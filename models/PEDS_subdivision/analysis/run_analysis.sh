#!/bin/bash
#SBATCH --job-name=study_analysis
#SBATCH --account=fc_neutronix
#SBATCH --partition=savio3
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
# Submit from the repo root (PEDS_NT):
#   sbatch models/PEDS_subdivision/analysis/run_analysis.sh
#SBATCH --output=models/PEDS_subdivision/analysis/slurm_logs/study_analysis_%j.out
#SBATCH --error=models/PEDS_subdivision/analysis/slurm_logs/study_analysis_%j.err

set -euo pipefail

# -----------------------------------------------------------------------------
# Slurm copies this script to a spool directory on compute nodes, so do NOT
# resolve paths from BASH_SOURCE. Use SLURM_SUBMIT_DIR (directory where sbatch
# was run) or fall back to a fixed repo path.
# -----------------------------------------------------------------------------

REPO_ROOT="${SLURM_SUBMIT_DIR:-/global/home/users/caterinafrau/PEDS_NT}"
# Analysis code lives with the model stack; study outputs live with the launchers.
MODELS_DIR="$REPO_ROOT/models"
CONFIG_RUN_DIR="$REPO_ROOT/config_and_run"
SLURM_LOG_DIR="$MODELS_DIR/PEDS_subdivision/analysis/slurm_logs"
ANALYSIS_PY="$MODELS_DIR/PEDS_subdivision/analysis/run_study_analysis.py"

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
# RUNS or RESULTS — parent folder under config_and_run/ that contains your study.
STUDY_PARENT_FOLDER="RESULTS"

# Study folder name, e.g. complete_strat or precise_param_strat.
STUDY_FOLDER="complete_strat"

# Run used for XS-stats inputs (must contain XS/test_final_xs.csv and
# XS/logratio_saturation.csv under config_and_run/<parent>/<study>/<run>/).
XS_SOURCE_RUN="train_1000_seed_2"

# Error metric mode:
#   "false"  → |Δρ| = |(k_pred - k_ref) / (k_pred·k_ref)| × 10^5 pcm  (reactivity, default)
#   "true"   → |Δk| = |k_pred - k_ref| × 10^5 pcm  (linear in keff, mode-insensitive)
# When "true", outputs go to separate *_dk subdirectories so both sets of
# results coexist and can be compared directly.
USE_DELTA_K="true"

# Tasks to run: "all" or comma-separated subset, e.g.:
#   TASKS="all"
#   TASKS="param-error,training-diagnostics"
#   TASKS="test-metrics"
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
