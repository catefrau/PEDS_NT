#!/bin/bash
# Evaluate held-out test metrics for one or more study roots.
#
#   cd utils && sbatch run_testset.sh
#
# LOGS_ROOTS entries are paths relative to the repo root,
# e.g. example_applications/main_results/PEDS. One array task per entry.
#SBATCH --job-name=eval_metrics
#SBATCH --account=fc_neutronix
#SBATCH --partition=savio3
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=010:00:00
#SBATCH --output=slurm_logs/eval_metrics_%A_%a.out
#SBATCH --error=slurm_logs/eval_metrics_%A_%a.err
#SBATCH --array=0        # <-- set to ${#LOGS_ROOTS[@]} - 1

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh

export OMP_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Define LOGS roots (relative to repo root or absolute) here — one array task per entry.
# Override without editing the file:
#   PEDS_LOGS_ROOTS="example_applications/a example_applications/b" sbatch --array=0-1 run_testset.sh
if [ -n "${PEDS_LOGS_ROOTS:-}" ]; then
  read -r -a LOGS_ROOTS <<< "$PEDS_LOGS_ROOTS"
else
  LOGS_ROOTS=(
    example_applications/main_results/PEDS
  )
fi

N_ROOTS=${#LOGS_ROOTS[@]}

# Safety check
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= N_ROOTS )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID (N_ROOTS=$N_ROOTS)"
  exit 1
fi

LOGS_ROOT=${LOGS_ROOTS[$SLURM_ARRAY_TASK_ID]}

# Slurm copies this script to a spool directory on the compute node, so paths
# must NOT be resolved from BASH_SOURCE. Use the directory sbatch was run from.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"   # utils/
REPO_ROOT="$(cd "$SUBMIT_DIR/.." && pwd)"
EVAL_PY="$REPO_ROOT/utils/analysis/evaluate_test_metrics.py"

cd "$REPO_ROOT"
mkdir -p "$SUBMIT_DIR/slurm_logs"

echo "Task $SLURM_ARRAY_TASK_ID -> LOGS_ROOT=$LOGS_ROOT"
echo "Evaluating with $EVAL_PY"

conda activate jax-env

python "$EVAL_PY" "$REPO_ROOT/$LOGS_ROOT" \
  --with-val-study \
  --results-dirname analysis/testset_results
