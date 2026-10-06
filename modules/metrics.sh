#!/bin/bash
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

source ~/miniconda3/etc/profile.d/conda.sh

export OMP_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Define LOGS roots here — one array task per entry
LOGS_ROOTS=(
  RUNS/agent_studies/r13_1k_elu6e4
)

N_ROOTS=${#LOGS_ROOTS[@]}

# Safety check
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= N_ROOTS )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID (N_ROOTS=$N_ROOTS)"
  exit 1
fi

LOGS_ROOT=${LOGS_ROOTS[$SLURM_ARRAY_TASK_ID]}

mkdir -p slurm_logs

echo "Task $SLURM_ARRAY_TASK_ID -> LOGS_ROOT=$LOGS_ROOT"

conda activate jax-env

python evaluate_test_metrics.py "$LOGS_ROOT" \
  --with-val-study \
  --results-dirname testset_results
