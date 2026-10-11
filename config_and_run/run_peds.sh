#!/bin/bash
# Launch a PEDS training study.
#
#   cd config_and_run && sbatch run_peds.sh
#
# Experiment knobs live in config_peds.py. The sweep variables below are
# exported as PEDS_* environment overrides, one Slurm array task per
# (train_size, seed, decay_epochs) combination.
#SBATCH --job-name=peds_precise
#SBATCH --account=fc_neutronix
#SBATCH --partition=savio3
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=010:00:00
#SBATCH --output=slurm_logs/peds_%A_%a.out
#SBATCH --error=slurm_logs/peds_%A_%a.err
#SBATCH --array=0-4         # optional: set to (N_TRAIN * N_SEEDS * N_DECAYS) - 1

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh

export OMP_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # don't grab all memory upfront
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

read -r -a TRAIN_SIZES <<< "${PEDS_SWEEP_TRAIN_SIZES:-1500}"
read -r -a SEEDS <<< "${PEDS_SWEEP_SEEDS:-0 1 2 3 4}"   # train+val seeds vary; TEST_SEED stays fixed
read -r -a DECAY_EPOCHS_LIST <<< "${PEDS_SWEEP_DECAY_EPOCHS:-70}"

N_TRAIN=${#TRAIN_SIZES[@]}
N_SEEDS=${#SEEDS[@]}
N_DECAYS=${#DECAY_EPOCHS_LIST[@]}

TOTAL=$(( N_TRAIN * N_SEEDS * N_DECAYS ))
# Accept both array and non-array submissions.
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
# Safety check
if (( TASK_ID < 0 || TASK_ID >= TOTAL )); then
  echo "Invalid task id=${TASK_ID} (TOTAL=$TOTAL)."
  echo "For array runs, set --array=0-$((TOTAL - 1)); for single runs, leave array unset."
  exit 1
fi
# Flattened index -> (train_size, seed, decay_epochs)
TS_IDX=$(( TASK_ID / (N_SEEDS * N_DECAYS) ))
REM=$(( TASK_ID % (N_SEEDS * N_DECAYS) ))
SEED_IDX=$(( REM / N_DECAYS ))
DECAY_IDX=$(( REM % N_DECAYS ))
export PEDS_TRAIN_SIZE=${TRAIN_SIZES[$TS_IDX]}
export PEDS_SEED=${SEEDS[$SEED_IDX]}
export PEDS_DECAY_EPOCHS=${DECAY_EPOCHS_LIST[$DECAY_IDX]}
# Optional: override the fixed test seed (default 0 in config_peds.py)
# export PEDS_TEST_SEED=0

# Slurm copies this script to a spool directory on the compute node, so paths
# must NOT be resolved from BASH_SOURCE. Use the directory sbatch was run from.
CONFIG_RUN_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$CONFIG_RUN_DIR"

mkdir -p slurm_logs

echo "Task $TASK_ID -> TRAIN_SIZE=$PEDS_TRAIN_SIZE SEED=$PEDS_SEED (train+val) DECAY_EPOCHS=$PEDS_DECAY_EPOCHS TEST_SEED=${PEDS_TEST_SEED:-0}"

conda activate jax-env

python ../models/PEDS.py
