#!/bin/bash
#SBATCH --job-name=peds_precise
#SBATCH --account=fc_neutronix
#SBATCH --partition=savio3
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=010:00:00
#SBATCH --output=slurm_logs/peds_%A_%a.out
#SBATCH --error=slurm_logs/peds_%A_%a.err
#SBATCH --array=0-4         # <-- set to N_TRAIN * N_SEEDS * N_DECAYS) - 1


source ~/miniconda3/etc/profile.d/conda.sh

export OMP_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # don't grab all memory upfront
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

TRAIN_SIZES=(1500)
SEEDS=(0 1 2 3 4)   # train+val seeds vary; TEST_SEED stays fixed (default 0 in PEDS.py)
# CHANGE THE ARRAY!!!
DECAY_EPOCHS_LIST=(70)   # <- study values

N_TRAIN=${#TRAIN_SIZES[@]}
N_SEEDS=${#SEEDS[@]}
N_DECAYS=${#DECAY_EPOCHS_LIST[@]}

TOTAL=$(( N_TRAIN * N_SEEDS * N_DECAYS ))
# Safety check
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TOTAL )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID (TOTAL=$TOTAL)"
  exit 1
fi
# Flattened index -> (train_size, seed, decay_epochs)
TS_IDX=$(( SLURM_ARRAY_TASK_ID / (N_SEEDS * N_DECAYS) ))
REM=$(( SLURM_ARRAY_TASK_ID % (N_SEEDS * N_DECAYS) ))
SEED_IDX=$(( REM / N_DECAYS ))
DECAY_IDX=$(( REM % N_DECAYS ))
export PEDS_TRAIN_SIZE=${TRAIN_SIZES[$TS_IDX]}
export PEDS_SEED=${SEEDS[$SEED_IDX]}
export PEDS_DECAY_EPOCHS=${DECAY_EPOCHS_LIST[$DECAY_IDX]}
# Optional: override fixed test seed (default 0 in PEDS.py)
# export PEDS_TEST_SEED=0


mkdir -p slurm_logs

echo "Task $SLURM_ARRAY_TASK_ID -> TRAIN_SIZE=$PEDS_TRAIN_SIZE SEED=$PEDS_SEED (train+val) DECAY_EPOCHS=$PEDS_DECAY_EPOCHS TEST_SEED=${PEDS_TEST_SEED:-0}"

conda activate jax-env

python PEDS.py