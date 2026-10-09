#!/bin/bash
# run_agent_studies_r5_1k.sh — scale the R4 winner (r07: ELU, lr_max=6e-4) to 1000 train
#
# 1 study x 3 seeds = 3 array tasks (0-2)
# Test set stays 300 with TEST_SEED=0 (same locked test as the 500-sample runs).
# steps_per_epoch is pinned at 16 inside PEDS_agent.py — LR trajectory is unchanged.
#
#   cd /global/home/users/caterinafrau/PEDS_NT/modules
#   sbatch RUNS/agent_studies/run_agent_studies_r5_1k.sh

#SBATCH --job-name=peds_1k
#SBATCH --account=fc_neutronix
#SBATCH --partition=savio3
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --output=RUNS/agent_studies/slurm_logs/r5_%A_%a.out
#SBATCH --error=RUNS/agent_studies/slurm_logs/r5_%A_%a.err
#SBATCH --array=0-2

SEEDS=(0 1 2)
export PEDS_STUDY_NAME=r13_1k_elu6e4
export PEDS_SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
export PEDS_TRAIN_SIZE=1000
export PEDS_VAL_SIZE=500
export PEDS_TEST_SIZE=300
export PEDS_TEST_SEED=0

mkdir -p RUNS/agent_studies/slurm_logs
echo "Task ${SLURM_ARRAY_TASK_ID} -> STUDY=${PEDS_STUDY_NAME}  SEED=${PEDS_SEED}  train=${PEDS_TRAIN_SIZE} val=${PEDS_VAL_SIZE} test=${PEDS_TEST_SIZE}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate jax-env
export OMP_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

python PEDS_agent.py
