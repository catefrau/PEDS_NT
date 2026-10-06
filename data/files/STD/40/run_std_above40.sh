#!/bin/bash
#SBATCH --job-name=std_above40_lf
#SBATCH --account=fc_neutronix
#SBATCH --partition=savio3
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/std_above40_%j.out
#SBATCH --error=slurm_logs/std_above40_%j.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate jax-env

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export PYTHONUNBUFFERED=1

mkdir -p slurm_logs

python run_std_diffusion_batch.py \
  --csv std_above40.csv \
  --out std_above40_with_solver.csv
