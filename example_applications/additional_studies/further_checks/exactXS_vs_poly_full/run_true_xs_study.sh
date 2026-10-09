#!/bin/bash
#SBATCH --job-name=full_true_xs_study
#SBATCH --account=fc_neutronix
#SBATCH --partition=savio3
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/full_true_xs_study_%j.out
#SBATCH --error=slurm_logs/full_true_xs_study_%j.err
#
# Full-population extension of run_5worstpar_study.py: for every geometry
# whose true OpenMC MGXS can be uniquely matched to an LHS sample, run the
# diffusion solver with (a) the true MGXS and (b) the poly-regression XS,
# and compare both to the OpenMC target keff.
#
# Submit from modules/MOREstudies/0small_studies/:
#   sbatch run_full_true_xs_study.sh

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate jax-env

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export PYTHONUNBUFFERED=1

mkdir -p slurm_logs

echo "Job ${SLURM_JOB_ID:-local} starting true-XS diffusion study"
python true_xs_study.py
echo "True-XS diffusion study complete."
