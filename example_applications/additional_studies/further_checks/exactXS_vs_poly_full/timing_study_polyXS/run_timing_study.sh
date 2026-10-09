#!/bin/bash
#SBATCH --job-name=poly_xs_timing
#SBATCH --account=fc_neutronix
#SBATCH --partition=savio3
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#SBATCH --output=slurm_logs/poly_xs_timing_%j.out
#SBATCH --error=slurm_logs/poly_xs_timing_%j.err
#
# Poly-regression XS only: time one diffusion solver run per matched
# geometry from the full true-XS study pool.
#
# Submit from modules/MOREstudies/0small_studies/:
#   sbatch run_poly_xs_timing_study.sh

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate jax-env

# Serial 1-D eigenproblem: keep BLAS single-threaded so the per-solve
# wall time is a clean "one solver run" number, not oversubscribed.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export PYTHONUNBUFFERED=1

mkdir -p slurm_logs
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

echo "Job ${SLURM_JOB_ID:-local} starting poly-XS solver timing study"
python run_poly_xs_timing_study.py
echo "Poly-XS solver timing study complete."
