#!/bin/bash
#SBATCH --job-name=peds_td_lhs_openmc
#SBATCH --account=fc_neutronix
#SBATCH --partition=savio3
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/td_generation_openmc_%j.out
#SBATCH --error=slurm_logs/td_generation_openmc_%j.err
#
# Submit from solvers/HF_openMC/:
#   sbatch run_many_openmc.sh

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mc-env

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENMC_CROSS_SECTIONS=/global/scratch/users/caterinafrau/openmc_data/endfb-viii.0-hdf5/cross_sections.xml
export PYTHONUNBUFFERED=1

# ── Run controls (edit only here) ────────────────────────────────────────────
N_SAMPLES=500
LHS_SEED=0

mkdir -p slurm_logs
cd /global/home/users/caterinafrau/PEDS_NT/solvers/HF_openMC

echo "Job ${SLURM_JOB_ID:-local} starting TD_generation.py (N_SAMPLES=${N_SAMPLES}, LHS_SEED=${LHS_SEED})"
N_SAMPLES="${N_SAMPLES}" LHS_SEED="${LHS_SEED}" python TD_generation.py

# Output path is defined only in TD_generation.py (DATA_DIR).
OUT_DIR="$(LHS_SEED="${LHS_SEED}" python -c 'from TD_generation import DATA_DIR; print(DATA_DIR)')"
echo "TD generation complete -> ${OUT_DIR}"
