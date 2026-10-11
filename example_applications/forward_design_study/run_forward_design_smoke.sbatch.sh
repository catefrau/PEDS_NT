#!/bin/bash
#SBATCH --job-name=fd_smoke100
#SBATCH --account=fc_neutronix
#SBATCH --partition=savio3
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/fd_smoke100_%j.out
#SBATCH --error=slurm_logs/fd_smoke100_%j.err

# End-to-end smoke test: screen (config in peds_ensemble_screen.py) +
# OpenMC verify + report + diffusion-only + parity plot.
# Submit from example_applications/forward_design_study/:
#   sbatch run_forward_design_smoke.sbatch.sh

set -euo pipefail

mkdir -p slurm_logs
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export PYTHONUNBUFFERED=1
export OPENMC_CROSS_SECTIONS="${OPENMC_CROSS_SECTIONS:-/global/scratch/users/caterinafrau/openmc_data/endfb-viii.0-hdf5/cross_sections.xml}"

bash run_forward_design.sh all
