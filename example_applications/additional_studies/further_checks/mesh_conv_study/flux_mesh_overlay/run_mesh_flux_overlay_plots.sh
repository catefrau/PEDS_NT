#!/bin/bash
#SBATCH --job-name=mesh_flux_overlay
#SBATCH --account=fc_neutronix
#SBATCH --partition=savio3
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/mesh_flux_overlay_%j.out
#SBATCH --error=slurm_logs/mesh_flux_overlay_%j.err
#
# For top-5 worst cases of mesh_conv_08091523:
#   diffusion fluxes (PEDS XS) at meshes 1, 0.5, 0.2, 0.1, 0.01 cm
#   + OpenMC reference flux overlays.
#
# Submit from modules/MOREstudies/0small_studies/:
#   sbatch run_mesh_flux_overlay_plots.sh

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export OPENMC_CROSS_SECTIONS=/global/scratch/users/caterinafrau/openmc_data/endfb-viii.0-hdf5/cross_sections.xml
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${PWD}/slurm_logs/mplconfig"
mkdir -p slurm_logs "$MPLCONFIGDIR"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

N_CASES=${N_CASES:-5}
MESHES=${MESHES:-"1 0.5 0.2 0.1 0.01"}

echo "Job ${SLURM_JOB_ID:-local} starting mesh-flux overlay study on $(hostname)"
echo "Started at : $(date)"
echo "N_CASES=${N_CASES}  MESHES=${MESHES}"

# ── Phase A: diffusion at each mesh (jax-env) ────────────────────────────────
conda activate jax-env
python run_mesh_flux_overlay_plots.py --phase diffusion --n-cases "${N_CASES}" --meshes ${MESHES}

# ── Phase B: OpenMC + final PNGs (mc-env) ────────────────────────────────────
conda activate mc-env
python run_mesh_flux_overlay_plots.py --phase openmc --n-cases "${N_CASES}" --meshes ${MESHES}

echo "Mesh-flux overlay study complete."
echo "Outputs → mesh_conv_08091523/flux_mesh_overlay/"
echo "Finished at: $(date)"
