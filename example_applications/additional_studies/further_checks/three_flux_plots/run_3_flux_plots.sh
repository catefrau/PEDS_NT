#!/bin/bash
#SBATCH --job-name=three_flux_normal
#SBATCH --account=fc_neutronix
#SBATCH --partition=savio3
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --output=../slurm_logs/three_flux_%j.out
#SBATCH --error=../slurm_logs/three_flux_%j.err
#
# Three-flux comparison for the top cases in ../mesh_conv_study/selected_cases.csv.
# OpenMC scratch is deleted after the flux arrays are stored in plot_data/.
#
# Submit from this folder:
#   sbatch run_3_flux_plots.sh

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export OPENMC_CROSS_SECTIONS=/global/scratch/users/caterinafrau/openmc_data/endfb-viii.0-hdf5/cross_sections.xml
export PYTHONUNBUFFERED=1

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -f "${SUBMIT_DIR}/3_flux_plots.py" ]]; then
  STUDY_DIR="${SUBMIT_DIR}"
else
  STUDY_DIR="${SUBMIT_DIR}/three_flux_plots"
fi
cd "${STUDY_DIR}"
mkdir -p "${STUDY_DIR}/../slurm_logs/mplconfig"
export MPLCONFIGDIR="${STUDY_DIR}/../slurm_logs/mplconfig"

echo "Job ${SLURM_JOB_ID:-local} starting three-flux study"
echo "Study dir: ${STUDY_DIR}"

# ── Phase A: baseline + PEDS fluxes (jax-env) ────────────────────────────────
conda activate jax-env
python 3_flux_plots.py --phase diffusion --n-cases 5

# ── Phase B: OpenMC reference + final PNGs (mc-env) ──────────────────────────
conda activate mc-env
python 3_flux_plots.py --phase openmc --n-cases 5

echo "Three-flux study complete."
echo "Outputs → ${STUDY_DIR}/flux_plots and ${STUDY_DIR}/plot_data"
