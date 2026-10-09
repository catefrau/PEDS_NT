#!/bin/bash
#SBATCH --job-name=mesh_conv_pps1
#SBATCH --account=fc_neutronix
#SBATCH --partition=savio3
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/mesh_conv_study_%j.out
#SBATCH --error=slurm_logs/mesh_conv_study_%j.err
#
# Mesh convergence study using precise_param_strat seed_1 test worst cases
# and the matching pretrained checkpoint.
#
# Submit from modules/MOREstudies/0small_studies/:
#   sbatch run_mesh_convergence_study.sh

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate jax-env

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_force_host_platform_device_count=${SLURM_CPUS_PER_TASK:-32}"
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu

mkdir -p slurm_logs
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

N_WORST=${N_WORST:-10}
MESHES=${MESHES:-"2 1 0.5 0.2 0.1 0.05 0.01"}
PEDS_RUN_DIR=${PEDS_RUN_DIR:-/global/home/users/caterinafrau/PEDS_NT/modules/pretrained_models/train_1000_seed_1}
KEFF_CMP_CSV=${KEFF_CMP_CSV:-/global/home/users/caterinafrau/PEDS_NT/modules/RUNS/precise_param_strat/testset_results/run_train1000_seed1_keff_comparison.csv}
NPZ_PATH=${NPZ_PATH:-/global/home/users/caterinafrau/PEDS_NT/data/highfidelity/17jul_0.8_1.2.npz}
OUT_DIR=${OUT_DIR:-}

echo "Job ${SLURM_JOB_ID:-local} starting mesh convergence study on $(hostname)"
echo "Started at : $(date)"
echo "Working dir: $(pwd)"
echo "N_WORST=${N_WORST}"
echo "MESHES=${MESHES}"
echo "PEDS_RUN_DIR=${PEDS_RUN_DIR}"
echo "KEFF_CMP_CSV=${KEFF_CMP_CSV}"
echo "NPZ_PATH=${NPZ_PATH}"

ARGS=(
  --n-worst "${N_WORST}"
  --meshes ${MESHES}
  --peds-run-dir "${PEDS_RUN_DIR}"
  --keff-cmp-csv "${KEFF_CMP_CSV}"
  --npz-path "${NPZ_PATH}"
  --study-source-name precise_param_strat_seed1
  --xs-sources poly peds
)
if [[ -n "${OUT_DIR}" ]]; then
  ARGS+=(--out-dir "${OUT_DIR}")
fi

python mesh_convergence_study.py "${ARGS[@]}"

echo "Mesh convergence study complete."
echo "Finished at: $(date)"
