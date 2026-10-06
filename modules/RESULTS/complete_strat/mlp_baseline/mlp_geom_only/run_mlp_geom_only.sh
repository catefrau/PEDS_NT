#!/bin/bash
#SBATCH --job-name=mlp_geom_only
#SBATCH --account=fc_neutronix
#SBATCH --partition=savio3_htc
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --chdir=/global/home/users/caterinafrau/PEDS_NT
#SBATCH --output=/global/home/users/caterinafrau/PEDS_NT/modules/RESULTS/complete_strat/mlp_baseline/mlp_geom_only/slurm_logs/mlp_geom_only_%j.out
#SBATCH --error=/global/home/users/caterinafrau/PEDS_NT/modules/RESULTS/complete_strat/mlp_baseline/mlp_geom_only/slurm_logs/mlp_geom_only_%j.err

set -euo pipefail

REPO_ROOT="/global/home/users/caterinafrau/PEDS_NT"
OUT_DIR="$REPO_ROOT/modules/RESULTS/complete_strat/mlp_baseline/mlp_geom_only"
LOG_DIR="$OUT_DIR/slurm_logs"

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

echo "Repo root:   $REPO_ROOT"
echo "Output dir:  $OUT_DIR"
echo "Started at:  $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate jax-env

export OMP_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

python modules/run_mlp_baseline_compare.py --output-dir "$OUT_DIR"

echo "Finished at: $(date)"
