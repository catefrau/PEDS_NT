#!/bin/bash
#SBATCH --job-name=mlp_phi_xs
#SBATCH --account=fc_neutronix
#SBATCH --partition=savio3
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --chdir=/global/home/users/caterinafrau/PEDS_NT
#SBATCH --output=/global/home/users/caterinafrau/PEDS_NT/modules/RESULTS/complete_strat/mlp_baseline/mlp_with_phi_xs/slurm_logs/mlp_phi_xs_%j.out
#SBATCH --error=/global/home/users/caterinafrau/PEDS_NT/modules/RESULTS/complete_strat/mlp_baseline/mlp_with_phi_xs/slurm_logs/mlp_phi_xs_%j.err

set -euo pipefail

REPO_ROOT="/global/home/users/caterinafrau/PEDS_NT"
OUT_DIR="$REPO_ROOT/modules/RESULTS/complete_strat/mlp_baseline/mlp_with_phi_xs"
GEOM_METRICS="$REPO_ROOT/modules/RESULTS/complete_strat/mlp_baseline/mlp_geom_only/testset_results/test_metrics_all_runs.csv"
LOG_DIR="$OUT_DIR/slurm_logs"

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

# Ensure both repo root (solvers/) and modules/ are importable.
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/modules${PYTHONPATH:+:$PYTHONPATH}"

echo "Repo root:   $REPO_ROOT"
echo "Output dir:  $OUT_DIR"
echo "PYTHONPATH:  $PYTHONPATH"
echo "Started at:  $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate jax-env

export OMP_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

python modules/run_mlp_augmented_compare.py \
  --output-dir "$OUT_DIR" \
  --geom-only-metrics "$GEOM_METRICS"

echo "Finished at: $(date)"
