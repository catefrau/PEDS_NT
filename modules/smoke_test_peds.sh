#!/bin/bash
# Minimal end-to-end PEDS run used to verify the repository layout after a move.
#
#   cd config_and_run && sbatch smoke_test_peds.sh
#
# Exercises the whole chain (config -> data loading -> grad check -> solver ->
# training -> plotting -> checkpoint reload -> test evaluation) on a tiny
# dataset. Writes to RUNS/_smoke_test/ so real studies are never touched.
#SBATCH --job-name=peds_smoke
#SBATCH --account=fc_neutronix
#SBATCH --partition=savio3
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:40:00
#SBATCH --output=slurm_logs/peds_smoke_%j.out
#SBATCH --error=slurm_logs/peds_smoke_%j.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh

export OMP_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# ── tiny-but-complete configuration ─────────────────────────────────────────
export PEDS_STUDY_NAME=_smoke_test
# Analysis scripts parse train_size/seed out of the folder name, so keep the
# train_<N>_seed_<S> convention or launch_metrics.sh will find no runs.
export PEDS_EXP_NAME=train_120_seed_0
export PEDS_TRAIN_SIZE=120
export PEDS_VAL_SIZE=60
export PEDS_TEST_SIZE=40
export PEDS_EPOCHS=2
export PEDS_DECAY_EPOCHS=2
export PEDS_PARAM_STRAT_BINS=3
# Save a "best" checkpoint from epoch 1 so the post-training reload has a file.
export PEDS_MIN_SAVE_EPOCH=1
export PEDS_PATIENCE=99
# Keep the custom-VJP vs finite-difference check on: it is the only thing that
# exercises PEDS_core/diagnostics.py.
export PEDS_RUN_GRAD_CHECK=true
# Plot on the final epoch; indices must stay inside the small val set.
export PEDS_XS_HEATMAP_EPOCHS=2
export PEDS_SUBPLOT_SAMPLE_INDICES="0 1"
export PEDS_TRACKED_SAMPLES="0 1"

# Slurm copies this script to a spool directory on the compute node, so paths
# must NOT be resolved from BASH_SOURCE. Use the directory sbatch was run from.
CONFIG_RUN_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$CONFIG_RUN_DIR"
mkdir -p slurm_logs

conda activate jax-env

python run_peds.py
echo "SMOKE TEST COMPLETED OK"
