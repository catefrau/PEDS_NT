#!/bin/bash
#SBATCH --job-name=xsstats_peds
#SBATCH --account=fc_neutronix
#SBATCH --partition=savio3
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/xs_stats_%j.out
#SBATCH --error=slurm_logs/xs_stats_%j.err

set -euo pipefail

mkdir -p slurm_logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-env

PROJECT_ROOT="/global/home/users/caterinafrau/PEDS_NT"
SCRIPT_PATH="$PROJECT_ROOT/modules/PEDS_subdivision/analysis/xs_stats_report.py"

python "$SCRIPT_PATH" \
  --study-parent-folder "RESULTS" \
  --study-folder "complete_strat/PEDS" \
  --xs-source-run "train_1000_seed_2"
