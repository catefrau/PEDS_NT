#!/bin/bash
#SBATCH --job-name=metrics
#SBATCH --account=fc_neutronix
#SBATCH --partition=savio3_htc
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=07:00:00
#SBATCH --output=output_%j.log

source ~/miniconda3/etc/profile.d/conda.sh

python density_analysis.py
