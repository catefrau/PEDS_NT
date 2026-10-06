#!/bin/bash
#SBATCH --job-name=inclusion_criteria
#SBATCH --account=fc_neutronix
#SBATCH --partition=savio3
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=07:00:00
#SBATCH --output=output_%j.log

source ~/miniconda3/etc/profile.d/conda.sh

python inclusion_criteria.py 
