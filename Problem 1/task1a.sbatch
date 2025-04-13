#!/bin/bash

#SBATCH --job-name=task1a
#SBATCH --output=task1a.out
#SBATCH --error=task1a.err
#SBATCH --account=macs30123
#SBATCH --partition=caslake
#SBATCH --ntasks=1

module load python/anaconda-2022.05

python ./task1a.py
