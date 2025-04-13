#!/bin/bash

#SBATCH --job-name=task1a_serial_numba
#SBATCH --output=task1a.out
#SBATCH --error=task1a.err
#SBATCH --partition=caslake
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --account=macs30123

# Load the required Python module
module load python/anaconda-2022.05

python3 task1a.py