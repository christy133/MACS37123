#!/bin/bash
#SBATCH --job-name=task2a
#SBATCH --output=task2a.out
#SBATCH --error=task2a.err
#SBATCH --account=macs30123
#SBATCH --partition=caslake
#SBATCH --ntasks=10
#SBATCH --nodes=1

module load python/anaconda-2022.05 mpich/3.2.1

mpirun python ./task2a.py