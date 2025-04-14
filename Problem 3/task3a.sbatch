#!/bin/bash
#SBATCH --job-name=task3a
#SBATCH --output=task3a.out
#SBATCH --error=task3a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=30G
#SBATCH --account=macs30123

module load cuda python

python3 ./task3a.py
