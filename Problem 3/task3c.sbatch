#!/bin/bash
#SBATCH --job-name=task3c
#SBATCH --output=task3c.out
#SBATCH --error=task3c.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=30G
#SBATCH --account=macs30123

module load cuda python
python3 ./task3c.py
