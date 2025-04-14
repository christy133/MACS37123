#!/bin/bash
#SBATCH --job-name=mpi_sim_job
#SBATCH --error=mpi_sim.err
#SBATCH --output=mpi_sim.out
#SBATCH --account=macs30123
#SBATCH --partition=caslake
#SBATCH --ntasks=20
#SBATCH --nodes=1

# Load Python and MPI
module load python/anaconda-2022.05  mpich/3.2.1

for i in {1..20}
do
    mpirun -n $i python3 task1b.py >> mpi_times.out
done
