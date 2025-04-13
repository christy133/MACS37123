from mpi4py import MPI
import numpy as np
import time
from sim_numba import simulate_health

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Parameters
S = 1000
T = 4160
mu = 3.0
rho = 0.5
sigma = 1.0
z_0 = mu

lives_per_proc = S // size

# Start timer
start = time.time()
np.random.seed(rank)
eps_local = np.random.normal(0, sigma, size=(T, lives_per_proc))

z_local = simulate_health(eps_local, rho, mu, z_0)

end = time.time()

if rank == 0:
    print(f"{size} cores took {end - start:.4f} seconds")
