from mpi4py import MPI
import numpy as np
import time
import sim2

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Parameters
S = 1000
T = 4160
mu = 3.0
z0 = mu
sigma = 1.0

if rank == 0:
    np.random.seed(0)
    eps_mat = np.random.normal(0, sigma, size=(T, S))
    rho_vals = np.linspace(-0.95, 0.95, 200)
    rho_chunks = np.array_split(rho_vals, size)
else:
    eps_mat = np.empty((T, S))
    rho_chunks = None

#Broadcast eps_mat and scatter rho_chunks
comm.Bcast(eps_mat, root=0)
rho_local = comm.scatter(rho_chunks, root=0)

results_local = []
start = time.time()

for rho in rho_local:
    fail_times = sim2.simulate_failure_times(eps_mat, rho, mu, z0)
    avg_time = np.mean(fail_times)
    results_local.append((rho, avg_time))

results_all = comm.gather(results_local, root=0)
end = time.time()

if rank == 0:
    flat_results = [item for sublist in results_all for item in sublist]
    best_rho, best_time = max(flat_results, key=lambda x: x[1])

    print("ρ values and their average failure times:")
    for rho, t in sorted(flat_results):
        print(f"{rho:.5f}, {t:.2f}")
    print(f"\nBest rho: {best_rho:.5f} (avg failure time {best_time:.2f})")
    print(f"\nTotal computation time: {end - start:.4f} seconds")
