import numpy as np
import scipy.stats as sts
import time
import sim_numba

# Set model parameters
rho = 0.5
mu = 3.0
sigma = 1.0
z_0 = mu

S = 1000
T = int(4160)

# -------------------------------------
# Original code
# -------------------------------------
np.random.seed(25)
eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, S))

z_mat_s = np.zeros((T, S))
start = time.time()

for s_ind in range(S):
    z_tm1 = z_0
    for t_ind in range(T):
        e_t = eps_mat[t_ind, s_ind]
        z_t = rho * z_tm1 + (1 - rho) * mu + e_t
        z_mat_s[t_ind, s_ind] = z_t
        z_tm1 = z_t

end = time.time()
print(f"Serial execution time: {end - start:.4f} seconds")

# -------------------------------------
# Numba
# -------------------------------------

start_numba = time.time()
z_mat_numba = sim_numba.simulate_health(eps_mat, rho, mu, z_0)
end_numba = time.time()
time_numba = end_numba - start_numba
print(f"Numba version time: {time_numba:.4f} seconds")
# -------------------------------------
speedup = (end - start) / time_numba
print(f"Speedup from numba: {speedup:.2f}x")
