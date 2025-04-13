import numpy as np
from numba.pycc import CC

cc = CC('sim_numba')

@cc.export('simulate_health', 'f8[:,:](f8[:,:], f8, f8, f8)')
def simulate_health(eps_mat, rho, mu, z_0):
    T, S = eps_mat.shape
    z_mat = np.zeros((T, S))
    for s_ind in range(S):
        z_tm1 = z_0
        for t_ind in range(T):
            e_t = eps_mat[t_ind, s_ind]
            z_t = rho * z_tm1 + (1 - rho) * mu + e_t
            z_mat[t_ind, s_ind] = z_t
            z_tm1 = z_t
    return z_mat

if __name__ == "__main__":
    cc.compile()