from numba.pycc import CC
import numpy as np

cc = CC("sim2")

@cc.export("simulate_failure_times", "f8[:](f8[:,:], f8, f8, f8)")
def simulate_failure_times(eps_mat, rho, mu, z0):
    T, S = eps_mat.shape
    first_fail = np.zeros(S)

    for s in range(S):
        z_tm1 = z0
        failed = False
        for t in range(T):
            z_t = rho * z_tm1 + (1 - rho) * mu + eps_mat[t, s]
            if not failed and z_t <= 0:
                first_fail[s] = t
                failed = True
            z_tm1 = z_t
        if not failed:
            first_fail[s] = T
    return first_fail

if __name__ == "__main__":
    cc.compile()
