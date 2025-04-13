# MACS37123

## Problem 1: Clocking CPU Parallelism
### Task 1(a): Numba Pre-Compilation
To improve performance, I moved the nested for-loop into a separate function and pre-compiled it using Numba’s `pycc`. I then benchmarked both the original serial code and the Numba-accelerated version on a single Midway CPU core.

**Results:**

- Serial execution time: 3.1207 seconds  
- Numba version time: 0.0337 seconds  
- **Speedup from Numba:** 92.68x

This speedup shows the effectiveness of ahead-of-time compilation using Numba's `pycc`, which removes the Python interpreter overhead and accelerates the computationally intensive loop.

### Task 1(b): Runtime vs. Number of Cores
In this task, I used MPI parallelism using `mpi4py`, distributing 1000 simulated lives evenly across 1 to 20 CPU cores. Each process generated its own share of random shocks using a seed based on its rank. I recorded the time taken for each run and plotted runtime against number of cores:

![Task 1(b) Plot](task1b_plot.png)

### Task 1(c) - Why Isn’t Speedup Linear?

As we can see from the plot, while the runtime decreases significantly as the number of cores increases, the speedup is not linear. Several factors contribute to this:

1. **Fixed Serial Overhead**: There's always some part of the code that cannot be parallelized, including initialization and final gathering.
2. **Communication Overhead**: As the number of processes increases, communication between cores (even minimal) becomes a larger portion of the total time.
3. **Diminishing Workload Per Core**: When splitting 1000 simulations across 20 cores, each core only runs 50 simulations — which may not fully utilize the core’s compute capacity due to under-parallelization.
4. **Startup Latency**: MPI initialization and task scheduling introduces latency, especially visible at lower runtimes.

Despite this, parallelism via MPI still provided excellent speedup and demonstrated scalable gains up to 20 cores.
