import matplotlib.pyplot as plt
cores = []
times = []

with open("mpi_times.out", "r") as f:
    for line in f:
        if "cores took" in line:
            parts = line.strip().split()
            cores.append(int(parts[0]))
            times.append(float(parts[-2]))

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(cores, times, marker='o')
plt.title("Runtime vs Number of Cores")
plt.xlabel("Number of Cores")
plt.ylabel("Computation Time (seconds)")
plt.grid(True)
plt.xticks(range(1, 21))
plt.savefig("task1b_plot.png")
plt.show()
