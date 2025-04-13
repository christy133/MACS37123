import matplotlib.pyplot as plt

rhos = []
avg_times = []

with open("task2a.out", "r") as f:
    for line in f:
        line = line.strip()
        if line == "" or "Best rho" in line or "Total computation time" in line:
            continue
        try:
            rho_str, time_str = line.split(",")
            rhos.append(float(rho_str.strip()))
            avg_times.append(float(time_str.strip()))
        except ValueError:
            continue

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(rhos, avg_times, marker='o', markersize=3)
plt.title("Average Time to First Negative Health Value by ρ")
plt.xlabel("ρ (Persistence)")
plt.ylabel("Avg. Time to First z ≤ 0 (Weeks)")
plt.grid(True)
plt.axvline(x=-0.03342, color='red', linestyle='--', label='Optimal ρ = -0.03342')
plt.legend()
plt.tight_layout()
plt.savefig("task2b_plot.png")
plt.show()
