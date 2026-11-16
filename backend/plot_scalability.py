import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/scalability_results.csv")

plt.figure()
plt.plot(df["Samples"], df["End_to_end_runtime_seconds"], marker="o")
plt.xlabel("Number of Samples")
plt.ylabel("Runtime (seconds)")
plt.title("Secure EM Scalability - Runtime")
plt.grid(True)
plt.savefig("results/scalability_runtime.png", dpi=300)
plt.show()

plt.figure()
plt.plot(
    df["Samples"],
    df["Last_iteration_global_communication_MB"],
    marker="o"
)
plt.xlabel("Number of Samples")
plt.ylabel("Communication (MB)")
plt.title("Secure EM Scalability - Communication")
plt.grid(True)
plt.savefig("results/scalability_communication.png", dpi=300)
plt.show()

print("Both scalability graphs saved successfully.")
