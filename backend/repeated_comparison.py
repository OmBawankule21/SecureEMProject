import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

SIZES = [600, 1200, 2400]
TRIALS = 5

all_runs = []

for size in SIZES:
    for trial in range(1, TRIALS + 1):

        print(f"\n========== SIZE {size} | TRIAL {trial}/5 ==========")

        # Run Centralized EM
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "backend" / "benchmark_centralized.py"),
                str(size),
            ],
            cwd=ROOT,
            check=True,
        )

        with open(RESULTS / "centralized_em_metrics.json") as f:
            central = json.load(f)

        # Run Secure EM
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "backend" / "run_secure_em_mpc.py"),
                str(size),
            ],
            cwd=ROOT,
            check=True,
        )

        with open(RESULTS / "secure_em_metrics.json") as f:
            secure = json.load(f)

        if central["dataset_samples"] != secure["dataset_samples"]:
            raise ValueError("Dataset size mismatch!")

        if central["iterations"] != secure["iterations"]:
            raise ValueError("Iteration count mismatch!")

        central_time = central["runtime_seconds"]
        secure_time = secure["algorithm_runtime_seconds"]

        all_runs.extend([
            {
                "Trial": trial,
                "Samples": size,
                "Algorithm": "Centralized EM",
                "Iterations": central["iterations"],
                "Runtime_seconds": central_time,
                "Communication_MB": 0.0,
            },
            {
                "Trial": trial,
                "Samples": size,
                "Algorithm": "Secure EM",
                "Iterations": secure["iterations"],
                "Runtime_seconds": secure_time,
                "Communication_MB": secure[
                    "total_global_communication_mb"
                ],
            },
        ])

        print("Centralized runtime:", central_time)
        print("Secure EM runtime:", secure_time)

raw = pd.DataFrame(all_runs)

raw.to_csv(
    RESULTS / "repeated_comparison_runs.csv",
    index=False
)

summary = raw.groupby(
    ["Samples", "Algorithm"]
).agg(
    Mean_Runtime=("Runtime_seconds", "mean"),
    Std_Runtime=("Runtime_seconds", "std"),
    Mean_Iterations=("Iterations", "mean"),
    Mean_Communication_MB=("Communication_MB", "mean"),
).reset_index()

summary.to_csv(
    RESULTS / "repeated_comparison_summary.csv",
    index=False
)

print("\n========== FINAL COMPARISON SUMMARY ==========")
print(summary.to_string(index=False))

print("\nSaved:")
print("results/repeated_comparison_runs.csv")
print("results/repeated_comparison_summary.csv")
