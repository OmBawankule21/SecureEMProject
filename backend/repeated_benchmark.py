import csv
import subprocess
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

all_runs = []

for trial in range(1, 6):
    print(f"\n========== TRIAL {trial}/5 ==========")

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "backend" / "run_scalability.py")
        ],
        cwd=ROOT,
        check=True
    )

    df = pd.read_csv(RESULTS / "scalability_results.csv")
    df["Trial"] = trial
    all_runs.append(df)

combined = pd.concat(all_runs, ignore_index=True)

combined.to_csv(
    RESULTS / "scalability_repeated_runs.csv",
    index=False
)

summary = combined.groupby("Samples").agg(
    Mean_Runtime=("End_to_end_runtime_seconds", "mean"),
    Std_Runtime=("End_to_end_runtime_seconds", "std"),
    Mean_Communication=("Last_iteration_global_communication_MB", "mean"),
    Mean_Iterations=("Iterations", "mean")
).reset_index()

summary.to_csv(
    RESULTS / "scalability_summary.csv",
    index=False
)

print("\n========== REPEATED BENCHMARK SUMMARY ==========")
print(summary.to_string(index=False))
print("\nSaved repeated benchmark results.")
