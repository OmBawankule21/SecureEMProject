import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

with open(RESULTS / "centralized_em_metrics.json") as f:
    centralized = json.load(f)

with open(RESULTS / "secure_em_metrics.json") as f:
    secure = json.load(f)

if centralized["dataset_samples"] != secure["dataset_samples"]:
    raise ValueError("Dataset sizes do not match!")

if centralized["iterations"] != secure["iterations"]:
    raise ValueError("Iteration counts do not match!")

central_time = centralized["runtime_seconds"]
secure_time = secure["algorithm_runtime_seconds"]

comparison = pd.DataFrame([
    {
        "Algorithm": "Centralized EM",
        "Samples": centralized["dataset_samples"],
        "Iterations": centralized["iterations"],
        "Runtime_seconds": central_time,
    },
    {
        "Algorithm": "Secure EM",
        "Samples": secure["dataset_samples"],
        "Iterations": secure["iterations"],
        "Runtime_seconds": secure_time,
    }
])

comparison.to_csv(
    RESULTS / "runtime_comparison.csv",
    index=False
)

print("\nRUNTIME COMPARISON")
print(comparison.to_string(index=False))

print(
    "\nSecure EM slowdown:",
    round(secure_time / central_time, 2),
    "times"
)

print("\nSaved: results/runtime_comparison.csv")
