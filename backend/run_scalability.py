import csv
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

SIZES = [600, 1200, 2400]
rows = []

for size in SIZES:
    print(f"\n========== DATASET: {size} ==========")

    start = time.perf_counter()

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "backend" / "run_secure_em_mpc.py"),
            str(size),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )

    elapsed = time.perf_counter() - start

    print(result.stdout)

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Secure EM failed for {size} samples")

    # Preserve results for each dataset size.
    shutil.copy2(
        RESULTS / "secure_em_results.npz",
        RESULTS / f"secure_em_{size}.npz",
    )

    iterations_match = re.search(
        r"Iterations:\s*(\d+)", result.stdout
    )

    iterations = (
        int(iterations_match.group(1))
        if iterations_match else None
    )

    # Read the global communication reported by Party 0.
    party_log = RESULTS / "mpc_party_0.log"
    communication = None
    mpc_time = None

    if party_log.exists():
        log = party_log.read_text()

        comm_match = re.search(
            r"Global data sent = ([0-9.]+) MB", log
        )
        time_match = re.search(
            r"Time = ([0-9.]+) seconds", log
        )

        if comm_match:
            communication = float(comm_match.group(1))

        if time_match:
            mpc_time = float(time_match.group(1))

    rows.append([
        size,
        iterations,
        round(elapsed, 6),
        mpc_time,
        communication,
    ])

    print("Total runtime:", round(elapsed, 6), "seconds")
    print("Iterations:", iterations)
    print("Last MPC execution communication:", communication, "MB")


with open(RESULTS / "scalability_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Samples",
        "Iterations",
        "End_to_end_runtime_seconds",
        "Last_iteration_party0_mpc_time_seconds",
        "Last_iteration_global_communication_MB",
    ])
    writer.writerows(rows)

print("\n========== SCALABILITY TEST COMPLETED ==========")
print("Saved:", RESULTS / "scalability_results.csv")
