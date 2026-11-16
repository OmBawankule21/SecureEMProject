
import time
import json
import numpy as np
from pathlib import Path

from app.algorithms.em import e_step, m_step
from run_secure_em_mpc import generate_hospital_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Same dataset generation as Secure EM
import sys
TOTAL_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 600
parties = generate_hospital_data(TOTAL_SAMPLES)
X = np.vstack(parties)

# Same initialization as Secure EM
rng = np.random.default_rng(100)

centers = np.array([
    [-4., -4., 0., 0.],
    [0., 4., 3., -2.],
    [4., -2., -3., 3.]
])

weights = np.full(3, 1 / 3)
means = centers + rng.normal(0, 0.5, centers.shape)

covariances = np.repeat(
    (2.0 * np.eye(4))[None, :, :], 3, axis=0
)

# Centralized EM execution
start = time.perf_counter()

for iteration in range(1, 21):

    resp, ll = e_step(X, weights, means, covariances)

    new_weights, new_means, new_covariances = m_step(X, resp)

    change = max(
        np.max(np.abs(new_weights - weights)),
        np.max(np.abs(new_means - means)),
        np.max(np.abs(new_covariances - covariances))
    )

    weights, means, covariances = (
        new_weights, new_means, new_covariances
    )

    if change < 1e-4:
        break

elapsed = time.perf_counter() - start

metrics = {
    "dataset_samples": TOTAL_SAMPLES,
    "iterations": iteration,
    "runtime_seconds": elapsed,
    "convergence_threshold": 1e-4
}

output_file = RESULTS / "centralized_em_metrics.json"

with open(output_file, "w") as f:
    json.dump(metrics, f, indent=4)

print("\nCentralized EM Results")
print("----------------------")
print("Samples:", TOTAL_SAMPLES)
print("Iterations:", iteration)
print("Runtime:", elapsed, "seconds")
print("Saved:", output_file)
