import csv
import numpy as np
from pathlib import Path

from app.algorithms.em import e_step, m_step
from run_secure_em_mpc import generate_hospital_data, PROJECT_ROOT

# Load Secure EM results
secure = np.load(PROJECT_ROOT / "results/secure_em_results.npz")

secure_weights = secure["weights"]
secure_means = secure["means"]
secure_covariances = secure["covariances"]

# Recreate same dataset
X = np.vstack(generate_hospital_data())

# Same initialization
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

for iteration in range(1, 21):
    responsibilities, ll = e_step(
        X, weights, means, covariances
    )

    new_weights, new_means, new_covariances = m_step(
        X, responsibilities
    )

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

rows = [
    ["Metric", "Centralized EM", "Secure EM", "Absolute Difference"],
    ["Iterations", iteration, int(secure["iterations"]),
     abs(iteration - int(secure["iterations"]))],
    ["Weights max error", "-", "-",
     np.max(np.abs(weights - secure_weights))],
    ["Means max error", "-", "-",
     np.max(np.abs(means - secure_means))],
    ["Covariance max error", "-", "-",
     np.max(np.abs(covariances - secure_covariances))]
]

output = PROJECT_ROOT / "results/validation_metrics.csv"

with open(output, "w", newline="") as file:
    csv.writer(file).writerows(rows)

print("Validation report saved:", output)
for row in rows:
    print(row)
