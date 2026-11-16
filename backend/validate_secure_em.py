import numpy as np

from run_secure_em_mpc import (
    generate_hospital_data,
    N_COMPONENTS,
    N_FEATURES,
    MAX_ITERATIONS,
    PROJECT_ROOT,
)

from app.algorithms.em import e_step, m_step


# Load actual Secure EM results
saved = np.load(PROJECT_ROOT / "results" / "secure_em_results.npz")

secure_weights = saved["weights"]
secure_means = saved["means"]
secure_covariances = saved["covariances"]
secure_iterations = int(saved["iterations"])


# Recreate exactly the same dataset
parties = generate_hospital_data()
X = np.vstack(parties)


# Recreate the same public initialization
rng = np.random.default_rng(100)

public_centers = np.array([
    [-4.0, -4.0, 0.0, 0.0],
    [0.0, 4.0, 3.0, -2.0],
    [4.0, -2.0, -3.0, 3.0],
])

weights = np.full(N_COMPONENTS, 1.0 / N_COMPONENTS)

means = public_centers + rng.normal(
    0, 0.5, public_centers.shape
)

covariances = np.repeat(
    (2.0 * np.eye(N_FEATURES))[None, :, :],
    N_COMPONENTS,
    axis=0
)


# Centralized EM using the same stopping criterion
for iteration in range(1, MAX_ITERATIONS + 1):

    responsibilities, ll = e_step(
        X, weights, means, covariances
    )

    new_weights, new_means, new_covariances = m_step(
        X, responsibilities
    )

    change = max(
        np.max(np.abs(new_weights - weights)),
        np.max(np.abs(new_means - means)),
        np.max(np.abs(new_covariances - covariances)),
    )

    weights = new_weights
    means = new_means
    covariances = new_covariances

    if change < 1e-4:
        break


# Numerical comparison
print("\n========== CENTRALIZED vs SECURE EM ==========")

print("\nIterations:")
print("Centralized:", iteration)
print("Secure:", secure_iterations)

print("\nMaximum absolute differences:")

print(
    "Weights:",
    np.max(np.abs(weights - secure_weights))
)

print(
    "Means:",
    np.max(np.abs(means - secure_means))
)

print(
    "Covariances:",
    np.max(np.abs(covariances - secure_covariances))
)

print("\nCentralized weights:")
print(weights)

print("\nSecure weights:")
print(secure_weights)

print("\nCentralized means:")
print(means)

print("\nSecure means:")
print(secure_means)

print("\nValidation completed.")
