import numpy as np


def compute_local_statistics(X, responsibilities):
    """
    Compute local sufficient statistics for Gaussian Mixture EM.

    X: Local dataset, shape (n_samples, n_features)
    responsibilities: Shape (n_samples, n_components)

    Returns:
        counts: (k,)
        sums: (k, d)
        second_moments: (k, d, d)
    """

    X = np.asarray(X, dtype=np.float64)
    responsibilities = np.asarray(
        responsibilities, dtype=np.float64
    )

    if X.ndim != 2 or responsibilities.ndim != 2:
        raise ValueError("Inputs must be 2D arrays")

    if X.shape[0] != responsibilities.shape[0]:
        raise ValueError("Sample counts must match")

    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values")

    if not np.all(np.isfinite(responsibilities)):
        raise ValueError("Responsibilities contain non-finite values")

    if np.any(responsibilities < 0):
        raise ValueError("Responsibilities cannot be negative")

    counts = responsibilities.sum(axis=0)

    sums = responsibilities.T @ X

    second_moments = np.einsum(
        "nk,ni,nj->kij",
        responsibilities,
        X,
        X
    )

    return counts, sums, second_moments


def aggregate_plaintext_statistics(parties):
    """
    Plaintext aggregation baseline for validation only.
    This function is NOT cryptographic MPC.
    """

    total_counts = None
    total_sums = None
    total_second_moments = None

    for counts, sums, second_moments in parties:

        if total_counts is None:
            total_counts = np.zeros_like(counts)
            total_sums = np.zeros_like(sums)
            total_second_moments = np.zeros_like(second_moments)

        total_counts += counts
        total_sums += sums
        total_second_moments += second_moments

    return total_counts, total_sums, total_second_moments


def update_global_parameters(counts, sums, second_moments):
    """
    Global M-step using aggregated sufficient statistics.
    """

    safe_counts = np.maximum(counts, 1e-8)

    weights = safe_counts / safe_counts.sum()

    means = sums / safe_counts[:, None]

    covariances = (
        second_moments / safe_counts[:, None, None]
        - np.einsum("ki,kj->kij", means, means)
    )

    d = means.shape[1]

    covariances = (
        0.5 * (covariances + covariances.transpose(0, 2, 1))
        + 1e-5 * np.eye(d)[None, :, :]
    )

    return weights, means, covariances
