import numpy as np


def log_gaussian(X, means, covariances):
    """Log probability of each sample under each Gaussian."""
    n, d = X.shape
    k = len(means)
    output = np.empty((n, k))
    identity = np.eye(d)

    for j in range(k):
        cov = covariances[j] + 1e-5 * identity
        sign, logdet = np.linalg.slogdet(cov)

        if sign <= 0:
            raise ValueError("Covariance matrix is not positive definite")

        diff = X - means[j]
        solved = np.linalg.solve(cov, diff.T).T
        quadratic = np.sum(diff * solved, axis=1)

        output[:, j] = -0.5 * (
            d * np.log(2 * np.pi) + logdet + quadratic
        )

    return output


def logsumexp(values):
    maximum = np.max(values, axis=1, keepdims=True)
    return maximum + np.log(
        np.sum(np.exp(values - maximum), axis=1, keepdims=True)
    )


def e_step(X, weights, means, covariances):
    """Estimate the probability that each sample belongs to each component."""
    log_prob = log_gaussian(X, means, covariances)
    log_prob += np.log(weights + 1e-15)[None, :]

    normalizer = logsumexp(log_prob)
    responsibilities = np.exp(log_prob - normalizer)
    log_likelihood = float(np.sum(normalizer))

    return responsibilities, log_likelihood


def m_step(X, responsibilities):
    """Update mixture weights, means and covariance matrices."""
    n, d = X.shape
    k = responsibilities.shape[1]

    counts = responsibilities.sum(axis=0) + 1e-8
    weights = counts / n
    means = (responsibilities.T @ X) / counts[:, None]
    covariances = np.empty((k, d, d))

    for j in range(k):
        diff = X - means[j]
        weighted = diff * responsibilities[:, j:j + 1]
        covariances[j] = (weighted.T @ diff) / counts[j]
        covariances[j] += 1e-5 * np.eye(d)

    return weights, means, covariances


def fit_em(X, n_components=3, max_iter=100, tol=1e-4, seed=42):
    """Fit a Gaussian Mixture Model using the EM algorithm."""
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 2 or len(X) < n_components:
        raise ValueError("X must be 2D with enough samples for all components")

    rng = np.random.default_rng(seed)
    n, d = X.shape

    means = X[rng.choice(n, n_components, replace=False)].copy()
    base_cov = np.cov(X, rowvar=False) + 1e-3 * np.eye(d)
    covariances = np.repeat(base_cov[None, :, :], n_components, axis=0)
    weights = np.full(n_components, 1.0 / n_components)

    history = []
    previous_ll = -np.inf

    for iteration in range(1, max_iter + 1):
        responsibilities, ll = e_step(
            X, weights, means, covariances
        )
        history.append(ll)

        weights, means, covariances = m_step(X, responsibilities)

        if np.isfinite(previous_ll):
            if abs(ll - previous_ll) <= tol * (1 + abs(previous_ll)):
                break

        previous_ll = ll

    responsibilities, final_ll = e_step(
        X, weights, means, covariances
    )

    return {
        "weights": weights,
        "means": means,
        "covariances": covariances,
        "responsibilities": responsibilities,
        "labels": np.argmax(responsibilities, axis=1),
        "log_likelihood": final_ll,
        "history": history,
        "iterations": iteration,
    }


def distributed_em(parties, n_components=3, max_iter=100, tol=1e-4, seed=42):
    """Distributed EM with plaintext aggregation of local sufficient statistics.

    This is a correctness baseline, NOT secure multiparty computation.
    Each party computes local E-step statistics; the coordinator aggregates them.
    """
    if len(parties) < 2:
        raise ValueError("At least two parties are required")

    parties = [np.asarray(p, dtype=np.float64) for p in parties]

    if any(p.ndim != 2 for p in parties):
        raise ValueError("Each party's data must be a 2D array")

    feature_counts = {p.shape[1] for p in parties}
    if len(feature_counts) != 1:
        raise ValueError("All parties must have the same number of features")

    X_all = np.concatenate(parties, axis=0)
    n, d = X_all.shape

    if n < n_components:
        raise ValueError("Not enough total samples for the components")

    rng = np.random.default_rng(seed)
    means = X_all[rng.choice(n, n_components, replace=False)].copy()
    base_cov = np.cov(X_all, rowvar=False) + 1e-3 * np.eye(d)
    covariances = np.repeat(base_cov[None, :, :], n_components, axis=0)
    weights = np.full(n_components, 1.0 / n_components)

    history = []
    previous_ll = -np.inf

    for iteration in range(1, max_iter + 1):
        total_counts = np.zeros(n_components)
        total_sums = np.zeros((n_components, d))
        total_second_moments = np.zeros((n_components, d, d))
        total_ll = 0.0

        for local_X in parties:
            responsibilities, local_ll = e_step(
                local_X, weights, means, covariances
            )
            total_ll += local_ll

            counts = responsibilities.sum(axis=0)
            sums = responsibilities.T @ local_X
            second_moments = np.einsum(
                "nk,ni,nj->kij",
                responsibilities,
                local_X,
                local_X,
            )

            total_counts += counts
            total_sums += sums
            total_second_moments += second_moments

        history.append(float(total_ll))

        # Stabilize components with very small effective counts.
        safe_counts = np.maximum(total_counts, 1e-8)
        weights = safe_counts / safe_counts.sum()
        means = total_sums / safe_counts[:, None]

        for j in range(n_components):
            covariances[j] = (
                total_second_moments[j] / safe_counts[j]
                - np.outer(means[j], means[j])
            )
            covariances[j] = (
                0.5 * (covariances[j] + covariances[j].T)
                + 1e-5 * np.eye(d)
            )

        if np.isfinite(previous_ll):
            if abs(total_ll - previous_ll) <= tol * (1 + abs(previous_ll)):
                break

        previous_ll = total_ll

    responsibilities, final_ll = e_step(
        X_all, weights, means, covariances
    )

    return {
        "weights": weights,
        "means": means,
        "covariances": covariances,
        "responsibilities": responsibilities,
        "labels": np.argmax(responsibilities, axis=1),
        "log_likelihood": final_ll,
        "history": history,
        "iterations": iteration,
        "n_parties": len(parties),
        "n_samples": n,
        "security": "plaintext aggregation; not cryptographic MPC",
    }
