import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from app.algorithms.em import fit_em, distributed_em


def main():
    X, _ = make_blobs(
        n_samples=600,
        centers=3,
        n_features=4,
        cluster_std=1.5,
        random_state=42,
    )
    X = StandardScaler().fit_transform(X)

    # Simulate three hospitals, each holding 200 records.
    parties = np.array_split(X, 3)

    centralized = fit_em(X, n_components=3, seed=42)
    distributed = distributed_em(
        parties, n_components=3, seed=42
    )

    print("\n=== CENTRALIZED EM ===")
    print("Log-likelihood:", round(centralized["log_likelihood"], 3))
    print("Iterations:", centralized["iterations"])

    print("\n=== DISTRIBUTED EM ===")
    print("Hospitals:", distributed["n_parties"])
    print("Total records:", distributed["n_samples"])
    print("Log-likelihood:", round(distributed["log_likelihood"], 3))
    print("Iterations:", distributed["iterations"])
    print("Cluster sizes:", np.bincount(distributed["labels"], minlength=3))
    print("Security:", distributed["security"])

    assert np.isclose(
        distributed["weights"].sum(), 1.0, atol=1e-5
    )
    assert np.allclose(
        distributed["responsibilities"].sum(axis=1),
        1.0,
        atol=1e-5,
    )
    assert np.isfinite(distributed["log_likelihood"])

    print("\nPASS: Distributed EM sanity checks succeeded.")


if __name__ == "__main__":
    main()
