import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

from app.algorithms.em import fit_em


def main():
    X, true_labels = make_blobs(
        n_samples=600,
        centers=3,
        n_features=4,
        cluster_std=1.5,
        random_state=42,
    )
    X = StandardScaler().fit_transform(X)

    custom = fit_em(X, n_components=3, max_iter=100, seed=42)

    reference = GaussianMixture(
        n_components=3,
        covariance_type="full",
        max_iter=200,
        reg_covar=1e-5,
        random_state=42,
        n_init=3,
    ).fit(X)

    print("\n=== CUSTOM EM BASELINE ===")
    print("Iterations:", custom["iterations"])
    print("Log-likelihood:", round(custom["log_likelihood"], 3))
    print("Cluster sizes:", np.bincount(custom["labels"], minlength=3))
    print("ARI against synthetic labels:",
          round(adjusted_rand_score(true_labels, custom["labels"]), 4))

    print("\n=== SCIKIT-LEARN REFERENCE ===")
    print("Iterations:", reference.n_iter_)
    print("Log-likelihood:", round(reference.score(X) * len(X), 3))
    print("ARI against synthetic labels:",
          round(adjusted_rand_score(true_labels, reference.predict(X)), 4))

    if not np.all(np.isfinite(custom["means"])):
        raise RuntimeError("EM produced non-finite means")

    if not np.isclose(custom["weights"].sum(), 1.0, atol=1e-5):
        raise RuntimeError("Mixture weights do not sum to one")

    if not np.allclose(
        custom["responsibilities"].sum(axis=1), 1.0, atol=1e-5
    ):
        raise RuntimeError("Responsibilities do not sum to one")

    print("\nPASS: EM output sanity checks succeeded.")


if __name__ == "__main__":
    main()
