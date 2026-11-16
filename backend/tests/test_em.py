import numpy as np
from sklearn.datasets import make_blobs

from app.algorithms.em import fit_em, e_step


def make_test_data(seed=7):
    X, _ = make_blobs(
        n_samples=180,
        centers=3,
        n_features=4,
        random_state=seed,
    )
    return X


def test_em_outputs_are_finite():
    result = fit_em(make_test_data(), n_components=3)

    assert np.all(np.isfinite(result["means"]))
    assert np.all(np.isfinite(result["covariances"]))
    assert np.all(np.isfinite(result["weights"]))
    assert np.isfinite(result["log_likelihood"])


def test_weights_sum_to_one():
    result = fit_em(make_test_data(), n_components=3)

    assert np.isclose(result["weights"].sum(), 1.0, atol=1e-5)


def test_responsibilities_sum_to_one():
    result = fit_em(make_test_data(), n_components=3)

    sums = result["responsibilities"].sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-5)


def test_responsibilities_are_valid_probabilities():
    result = fit_em(make_test_data(), n_components=3)
    r = result["responsibilities"]

    assert np.all(r >= 0.0)
    assert np.all(r <= 1.0)


def test_e_step_shapes_and_likelihood():
    X = make_test_data()
    result = fit_em(X, n_components=3)

    responsibilities, ll = e_step(
        X,
        result["weights"],
        result["means"],
        result["covariances"],
    )

    assert responsibilities.shape == (180, 3)
    assert np.isfinite(ll)
    assert np.allclose(responsibilities.sum(axis=1), 1.0, atol=1e-5)
