import numpy as np
import pytest
from sklearn.datasets import make_blobs

from app.algorithms.em import fit_em, distributed_em


@pytest.fixture
def dataset():
    X, _ = make_blobs(
        n_samples=240,
        centers=3,
        n_features=4,
        random_state=42,
    )
    return X


def test_three_party_training(dataset):
    parties = np.array_split(dataset, 3)

    result = distributed_em(
        parties,
        n_components=3,
        seed=42,
    )

    assert result["n_parties"] == 3
    assert result["n_samples"] == 240
    assert result["responsibilities"].shape == (240, 3)


def test_weights_sum_to_one(dataset):
    result = distributed_em(
        np.array_split(dataset, 3),
        n_components=3,
        seed=42,
    )

    assert np.isclose(result["weights"].sum(), 1.0, atol=1e-5)


def test_responsibilities_are_normalized(dataset):
    result = distributed_em(
        np.array_split(dataset, 3),
        n_components=3,
        seed=42,
    )

    sums = result["responsibilities"].sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-5)


def test_distributed_matches_centralized(dataset):
    parties = np.array_split(dataset, 3)

    centralized = fit_em(dataset, n_components=3, seed=42)
    distributed = distributed_em(parties, n_components=3, seed=42)

    assert np.isclose(
        centralized["log_likelihood"],
        distributed["log_likelihood"],
        atol=1e-3,
    )


def test_requires_multiple_parties(dataset):
    with pytest.raises(ValueError):
        distributed_em([dataset], n_components=3)
