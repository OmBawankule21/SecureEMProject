
import numpy as np
import pytest # type: ignore

from app.algorithms.fixed_point import ( # type: ignore
    encode_statistics,
    decode_statistics,
    MAX_SAFE_INTEGER,
)


def test_fixed_point_encode_decode():
    counts = np.array([100.0, 200.0, 300.0])
    sums = np.ones((3, 4)) * 10
    second = np.ones((3, 4, 4)) * 20

    encoded = encode_statistics(counts, sums, second)

    decoded_counts, decoded_sums, decoded_second = (
        decode_statistics(
            encoded,
            counts.shape,
            sums.shape,
            second.shape,
        )
    )

    np.testing.assert_allclose(decoded_counts, counts, atol=1e-6)
    np.testing.assert_allclose(decoded_sums, sums, atol=1e-6)
    np.testing.assert_allclose(decoded_second, second, atol=1e-6)


def test_non_finite_input_rejected():
    counts = np.array([np.nan])
    sums = np.zeros((1, 1))
    second = np.zeros((1, 1, 1))

    with pytest.raises(ValueError):
        encode_statistics(counts, sums, second)


def test_invalid_scale_rejected():
    with pytest.raises(ValueError):
        encode_statistics(
            np.array([1.0]),
            np.zeros((1, 1)),
            np.zeros((1, 1, 1)),
            scale=0,
        )


def test_aggregated_integer_overflow_rejected():
    encoded = np.array(
        [MAX_SAFE_INTEGER + 1],
        dtype=np.int64,
    )

    with pytest.raises(OverflowError):
        decode_statistics(
            encoded,
            (1,),
            (0, 1),
            (0, 1, 1),
        )
