import numpy as np

DEFAULT_SCALE = 1_000_000
MAX_SAFE_INTEGER = (2**62) - 1


def encode_statistics(counts, sums, second_moments, scale=DEFAULT_SCALE):
    """Flatten and encode local EM statistics as fixed-point integers."""

    if scale <= 0:
        raise ValueError("Scale must be positive")

    arrays = [
        np.asarray(counts, dtype=np.float64),
        np.asarray(sums, dtype=np.float64),
        np.asarray(second_moments, dtype=np.float64),
    ]

    if any(not np.all(np.isfinite(a)) for a in arrays):
        raise ValueError("Statistics contain non-finite values")

    flat = np.concatenate([a.ravel() for a in arrays])

    scaled = np.rint(flat * scale)

    if np.any(np.abs(scaled) > MAX_SAFE_INTEGER):
        raise OverflowError("Fixed-point values exceed safe integer range")

    return scaled.astype(np.int64)


def decode_statistics(encoded, counts_shape, sums_shape,
                      second_moments_shape, scale=DEFAULT_SCALE):
    """Decode an aggregated fixed-point vector into EM statistics."""

    if scale <= 0:
        raise ValueError("Scale must be positive")

    encoded = np.asarray(encoded)

    if encoded.ndim != 1 or not np.issubdtype(encoded.dtype, np.integer):
        raise ValueError("Encoded statistics must be a 1D integer array")

    if np.any(np.abs(encoded.astype(object)) > MAX_SAFE_INTEGER):
        raise OverflowError("Aggregated values exceed safe integer range")

    sizes = [
        int(np.prod(counts_shape)),
        int(np.prod(sums_shape)),
        int(np.prod(second_moments_shape)),
    ]

    if len(encoded) != sum(sizes):
        raise ValueError("Encoded vector has an unexpected length")

    values = encoded.astype(np.float64) / scale

    a, b, c = np.split(values, np.cumsum(sizes)[:-1])

    return (
        a.reshape(counts_shape),
        b.reshape(sums_shape),
        c.reshape(second_moments_shape),
    )
