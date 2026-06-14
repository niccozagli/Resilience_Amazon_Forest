from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from scipy.signal import correlate, correlation_lags


def cross_correlation(
    x: np.ndarray,
    y: np.ndarray,
    dt: float,
    max_lag: int | None = None,
    demean: bool = True,
    normalization: str = "unbiased",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cross-correlation using scipy.signal.correlate.

    Note: for complex inputs, scipy.signal.correlate computes the conjugate
    correlation by conjugating the second input.

    Normalization:
        - "unbiased": divide each lag by (N - lag) (default, fewer samples at long lags)
        - "biased": divide each lag by N

    Parameters:
        x: 1D signal (real or complex).
        y: 1D signal (real or complex).
        dt: sampling interval.
        max_lag: maximum lag (in samples). Defaults to len(x) - 1.
        demean: if True, subtract mean from x and y before correlation.
        normalization: "unbiased" or "biased".

    Returns:
        lags: time lags (same units as dt).
        corr: cross-correlation values for non-negative lags.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same length")

    n = x.shape[0]
    if n == 0:
        raise ValueError("x and y must have at least one sample")

    if max_lag is None:
        max_lag = n - 1
    if max_lag < 0:
        raise ValueError("max_lag must be >= 0")

    if demean:
        x = x - np.mean(x)
        y = y - np.mean(y)

    corr_full = correlate(x, y, mode="full", method="auto")
    lags_full = correlation_lags(n, n, mode="full")

    nonneg = lags_full >= 0
    lags = lags_full[nonneg][: max_lag + 1]
    corr = corr_full[nonneg][: max_lag + 1]

    normalization = normalization.lower()
    if normalization == "unbiased":
        denom = (n - lags).astype(float)
        corr = corr / denom
    elif normalization == "biased":
        corr = corr / float(n)
    else:
        raise ValueError("normalization must be 'unbiased' or 'biased'")

    return lags * dt, corr


def find_index(indices: Sequence[Tuple[int, ...]], target: Tuple[int, ...]) -> int:
    try:
        return indices.index(target)
    except ValueError:
        return -1
