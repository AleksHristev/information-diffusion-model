"""Maritz-Jarrett standard error of a sample quantile."""

from __future__ import annotations

import numpy as np
from scipy import stats


def standard_error(draws, probability: float) -> float:
    """Compute the Maritz-Jarrett standard error of a sample quantile.

    With m = floor(n q + 0.5) and beta weights W_i = I_{i/n}(m - 1, n - m) - I_{(i-1)/n}(m - 1, n - m), it returns
    sqrt(sum_i W_i (X_(i) - C_1)^2) with C_1 = sum_i W_i X_(i).
    Arguments: draws are the sample values; probability is the quantile level q.
    Returns: the standard error.
    Reference: Maritz and Jarrett (1978), J. Amer. Statist. Assoc.
    """
    ordered = np.sort(np.asarray(draws, dtype=float))
    n = ordered.size
    m = int(np.floor(n * probability + 0.5))
    if not 1 < m < n:
        raise ValueError("the quantile must have order statistics on both sides of it")
    weights = np.diff(stats.beta.cdf(np.arange(n + 1) / n, m - 1, n - m))
    centre = float(weights @ ordered)
    return float(np.sqrt(weights @ (ordered - centre) ** 2))
