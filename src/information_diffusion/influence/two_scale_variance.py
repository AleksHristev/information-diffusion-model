"""Two-scale realised variance, a noise-robust variance estimator for tick data."""

from __future__ import annotations

import numpy as np


def two_scale_variance(values) -> float:
    """Two-scale realised variance TSRV = (RV_K - (n_bar / n) RV_all) / ((n - K + 1) / n - n_bar / n).

    RV_all = sum_i (y_i - y_{i-1})^2, RV_K = (1 / K) sum_i (y_{i+K} - y_i)^2, n_bar = (n - K + 1) / K and K = round(n^(2/3))
    (at least 2, at most n - 1). The denominator is the exact finite-sample factor, which accounts for the windows the
    slow scale loses at each edge.
    Arguments: values are observed log prices in time order.
    Returns: the variance estimate; NaN with fewer than four returns.
    Reference: Zhang, Mykland and Ait-Sahalia (2005), JASA.
    """
    y = np.asarray(values, dtype=float)
    n = y.size - 1
    if n < 4:
        return float("nan")
    slow = int(max(2, round(n ** (2.0 / 3.0))))
    slow = min(slow, n - 1)
    n_bar = (n - slow + 1) / slow
    fast_sum = np.sum(np.diff(y) ** 2)
    slow_sum = np.sum((y[slow:] - y[:-slow]) ** 2) / slow
    return float((slow_sum - (n_bar / n) * fast_sum) / ((n - slow + 1) / n - n_bar / n))
