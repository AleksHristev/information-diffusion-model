"""Hayashi-Yoshida covariance of two asynchronously observed series."""

from __future__ import annotations

import numpy as np


def covariance(times_x, values_x, times_y, values_y) -> float:
    """Hayashi-Yoshida covariance HY = sum_i sum_j (X_i - X_{i-1}) (Y_j - Y_{j-1}) 1{I_i intersects J_j}.

    I_i = (t_{i-1}, t_i] and J_j = (s_{j-1}, s_j] are the observation intervals; intervals that only touch at an endpoint
    do not overlap. The cost is O(min(n, m) log max(n, m)).
    Arguments: times_x, values_x and times_y, values_y are the observation times and values of the two series, each
    sorted by time.
    Returns: the covariance; 0 when either series has fewer than two observations.
    Reference: Hayashi and Yoshida (2005), Bernoulli.
    """
    tx, x, ty, y = (np.asarray(array, dtype=float) for array in (times_x, values_x, times_y, values_y))
    if tx.size < 2 or ty.size < 2:
        return 0.0
    if tx.size > ty.size:
        tx, x, ty, y = ty, y, tx, x
    first = np.searchsorted(ty[1:], tx[:-1], side="right")
    last = np.searchsorted(ty[:-1], tx[1:], side="left") - 1
    valid = last >= first
    return float(np.sum(np.diff(x)[valid] * (y[last[valid] + 1] - y[first[valid]])))


def pooled_covariance(events) -> float:
    """Sum of Hayashi-Yoshida covariances over independent events, each given as (times_x, values_x, times_y, values_y)."""
    return float(sum(covariance(*event) for event in events))
