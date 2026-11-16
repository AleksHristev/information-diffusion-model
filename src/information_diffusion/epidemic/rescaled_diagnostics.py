"""Goodness-of-fit summaries of rescaled intervals: Ljung-Box test and Q-Q summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

LAGS = 20
QQ_GRID = np.linspace(0.01, 0.99, 99)


@dataclass(frozen=True)
class LjungBox:
    """Ljung-Box test result. Fields: statistic Q, its chi-square pvalue, and the number of lags used."""

    statistic: float
    pvalue: float
    lags: int


def ljung_box(values: np.ndarray, lags: int = LAGS) -> LjungBox:
    """Ljung-Box test of no autocorrelation, Q = n (n + 2) sum_{k=1}^m r_k^2 / (n - k), compared with chi-square(m).

    Arguments: values is the series (for example z_i = 1 - exp(-tau_i)); lags is m (default 20, at most n - 1).
    Returns: a LjungBox; statistic and p-value are NaN when the series is constant or too short.
    Reference: Ljung and Box (1978), Biometrika.
    """
    x = np.asarray(values, dtype=float)
    n = x.size
    m = min(lags, n - 1)
    centred = x - x.mean()
    denominator = float(centred @ centred)
    if m < 1 or denominator == 0.0:
        return LjungBox(float("nan"), float("nan"), m)
    autocorrelations = np.array([float(centred[k:] @ centred[:-k]) / denominator for k in range(1, m + 1)])
    statistic = float(n * (n + 2) * np.sum(autocorrelations**2 / (n - np.arange(1, m + 1))))
    return LjungBox(statistic, float(stats.chi2.sf(statistic, m)), m)


def uniform_values(intervals: np.ndarray) -> np.ndarray:
    """Uniform values z_i = 1 - exp(-tau_i) from the rescaled intervals tau_i; i.i.d. Uniform(0, 1) under a correct model."""
    return -np.expm1(-np.asarray(intervals, dtype=float))


def qq_gap(values: np.ndarray) -> float:
    """Largest gap |z_(i) - (i - 0.5) / n| between the sorted values and their Uniform(0, 1) plotting positions; NaN when empty."""
    ordered = np.sort(np.asarray(values, dtype=float))
    return float(np.max(np.abs(ordered - (np.arange(ordered.size) + 0.5) / ordered.size))) if ordered.size else float("nan")


def qq_quantiles(values: np.ndarray) -> np.ndarray:
    """Empirical quantiles of the values at the probabilities QQ_GRID (0.01 to 0.99 in steps of 0.01), for a Q-Q plot."""
    return np.quantile(np.asarray(values, dtype=float), QQ_GRID)
