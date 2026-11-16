"""Minute-scale lead-lag between two prediction-market venues after a scheduled decision, by the centroid of pooled cross-covariances."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_LAG = 5
WINDOWS = {"statement": (1, 30), "press_conference": (31, 90)}
MINUTES = 90


def minute_changes(prices: np.ndarray) -> np.ndarray:
    """Minute changes r_m = p_m - p_{m-1}, m = 1, ..., M, from probabilities at minutes 0, ..., M; zero where either end is undefined."""
    prices = np.asarray(prices, dtype=float)
    changes = np.diff(prices)
    return np.where(np.isfinite(changes), changes, 0.0)


def lags(max_lag: int = MAX_LAG) -> np.ndarray:
    """Integer lags -max_lag, ..., max_lag in minutes (default 5)."""
    return np.arange(-max_lag, max_lag + 1)


def cross_products(changes_kalshi: np.ndarray, changes_polymarket: np.ndarray, window: tuple[int, int], max_lag: int = MAX_LAG) -> np.ndarray:
    """Cross-products C(l) = sum_m r_{K,m} r_{P,m+l} over minutes m and m + l both inside the window, l = -L, ..., L.

    Arguments: changes_kalshi and changes_polymarket are the minute changes of the first venue (K) and the second
    venue (P), minute m at position m - 1; window is (first, last) minute, inclusive; max_lag is L (default 5).
    Returns: C at each lag.
    """
    first, last = window
    k = np.asarray(changes_kalshi, dtype=float)[first - 1:last]
    p = np.asarray(changes_polymarket, dtype=float)[first - 1:last]
    n = k.size
    return np.array([float(np.dot(k[max(0, -lag):n - max(0, lag)], p[max(0, lag):n - max(0, -lag)])) for lag in lags(max_lag)])


def window_variances(changes_kalshi: np.ndarray, changes_polymarket: np.ndarray, window: tuple[int, int]) -> tuple[float, float]:
    """Sums of squared minute changes of each venue inside the window (first, last), inclusive. Returns (sum r_K^2, sum r_P^2)."""
    first, last = window
    k = np.asarray(changes_kalshi, dtype=float)[first - 1:last]
    p = np.asarray(changes_polymarket, dtype=float)[first - 1:last]
    return float(k @ k), float(p @ p)


def centroid(cross: np.ndarray, max_lag: int = MAX_LAG) -> float:
    """Lead-lag centroid lambda = sum_l l C(l) / sum_l C(l) in minutes, positive when the first venue moves first.

    For a pure delay of d minutes inside the lag window, lambda = d.
    Arguments: cross is C on the lags -max_lag, ..., max_lag.
    Returns: lambda; NaN when sum C(l) is not positive.
    Reference: de Jong and Nijman (1997), Journal of Empirical Finance; Hoffmann, Rosenbaum and Yoshida (2013),
    Bernoulli.
    """
    cross = np.asarray(cross, dtype=float)
    total = cross.sum()
    return float(lags(max_lag) @ cross / total) if total > 0 else np.nan


def peak_lag(cross: np.ndarray, max_lag: int = MAX_LAG) -> float:
    """Lag (minutes) of the largest C(l) over -max_lag, ..., max_lag; NaN when every C(l) is zero."""
    cross = np.asarray(cross, dtype=float)
    return float(lags(max_lag)[int(np.argmax(cross))]) if np.any(cross != 0) else np.nan


def correlation(cross: np.ndarray, variance_kalshi: float, variance_polymarket: float) -> np.ndarray:
    """Cross-correlation rho(l) = C(l) / sqrt(sum r_K^2 sum r_P^2); NaN when either sum of squares is zero."""
    scale = np.sqrt(variance_kalshi * variance_polymarket)
    return np.asarray(cross, dtype=float) / scale if scale > 0 else np.full(np.shape(cross), np.nan)


@dataclass(frozen=True)
class DecisionTerms:
    """Per-decision sums for one window, each summed over the decision's outcomes.

    Fields: cross, shape (decisions, 2L + 1), the cross-products; variance_kalshi and variance_polymarket, shape
    (decisions,), the sums of squared changes.
    """

    cross: np.ndarray
    variance_kalshi: np.ndarray
    variance_polymarket: np.ndarray

    def pooled(self, weights: np.ndarray | None = None) -> tuple[np.ndarray, float, float]:
        """Weighted sums over decisions (equal weights when None). Returns (C, sum r_K^2, sum r_P^2)."""
        w = np.ones(self.cross.shape[0]) if weights is None else weights
        return w @ self.cross, float(w @ self.variance_kalshi), float(w @ self.variance_polymarket)


def estimate(terms: DecisionTerms, max_lag: int = MAX_LAG) -> dict:
    """Pooled estimates for one window: dict of lead_lag_minutes (centroid), peak_lag_minutes, cross (C) and rho (cross-correlation)."""
    cross, vk, vp = terms.pooled()
    return {"lead_lag_minutes": centroid(cross, max_lag), "peak_lag_minutes": peak_lag(cross, max_lag), "cross": cross, "rho": correlation(cross, vk, vp)}


def decision_bootstrap(terms: DecisionTerms, draws: int, rng: np.random.Generator, max_lag: int = MAX_LAG) -> dict:
    """Bootstrap draws of the centroid, the peak lag and rho(l), resampling whole decisions with replacement.

    Arguments: terms are the per-decision sums; draws the number of draws; rng the generator; max_lag is L.
    Returns: a dict of arrays lead_lag_minutes, peak_lag_minutes and rho (draws, 2L + 1).
    Reference: Efron and Tibshirani (1993), An Introduction to the Bootstrap.
    """
    count = terms.cross.shape[0]
    centroids, peaks = np.full(draws, np.nan), np.full(draws, np.nan)
    rho = np.full((draws, 2 * max_lag + 1), np.nan)
    for draw in range(draws):
        weights = np.bincount(rng.integers(0, count, size=count), minlength=count).astype(float)
        cross, vk, vp = terms.pooled(weights)
        centroids[draw], peaks[draw] = centroid(cross, max_lag), peak_lag(cross, max_lag)
        rho[draw] = correlation(cross, vk, vp)
    return {"lead_lag_minutes": centroids, "peak_lag_minutes": peaks, "rho": rho}


def percentile_interval(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval at the given level over the finite values. Returns (lower, upper), NaN when none are finite."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    tail = 100.0 * (1.0 - level) / 2.0
    return float(np.percentile(values, tail)), float(np.percentile(values, 100.0 - tail))
