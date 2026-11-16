"""Lead-lag estimation by cross-products of previous-tick returns on a 1-second grid."""

from __future__ import annotations

import numpy as np

GRID_SECONDS = 1.0
MAX_LAG = 5


def lag_grid(max_lag: int = MAX_LAG) -> np.ndarray:
    """Integer lags -max_lag, ..., max_lag (default 5) in seconds."""
    return np.arange(-max_lag, max_lag + 1, dtype=float) * GRID_SECONDS


def previous_tick_returns(times, values, window_seconds: float) -> np.ndarray:
    """Previous-tick returns on the 1-second grid over [0, window_seconds].

    Each grid point takes the last observation at or before it (the first observation before any is seen).
    Arguments: times are sorted observation times in seconds; values the observed (log) prices; window_seconds the window
    length in seconds.
    Returns: the grid increments; zeros when there are no observations.
    Reference: Epps (1979), JASA (the effect of asynchronous sampling).
    """
    times, values = np.asarray(times, dtype=float), np.asarray(values, dtype=float)
    if times.size == 0:
        return np.zeros(int(round(window_seconds)))
    grid = np.arange(0.0, window_seconds + GRID_SECONDS, GRID_SECONDS)
    position = np.searchsorted(times, grid, side="right") - 1
    sampled = values[np.maximum(position, 0)]
    return np.diff(sampled)


def event_cross_products(returns_x: np.ndarray, returns_y: np.ndarray, max_lag: int = MAX_LAG) -> np.ndarray:
    """Cross-products C(l) = sum_k r_x(k) r_y(k + l) for one event at integer lags -max_lag, ..., max_lag (default 5).

    Arguments: returns_x and returns_y are grid returns of equal length.
    Returns: C at each lag, in order.
    """
    n = returns_x.size
    return np.array([np.sum(returns_x[max(0, -lag) : n - max(0, lag)] * returns_y[max(0, lag) : n - max(0, -lag)]) for lag in range(-max_lag, max_lag + 1)])


def estimate_lag(cross_products, lags) -> float:
    """Lead-lag estimate: the lag (seconds) maximising |C|, positive meaning X leads Y.

    Arguments: cross_products is C on lags, or an (events, lags) array summed over events; lags is the lag grid.
    Returns: the lag in seconds.
    """
    pooled = np.asarray(cross_products, dtype=float)
    if pooled.ndim == 2:
        pooled = pooled.sum(axis=0)
    return float(np.asarray(lags)[np.argmax(np.abs(pooled))])


def bootstrap_lags(event_values: np.ndarray, lags, draws: int, rng: np.random.Generator) -> np.ndarray:
    """Event-bootstrap lag estimates: resample events with replacement and take the lag maximising the pooled |C|.

    Arguments: event_values is the (events, lags) array of cross-products; lags the lag grid; draws the number of
    resamples; rng the generator.
    Returns: one lag (seconds) per resample.
    """
    events = event_values.shape[0]
    counts = np.apply_along_axis(lambda draw: np.bincount(draw, minlength=events), 1, rng.integers(0, events, size=(draws, events)))
    return np.asarray(lags, dtype=float)[np.argmax(np.abs(counts @ event_values), axis=1)]
