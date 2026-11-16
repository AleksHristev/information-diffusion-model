"""Lagged leader returns and follower grid returns by exact previous-tick lookups in integer nanoseconds."""

from __future__ import annotations

import numpy as np

NS_PER_SECOND = 1_000_000_000
STEP_SECONDS = 5


def to_ns(seconds) -> np.ndarray | int:
    """Seconds (scalar or array) to integer nanoseconds by rounding; exact for whole-nanosecond times below 2^22 seconds (about 48 days), where float64 seconds resolve one nanosecond."""
    value = np.rint(np.asarray(seconds, dtype=float) * NS_PER_SECOND).astype(np.int64)
    return int(value) if value.ndim == 0 else value






def previous_tick_ns(times_ns: np.ndarray, values: np.ndarray, query_ns: np.ndarray) -> np.ndarray:
    """Previous-tick value at each query time by exact integer comparison.

    Arguments: times_ns are sorted observation times and query_ns the query times, both integer nanoseconds from the
    window start; values the observed values.
    Returns: the last value at or before each query time; NaN before the first observation.
    """
    times_ns = np.asarray(times_ns, dtype=np.int64)
    query_ns = np.asarray(query_ns, dtype=np.int64)
    if times_ns.size == 0:
        return np.full(query_ns.size, np.nan)
    index = np.searchsorted(times_ns, query_ns, side="right") - 1
    out = np.asarray(values, dtype=float)[np.maximum(index, 0)].copy()
    out[index < 0] = np.nan
    return out


def leader_return_ns(path: tuple[np.ndarray, np.ndarray], lag_seconds: float, steps: int, step_seconds: int = STEP_SECONDS) -> np.ndarray:
    """Leader's most recent return at a lag, x_k = P(k step - theta) - P(k step - theta - step), k = 0, ..., steps - 1.

    theta = max(lag, 0), so the leader's future is never used; all times are integer nanoseconds.
    Arguments: path is (times_ns, log prices) of the leader; lag_seconds is the lag; steps the number of steps;
    step_seconds the step length (default 5 seconds).
    Returns: x_k for each step, NaN where no observation precedes the lookup time.
    """
    theta = to_ns(max(float(lag_seconds), 0.0))
    step = to_ns(step_seconds)
    origins = np.arange(steps, dtype=np.int64) * step - theta
    times, values = path
    return previous_tick_ns(times, values, origins) - previous_tick_ns(times, values, origins - step)


def grid_returns_ns(path: tuple[np.ndarray, np.ndarray], steps: int, step_seconds: int = STEP_SECONDS) -> np.ndarray:
    """Follower's return over each step, y_k = P((k + 1) step) - P(k step), k = 0, ..., steps - 1.

    Arguments: path is (times_ns, log prices) of the follower; steps the number of steps; step_seconds the step length
    (default 5 seconds).
    Returns: y_k for each step, NaN where no observation precedes the grid time.
    """
    step = to_ns(step_seconds)
    grid = np.arange(steps + 1, dtype=np.int64) * step
    prices = previous_tick_ns(*path, grid)
    return np.diff(prices)
