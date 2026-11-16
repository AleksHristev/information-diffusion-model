"""Hoffmann-Rosenbaum-Yoshida lead-lag estimator on asynchronous observations, pooled across events."""

from __future__ import annotations

import numpy as np

from information_diffusion.compiled import njit

DEFAULT_STEP = 0.001
DEFAULT_MAX_LAG = 5.0


def lag_grid(step: float = DEFAULT_STEP, max_lag: float = DEFAULT_MAX_LAG) -> np.ndarray:
    """Symmetric uniform lag grid containing 0, in seconds, with the given step (default 0.001) out to max_lag (default 5)."""
    half = int(round(max_lag / step))
    return np.arange(-half, half + 1, dtype=float) * step


@njit(cache=True)
def _contrast_differences(tx, ty, dx, dy, first, pairs, start, step, count, batch_pairs):
    difference = np.zeros(count + 1)
    entering = np.zeros(count + 1)
    leaving = np.zeros(count + 1)
    size = dx.size
    begin = 0
    while begin < size:
        total = 0
        end = begin
        while end < size:
            total += pairs[end]
            end += 1
            if total > batch_pairs:
                break
        for i in range(begin, end):
            for j in range(first[i], first[i] + pairs[i]):
                lower = int(np.floor((ty[j] - tx[i + 1] - start) / step)) + 1
                upper = int(np.ceil((ty[j + 1] - tx[i] - start) / step)) - 1
                if lower < 0:
                    lower = 0
                if upper > count - 1:
                    upper = count - 1
                if lower <= upper:
                    weight = dx[i] * dy[j]
                    entering[lower] += weight
                    leaving[upper + 1] += weight
        for index in range(count + 1):
            difference[index] += entering[index]
            entering[index] = 0.0
        for index in range(count + 1):
            difference[index] -= leaving[index]
            leaving[index] = 0.0
        begin = end
    return difference


def event_contrast(times_x, values_x, times_y, values_y, lags, *, batch_pairs: int = 1_500_000) -> np.ndarray:
    """Shifted Hayashi-Yoshida contrast U(theta) = sum_i sum_j (X_i - X_{i-1}) (Y_j - Y_{j-1}) 1{I_i intersects (J_j - theta)} for one event.

    The pair (i, j) contributes exactly for theta in (s_{j-1} - t_i, s_j - t_{i-1}), so U is built on the whole grid from
    a difference array, in time proportional to the overlapping pairs.
    Arguments: times_x, values_x and times_y, values_y are the observation times (seconds) and log prices of the two
    series; lags is a uniform ascending lag grid in seconds; batch_pairs is the number of pairs processed per batch.
    Returns: U at each lag.
    """
    tx, x, ty, y = (np.asarray(array, dtype=float) for array in (times_x, values_x, times_y, values_y))
    grid = np.asarray(lags, dtype=float)
    count = grid.size
    if tx.size < 2 or ty.size < 2:
        return np.zeros(count)
    start, step = float(grid[0]), float(grid[1] - grid[0])
    stop = float(grid[-1])
    first = np.searchsorted(ty[1:], tx[:-1] + start, side="right")
    last = np.searchsorted(ty[:-1], tx[1:] + stop, side="left") - 1
    pairs = np.maximum(last - first + 1, 0)
    return np.cumsum(_contrast_differences(tx, ty, np.diff(x), np.diff(y), first, pairs, start, step, count, batch_pairs))[:count]


def _plateau_midpoint(magnitude: np.ndarray) -> int:
    peak = int(np.argmax(magnitude))
    top = magnitude[peak]
    if top <= 0.0 or not np.isfinite(top):
        return peak
    tied = np.flatnonzero(magnitude >= top * (1.0 - 1e-12))
    runs = np.split(tied, np.flatnonzero(np.diff(tied) != 1) + 1)
    run = next(candidate for candidate in runs if candidate[0] <= peak <= candidate[-1])
    return int((run[0] + run[-1]) // 2)


def estimate_lag(contrast, lags) -> float:
    """Estimate the lead-lag time as the lag maximising |U|, positive meaning X leads Y.

    U is piecewise constant, so the midpoint of the run of maximal values holding the first maximiser is returned.
    Arguments: contrast is U on lags, or an (events, lags) array summed over events; lags is the grid in seconds.
    Returns: the lag in seconds.
    Reference: Hoffmann, Rosenbaum and Yoshida (2013), Bernoulli.
    """
    pooled = np.asarray(contrast, dtype=float)
    if pooled.ndim == 2:
        pooled = pooled.sum(axis=0)
    return float(np.asarray(lags, dtype=float)[_plateau_midpoint(np.abs(pooled))])


def bootstrap_lags(event_contrasts: np.ndarray, lags, draws: int, rng: np.random.Generator) -> np.ndarray:
    """Event-bootstrap lag estimates: resample events with replacement and estimate the lag from the pooled contrast.

    Arguments: event_contrasts is the (events, lags) contrast array; lags the grid in seconds; draws the number of
    resamples; rng the generator.
    Returns: one lag (seconds) per resample.
    """
    events = event_contrasts.shape[0]
    grid = np.asarray(lags, dtype=float)
    counts = np.apply_along_axis(lambda draw: np.bincount(draw, minlength=events), 1, rng.integers(0, events, size=(draws, events)))
    pooled = counts @ event_contrasts
    magnitude = np.abs(pooled)
    return grid[[_plateau_midpoint(row) for row in magnitude]]
