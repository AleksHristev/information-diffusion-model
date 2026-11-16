"""Giacomini-White test of equal predictive ability on event-level loss differentials from rolling windows."""

from __future__ import annotations

import functools
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from scipy import stats

from information_diffusion.statistical_framework import hac
from information_diffusion.statistical_framework.giacomini_white_critical_values import CRITICAL_VALUES, PROBABILITIES

DEFAULT_LAG = 1
NULL_SIMULATIONS = 4_000_000
NULL_SEED = 20261110
_CHUNK = 100_000


@dataclass(frozen=True)
class GiacominiWhite:
    """Result of a Giacomini-White test.

    Fields: the t-statistic, p-value, mean loss differential, long-run variance, number of out-of-sample events,
    degrees of freedom n - 1, number of lags, the alternative and the null distribution used ("simulated" or
    "student").
    """

    statistic: float
    p_value: float
    mean_differential: float
    long_run_variance: float
    n_events: int
    degrees_of_freedom: int
    lags: int
    alternative: str
    null: str


def rolling_splits(n_events: int, window: int) -> Iterator[tuple[np.ndarray, int]]:
    """Generate rolling splits: each event from index window on is forecast from the window events just before it.

    Arguments: n_events is the number of events; window is the rolling window length in events.
    Returns: an iterator of (estimation indices, test index).
    """
    if not 0 < window < n_events:
        raise ValueError("window must lie between 1 and the number of events minus one")
    for test in range(window, n_events):
        yield np.arange(test - window, test), test


def out_of_sample_events(n_events: int, window: int) -> int:
    """Return the number of out-of-sample events, n_events minus window."""
    return n_events - window


def statistic(differentials, lags: int = DEFAULT_LAG) -> np.ndarray:
    """Compute t = dbar / sqrt(V / n), with V the Newey-West long-run variance of the differentials.

    Arguments: differentials holds loss differentials on the last axis (other axes are replications); lags is the
    Newey-West lag (1 by default).
    Returns: the statistic for each replication.
    """
    d = np.asarray(differentials, dtype=float)
    n = d.shape[-1]
    return d.mean(axis=-1) / np.sqrt(hac.long_run_variance(d, lags) / n)


def p_value_from_statistic(t, n_events: int, *, alternative: str = "two-sided") -> np.ndarray:
    """Compute the p-value of t against Student t(n_events - 1).

    Arguments: t is the statistic; n_events is the number of out-of-sample events; alternative is "two-sided",
    "greater" or "less".
    Returns: the p-value.
    """
    t = np.asarray(t, dtype=float)
    df = n_events - 1
    if alternative == "two-sided":
        return 2 * stats.t.sf(np.abs(t), df)
    if alternative == "greater":
        return stats.t.sf(t, df)
    if alternative == "less":
        return stats.t.cdf(t, df)
    raise ValueError("alternative must be two-sided, greater or less")


def null_noise(rng: np.random.Generator, sims: int, n: int) -> np.ndarray:
    """Draw null loss differentials: mean-zero AR(1) series with coefficient 0.1 and unit-variance Student t(5) shocks.

    Arguments: rng is the random generator; sims is the number of series; n is the series length.
    Returns: an array (sims, n), scaled to unit stationary variance.
    """
    shocks = rng.standard_t(5, size=(sims, n)) / np.sqrt(5 / 3)
    series = shocks.copy()
    for t in range(1, n):
        series[:, t] = 0.1 * series[:, t - 1] + shocks[:, t]
    return series / np.sqrt(1 / (1 - 0.01))


def simulate_critical_values(n_events: int, seed: int, sims: int = NULL_SIMULATIONS) -> np.ndarray:
    """Simulate the critical values of |t| under the null process of null_noise.

    Arguments: n_events is the number of events; seed is the random seed; sims is the number of simulated series,
    drawn in whole chunks of 100,000 (sims is rounded down to a multiple of 100,000, with at least one chunk).
    Returns: the quantiles of |t| at one minus each tail probability in PROBABILITIES.
    """
    rng = np.random.default_rng(seed)
    draws = np.concatenate([np.abs(statistic(null_noise(rng, _CHUNK, n_events))) for _ in range(max(1, sims // _CHUNK))])
    return np.quantile(draws, 1 - np.asarray(PROBABILITIES))


@functools.lru_cache(maxsize=None)
def _critical_values(n_events: int) -> tuple[float, ...]:
    if n_events in CRITICAL_VALUES:
        return tuple(CRITICAL_VALUES[n_events])
    return tuple(simulate_critical_values(n_events, NULL_SEED + n_events))


def design_p_value(t, draws) -> np.ndarray:
    """Compute the Monte Carlo p-value (1 + draws with |T*| >= |t|) / (R + 1), which is never zero.

    Arguments: t is the statistic; draws are R null draws of |T*|.
    Returns: the two-sided p-value.
    Reference: Phipson and Smyth (2010).
    """
    ordered = np.sort(np.asarray(draws, dtype=float))
    at_least_as_extreme = ordered.size - np.searchsorted(ordered, np.abs(np.asarray(t, dtype=float)), side="left")
    return (1.0 + at_least_as_extreme) / (ordered.size + 1.0)




def simulated_p_value(t, n_events: int, *, alternative: str = "two-sided", reference_abs_t=None) -> np.ndarray:
    """Compute the p-value of t against its simulated null distribution at exactly n_events events.

    Without reference_abs_t, |t| is interpolated in the critical values of CRITICAL_VALUES, with the p-value floored at
    the smallest stored probability. For an n without a stored table (tables are stored for 6, 8 and 21 events), the
    critical values are first simulated by simulate_critical_values with NULL_SIMULATIONS series on the seed
    NULL_SEED + n and kept for the rest of the session, which takes a while the first time. With reference_abs_t, the
    two-sided p-value is design_p_value on those draws. One-sided p-values halve the two-sided tail.
    Arguments: t is the statistic; n_events is the number of out-of-sample events; alternative is "two-sided",
    "greater" or "less"; reference_abs_t is an optional array of null draws of |T*|.
    Returns: the p-value.
    """
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("alternative must be two-sided, greater or less")
    t = np.asarray(t, dtype=float)
    if reference_abs_t is not None:
        two_sided = design_p_value(t, reference_abs_t)
    else:
        critical = np.asarray(_critical_values(n_events))
        two_sided = np.exp(np.interp(np.abs(t), critical[::-1], np.log(np.asarray(PROBABILITIES))[::-1]))
    if alternative == "two-sided":
        return two_sided
    upper = np.where(t >= 0, two_sided / 2, 1 - two_sided / 2)
    return upper if alternative == "greater" else 1 - upper


def giacomini_white(differentials, *, alternative: str = "two-sided", lags: int = DEFAULT_LAG, null: str = "simulated", reference_abs_t=None) -> GiacominiWhite:
    """Run the Giacomini-White test with a constant test function on out-of-sample event-level loss differentials.

    The statistic is t = dbar / sqrt(V / n), with V the Newey-West long-run variance; a numerically zero variance
    raises an error.
    Arguments: differentials are the loss differentials, benchmark minus model, so positive favours the model;
    alternative is "two-sided", "greater" or "less"; lags is the Newey-West lag; null is "simulated" (simulated null
    distribution, see simulated_p_value) or "student" (Student t(n - 1)); reference_abs_t is an optional array of null
    draws of |T*|.
    Returns: a GiacominiWhite.
    Reference: Giacomini and White (2006).
    """
    if null not in ("simulated", "student"):
        raise ValueError('null must be "simulated" or "student"')
    d = np.asarray(differentials, dtype=float)
    if d.ndim != 1 or not np.all(np.isfinite(d)):
        raise ValueError("differentials must be a finite one-dimensional array")
    n = d.size
    if n < 3:
        raise ValueError("at least three out-of-sample events are required")
    variance = float(hac.long_run_variance(d, lags))
    if variance <= n * (np.finfo(float).eps * np.abs(d).max()) ** 2:
        raise ValueError("the variance is numerically zero, so the statistic is undefined")
    t = float(d.mean() / np.sqrt(variance / n))
    p_value = simulated_p_value(t, n, alternative=alternative, reference_abs_t=reference_abs_t) if null == "simulated" else p_value_from_statistic(t, n, alternative=alternative)
    return GiacominiWhite(t, float(p_value), float(d.mean()), variance, n, n - 1, lags, alternative, null)


def supports_stated_sign(p_value: float, estimate: float, stated_sign: int, *, alpha: float = 0.05) -> bool:
    """Decide whether a rejection supports a directional claim.

    It does when the p-value is at most alpha and the estimate has the stated sign.
    Arguments: p_value is the (possibly adjusted) p-value; estimate is the effect estimate; stated_sign is +1 or -1;
    alpha is the significance level (0.05 by default).
    Returns: True when the rejection supports the stated sign.
    """
    if stated_sign not in (-1, 1):
        raise ValueError("stated_sign must be +1 or -1")
    return bool(p_value <= alpha and np.sign(estimate) == stated_sign)
