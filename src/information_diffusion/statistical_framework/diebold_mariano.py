"""Diebold-Mariano test of equal forecast accuracy, with the Harvey-Leybourne-Newbold correction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class DieboldMariano:
    """Result of a Diebold-Mariano test.

    Fields: statistic, p_value, mean loss differential, variance of the mean, sample size n, forecast horizon, whether
    the corrected statistic was used, and the alternative.
    """

    statistic: float
    p_value: float
    mean_differential: float
    variance_of_mean: float
    n: int
    horizon: int
    corrected: bool
    alternative: str


def _variance_of_mean(differentials: np.ndarray, horizon: int) -> np.ndarray:
    n = differentials.shape[-1]
    centred = differentials - differentials.mean(axis=-1, keepdims=True)
    variance = np.sum(centred * centred, axis=-1) / n
    for k in range(1, horizon):
        variance = variance + 2.0 * np.sum(centred[..., k:] * centred[..., :-k], axis=-1) / n
    return variance / n


def _correction(n: int, horizon: int) -> float:
    return float(np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n))


def dm_statistic(differentials, horizon: int = 1, *, corrected: bool = False) -> np.ndarray:
    """Compute DM = dbar / sqrt(Var(dbar)) with the rectangular-kernel variance over horizon - 1 lags.

    With corrected=True it returns DM* = DM sqrt((n + 1 - 2h + h(h - 1)/n) / n).
    Arguments: differentials holds loss differentials on the last axis (other axes are replications); horizon is the
    forecast horizon h; corrected selects the corrected statistic.
    Returns: the statistic for each replication.
    """
    d = np.asarray(differentials, dtype=float)
    n = d.shape[-1]
    if horizon < 1 or n <= horizon:
        raise ValueError("the horizon must be at least 1 and smaller than the sample")
    with np.errstate(invalid="ignore"):
        statistic = d.mean(axis=-1) / np.sqrt(_variance_of_mean(d, horizon))
    return statistic * _correction(n, horizon) if corrected else statistic


def p_value_from_statistic(statistic, n: int, *, corrected: bool = False, alternative: str = "two-sided") -> np.ndarray:
    """Compute the p-value of a statistic.

    Arguments: statistic is the DM or DM* statistic; n is the sample size; corrected selects Student t(n - 1) instead
    of N(0, 1); alternative is "two-sided", "greater" or "less".
    Returns: the p-value.
    """
    dist = stats.t(n - 1) if corrected else stats.norm()
    statistic = np.asarray(statistic, dtype=float)
    if alternative == "two-sided":
        return 2 * dist.sf(np.abs(statistic))
    if alternative == "greater":
        return dist.sf(statistic)
    if alternative == "less":
        return dist.cdf(statistic)
    raise ValueError("alternative must be two-sided, greater or less")


def diebold_mariano(differentials, horizon: int = 1, *, corrected: bool = False, alternative: str = "two-sided") -> DieboldMariano:
    """Run the Diebold-Mariano test on one series of loss differentials d_t = L_1 - L_2.

    Arguments: differentials is a finite one-dimensional array; horizon is the forecast horizon; corrected applies the
    Harvey-Leybourne-Newbold correction; alternative is "two-sided", "greater" or "less".
    Returns: a DieboldMariano.
    Reference: Diebold and Mariano (1995); Harvey, Leybourne and Newbold (1997).
    """
    d = np.asarray(differentials, dtype=float)
    if d.ndim != 1 or not np.all(np.isfinite(d)):
        raise ValueError("differentials must be a finite one-dimensional array")
    n = d.size
    if horizon < 1 or n <= horizon:
        raise ValueError("the horizon must be at least 1 and smaller than the sample")
    variance = float(_variance_of_mean(d, horizon))
    if not variance > (np.finfo(float).eps * np.abs(d).max()) ** 2:
        raise ValueError("the rectangular-kernel variance is not positive, so the statistic is undefined")
    statistic = float(dm_statistic(d, horizon, corrected=corrected))
    return DieboldMariano(statistic, float(p_value_from_statistic(statistic, n, corrected=corrected, alternative=alternative)), float(d.mean()), variance, n, horizon, corrected, alternative)
