"""Augmented Dickey-Fuller test on a series observed through a keep mask, with MacKinnon and simulated critical values."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_MAX_LAGS = 3
LEVELS = (0.01, 0.05, 0.10)
MACKINNON_2010_CONSTANT = {0.01: (-3.43035, -6.5393, -16.786, -79.433), 0.05: (-2.86154, -2.8903, -4.234, -40.040), 0.10: (-2.56677, -1.5384, -2.809, 0.0)}
MIN_ROWS = 20


def mackinnon_critical_value(level: float, rows: int) -> float:
    """Response-surface critical value c_p(T) = b_inf + b_1 / T + b_2 / T^2 + b_3 / T^3 for a constant and no trend.

    Arguments: level is 0.01, 0.05 or 0.10; rows is T, the number of regression rows.
    Returns: the critical value, which assumes a contiguous sample.
    Reference: MacKinnon (2010), Critical values for cointegration tests, Queen's Economics Department.
    """
    b_inf, b1, b2, b3 = MACKINNON_2010_CONSTANT[level]
    inverse = 1.0 / rows
    return b_inf + b1 * inverse + b2 * inverse**2 + b3 * inverse**3


def run_position(keep: np.ndarray) -> np.ndarray:
    """Number of consecutive kept points ending at each grid index of the boolean mask keep (0 where the point is dropped)."""
    keep = np.asarray(keep, dtype=bool)
    index = np.arange(keep.size)
    last_dropped = np.maximum.accumulate(np.where(keep, -1, index))
    return index - last_dropped


def valid_rows(keep: np.ndarray, lags: int) -> np.ndarray:
    """Grid indices t whose regression row at the given lag order uses only kept points t - lags - 1, ..., t, given the boolean mask keep."""
    return np.flatnonzero(run_position(keep) >= lags + 2)


@dataclass(frozen=True)
class AdfResult:
    """ADF regression result. Fields: statistic tau = rho_hat / se(rho_hat); rho, the coefficient on s_{t-1}; lags, the lag order; rows, the number of regression rows; intercept."""

    statistic: float
    rho: float
    lags: int
    rows: int
    intercept: float

    def mackinnon(self, level: float = 0.05) -> float:
        """MacKinnon critical value at level for this result's number of rows."""
        return mackinnon_critical_value(level, self.rows)


def _design(series: np.ndarray, rows: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    difference = np.empty_like(series)
    difference[0] = np.nan
    difference[1:] = np.diff(series)
    columns = [np.ones(rows.size), series[rows - 1]]
    columns += [difference[rows - j] for j in range(1, lags + 1)]
    return np.column_stack(columns), difference[rows]


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    gram = x.T @ x
    coefficients = np.linalg.solve(gram, x.T @ y)
    residual = y - x @ coefficients
    return coefficients, float(residual @ residual), gram


def adf_at(series: np.ndarray, rows: np.ndarray, lags: int) -> AdfResult:
    """ADF regression ds_t = a + rho s_{t-1} + sum_{j=1}^k g_j ds_{t-j} + e_t at a fixed lag order, with tau = rho_hat / se(rho_hat).

    A unit root (rho = 0) is rejected at level p when tau is below the critical value c_p.
    Arguments: series is the series on its regular grid; rows the grid indices used as regression rows; lags is k.
    Returns: an AdfResult; raises ValueError when there are too few rows.
    Reference: Dickey and Fuller (1979), JASA; Said and Dickey (1984), Biometrika.
    """
    x, y = _design(np.asarray(series, dtype=float), rows, lags)
    coefficients, rss, gram = _ols(x, y)
    degrees = rows.size - x.shape[1]
    if degrees <= 0:
        raise ValueError("too few rows for the ADF regression")
    sigma2 = rss / degrees
    variance = sigma2 * np.linalg.inv(gram)[1, 1]
    return AdfResult(float(coefficients[1] / np.sqrt(variance)), float(coefficients[1]), lags, int(rows.size), float(coefficients[0]))


@dataclass(frozen=True)
class MaskPlan:
    """Valid regression rows at each lag order for one keep mask. Fields: rows, one index array per lag order 0, ..., max_lags; max_lags."""

    rows: tuple[np.ndarray, ...]
    max_lags: int

    @property
    def usable(self) -> bool:
        """True when at least 20 rows are valid at the largest lag order."""
        return self.rows[self.max_lags].size >= MIN_ROWS


def plan(keep: np.ndarray, max_lags: int = DEFAULT_MAX_LAGS) -> MaskPlan:
    """MaskPlan of the valid rows at each lag order 0, ..., max_lags (default 3) for the boolean mask keep."""
    position = run_position(keep)
    return MaskPlan(tuple(np.flatnonzero(position >= lags + 2) for lags in range(max_lags + 1)), max_lags)


def select_lags_planned(series: np.ndarray, mask_plan: MaskPlan) -> int:
    """Lag order minimising BIC = n log(RSS / n) + (k + 2) log n on the common rows valid at the largest lag order.

    Arguments: series is the series on its grid; mask_plan the MaskPlan of its keep mask.
    Returns: the chosen lag order.
    Reference: Schwarz (1978), Annals of Statistics.
    """
    common = mask_plan.rows[mask_plan.max_lags]
    if common.size < MIN_ROWS:
        raise ValueError("too few rows to select the lag order")
    best, best_lags = np.inf, None
    for lags in range(mask_plan.max_lags + 1):
        x, y = _design(series, common, lags)
        try:
            _, rss, _ = _ols(x, y)
        except np.linalg.LinAlgError:
            continue
        criterion = common.size * np.log(rss / common.size) + x.shape[1] * np.log(common.size)
        if criterion < best:
            best, best_lags = criterion, lags
    if best_lags is None:
        raise np.linalg.LinAlgError("no lag order gives a non-singular ADF regression")
    return best_lags


def adf_planned(series: np.ndarray, mask_plan: MaskPlan, lags: int | None = None) -> AdfResult:
    """ADF test with a precomputed MaskPlan: BIC lag selection (or the fixed lags), then the regression on all rows valid at that order.

    Arguments: series is the series on its grid; mask_plan the MaskPlan of its keep mask; lags a fixed lag order, or
    None for BIC selection up to mask_plan.max_lags.
    Returns: an AdfResult; raises ValueError when fewer than 20 rows are valid.
    """
    series = np.asarray(series, dtype=float)
    chosen = select_lags_planned(series, mask_plan) if lags is None else lags
    rows = mask_plan.rows[chosen]
    if rows.size < MIN_ROWS:
        raise ValueError("too few rows for the ADF regression")
    return adf_at(series, rows, chosen)


def select_lags(series: np.ndarray, keep: np.ndarray, max_lags: int = DEFAULT_MAX_LAGS) -> int:
    """BIC lag order of series observed through the boolean mask keep, on the rows valid at max_lags (default 3)."""
    return select_lags_planned(np.asarray(series, dtype=float), plan(keep, max_lags))


def adf(series: np.ndarray, keep: np.ndarray | None = None, max_lags: int = DEFAULT_MAX_LAGS, lags: int | None = None) -> AdfResult:
    """Augmented Dickey-Fuller test with a constant and no trend on a gridded series, using only kept, finite points.

    A regression row at t needs t, t - 1, ..., t - k - 1 all kept, so no difference or lag spans a dropped point.
    Arguments: series is the series on a regular grid; keep an optional boolean mask; max_lags the largest lag order
    for BIC selection (default 3); lags a fixed lag order instead of selection.
    Returns: an AdfResult; raises ValueError when fewer than 20 rows are valid.
    """
    series = np.asarray(series, dtype=float)
    keep = np.isfinite(series) if keep is None else np.asarray(keep, dtype=bool) & np.isfinite(series)
    return adf_planned(series, plan(keep, max(max_lags, lags or 0)), lags)


def simulated_null(keep: np.ndarray, draws: int, rng: np.random.Generator, max_lags: int = DEFAULT_MAX_LAGS) -> np.ndarray:
    """Simulated null distribution of tau under the mask: ADF statistics of Gaussian random walks observed through keep.

    The same BIC lag selection is applied to each walk. Under a unit root the level after each gap carries the increments
    of the gap, so the distribution depends on the gap pattern and can differ from MacKinnon's values.
    Arguments: keep is the boolean mask; draws the number of walks; rng the generator; max_lags the largest lag order.
    Returns: the simulated statistics.
    """
    keep = np.asarray(keep, dtype=bool)
    mask_plan = plan(keep, max_lags)
    statistics = np.empty(draws)
    for draw in range(draws):
        statistics[draw] = adf_planned(np.cumsum(rng.standard_normal(keep.size)), mask_plan).statistic
    return statistics


def simulated_critical_value(null: np.ndarray, level: float = 0.05) -> float:
    """Lower-tail critical value: the level quantile (default 0.05) of the simulated null statistics."""
    return float(np.quantile(np.asarray(null, dtype=float), level))


def simulated_p_value(statistic: float, null: np.ndarray) -> float:
    """Monte Carlo p-value (1 + #{tau* <= tau}) / (R + 1) of statistic tau against R simulated null statistics."""
    null = np.asarray(null, dtype=float)
    return float((1 + np.sum(null <= statistic)) / (null.size + 1))
