"""Freedman-Lane permutation test of a regression coefficient with nuisance covariates."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

DEFAULT_DRAWS = 9999
EXACT_LIMIT = 9


@dataclass(frozen=True)
class FreedmanLane:
    """Result of a Freedman-Lane test.

    Fields: the estimated coefficient, its t-statistic, the permutation p-value, the number of permutations, whether
    they were fully enumerated, the alternative and the residual degrees of freedom.
    """

    coefficient: float
    statistic: float
    p_value: float
    draws: int
    exact: bool
    alternative: str
    degrees_of_freedom: int


def _design(x: np.ndarray, nuisance, add_intercept: bool) -> np.ndarray:
    n = x.shape[0]
    columns = [] if nuisance is None else [np.asarray(nuisance, dtype=float).reshape(n, -1)]
    if add_intercept:
        columns.insert(0, np.ones((n, 1)))
    if not columns:
        raise ValueError("the reduced model must contain at least one column")
    return np.column_stack(columns)


def _t_statistics(all_y: np.ndarray, Z: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    n = Z.shape[0]
    X = np.column_stack([Z, x])
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError("the design [Z, x] is rank deficient")
    df = n - X.shape[1]
    if df < 1:
        raise ValueError("the full model leaves no residual degrees of freedom")
    Q_z, _ = np.linalg.qr(Z)
    Q_full, _ = np.linalg.qr(X)
    w = x - Q_z @ (Q_z.T @ x)
    ww = float(w @ w)
    slopes = w @ all_y / ww
    residuals = all_y - Q_full @ (Q_full.T @ all_y)
    rss = np.sum(residuals * residuals, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = slopes / np.sqrt(rss / df / ww)
    return t, slopes, df


def freedman_lane(y, x, nuisance=None, *, draws: int = DEFAULT_DRAWS, rng: np.random.Generator | None = None, alternative: str = "two-sided",
                  add_intercept: bool = True, permutations=None, exact: bool = False) -> FreedmanLane:
    """Test b = 0 in y = x b + Z g + e by permuting the residuals of the reduced model of y on Z.

    Each permuted response f + P r is refitted on [Z, x] and the t-statistic of x is compared with the observed one;
    p = (1 + hits) / (1 + draws) for random permutations and hits / n! under full enumeration.
    Arguments: y is the response; x is the tested regressor; nuisance is the nuisance matrix Z; draws is the number of
    random permutations; rng is the random generator; alternative is "two-sided", "greater" or "less"; add_intercept
    adds an intercept to Z; permutations is an optional (draws, n) integer array of explicit permutations; exact
    enumerates all n! permutations (n at most 9).
    Returns: a FreedmanLane.
    Reference: Freedman and Lane (1983); Winkler, Ridgway, Webster, Smith and Nichols (2014).
    """
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("alternative must be two-sided, greater or less")
    response = np.asarray(y, dtype=float)
    tested = np.asarray(x, dtype=float)
    n = response.shape[0]
    if response.ndim != 1 or tested.shape != (n,) or not (np.all(np.isfinite(response)) and np.all(np.isfinite(tested))):
        raise ValueError("y and x must be finite one-dimensional arrays of equal length")
    Z = _design(tested, nuisance, add_intercept)
    Q_z, _ = np.linalg.qr(Z)
    fitted = Q_z @ (Q_z.T @ response)
    residual = response - fitted
    if np.sum(residual**2) <= n * (np.finfo(float).eps * np.abs(response).max()) ** 2:
        raise ValueError("the reduced model fits exactly, so there is nothing to permute")
    if exact:
        if n > EXACT_LIMIT:
            raise ValueError(f"full enumeration is limited to n <= {EXACT_LIMIT}")
        index = np.array(list(itertools.permutations(range(n))), dtype=np.intp)
    elif permutations is not None:
        index = np.asarray(permutations, dtype=np.intp)
        if index.ndim != 2 or index.shape[1] != n:
            raise ValueError("permutations must be an array of shape (draws, n)")
    else:
        if rng is None:
            raise ValueError("provide rng, permutations or exact=True")
        if draws < 1:
            raise ValueError("draws must be positive")
        index = rng.permuted(np.tile(np.arange(n), (draws, 1)), axis=1)
    all_y = np.vstack([response, fitted + residual[index]]).T
    t, slopes, df = _t_statistics(all_y, Z, tested)
    if not np.isfinite(t[0]):
        raise ValueError("the observed t-statistic is not finite")
    observed, permuted = t[0], t[1:]
    tolerance = 1e-12 * max(1.0, abs(observed))
    if alternative == "two-sided":
        hits = np.abs(permuted) >= abs(observed) - tolerance
    elif alternative == "greater":
        hits = permuted >= observed - tolerance
    else:
        hits = permuted <= observed + tolerance
    count = int(np.sum(hits))
    total = permuted.size
    if exact:
        p = count / total
        if total != math.factorial(n):
            raise AssertionError("full enumeration must contain n! permutations")
    else:
        p = (count + 1) / (total + 1)
    return FreedmanLane(float(slopes[0]), float(observed), float(p), int(total), bool(exact), alternative, int(df))
