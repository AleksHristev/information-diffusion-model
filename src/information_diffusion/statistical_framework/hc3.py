"""Ordinary least squares with HC3 heteroscedasticity-consistent standard errors and linear contrasts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Regression:
    """OLS fit with HC3 covariance.

    Fields: coefficients, HC3 covariance, residuals, number of observations n and number of columns k.
    """

    coefficients: np.ndarray
    covariance: np.ndarray
    residuals: np.ndarray
    n: int
    k: int


@dataclass(frozen=True)
class Contrast:
    """Test of a linear combination of coefficients against zero.

    Fields: estimate, standard error, t-statistic, two-sided p-value and degrees of freedom n - k.
    """

    estimate: float
    stderr: float
    t: float
    pvalue: float
    degrees_of_freedom: int


def ols_hc3(y, X) -> Regression:
    """Fit OLS with the HC3 covariance (X'X)^-1 X' diag(r_i^2 / (1 - h_i)^2) X (X'X)^-1.

    Arguments: y is the response; X is the design, of full column rank with more rows than columns.
    Returns: a Regression.
    Reference: MacKinnon and White (1985); Long and Ervin (2000).
    """
    y, X = np.asarray(y, dtype=float), np.asarray(X, dtype=float)
    n, k = X.shape
    if n <= k or np.linalg.matrix_rank(X) < k:
        raise ValueError("the design must have full column rank and more rows than columns")
    bread = np.linalg.inv(X.T @ X)
    coefficients = bread @ X.T @ y
    residuals = y - X @ coefficients
    leverage = np.einsum("ij,jk,ik->i", X, bread, X)
    if np.any(leverage >= 1.0 - 1e-10):
        raise ValueError("an observation has leverage one, so HC3 is undefined")
    meat = (X * (residuals**2 / (1.0 - leverage) ** 2)[:, None]).T @ X
    return Regression(coefficients, bread @ meat @ bread, residuals, n, k)


def contrast(regression: Regression, weights) -> Contrast:
    """Test a linear combination c'b of the coefficients against zero, with variance c'Vc and Student t(n - k).

    Arguments: regression is the Regression; weights is the vector c.
    Returns: a Contrast.
    """
    c = np.asarray(weights, dtype=float)
    estimate = float(c @ regression.coefficients)
    stderr = float(np.sqrt(c @ regression.covariance @ c))
    t = estimate / stderr
    degrees = regression.n - regression.k
    return Contrast(estimate, stderr, t, float(2.0 * stats.t.sf(abs(t), degrees)), degrees)
