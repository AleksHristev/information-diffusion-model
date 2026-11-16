"""Newey-West HAC variance with a Bartlett kernel, for a mean and for OLS coefficients."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HacRegression:
    """OLS fit with Newey-West covariance.

    Fields: coefficients, covariance, standard errors, residuals and the number of lags.
    """

    coefficients: np.ndarray
    covariance: np.ndarray
    standard_errors: np.ndarray
    residuals: np.ndarray
    lags: int


def bartlett_weights(lags: int) -> np.ndarray:
    """Return the Bartlett weights w_j = 1 - j / (lags + 1) for j = 1 to lags."""
    if lags < 0:
        raise ValueError("lags must be non-negative")
    return 1.0 - np.arange(1, lags + 1) / (lags + 1)


def long_run_variance(x, lags: int) -> np.ndarray:
    """Compute the Newey-West long-run variance gamma_0 + 2 sum_j w_j gamma_j.

    Arguments: x holds the series on the last axis (other axes are replications); lags is the number of lags.
    Returns: the long-run variance of each series.
    Reference: Newey and West (1987).
    """
    series = np.asarray(x, dtype=float)
    n = series.shape[-1]
    if not 0 <= lags < n:
        raise ValueError("lags must lie between 0 and the series length minus one")
    if not np.all(np.isfinite(series)):
        raise ValueError("series must be finite")
    centred = series - series.mean(axis=-1, keepdims=True)
    variance = np.sum(centred * centred, axis=-1) / n
    for j, weight in enumerate(bartlett_weights(lags), start=1):
        variance = variance + 2.0 * weight * np.sum(centred[..., j:] * centred[..., :-j], axis=-1) / n
    return variance


def variance_of_mean(x, lags: int) -> np.ndarray:
    """Return the Newey-West variance of the sample mean of x: the long-run variance divided by n."""
    return long_run_variance(x, lags) / np.asarray(x).shape[-1]


def regression_covariance(X, residuals, lags: int) -> np.ndarray:
    """Compute the Newey-West covariance (X'X)^-1 S (X'X)^-1 of OLS coefficients, without small-sample correction.

    Arguments: X is the design; residuals are the OLS residuals; lags is the number of lags.
    Returns: the covariance matrix.
    """
    design = np.asarray(X, dtype=float)
    errors = np.asarray(residuals, dtype=float)
    n = design.shape[0]
    if not 0 <= lags < n:
        raise ValueError("lags must lie between 0 and the sample size minus one")
    scores = design * errors[:, None]
    meat = scores.T @ scores
    for j, weight in enumerate(bartlett_weights(lags), start=1):
        cross = scores[j:].T @ scores[:-j]
        meat = meat + weight * (cross + cross.T)
    bread = np.linalg.inv(design.T @ design)
    return bread @ meat @ bread


def ols_newey_west(y, X, lags: int, *, small_sample: bool = False) -> HacRegression:
    """Fit OLS of y on X with Newey-West covariance.

    Arguments: y is the response; X is the design, including any intercept; lags is the number of lags; small_sample
    multiplies the covariance by n / (n - k).
    Returns: a HacRegression.
    """
    response = np.asarray(y, dtype=float)
    design = np.asarray(X, dtype=float)
    if design.ndim != 2 or design.shape[0] != response.shape[0]:
        raise ValueError("X must be n by k and y must have length n")
    n, k = design.shape
    if np.linalg.matrix_rank(design) < k:
        raise ValueError("X is rank deficient")
    coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
    residuals = response - design @ coefficients
    covariance = regression_covariance(design, residuals, lags)
    if small_sample:
        covariance = covariance * n / (n - k)
    return HacRegression(coefficients, covariance, np.sqrt(np.diag(covariance)), residuals, lags)
