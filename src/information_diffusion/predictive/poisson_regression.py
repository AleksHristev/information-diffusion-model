"""Poisson regression with a log link, fitted by Newton-Raphson."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.special import gammaln


@dataclass(frozen=True)
class PoissonFit:
    """Fitted Poisson regression.

    Fields: coefficients, covariance (inverse information at the estimate), converged, iterations, log_likelihood,
    deviance and the number of observations n.
    """

    coefficients: np.ndarray
    covariance: np.ndarray
    converged: bool
    iterations: int
    log_likelihood: float
    deviance: float
    n: int

    @property
    def standard_errors(self) -> np.ndarray:
        """Standard errors of the coefficients, the square roots of the covariance diagonal."""
        return np.sqrt(np.diag(self.covariance))

    def interval(self, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """Compute Wald intervals for the coefficients.

        Arguments: level is the two-sided coverage.
        Returns: the lower and upper bounds.
        """
        z = stats.norm.ppf(0.5 + level / 2)
        return self.coefficients - z * self.standard_errors, self.coefficients + z * self.standard_errors

    def predict(self, X, offset=None) -> np.ndarray:
        """Compute the fitted means exp(offset + X b).

        Arguments: X is the design matrix; offset is an optional offset.
        Returns: the predicted means.
        """
        X = np.asarray(X, dtype=float)
        eta = X @ self.coefficients + (0.0 if offset is None else np.asarray(offset, dtype=float))
        return np.exp(eta)


def log_likelihood(y, mu) -> float:
    """Poisson log-likelihood sum(y log mu - mu - log y!) of counts y at means mu."""
    y, mu = np.asarray(y, dtype=float), np.asarray(mu, dtype=float)
    return float(np.sum(y * np.log(mu) - mu - gammaln(y + 1.0)))


def deviance(y, mu) -> float:
    """Poisson deviance 2 sum(y log(y / mu) - (y - mu)) of counts y at means mu, with y log y = 0 at y = 0."""
    y, mu = np.asarray(y, dtype=float), np.asarray(mu, dtype=float)
    ratio = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
    return float(2.0 * np.sum(ratio - (y - mu)))


def fit(y, X, offset=None, *, tol: float = 1e-9, max_iter: int = 100) -> PoissonFit:
    """Fit a Poisson regression by Newton-Raphson with step halving.

    Arguments: y is the vector of counts; X is the design matrix, of full column rank; offset is an optional offset;
    tol is the convergence threshold on the largest absolute step; max_iter is the iteration limit.
    Returns: a PoissonFit.
    Reference: McCullagh and Nelder (1989), Generalized Linear Models, 2nd edition.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    o = np.zeros(y.size) if offset is None else np.asarray(offset, dtype=float)
    if X.ndim != 2 or X.shape[0] != y.size or o.size != y.size:
        raise ValueError("y, X and offset must align")
    if np.any(y < 0) or not np.all(np.isfinite(y)) or not np.all(np.isfinite(X)) or not np.all(np.isfinite(o)):
        raise ValueError("counts must be non-negative and every input finite")
    b = np.linalg.lstsq(X, np.log(y + 0.5) - o, rcond=None)[0]
    current = log_likelihood(y, np.exp(np.clip(o + X @ b, -700, 700)))
    converged = False
    for iteration in range(1, max_iter + 1):
        mu = np.exp(np.clip(o + X @ b, -700, 700))
        information = X.T @ (X * mu[:, None])
        step = np.linalg.solve(information, X.T @ (y - mu))
        scale = 1.0
        while True:
            candidate = b + scale * step
            value = log_likelihood(y, np.exp(np.clip(o + X @ candidate, -700, 700)))
            if value >= current - 1e-12 * abs(current) or scale < 1e-8:
                break
            scale /= 2.0
        b, current = candidate, value
        if np.max(np.abs(scale * step)) < tol:
            converged = True
            break
    mu = np.exp(np.clip(o + X @ b, -700, 700))
    covariance = np.linalg.inv(X.T @ (X * mu[:, None]))
    return PoissonFit(b, covariance, converged, iteration, log_likelihood(y, mu), deviance(y, mu), int(y.size))
