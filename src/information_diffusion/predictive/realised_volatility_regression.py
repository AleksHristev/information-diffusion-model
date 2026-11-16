"""Heterogeneous autoregression of log realised variance over intraday steps, with instrument fixed effects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

_Z = float(norm.ppf(0.975))
_FLOOR = 1e-16


@dataclass(frozen=True)
class RealisedVolatilityFit:
    """Fitted regression log RV_k = alpha_i + sum_m beta_m (mean of log RV over the last m steps) + eps.

    Fields: instrument fixed effects alpha, lag coefficients beta, their standard errors, the residual variance sigma2
    and a convergence flag.
    Reference: Corsi (2009); Andersen, Bollerslev, Diebold and Labys (2003), Econometrica.
    """

    alpha: np.ndarray
    beta: np.ndarray
    alpha_se: np.ndarray
    beta_se: np.ndarray
    sigma2: float
    converged: bool = True


def lagged_design(rv: np.ndarray, lags: tuple[int, ...] = (1, 5, 20)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the lagged design; lags do not cross events and zero variance is floored at 1e-16.

    Arguments: rv is the realised variance, shape (instruments, events, steps); lags are the averaging windows in steps.
    Returns: the response log RV, the instrument fixed-effect design and the lag columns.
    """
    log_rv = np.log(np.maximum(np.asarray(rv, dtype=float), _FLOOR))
    if log_rv.ndim != 3:
        raise ValueError("rv must be (instruments, events, steps)")
    longest = max(lags)
    if log_rv.shape[2] <= longest:
        raise ValueError("need more steps than the longest lag")
    n_instruments, n_events, n_steps = log_rv.shape
    usable = np.arange(longest, n_steps)
    y = log_rv[:, :, usable].ravel()
    instrument = np.repeat(np.arange(n_instruments), n_events * usable.size)
    fixed_effects = np.eye(n_instruments)[instrument]
    columns = []
    for m in lags:
        mean_last_m = np.mean(np.stack([log_rv[:, :, longest - j : n_steps - j] for j in range(1, m + 1)], axis=-1), axis=-1)
        columns.append(mean_last_m.ravel())
    return y, fixed_effects, np.column_stack(columns)


def fit(rv: np.ndarray, lags: tuple[int, ...] = (1, 5, 20)) -> RealisedVolatilityFit:
    """Fit the regression by least squares on all rows with a full set of lags.

    Arguments: rv is the realised variance, shape (instruments, events, steps); lags are the averaging windows in steps.
    Returns: a RealisedVolatilityFit with standard errors from sigma^2 (X'X)^-1.
    """
    y, fixed_effects, components = lagged_design(rv, lags)
    n_instruments = fixed_effects.shape[1]
    design = np.column_stack([fixed_effects, components])
    q, r = np.linalg.qr(design)
    coefficients = np.linalg.solve(r, q.T @ y)
    residual = y - design @ coefficients
    sigma2 = float(residual @ residual / (len(y) - design.shape[1]))
    r_inverse = np.linalg.inv(r)
    covariance = sigma2 * (r_inverse @ r_inverse.T)
    stderr = np.sqrt(np.diag(covariance))
    return RealisedVolatilityFit(alpha=coefficients[:n_instruments], beta=coefficients[n_instruments:], alpha_se=stderr[:n_instruments], beta_se=stderr[n_instruments:], sigma2=sigma2)


def wald_interval(estimate: np.ndarray, stderr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the 95% Wald interval estimate -/+ z stderr as lower and upper bounds."""
    return estimate - _Z * stderr, estimate + _Z * stderr


def simulate(alpha: np.ndarray, beta: np.ndarray, sigma: float, lags: tuple[int, ...], n_events: int, n_steps: int, rng: np.random.Generator, *, burn_in: int = 200) -> np.ndarray:
    """Simulate realised variance from the model after a discarded burn-in.

    Arguments: alpha and beta are the coefficients; sigma is the residual standard deviation; lags are the averaging
    windows; n_events and n_steps set the shape; rng is the random generator; burn_in is the number of burn-in steps.
    Returns: realised variance on the level scale, shape (instruments, events, steps).
    """
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    longest = max(lags)
    total = n_steps + burn_in
    log_rv = np.empty((alpha.size, n_events, total + longest))
    log_rv[:, :, :longest] = (alpha / (1.0 - beta.sum()))[:, None, None] + sigma * rng.standard_normal((alpha.size, n_events, longest))
    for step in range(longest, total + longest):
        means = np.stack([log_rv[:, :, step - m : step].mean(axis=2) for m in lags], axis=-1)
        log_rv[:, :, step] = alpha[:, None] + means @ beta + sigma * rng.standard_normal((alpha.size, n_events))
    return np.exp(log_rv[:, :, longest + burn_in:])
