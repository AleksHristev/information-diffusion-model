"""Autoregressive Poisson count model on lagged counts around a spline baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from information_diffusion.epidemic.spline_baseline import SplineBaseline
from information_diffusion.predictive.spline_poisson import DEFAULT_STEP_SECONDS, PoissonFit, StepQuadrature, fit_with_baseline, step_quadrature


@dataclass(frozen=True)
class AutoregressivePoissonFit:
    """Fitted log-linear Poisson autoregression mu = S_k(gamma) exp(alpha_i + sum_l phi_l log(1 + y_{k-l})).

    Fields: instrument fixed effects alpha, lag coefficients phi, baseline spline coefficients gamma,
    their standard errors, the step quadrature of the baseline and the underlying penalised Poisson fit.
    Reference: Zeger and Qaqish (1988); Fokianos and Tjostheim (2011).
    """

    alpha: np.ndarray
    phi: np.ndarray
    gamma: np.ndarray
    alpha_se: np.ndarray
    phi_se: np.ndarray
    gamma_se: np.ndarray
    quadrature: StepQuadrature
    fit: PoissonFit

    def rate(self, instrument: np.ndarray, step: np.ndarray, lags: np.ndarray) -> np.ndarray:
        """Compute the conditional mean count.

        Arguments: instrument and step are integer index arrays; lags holds the raw counts at lags 1 to p, shape (n, p).
        Returns: the conditional mean count of each row.
        """
        return self.quadrature.integrals(self.gamma)[np.asarray(step, dtype=int)] * np.exp(self.alpha[np.asarray(instrument, dtype=int)] + np.log1p(np.asarray(lags, dtype=float)) @ self.phi)


def lagged_design(counts: np.ndarray, n_lags: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the lagged design of the autoregression; lags do not cross events.

    Arguments: counts is an array (instruments, events, steps); n_lags is the number of lags p.
    Returns: the response y, the instrument fixed-effect design, the log(1 + lagged count) columns and the step index.
    """
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 3:
        raise ValueError("counts must be (instruments, events, steps)")
    if not 1 <= n_lags < counts.shape[2]:
        raise ValueError("need 1 <= n_lags < number of steps")
    n_instruments, n_events, n_steps = counts.shape
    y = counts[:, :, n_lags:].ravel()
    instrument = np.repeat(np.arange(n_instruments), n_events * (n_steps - n_lags))
    fixed_effects = np.eye(n_instruments)[instrument]
    lag_columns = [np.log1p(counts[:, :, n_lags - l : n_steps - l].ravel()) for l in range(1, n_lags + 1)]
    steps = np.tile(np.arange(n_lags, n_steps), n_instruments * n_events)
    return y, fixed_effects, np.column_stack(lag_columns), steps


def fit(counts: np.ndarray, baseline: SplineBaseline, n_lags: int = 3, *, step_seconds: float = DEFAULT_STEP_SECONDS) -> AutoregressivePoissonFit:
    """Fit the model by penalised maximum likelihood on all rows with a full set of lags.

    Arguments: counts is an array (instruments, events, steps) with one column per step of the baseline's window;
    baseline is the spline baseline whose knots and penalty are used; n_lags is the number of lags (3 by default);
    step_seconds is the step length in seconds.
    Returns: an AutoregressivePoissonFit.
    """
    quadrature = step_quadrature(baseline, step_seconds)
    if np.asarray(counts).shape[2] != quadrature.n_steps:
        raise ValueError("the counts do not have one column per forecast step of the baseline's window")
    y, fixed_effects, lags, steps = lagged_design(counts, n_lags)
    n_instruments = fixed_effects.shape[1]
    result = fit_with_baseline(y, np.column_stack([fixed_effects, lags]), steps, quadrature, baseline.penalty_matrix[1:, 1:])
    stderr = np.sqrt(np.diag(result.covariance))
    split = n_instruments + n_lags
    return AutoregressivePoissonFit(alpha=result.coefficients[:n_instruments], phi=result.coefficients[n_instruments:split], gamma=result.coefficients[split:], alpha_se=stderr[:n_instruments], phi_se=stderr[n_instruments:split], gamma_se=stderr[split:], quadrature=quadrature, fit=result)


def simulate(alpha: np.ndarray, gamma: np.ndarray, phi: np.ndarray, baseline: SplineBaseline, n_events: int, rng: np.random.Generator, *, step_seconds: float = DEFAULT_STEP_SECONDS, burn_in: int = 200) -> np.ndarray:
    """Simulate counts from the model after a discarded burn-in at the first step's baseline.

    Arguments: alpha, gamma and phi are the model coefficients; baseline is the spline baseline; n_events is the number
    of events; rng is the random generator; step_seconds is the step length in seconds; burn_in is the number of
    burn-in steps (200 by default).
    Returns: simulated counts, shape (instruments, events, steps).
    """
    alpha = np.asarray(alpha, dtype=float)
    phi = np.asarray(phi, dtype=float)
    n_lags = phi.size
    quadrature = step_quadrature(baseline, step_seconds)
    integrals = quadrature.integrals(np.asarray(gamma, dtype=float))
    log_baseline = np.concatenate((np.full(burn_in, np.log(integrals[0])), np.log(integrals)))
    total = log_baseline.size
    long_run = (alpha + log_baseline[0]) / (1.0 - phi.sum())
    previous = np.maximum(np.expm1(np.full((alpha.size, n_events, n_lags), np.maximum(long_run, 0.0)[:, None, None])), 0.0)
    counts = np.empty((alpha.size, n_events, total))
    for step in range(total):
        log_mean = alpha[:, None] + log_baseline[step] + np.log1p(previous) @ phi
        counts[:, :, step] = rng.poisson(np.exp(np.clip(log_mean, -30.0, 30.0)))
        previous = np.concatenate([counts[:, :, step][:, :, None], previous[:, :, :-1]], axis=2)
    return counts[:, :, burn_in:]
