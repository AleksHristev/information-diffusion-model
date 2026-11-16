"""Inhomogeneous Poisson count model whose rate is a penalised B-spline baseline integrated over each step."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.special import gammaln
from scipy.stats import norm

from information_diffusion.epidemic.spline_baseline import GAUSS_NODES, SplineBaseline

_Z = float(norm.ppf(0.975))
DEFAULT_STEP_SECONDS = 30.0


def deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Compute the Poisson deviance 2 sum(y log(y / mu) - (y - mu)) of counts y at means mu, with the y = 0 term 2 mu."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    positive = y > 0
    term = -(y - mu)
    term[positive] += y[positive] * np.log(y[positive] / mu[positive])
    return float(2.0 * term.sum())


@dataclass(frozen=True)
class StepQuadrature:
    """Gauss-Legendre quadrature nodes of a spline baseline, grouped by forecast step.

    Fields: design holds the basis values at the nodes without the first basis; weights are the node weights; starts is
    the first node of each step; n_steps is the number of steps.
    """

    design: np.ndarray
    weights: np.ndarray
    starts: np.ndarray
    n_steps: int

    def integrals(self, gamma: np.ndarray) -> np.ndarray:
        """Compute S_k(gamma), the integral of exp(sum_j gamma_j B_j(t)) over each step k.

        Arguments: gamma is the vector of spline coefficients.
        Returns: the integral of each step.
        """
        return np.add.reduceat(self.weights * np.exp(self.design @ gamma), self.starts)

    def derivatives(self, gamma: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the step integrals and their gradients.

        Arguments: gamma is the vector of spline coefficients.
        Returns: S_k for each step, G_k = dS_k/dgamma for each step, and the weighted node values w exp(D gamma).
        """
        values = self.weights * np.exp(self.design @ gamma)
        return np.add.reduceat(values, self.starts), np.add.reduceat(values[:, None] * self.design, self.starts, axis=0), values


def step_quadrature(baseline: SplineBaseline, step_seconds: float = DEFAULT_STEP_SECONDS) -> StepQuadrature:
    """Build the step quadrature of a baseline: its Gauss-Legendre rule on each piece between knots and step edges.

    Arguments: baseline is the spline baseline; step_seconds is the step length in seconds, which must divide the window.
    Returns: a StepQuadrature.
    Reference: De Boor (2001), A Practical Guide to Splines.
    """
    n_steps = int(round(baseline.window / step_seconds))
    if abs(n_steps * step_seconds - baseline.window) > 1e-9:
        raise ValueError("the window must be a whole number of steps")
    edges = np.arange(n_steps + 1) * step_seconds
    breaks = np.unique(np.concatenate((np.unique(baseline.knots), edges)))
    nodes, weights = np.polynomial.legendre.leggauss(GAUSS_NODES)
    lengths = np.diff(breaks)
    points = (breaks[:-1, None] + 0.5 * lengths[:, None] * (nodes + 1.0)).ravel()
    node_weights = (0.5 * lengths[:, None] * weights).ravel()
    step = np.minimum(np.floor(points / step_seconds).astype(np.int64), n_steps - 1)
    return StepQuadrature(baseline.design(points).toarray()[:, 1:], node_weights, np.searchsorted(step, np.arange(n_steps)), n_steps)


@dataclass(frozen=True)
class PoissonFit:
    """Fitted penalised Poisson model with a spline baseline, returned by fit_with_baseline.

    This is a different class from poisson_regression.PoissonFit. Fields: coefficients (beta, gamma), covariance
    (inverse observed information), deviance, log_likelihood, the convergence flag and the number of iterations.
    """

    coefficients: np.ndarray
    covariance: np.ndarray
    deviance: float
    log_likelihood: float
    converged: bool
    iterations: int


def _penalised(y, offset, x, step, quadrature: StepQuadrature, penalty: np.ndarray, beta, gamma) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    exposure = np.exp(offset + x @ beta)
    integrals = quadrature.integrals(gamma)
    mean = exposure * integrals[step]
    with np.errstate(divide="ignore", invalid="ignore"):
        value = float(np.sum(np.where(y > 0.0, y * np.log(mean), 0.0)) - mean.sum() - 0.5 * gamma @ penalty @ gamma)
    return value, mean, exposure, integrals


def _score_and_information(y, x, step, quadrature: StepQuadrature, penalty: np.ndarray, gamma, mean, exposure):
    p, q = x.shape[1], gamma.size
    integrals, gradients, values = quadrature.derivatives(gamma)
    step_y = np.bincount(step, weights=y, minlength=quadrature.n_steps)
    step_exposure = np.bincount(step, weights=exposure, minlength=quadrature.n_steps)
    score_beta = x.T @ (y - mean)
    multiplier = step_y / integrals - step_exposure
    score_gamma = multiplier @ gradients - penalty @ gamma
    bb = (x.T * mean) @ x
    bg = (x.T * exposure) @ gradients[step]
    expected_gg = (gradients.T * (step_exposure / integrals)) @ gradients + penalty
    node_multiplier = np.repeat(multiplier, np.diff(np.concatenate((quadrature.starts, [values.size]))))
    observed_gg = (gradients.T * (step_y / integrals ** 2)) @ gradients - (quadrature.design.T * (node_multiplier * values)) @ quadrature.design + penalty
    build = lambda gg: np.block([[bb, bg], [bg.T, gg]])
    return np.concatenate((score_beta, score_gamma)), build(observed_gg), build(expected_gg)


def fit_with_baseline(y: np.ndarray, x: np.ndarray, step: np.ndarray, quadrature: StepQuadrature, penalty: np.ndarray, *, offset: np.ndarray | None = None, max_iterations: int = 200, tolerance: float = 1e-9) -> PoissonFit:
    """Maximise sum(y log m - m) - 0.5 gamma' P gamma with mean m = exp(offset + x beta) S_step(gamma).

    Uses Newton steps on the observed information with step halving, falling back to Fisher scoring when the observed
    information is not positive definite.
    Arguments: y is the vector of counts; x is the log-linear design; step is the step of each row; quadrature is the
    StepQuadrature; penalty is the penalty matrix P of gamma; offset is an optional offset; max_iterations and tolerance
    control the iteration.
    Returns: a PoissonFit with coefficients (beta, gamma).
    Reference: Eilers and Marx (1996); McCullagh and Nelder (1989).
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    step = np.asarray(step, dtype=np.int64)
    offset = np.zeros(y.size) if offset is None else np.asarray(offset, dtype=float)
    p, q = x.shape[1], penalty.shape[0]
    gamma = np.zeros(q)
    integrals = quadrature.integrals(gamma)
    beta = np.linalg.lstsq(x, np.log(y + 0.5) - offset - np.log(integrals[step]), rcond=None)[0]
    value, mean, exposure, integrals = _penalised(y, offset, x, step, quadrature, penalty, beta, gamma)
    converged = False
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        score, observed, expected = _score_and_information(y, x, step, quadrature, penalty, gamma, mean, exposure)
        direction = None
        for information in (observed, expected):
            try:
                direction = cho_solve(cho_factor(information), score)
                break
            except LinAlgError:
                continue
        if direction is None:
            break
        decrement = float(score @ direction)
        if decrement < tolerance:
            converged = True
            break
        size = 1.0
        for _ in range(40):
            beta_new, gamma_new = beta + size * direction[:p], gamma + size * direction[p:]
            new_value, new_mean, new_exposure, new_integrals = _penalised(y, offset, x, step, quadrature, penalty, beta_new, gamma_new)
            if np.isfinite(new_value) and new_value >= value - 1e-12 * max(abs(value), 1.0):
                break
            size *= 0.5
        else:
            break
        beta, gamma, value, mean, exposure, integrals = beta_new, gamma_new, new_value, new_mean, new_exposure, new_integrals
    _, observed, _ = _score_and_information(y, x, step, quadrature, penalty, gamma, mean, exposure)
    covariance = np.linalg.inv(observed)
    log_likelihood = float(np.sum(np.where(y > 0.0, y * np.log(mean), 0.0)) - mean.sum() - gammaln(y + 1.0).sum())
    return PoissonFit(coefficients=np.concatenate((beta, gamma)), covariance=covariance, deviance=deviance(y, mean), log_likelihood=log_likelihood, converged=converged, iterations=iteration)


@dataclass(frozen=True)
class SplinePoissonFit:
    """Fitted model y ~ Poisson(exp(alpha_i) S_k(gamma)).

    Fields: instrument fixed effects alpha, spline coefficients gamma, their standard errors, the step quadrature and the
    underlying PoissonFit.
    """

    alpha: np.ndarray
    gamma: np.ndarray
    alpha_se: np.ndarray
    gamma_se: np.ndarray
    quadrature: StepQuadrature
    fit: PoissonFit

    def rate(self, instrument: np.ndarray, step: np.ndarray) -> np.ndarray:
        """Compute the mean count per step.

        Arguments: instrument and step are integer index arrays.
        Returns: the mean count of each row.
        """
        return np.exp(self.alpha[np.asarray(instrument, dtype=int)]) * self.quadrature.integrals(self.gamma)[np.asarray(step, dtype=int)]


def fit(counts: np.ndarray, baseline: SplineBaseline, *, step_seconds: float = DEFAULT_STEP_SECONDS) -> SplinePoissonFit:
    """Fit the model on counts summed over events, with a log event-count offset.

    Arguments: counts is an array (instruments, events, steps) with one column per step of the baseline's window;
    baseline is the spline baseline whose knots and penalty are used; step_seconds is the step length in seconds.
    Returns: a SplinePoissonFit.
    Reference: Hawkes (1971), Biometrika (the baseline is the Hawkes model's); McCullagh and Nelder (1989), Generalized
    Linear Models.
    """
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 3:
        raise ValueError("counts must be (instruments, events, steps)")
    n_instruments, n_events, n_steps = counts.shape
    quadrature = step_quadrature(baseline, step_seconds)
    if n_steps != quadrature.n_steps:
        raise ValueError("the counts do not have one column per forecast step of the baseline's window")
    instruments = np.repeat(np.arange(n_instruments), n_steps)
    steps = np.tile(np.arange(n_steps), n_instruments)
    fixed_effects = np.eye(n_instruments)[instruments]
    total = counts.sum(axis=1).ravel()
    result = fit_with_baseline(total, fixed_effects, steps, quadrature, baseline.penalty_matrix[1:, 1:], offset=np.full(total.size, np.log(n_events)))
    stderr = np.sqrt(np.diag(result.covariance))
    return SplinePoissonFit(alpha=result.coefficients[:n_instruments], gamma=result.coefficients[n_instruments:], alpha_se=stderr[:n_instruments], gamma_se=stderr[n_instruments:], quadrature=quadrature, fit=result)


def wald_interval(estimate: np.ndarray, stderr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the normal 95% interval estimate -/+ z stderr as lower and upper bounds."""
    return estimate - _Z * stderr, estimate + _Z * stderr


def simulate(alpha: np.ndarray, gamma: np.ndarray, baseline: SplineBaseline, n_events: int, rng: np.random.Generator, *, step_seconds: float = DEFAULT_STEP_SECONDS) -> np.ndarray:
    """Simulate counts from the model with known coefficients.

    Arguments: alpha and gamma are the coefficients; baseline is the spline baseline; n_events is the number of events;
    rng is the random generator; step_seconds is the step length in seconds.
    Returns: counts of shape (instruments, events, steps).
    """
    alpha = np.asarray(alpha, dtype=float)
    quadrature = step_quadrature(baseline, step_seconds)
    mean = np.exp(alpha)[:, None] * quadrature.integrals(np.asarray(gamma, dtype=float))
    return rng.poisson(mean[:, None, :], size=(alpha.size, n_events, quadrature.n_steps)).astype(float)
