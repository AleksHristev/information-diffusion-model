"""Hawkes branching ratio with a parametric bootstrap interval."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg
from scipy.special import expit, logit

from information_diffusion.epidemic.event_merging import merge_events
from information_diffusion.epidemic.hawkes_exponential import ETA_CEILING, FitResult
from information_diffusion.simulation.hawkes import ExplosionError

MAX_FAILED_FRACTION = 0.10


@dataclass
class BootstrapInterval:
    """Bootstrap interval of a statistic.

    Fields: estimate at the fit; lower and upper bounds (NaN when the interval is not available); draws, the finished
    bootstrap values; failed, the number of failed replicates; replicates, the number attempted.
    """

    estimate: float
    lower: float
    upper: float
    draws: np.ndarray
    failed: int
    replicates: int


def bootstrap_branching(fit: FitResult, replicates: int, rng: np.random.Generator, *, level: float = 0.95, method: str = "basic_logit", event_guard: float = 60.0, statistic=None) -> BootstrapInterval:
    """Parametric bootstrap of the branching ratio eta = integral_0^infty phi(t) dt, or of another statistic.

    At the fitted parameters, each replicate simulates a new event set, merges it to one event per millisecond, refits
    it from a warm start and evaluates the statistic.
    Arguments: fit is the Hawkes fit; replicates the number of replicates; rng the generator; level the coverage; method
    as in interval_from_draws; event_guard caps each simulation at max(event_guard * n, 2,000) events; statistic is a
    function of (problem, x), the branching ratio by default (for a statistic outside (0, 0.9995) use method
    "percentile" or "basic").
    Returns: a BootstrapInterval; its bounds are NaN without a usable Hessian or when more than 10% of replicates fail.
    Reference: Efron and Tibshirani (1993), An Introduction to the Bootstrap.
    """
    problem = fit.problem
    statistic = (lambda p, x: p.branching(x)) if statistic is None else statistic
    estimate = statistic(problem, fit.x)
    invalid = BootstrapInterval(estimate, np.nan, np.nan, np.empty(0), replicates, replicates)
    if fit.hessian is None:
        return invalid
    try:
        chol = linalg.cho_factor(fit.hessian)
    except linalg.LinAlgError:
        return invalid
    guard = int(max(event_guard * problem.n, 2_000.0))
    draws = np.full(replicates, np.nan)
    for b in range(replicates):
        try:
            raw = problem.simulate(fit.x, rng, max_events=guard)
        except ExplosionError:
            continue
        times, marks = raw if isinstance(raw, tuple) else (raw, None)
        merged, merged_marks = merge_events(times, marks)
        if merged.size < 5:
            continue
        refitted = problem.spawn(merged, marks=merged_marks if problem.marks is not None else None)
        x = refitted.refit_warm(fit.x, fit.hessian, chol=chol)
        if x is not None:
            draws[b] = statistic(refitted, x)
    finished = np.isfinite(draws)
    if finished.sum() < (1.0 - MAX_FAILED_FRACTION) * replicates:
        return BootstrapInterval(estimate, np.nan, np.nan, draws[finished], int((~finished).sum()), replicates)
    lower, upper = interval_from_draws(estimate, draws[finished], level, method)
    return BootstrapInterval(estimate, lower, upper, draws[finished], int((~finished).sum()), replicates)


def interval_from_draws(estimate: float, draws: np.ndarray, level: float = 0.95, method: str = "basic_logit"):
    """Bootstrap interval from draws: percentile or reflected ("basic"), on the natural or the logit scale.

    The basic logit interval, with L = logit(eta / ceiling) and q the draw quantiles, is
    [2 L(eta_hat) - q(1 - alpha/2), 2 L(eta_hat) - q(alpha/2)] mapped back; it stays inside (0, ceiling). The ceiling
    is 0.9995, so the logit methods suit a branching ratio and not a statistic that can leave (0, 0.9995).
    Arguments: estimate is the point estimate; draws the bootstrap values; level the coverage; method is one of
    "percentile", "percentile_logit", "basic_logit" or "basic".
    Returns: (lower, upper).
    """
    alpha = (1.0 - level) / 2.0
    if method == "percentile":
        return float(np.quantile(draws, alpha)), float(np.quantile(draws, 1.0 - alpha))
    if method == "percentile_logit" or method == "basic_logit":
        limit = 1e-9
        transform = lambda eta: logit(np.clip(eta / ETA_CEILING, limit, 1.0 - limit))
        centre = transform(estimate)
        q_low, q_high = np.quantile(transform(draws), [alpha, 1.0 - alpha])
        if method == "percentile_logit":
            return float(ETA_CEILING * expit(q_low)), float(ETA_CEILING * expit(q_high))
        return float(ETA_CEILING * expit(2.0 * centre - q_high)), float(ETA_CEILING * expit(2.0 * centre - q_low))
    if method == "basic":
        q_low, q_high = np.quantile(draws, [alpha, 1.0 - alpha])
        return float(2.0 * estimate - q_high), float(2.0 * estimate - q_low)
    raise ValueError(f"unknown interval method {method}")
