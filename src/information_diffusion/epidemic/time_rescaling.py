"""Goodness of fit of Hawkes models by the time-rescaling theorem, in continuous or discrete (merged-event) time."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

MERGE_RESOLUTION_SECONDS = 0.001



@dataclass(frozen=True)
class RescalingResult:
    """Kolmogorov-Smirnov test result. Fields: statistic D, its asymptotic pvalue, and n, the number of intervals."""

    statistic: float
    pvalue: float
    n: int

    def rejected(self, level: float = 0.05) -> bool:
        """True when the p-value is below level (default 0.05)."""
        return self.pvalue < level


def rescaled_intervals(fit, *, rng: np.random.Generator | None = None, resolution: float | None = MERGE_RESOLUTION_SECONDS) -> np.ndarray:
    """Rescaled intervals tau_i = Lambda(t_i) - Lambda(t_{i-1}) of a fitted Hawkes model, i.i.d. Exp(1) under the model.

    With a resolution d the discrete-time form is used: tau_i sums the intensity mass of the bins strictly between events
    i-1 and i and adds -log(1 - U_i p_i), U_i ~ Uniform(0, 1), p_i = 1 - exp(-r_i), r_i the mass of event i's bin from
    earlier events only.
    Arguments: fit is a FitResult; rng the generator for U_i (fixed seed when None); resolution is d in seconds
    (default 0.001; None for continuous time).
    Returns: one interval per event.
    Reference: Haslinger, Pipa and Brown (2010), Neural Computation.
    """
    problem = fit.problem
    b = fit.x[: problem.K]
    cumulative = problem.baseline.cumulative(b, problem.t)
    baseline_increments = np.diff(np.concatenate(([0.0], cumulative)))
    excitation_increments = problem.excitation_integral_increments(fit.x)
    continuous = baseline_increments + excitation_increments
    if resolution is None:
        return continuous
    rng = np.random.default_rng(987654321) if rng is None else rng
    bin_base = problem.baseline.cumulative(b, problem.t + resolution) - cumulative
    past_mass, own_mass = problem.event_bin_masses(fit.x, resolution)
    interval_mass = continuous.copy()
    interval_mass[1:] -= bin_base[:-1] + own_mass[:-1]
    interval_mass = np.maximum(interval_mass, 0.0)
    r_event = bin_base + past_mass
    uniforms = rng.random(problem.n)
    return interval_mass - np.log1p(-uniforms * -np.expm1(-r_event))


def ks_test(intervals: np.ndarray) -> RescalingResult:
    """Kolmogorov-Smirnov test of z = 1 - exp(-tau) against Uniform(0, 1).

    Arguments: intervals are the rescaled intervals tau.
    Returns: a RescalingResult; NaN when the intervals are empty, negative or not finite.
    Reference: Brown, Barbieri, Ventura, Kass and Frank (2002), Neural Computation.
    """
    tau = np.asarray(intervals, dtype=float)
    if tau.size == 0 or np.any(~np.isfinite(tau)) or np.any(tau < 0.0):
        return RescalingResult(np.nan, np.nan, tau.size)
    z = -np.expm1(-tau)
    result = stats.kstest(z, "uniform", method="asymp")
    return RescalingResult(float(result.statistic), float(result.pvalue), tau.size)


def test_fit(fit, *, rng: np.random.Generator | None = None, resolution: float | None = MERGE_RESOLUTION_SECONDS) -> RescalingResult:
    """Time-rescaling Kolmogorov-Smirnov test of a fitted Hawkes model; arguments as in rescaled_intervals. Returns a RescalingResult."""
    return ks_test(rescaled_intervals(fit, rng=rng, resolution=resolution))


test_fit.__test__ = False
