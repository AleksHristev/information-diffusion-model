"""Diagnostic of autocorrelation and tail weight of loss differentials, with a parametric Giacomini-White p-value."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special

from information_diffusion.statistical_framework import giacomini_white as gw

AR_TRIGGER = 0.2
TAIL_TRIGGER = 3.0
NU_BOUNDS = (2.05, 200.0)
SIMULATIONS = 1_000_000
SEED = 20261111
_RHO_LIMIT = 0.95
_BURN_IN = 50
_CHUNK = 100_000


@dataclass(frozen=True)
class Diagnostic:
    """Result of the diagnostic.

    Fields: the number of events, the AR(1) coefficient, the tail index, whether the diagnostic triggered, and the
    parametric p-value (None when not triggered).
    """

    n_events: int
    ar1: float
    tail_index: float
    triggered: bool
    parametric_p_value: float | None


def ar1_coefficient(differentials) -> float:
    """Return the AR(1) coefficient sum c_t c_{t-1} / sum c_t^2 of the centred differentials c_t."""
    d = np.asarray(differentials, dtype=float)
    c = d - d.mean()
    return float(np.sum(c[1:] * c[:-1]) / np.sum(c**2))


def _profile_loglik(nu: float, e: np.ndarray) -> float:
    s2 = float(np.mean(e**2)) or 1e-300
    for _ in range(200):
        weights = (nu + 1) / (nu + e**2 / s2)
        updated = float(np.mean(weights * e**2))
        if abs(updated - s2) <= 1e-10 * s2:
            s2 = updated
            break
        s2 = updated
    return float(e.size * (special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * np.log(np.pi * nu * s2)) - (nu + 1) / 2 * np.sum(np.log1p(e**2 / (nu * s2))))


def tail_index(differentials, rho: float | None = None) -> float:
    """Estimate the tail index as the maximum-likelihood degrees of freedom of a Student t with free scale.

    The t is fitted to the AR(1)-filtered residuals c_t - rho c_{t-1}, with the degrees of freedom bounded by NU_BOUNDS.
    Arguments: differentials is the series; rho is the AR(1) coefficient (estimated when None).
    Returns: the tail index.
    Reference: Lange, Little and Taylor (1989).
    """
    d = np.asarray(differentials, dtype=float)
    c = d - d.mean()
    rho = ar1_coefficient(d) if rho is None else rho
    e = c[1:] - rho * c[:-1]
    result = optimize.minimize_scalar(lambda log_nu: -_profile_loglik(float(np.exp(log_nu)), e), bounds=tuple(np.log(NU_BOUNDS)), method="bounded", options={"xatol": 1e-4})
    return float(np.exp(result.x))


def is_triggered(rho: float, nu: float) -> bool:
    """Return True when the AR(1) coefficient rho exceeds AR_TRIGGER or the tail index nu is below TAIL_TRIGGER."""
    return bool(rho > AR_TRIGGER or nu < TAIL_TRIGGER)


def _simulate(rng: np.random.Generator, sims: int, n: int, rho: float, nu: float) -> np.ndarray:
    shocks = rng.standard_t(nu, size=(sims, n + _BURN_IN))
    series = shocks.copy()
    for t in range(1, n + _BURN_IN):
        series[:, t] = rho * series[:, t - 1] + shocks[:, t]
    return series[:, _BURN_IN:]


def parametric_p_value(t: float, n_events: int, rho: float, nu: float, *, alternative: str = "two-sided", sims: int = SIMULATIONS, seed: int = SEED) -> float:
    """Compute the p-value of the statistic against simulated mean-zero AR(1) series with Student t(nu) shocks.

    Arguments: t is the observed statistic; n_events is the series length; rho is the AR(1) coefficient (clipped to
    +/-0.95); nu is the tail index; alternative is "two-sided", "greater" or "less"; sims is the number of simulated
    series, drawn in whole chunks of 100,000 (rounded down, with at least one chunk); seed is the random seed.
    Returns: the share of simulated statistics at least as extreme as t.
    Reference: Giacomini and White (2006), Econometrica.
    """
    rho = float(np.clip(rho, -_RHO_LIMIT, _RHO_LIMIT))
    rng = np.random.default_rng(seed)
    draws = np.concatenate([gw.statistic(_simulate(rng, _CHUNK, n_events, rho, nu)) for _ in range(max(1, sims // _CHUNK))])
    if alternative == "two-sided":
        return float(np.mean(np.abs(draws) >= abs(t)))
    if alternative == "greater":
        return float(np.mean(draws >= t))
    if alternative == "less":
        return float(np.mean(draws <= t))
    raise ValueError("alternative must be two-sided, greater or less")


def diagnose(differentials, *, alternative: str = "two-sided", sims: int = SIMULATIONS, seed: int = SEED) -> Diagnostic:
    """Run the diagnostic on one series of loss differentials.

    Arguments: differentials is the series; alternative, sims and seed are passed to parametric_p_value.
    Returns: a Diagnostic, whose parametric p-value is computed only when the diagnostic triggers.
    Reference: Giacomini and White (2006), Econometrica.
    """
    d = np.asarray(differentials, dtype=float)
    rho = ar1_coefficient(d)
    nu = tail_index(d, rho)
    triggered = is_triggered(rho, nu)
    p_value = parametric_p_value(float(gw.statistic(d)), d.size, rho, nu, alternative=alternative, sims=sims, seed=seed) if triggered else None
    return Diagnostic(d.size, rho, nu, triggered, p_value)
