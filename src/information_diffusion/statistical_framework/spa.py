"""Hansen's test of superior predictive ability, with White's Reality Check, by block bootstrap."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from information_diffusion.statistical_framework import block_bootstrap


@dataclass(frozen=True)
class SuperiorPredictiveAbility:
    """Result of the superior predictive ability test.

    Fields: the statistic T, the consistent, lower and upper (Reality Check) p-values, the mean differentials of each
    alternative, their bootstrap scales omega and the number of replications.
    """

    statistic: float
    p_consistent: float
    p_lower: float
    p_upper: float
    mean_differentials: np.ndarray
    omega: np.ndarray
    replications: int

    @property
    def p_value(self) -> float:
        """Return the consistent p-value, SPA_c."""
        return self.p_consistent


def _indices(n: int, block_length: float, rng: np.random.Generator, scheme: str) -> np.ndarray:
    if scheme == "stationary":
        return block_bootstrap.stationary_indices(n, block_length, rng)
    if scheme == "moving":
        return block_bootstrap.moving_block_indices(n, int(block_length), rng)
    raise ValueError("scheme must be stationary or moving")


def spa(benchmark, alternatives, *, block_length: float, rng: np.random.Generator, replications: int = 999, scheme: str = "stationary") -> SuperiorPredictiveAbility:
    """Test the null max_k E[L0 - Lk] <= 0 that no alternative beats the benchmark.

    T = max(0, max_k sqrt(n) dbar_k / omega_k); p = share of bootstrap draws with T* >= T, under the lower, consistent
    and upper recentrings. An alternative with zero bootstrap variance gets t = 0.
    Arguments: benchmark is the benchmark loss series; alternatives is an n by k array of alternative losses;
    block_length is the (mean) block length; rng is the random generator; replications is the number of bootstrap
    draws; scheme is "stationary" or "moving".
    Returns: a SuperiorPredictiveAbility.
    Reference: Hansen (2005); White (2000).
    """
    L0 = np.asarray(benchmark, dtype=float)
    Lk = np.asarray(alternatives, dtype=float)
    if Lk.ndim == 1:
        Lk = Lk[:, None]
    if L0.ndim != 1 or Lk.ndim != 2 or Lk.shape[0] != L0.size or not (np.all(np.isfinite(L0)) and np.all(np.isfinite(Lk))):
        raise ValueError("benchmark must be a finite vector and alternatives a finite n by k array")
    n = L0.size
    if n < 3:
        raise ValueError("at least three observations are required")
    d = L0[:, None] - Lk
    dbar = d.mean(axis=0)
    k = d.shape[1]
    dstar = np.empty((replications, k))
    for b in range(replications):
        dstar[b] = d[_indices(n, block_length, rng, scheme)].mean(axis=0)
    omega = np.sqrt(n * np.mean((dstar - dbar) ** 2, axis=0))
    root_n = np.sqrt(n)
    t = _studentise(root_n * dbar, omega)
    statistic = float(max(t.max(), 0.0))
    threshold = -np.sqrt(omega**2 * 2.0 * np.log(np.log(n)) / n)
    centres = {"lower": np.maximum(dbar, 0.0), "consistent": np.where(dbar >= threshold, dbar, 0.0), "upper": dbar}
    p = {name: float(np.mean(_bootstrap_statistic(root_n * (dstar - centre), omega) >= statistic)) for name, centre in centres.items()}
    return SuperiorPredictiveAbility(statistic, p["consistent"], p["lower"], p["upper"], dbar, omega, replications)


def _studentise(values: np.ndarray, omega: np.ndarray) -> np.ndarray:
    return np.divide(values, omega, out=np.zeros_like(values), where=omega > 0)


def _bootstrap_statistic(scaled: np.ndarray, omega: np.ndarray) -> np.ndarray:
    return np.maximum(np.max(_studentise(scaled, np.broadcast_to(omega, scaled.shape)), axis=1), 0.0)
