"""Fit a Hawkes process to simulated event times, then measure absorption from simulated quotes and fit an SI model.

The first half simulates trades from an exponential Hawkes process with a known branching ratio and fits it back, with a
parametric bootstrap interval for the branching ratio.

The second half simulates the spread of a shock across 30 instruments as an SI epidemic, turns each instrument's absorption
time into a quote stream, measures the absorption times back from the quotes, builds the absorption curves and fits
the SI model to them.
"""

import numpy as np

from information_diffusion.epidemic import absorption, compartment_common, compartment_si, event_merging
from information_diffusion.epidemic.branching_ratio import bootstrap_branching
from information_diffusion.epidemic.hawkes_exponential import ExponentialHawkes
from information_diffusion.epidemic.spline_baseline import ConstantBaseline
from information_diffusion.simulation import hawkes

NS = 1_000_000_000
rng = np.random.default_rng(2026)

window, mu, eta, beta = 3600.0, 0.5, 0.6, 1.0
immigrants = hawkes.FlatRate(mu, window).sample(rng)
raw = hawkes.simulate_cluster(immigrants, hawkes.ExponentialKernel(beta), eta, window, rng)
times, _ = event_merging.merge_events(raw)
fit = ExponentialHawkes(times, window, ConstantBaseline(window)).fit()
interval = bootstrap_branching(fit, 99, rng)
print("Exponential Hawkes process")
print(f"  events                      {times.size}")
print(f"  branching ratio, true       {eta:.3f}")
print(f"  branching ratio, estimate   {fit.branching:.3f}   95% bootstrap interval [{interval.lower:.3f}, {interval.upper:.3f}]")
print(f"  baseline rate, true         {mu:.3f} per second")
print(f"  baseline rate, estimate     {fit.mean_baseline():.3f} per second")


def si_absorption_times(rate: float, n: int, horizon: float) -> np.ndarray:
    """Absorption times of n instruments under an SI epidemic seeded at time 0, S -> I at total rate rate * S * I / n; NaN if not reached by horizon."""
    out = np.full(n, np.nan)
    order = rng.permutation(n)
    out[order[0]] = 0.0
    now, infected = 0.0, 1
    while infected < n:
        now += rng.exponential(1.0 / (rate * (n - infected) * infected / n))
        if now > horizon:
            break
        out[order[infected]] = now
        infected += 1
    return out


def quotes_for(tau: float, t0_ns: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One instrument's quotes: a one-cent range for five minutes before t0, the same range until tau, then a move."""
    before = t0_ns + np.arange(-300, 0) * NS
    after = t0_ns + np.arange(0, 3600, 5) * NS
    t = np.concatenate([before, after])
    mid = np.where(np.arange(t.size) % 2 == 0, 100.005, 100.015)
    if np.isfinite(tau):
        exit_ns = t0_ns + int(round(tau * NS))
        position = np.searchsorted(t, exit_ns)
        t = np.insert(t, position, exit_ns)
        mid = np.insert(mid, position, 100.045)
        mid[position:] = 100.045
    return t, mid + 0.005, mid - 0.005


true_rate, n_events, n_instruments, t0_ns = 0.01, 20, 30, 10_000 * NS
curves, errors = [], []
for _ in range(n_events):
    taus = si_absorption_times(true_rate, n_instruments, 3600.0)
    measured = np.array([absorption.absorption_time(*quotes_for(tau, t0_ns), t0_ns)["absorption_seconds"] for tau in taus])
    finite = np.isfinite(taus)
    errors.append(np.max(np.abs(measured[finite] - taus[finite])) if finite.any() else 0.0)
    curves.append(absorption.absorption_curve(measured))
curves = np.array(curves)
si = compartment_si.fit(curves, bootstrap=99, rng=rng)
grid = compartment_common.TIME_GRID
print("Absorption across 30 instruments and the SI model")
print(f"  largest error of the measured absorption times   {max(errors):.2e} seconds")
for seconds in (60, 300, 600, 1200, 3600):
    k = int(np.searchsorted(grid, seconds))
    print(f"  mean share absorbed by {seconds:>4} s                  {curves[:, k].mean():.3f}")
print(f"  SI infection rate, true       {true_rate:.4f} per second")
print(f"  SI infection rate, estimate   {si.estimate[0]:.4f} per second   95% bootstrap interval [{si.lower[0]:.4f}, {si.upper[0]:.4f}]")
