"""Estimate the lead-lag time between two simulated price series observed at irregular, asynchronous times.

Both series follow one latent log-price path, with volatility 1 basis point per root second; the second sees it
0.25 s late. Each is observed at its own Poisson times with microstructure noise. The Hoffmann-Rosenbaum-Yoshida
contrast, summed over independent events, gives the lag; a bootstrap over events gives its interval. The
Hayashi-Yoshida estimator gives the correlation without synchronising the two clocks, at zero lag and after shifting
the clock of the second series by the estimated lag.
"""

import numpy as np

from information_diffusion.influence import hayashi_yoshida
from information_diffusion.influence import hoffmann_rosenbaum_yoshida as hry

rng = np.random.default_rng(7)
true_lag, seconds, n_events = 0.25, 300.0, 12
sigma, noise = 1e-4, 1e-5


def simulated_event(rate_x: float = 10.0, rate_y: float = 4.0):
    """One event: observation times and log prices of X and of Y, which follows X with a delay of true_lag seconds."""
    grid = np.arange(-5.0, seconds + 1e-9, 0.001)
    path = np.cumsum(rng.standard_normal(grid.size) * sigma * np.sqrt(0.001))
    tx = np.sort(rng.uniform(0.0, seconds, rng.poisson(rate_x * seconds)))
    ty = np.sort(rng.uniform(0.0, seconds, rng.poisson(rate_y * seconds)))
    x = path[np.searchsorted(grid, tx)] + noise * rng.standard_normal(tx.size)
    y = path[np.searchsorted(grid, ty - true_lag)] + noise * rng.standard_normal(ty.size)
    return tx, x, ty, y


def correlation(events, shift: float = 0.0) -> float:
    """Pooled Hayashi-Yoshida correlation with the clock of Y moved back by shift seconds."""
    cross = hayashi_yoshida.pooled_covariance([(tx, x, ty - shift, y) for tx, x, ty, y in events])
    var_x = hayashi_yoshida.pooled_covariance([(tx, x, tx, x) for tx, x, _, _ in events])
    var_y = hayashi_yoshida.pooled_covariance([(ty, y, ty, y) for _, _, ty, y in events])
    return cross / np.sqrt(var_x * var_y)


events = [simulated_event() for _ in range(n_events)]
lags = hry.lag_grid(step=0.001, max_lag=1.0)
contrasts = np.array([hry.event_contrast(*event, lags) for event in events])
estimate = hry.estimate_lag(contrasts, lags)
draws = hry.bootstrap_lags(contrasts, lags, 499, rng)
lower, upper = np.percentile(draws, [2.5, 97.5])

print(f"events                              {n_events} of {seconds:.0f} s")
print(f"observations per event, X and Y     {np.mean([e[0].size for e in events]):.0f} and {np.mean([e[2].size for e in events]):.0f}")
print(f"lag of Y behind X, true             {true_lag:.3f} s")
print(f"lag of Y behind X, HRY estimate     {estimate:.3f} s   95% bootstrap interval [{lower:.3f}, {upper:.3f}]")
print(f"Hayashi-Yoshida correlation         {correlation(events):.3f} at zero lag, {correlation(events, estimate):.3f} at the estimated lag")
