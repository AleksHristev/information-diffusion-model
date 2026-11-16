"""Simulators for two-venue price discovery: a VECM with known shares and minute-scale prices after a decision with venue delays."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def vecm_parameters(gg_kalshi: float, half_life_steps: float, sigma_kalshi: float, sigma_polymarket: float, correlation: float) -> tuple[np.ndarray, np.ndarray]:
    """VECM adjustment coefficients and innovation covariance with a given first-venue component share.

    kappa = 1 - 0.5^(1/h), alpha_P = g kappa and alpha_K = (g - 1) kappa, so CS_K = g and the spread has half-life h steps.
    Arguments: gg_kalshi is g; half_life_steps is h; sigma_kalshi and sigma_polymarket the innovation standard
    deviations; correlation their correlation.
    Returns: (alpha, Sigma).
    """
    kappa = 1.0 - 0.5 ** (1.0 / half_life_steps)
    alpha = np.array([(gg_kalshi - 1.0) * kappa, gg_kalshi * kappa])
    covariance = correlation * sigma_kalshi * sigma_polymarket
    sigma = np.array([[sigma_kalshi**2, covariance], [covariance, sigma_polymarket**2]])
    return alpha, sigma


def simulate_vecm(alpha: np.ndarray, gamma1: np.ndarray, sigma: np.ndarray, steps: int, series: int, rng: np.random.Generator, *, df: float = 5.0, burn: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Simulate dy_t = alpha (x_K - x_P)_{t-1} + Gamma_1 dy_{t-1} + e_t, e_t = F z_t with F F' = Sigma and z_t unit-variance Student t.

    Arguments: alpha, gamma1 (2 x 2) and sigma are the parameters; steps the path length; series the number of paths;
    rng the generator; df the Student t degrees of freedom (default 5); burn the burn-in steps (default 500).
    Returns: (first-venue paths, second-venue paths), each of shape (series, steps).
    """
    total = steps + burn
    scale = np.sqrt(df / (df - 2.0))
    shocks = rng.standard_t(df, size=(total, series, 2)) / scale @ np.linalg.cholesky(sigma).T
    levels = np.zeros((series, 2))
    change = np.zeros((series, 2))
    out = np.empty((total, series, 2))
    gamma_t = np.asarray(gamma1, dtype=float).T
    alpha = np.asarray(alpha, dtype=float)
    for step in range(total):
        spread = levels[:, 0] - levels[:, 1]
        change = spread[:, None] * alpha[None, :] + change @ gamma_t + shocks[step]
        levels = levels + change
        out[step] = levels
    kept = out[burn:]
    return kept[:, :, 0].T.copy(), kept[:, :, 1].T.copy()


def trade_times(counts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sorted trade times in seconds after t0: one uniform trade in the minute before t0, then counts[m - 1] uniform trades in minute m, using the generator rng."""
    counts = np.asarray(counts, dtype=np.int64)
    minutes = np.repeat(np.arange(counts.size), counts)
    inside = 60.0 * minutes + 60.0 * rng.random(minutes.size)
    return np.sort(np.concatenate(([-60.0 * rng.random()], inside)))


def simulate_post_decision(counts: Mapping[tuple, np.ndarray], delays: Mapping[str, float], rng: np.random.Generator, *, sigma: float, jump_sd: tuple[float, float], noise_sd: float, minutes: int = 90) -> dict[tuple, np.ndarray]:
    """Simulate minute prices after a decision on several venues that see one efficient probability with delays.

    The efficient probability is a random walk on a 1-second clock with standard deviation sigma per second, with a jump
    N(0, jump_sd[0]^2) in the first minute after t0 and another N(0, jump_sd[1]^2) in the first minute after t0 + 30
    minutes. Venue v's trade at time t prices the walk at floor(t) - d_v plus N(0, noise_sd^2) noise; the minute price
    is the last trade at or before the minute's end.
    Arguments: counts maps (decision, outcome, venue) to trades per minute for m = 1, ..., minutes; delays maps venue to
    its delay d_v in seconds (rounded to whole seconds); rng the generator; sigma, jump_sd and noise_sd as above;
    minutes (default 90).
    Returns: a dict from each key of counts to its prices at m = 0, ..., minutes.
    """
    longest = int(np.ceil(max(delays.values())))
    start = -(120 + longest)
    length = 60 * minutes - start + 1
    prices: dict[tuple, np.ndarray] = {}
    for decision, outcome in sorted({(key[0], key[1]) for key in counts}):
        path = np.cumsum(sigma * rng.standard_normal(length))
        for shock, size in zip((0, 30 * 60), jump_sd, strict=True):
            jump_at = shock + int(rng.integers(0, 60))
            path[jump_at - start:] += size * rng.standard_normal()
        for venue, delay in delays.items():
            key = (decision, outcome, venue)
            if key not in counts:
                continue
            times = trade_times(counts[key], rng)
            index = np.floor(times).astype(np.int64) - int(round(delay)) - start
            traded = path[index] + noise_sd * rng.standard_normal(times.size)
            grid = 60.0 * np.arange(minutes + 1)
            position = np.searchsorted(times, grid, side="right") - 1
            prices[key] = traded[position]
    return prices
