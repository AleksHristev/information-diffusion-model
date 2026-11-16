"""Absorption of news across instruments: absorption times and curves, half-life, R0 interval and the absorption clock."""

from __future__ import annotations

import numpy as np

from information_diffusion.epidemic import compartment_common as common
from information_diffusion.epidemic import compartment_sir
from information_diffusion.statistical_framework import block_bootstrap

FIVE_MINUTES_NS = 5 * 60 * 1_000_000_000

NS = 1_000_000_000
HORIZON_NS = int(common.HORIZON_SECONDS * NS)
TICK = 0.01
MIN_MOVE_TICKS = 2
TOLERANCE = 1e-9
OBSERVATION_SECONDS = (60.0, 300.0, 900.0)
CLOCK_HORIZON_SECONDS = (1800.0, 3600.0)
HALF_LIFE_OK = "ok"
NO_MIDPOINT_BEFORE_T0 = "no_midpoint_before_t0"
NO_MIDPOINT_AT_HORIZON = "no_midpoint_at_horizon"
MOVE_BELOW_TWO_TICKS = "move_below_two_ticks"


def absorption_time(ts_ns: np.ndarray, ask: np.ndarray, bid: np.ndarray, t0_ns: int, *, horizon_ns: int = HORIZON_NS) -> dict[str, float | int | bool]:
    """Absorption time of one instrument: seconds from t0 to the first valid midpoint outside its prior five-minute range.

    Valid quotes have ask > 0, bid > 0 and ask >= bid; the range is the minimum and maximum midpoint over [t0 - 300 s, t0).
    Arguments: ts_ns are quote times in nanoseconds; ask and bid the quotes; t0_ns the shock time in nanoseconds;
    horizon_ns the search horizon in nanoseconds (default 3,600 seconds).
    Returns: a dict with absorption_seconds (NaN when no midpoint leaves the range in [t0, t0 + horizon]), quotes_prior,
    quotes_after (the numbers of valid quotes in the range window and the search window) and included (False when
    there is no valid quote in the five minutes before t0, so the instrument has no range).
    """
    ts_ns, ask, bid = np.asarray(ts_ns), np.asarray(ask, dtype=float), np.asarray(bid, dtype=float)
    valid = (ask > 0) & (bid > 0) & (ask >= bid)
    ts, mid = ts_ns[valid], (ask[valid] + bid[valid]) / 2
    before = (ts >= t0_ns - FIVE_MINUTES_NS) & (ts < t0_ns)
    after = (ts >= t0_ns) & (ts <= t0_ns + horizon_ns)
    result: dict[str, float | int | bool] = {"absorption_seconds": float("nan"), "quotes_prior": int(before.sum()), "quotes_after": int(after.sum()), "included": bool(before.any())}
    if not before.any() or not after.any():
        return result
    low, high = mid[before].min(), mid[before].max()
    moved = (mid[after] < low) | (mid[after] > high)
    if moved.any():
        result["absorption_seconds"] = float((int(ts[after][moved].min()) - t0_ns) / NS)
    return result


def absorption_curve(absorption_seconds: np.ndarray, *, times: np.ndarray = common.TIME_GRID, observable_seconds: float | None = None) -> np.ndarray:
    """Absorption curve y(t) = (1 / n) sum_i 1{tau_i <= t}: the share of instruments absorbed by each grid time.

    Arguments: absorption_seconds holds one absorption time per included instrument, NaN if it never absorbed; times is
    the grid in seconds; observable_seconds, if given, is the last observable time.
    Returns: the curve on the grid; all NaN for no instruments, and NaN at grid times past observable_seconds.
    """
    seconds = np.asarray(absorption_seconds, dtype=float)
    times = np.asarray(times, dtype=float)
    if seconds.size == 0:
        return np.full(times.size, np.nan)
    curve = np.sum(seconds[:, None] <= times[None, :], axis=0) / seconds.size
    if observable_seconds is not None:
        curve = np.where(times > observable_seconds + TOLERANCE, np.nan, curve.astype(float))
    return curve.astype(float)


def half_life(elapsed_seconds: np.ndarray, closes: np.ndarray, *, horizon_seconds: float = common.HORIZON_SECONDS, tick: float = TICK) -> tuple[float, str]:
    """Absorption half-life: the first elapsed time at which the price has covered half of its move from t0 to the horizon.

    With D = m_H - m_0, half-life = min{s_k : 0 < s_k <= H, sign(D) (m_k - m_0) >= |D| / 2}.
    Arguments: elapsed_seconds and closes are a grid of midpoint closes, including s = 0 and s = H; horizon_seconds is H;
    tick is the tick size, and moves smaller than two ticks give no half-life.
    Returns: (half-life in seconds or NaN, status), the status being "ok" or the reason it is missing.
    """
    elapsed = np.asarray(elapsed_seconds, dtype=float)
    closes = np.asarray(closes, dtype=float)
    at_start = np.flatnonzero(np.abs(elapsed) < 1e-6)
    at_end = np.flatnonzero(np.abs(elapsed - horizon_seconds) < 1e-6)
    if at_start.size == 0 or not np.isfinite(closes[at_start[0]]):
        return float("nan"), NO_MIDPOINT_BEFORE_T0
    if at_end.size == 0 or not np.isfinite(closes[at_end[0]]):
        return float("nan"), NO_MIDPOINT_AT_HORIZON
    start, move = closes[at_start[0]], closes[at_end[0]] - closes[at_start[0]]
    if abs(move) < MIN_MOVE_TICKS * tick - TOLERANCE:
        return float("nan"), MOVE_BELOW_TWO_TICKS
    after = (elapsed > 0) & (elapsed <= horizon_seconds + 1e-6)
    covered = np.sign(move) * (closes[after] - start)
    return float(elapsed[after][np.argmax(covered >= abs(move) / 2.0 - TOLERANCE)]), HALF_LIFE_OK


def r0_with_interval(beta_hat: float, gamma_hat: float, bootstrap_log: np.ndarray, *, max_failed_share: float = common.MAX_FAILED_DRAWS) -> tuple[float, float, float]:
    """Basic reproduction number of the SIR model, R0 = beta / gamma, with a 95% percentile interval.

    Arguments: beta_hat and gamma_hat are the estimates; bootstrap_log holds event-bootstrap draws of (log beta, log gamma)
    as columns; max_failed_share is the largest share of failed draws allowed.
    Returns: (estimate, lower, upper); the bounds are NaN when too many draws failed.
    """
    estimate = float(beta_hat / gamma_hat)
    draws = np.asarray(bootstrap_log, dtype=float)
    if draws.size == 0 or np.mean(~np.isfinite(draws).all(axis=1)) > max_failed_share:
        return estimate, float("nan"), float("nan")
    lower, upper = np.exp(np.nanquantile(draws[:, 0] - draws[:, 1], [0.025, 0.975]))
    return estimate, float(lower), float(upper)


def clock_share(observed_share: float, model_at_observation: float, model_at_horizon: float) -> float:
    """Absorption clock: predicted share at the horizon, yhat(H | s) = 1 - (1 - y(s)) (1 - m(H)) / (1 - m(s)).

    Arguments: observed_share is y(s); model_at_observation and model_at_horizon are the fitted curve m at s and at H.
    Returns: the predicted share; y(s) when m(s) = 1, NaN when an input is not finite.
    Reference: Kalbfleisch and Prentice (2002), The Statistical Analysis of Failure Time Data (conditional survival).
    """
    observed, model_seen, model_ahead = float(observed_share), float(model_at_observation), float(model_at_horizon)
    if not np.isfinite(observed) or not np.isfinite(model_seen) or not np.isfinite(model_ahead):
        return float("nan")
    if model_seen >= 1.0 - 1e-12:
        return observed
    remaining = np.clip((1.0 - model_ahead) / (1.0 - model_seen), 0.0, 1.0)
    return float(1.0 - (1.0 - observed) * remaining)


def clock_step(train_curves: np.ndarray, target_curve: np.ndarray, *, mean_model: str = "chain", n_nodes: int = common.N_NODES, times: np.ndarray = common.TIME_GRID, observation_seconds: tuple[float, ...] = OBSERVATION_SECONDS, horizon_seconds: tuple[float, ...] = CLOCK_HORIZON_SECONDS, n_starts: int | None = None) -> list[dict]:
    """One step of the absorption clock: fit SIR to the mean of earlier curves and predict a new curve.

    The benchmark prediction is the mean of the earlier curves at the horizon.
    Arguments: train_curves are the earlier events' curves (one row each); target_curve the curve to predict; mean_model,
    n_nodes, times and n_starts as in compartment_common.fit_curve; observation_seconds and horizon_seconds the elapsed
    times at which the share is observed and predicted.
    Returns: one dict per observation-horizon pair with the clock and benchmark predictions, the actual share, their
    absolute errors, the fitted beta and gamma, convergence and the number of fitted events.
    Reference: Kermack and McKendrick (1927), Proc. R. Soc. A (the SIR curve).
    """
    train = np.asarray(train_curves, dtype=float)
    target = np.asarray(target_curve, dtype=float)
    historical = common.mean_curve(train)
    u_hat, _, _, converged = common.fit_curve(compartment_sir.SPEC, historical, mean_model=mean_model, n_nodes=n_nodes, times=times, n_starts=n_starts)
    theta = np.exp(u_hat)
    model = common.chain_for(compartment_sir.SPEC, n_nodes).mean_curve(theta, times) if mean_model == "chain" else common.mean_field_curve(compartment_sir.SPEC, theta, times, n_nodes=n_nodes)
    rows = []
    for observation in observation_seconds:
        seen_index = int(np.searchsorted(times, observation))
        for horizon in horizon_seconds:
            ahead_index = int(np.searchsorted(times, horizon))
            actual = float(target[ahead_index])
            clock = clock_share(target[seen_index], model[seen_index], model[ahead_index])
            benchmark = float(historical[ahead_index])
            rows.append({"observation_seconds": float(observation), "horizon_seconds": float(horizon), "observed_share": float(target[seen_index]), "clock_prediction": clock, "benchmark_prediction": benchmark, "actual": actual,
                         "clock_abs_error": abs(clock - actual), "benchmark_abs_error": abs(benchmark - actual), "beta": float(theta[0]), "gamma": float(theta[1]), "converged": bool(converged), "fit_events": int(train.shape[0])})
    return rows


def paired_mae_difference(clock_errors: np.ndarray, benchmark_errors: np.ndarray, *, rng: np.random.Generator, draws: int = 1999) -> dict[str, float]:
    """Mean absolute errors of two predictors and their paired difference, with an event-bootstrap 95% percentile interval.

    Arguments: clock_errors and benchmark_errors are absolute errors per event (pairs with a non-finite value are dropped);
    rng the generator; draws the number of bootstrap draws (default 1,999).
    Returns: a dict with n_events, mae_clock, mae_benchmark, mae_difference (clock minus benchmark), difference_lower and
    difference_upper.
    """
    clock, benchmark = np.asarray(clock_errors, dtype=float), np.asarray(benchmark_errors, dtype=float)
    keep = np.isfinite(clock) & np.isfinite(benchmark)
    clock, benchmark = clock[keep], benchmark[keep]
    n_events = clock.size
    if n_events == 0:
        return {"n_events": 0, "mae_clock": float("nan"), "mae_benchmark": float("nan"), "mae_difference": float("nan"), "difference_lower": float("nan"), "difference_upper": float("nan")}
    resamples = np.stack([block_bootstrap.event_indices(n_events, rng) for _ in range(draws)])
    differences = np.mean(clock[resamples] - benchmark[resamples], axis=1)
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return {"n_events": n_events, "mae_clock": float(clock.mean()), "mae_benchmark": float(benchmark.mean()), "mae_difference": float(clock.mean() - benchmark.mean()), "difference_lower": float(lower), "difference_upper": float(upper)}
