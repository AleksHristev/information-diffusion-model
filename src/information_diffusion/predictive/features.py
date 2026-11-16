"""Step features built only from data stamped before each step starts, free of look-ahead."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

STEP_SECONDS = 30.0
TARGET_COLUMNS = ("y_count", "y_rv")


@dataclass(frozen=True)
class FeatureConfig:
    """Feature settings.

    Fields: count_lags are the lags of the step counts; rolling_steps are the widths in steps of the rolling counts;
    rv_lags are the lags of the realised variance; step_seconds is the step length in seconds (30 by default).
    """

    count_lags: tuple[int, ...] = (1, 2, 3, 4)
    rolling_steps: tuple[int, ...] = (4, 10)
    rv_lags: tuple[int, ...] = (1, 2, 3)
    step_seconds: float = STEP_SECONDS


def _last_before(values: np.ndarray, times: np.ndarray, edges: np.ndarray) -> np.ndarray:
    position = np.searchsorted(times, edges, side="left") - 1
    result = np.full((edges.size,) + values.shape[1:], np.nan)
    valid = position >= 0
    result[valid] = values[position[valid]]
    return result


def build_features(event_times: np.ndarray, n_steps: int, *, window_start: float = 0.0, release_time: float = 1800.0, config: FeatureConfig = FeatureConfig(), history_start: float | None = None, intensity_times: np.ndarray | None = None, intensity_values: np.ndarray | None = None, network_times: np.ndarray | None = None, network_values: np.ndarray | None = None, network_names: tuple[str, ...] | None = None, price_times: np.ndarray | None = None, log_prices: np.ndarray | None = None) -> pd.DataFrame:
    """Build the features of each step from data stamped strictly before the step start.

    Step k covers [t_k, t_{k+1}) with t_k = window_start + k * step_seconds. Feature columns: lag_count_l, the event
    count of step k - l; roll_count_m, the event count of the m steps before step k; sec_since_last, the time to t_k
    from the last earlier event (from history_start when there is none); sec_from_t0, t_k - release_time; and, when
    their inputs are given, intensity and net_<name>, the last values stamped before t_k, and lag_rv_l, the realised
    variance of step k - l. The other columns are step and the targets y_count, the event count of step k, and y_rv,
    its realised variance (the sum of squared log-price changes in the step) when prices are given.
    Arguments: event_times are the event times and intensity_times, network_times and price_times the sorted times of
    the series, all in seconds on the clock of window_start (earlier history allowed); n_steps is the number of steps;
    window_start is the start of step 0 (0 by default); release_time is the scheduled release time (1800 by default); config is the
    FeatureConfig; history_start is the start of the event history (window_start by default); intensity_values,
    network_values (one column per name; network_names defaults to "0", "1", ...) and log_prices are the series to
    sample.
    Returns: a DataFrame with one row per step.
    """
    step = config.step_seconds
    events = np.sort(np.asarray(event_times, dtype=float))
    longest = max(config.count_lags + config.rolling_steps + config.rv_lags + (1,))
    grid = window_start + step * np.arange(-longest, n_steps + 1, dtype=float)
    before = np.searchsorted(events, grid, side="left")
    start_edges = grid[longest:]
    n_before = before[longest:]
    edges = start_edges[:-1]
    frame: dict[str, np.ndarray] = {"step": np.arange(n_steps)}
    for lag in config.count_lags:
        i = longest + np.arange(n_steps) - lag
        frame[f"lag_count_{lag}"] = (before[i + 1] - before[i]).astype(float)
    for width in config.rolling_steps:
        i = longest + np.arange(n_steps)
        frame[f"roll_count_{width}"] = (before[i] - before[i - width]).astype(float)
    last_event = _last_before(events[:, None], events, edges)[:, 0] if events.size else np.full(n_steps, np.nan)
    history = window_start if history_start is None else float(history_start)
    since = edges - np.where(np.isnan(last_event), history, last_event)
    frame["sec_since_last"] = np.maximum(since, 0.0)
    frame["sec_from_t0"] = edges - float(release_time)
    if intensity_values is not None:
        times = np.asarray(intensity_times, dtype=float)
        values = np.asarray(intensity_values, dtype=float)
        frame["intensity"] = _last_before(values, times, edges)
    if network_values is not None:
        times = np.asarray(network_times, dtype=float)
        values = np.asarray(network_values, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        names = tuple(f"{j}" for j in range(values.shape[1])) if network_names is None else network_names
        latest = _last_before(values, times, edges)
        for j, name in enumerate(names):
            frame[f"net_{name}"] = latest[:, j]
    if log_prices is not None:
        stamps = np.asarray(price_times, dtype=float)
        prices = np.asarray(log_prices, dtype=float)
        squared = np.concatenate([[0.0], np.square(np.diff(prices))]) if prices.size else np.zeros(0)
        cumulative = np.concatenate([[0.0], np.cumsum(squared)])
        through = np.searchsorted(stamps, grid, side="left")
        for lag in config.rv_lags:
            i = longest + np.arange(n_steps) - lag
            frame[f"lag_rv_{lag}"] = cumulative[through[i + 1]] - cumulative[through[i]]
    frame["y_count"] = (n_before[1:] - n_before[:-1]).astype(float)
    if log_prices is not None:
        i = longest + np.arange(n_steps)
        frame["y_rv"] = cumulative[through[i + 1]] - cumulative[through[i]]
    return pd.DataFrame(frame)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return the feature column names of a frame, excluding the step index and the targets."""
    return [name for name in frame.columns if name not in ("step", *TARGET_COLUMNS)]
