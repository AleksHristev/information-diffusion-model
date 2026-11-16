"""Checks of feature timing: every feature at step k uses data strictly before t_k."""

import numpy as np
import pandas as pd
import pytest

from information_diffusion.predictive import features


def test_event_exactly_at_the_step_start_belongs_to_the_step_not_its_features():
    frame = features.build_features(np.array([0.0, 30.0, 30.5, 60.0]), 3, window_start=0.0, release_time=0.0)
    assert frame.lag_count_1.tolist() == [0.0, 1.0, 2.0]
    assert frame.y_count.tolist() == [1.0, 2.0, 1.0]


def test_lag_and_rolling_counts_look_only_backwards():
    events = np.array([-40.0, -10.0, 5.0, 25.0, 35.0, 65.0, 95.0, 125.0])
    frame = features.build_features(events, 5, window_start=0.0, release_time=0.0, history_start=-60.0)
    assert frame.lag_count_2.iloc[2] == 2.0
    assert frame.roll_count_4.iloc[4] == 5.0
    assert frame.sec_since_last.iloc[0] == pytest.approx(10.0)


def test_step_zero_uses_pre_window_history():
    frame = features.build_features(np.array([-5.0]), 2, window_start=0.0, release_time=0.0, history_start=-60.0)
    assert frame.lag_count_1.iloc[0] == 1.0


def test_censored_time_since_last_event_uses_history_start():
    frame = features.build_features(np.array([100.0]), 3, window_start=0.0, release_time=0.0, history_start=-60.0)
    assert frame.sec_since_last.iloc[0] == pytest.approx(60.0) and frame.sec_since_last.iloc[1] == pytest.approx(90.0)


def test_intensity_uses_the_last_value_strictly_before_the_step():
    times = np.array([-5.0, 29.0, 30.0, 61.0])
    values = np.array([1.0, 2.0, 99.0, 3.0])
    frame = features.build_features(np.array([1.0]), 3, window_start=0.0, release_time=0.0, intensity_times=times, intensity_values=values, history_start=-60.0)
    assert frame.intensity.tolist() == [1.0, 2.0, 99.0]


def test_network_features_are_carried_forward_and_named():
    times = np.array([-1.0, 15.0, 30.0])
    values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    frame = features.build_features(np.array([5.0]), 3, network_times=times, network_values=values, network_names=("leader", "peer"), release_time=0.0, history_start=-60.0)
    assert frame.net_leader.iloc[1] == 2.0 and frame.net_peer.iloc[0] == 10.0


def test_realised_variance_uses_only_returns_stamped_before_the_step():
    times = np.array([-10.0, 10.0, 20.0, 40.0])
    log_prices = np.array([0.0, 0.1, 0.3, 1.3])
    frame = features.build_features(np.array([1.0]), 2, price_times=times, log_prices=log_prices, release_time=0.0, history_start=-60.0)
    assert frame.y_rv.iloc[0] == pytest.approx(0.1**2 + 0.2**2)
    assert frame.lag_rv_1.iloc[1] == pytest.approx(0.1**2 + 0.2**2)
    assert frame.lag_rv_1.iloc[0] == 0.0


def _rich_inputs(rng):
    events = np.sort(rng.uniform(-300, 1500, 2000))
    intensity_times = np.arange(-300.0, 1500.0, 2.0)
    network_times = np.arange(-300.0, 1500.0, 5.0)
    price_times = np.sort(rng.uniform(-300, 1500, 1000))
    return {"event_times": events, "intensity_times": intensity_times, "intensity_values": rng.lognormal(0, 1, intensity_times.size), "network_times": network_times, "network_values": rng.normal(size=(network_times.size, 2)), "price_times": price_times, "log_prices": rng.normal(size=price_times.size).cumsum() * 1e-3}


def test_altering_data_at_or_after_a_step_leaves_features_unchanged():
    rng = np.random.default_rng(0)
    inputs = _rich_inputs(rng)
    cut = 20
    cut_time = cut * features.STEP_SECONDS
    kwargs = {"n_steps": 50, "history_start": -300.0, "release_time": 600.0, "network_names": ("a", "b")}
    original = features.build_features(**inputs, **kwargs)
    altered = {name: value.copy() for name, value in inputs.items()}
    altered["event_times"] = np.sort(np.concatenate([inputs["event_times"][inputs["event_times"] < cut_time], [cut_time], rng.uniform(cut_time, cut_time + 500, 900)]))
    for values, times in (("intensity_values", "intensity_times"), ("network_values", "network_times"), ("log_prices", "price_times")):
        later = inputs[times] >= cut_time
        altered[values][later] = rng.normal(size=altered[values][later].shape) * 10.0
    changed = features.build_features(**altered, **kwargs)
    columns = features.feature_columns(original)
    assert "y_count" not in columns and "y_rv" not in columns
    pd.testing.assert_frame_equal(original.loc[original.step <= cut, columns], changed.loc[changed.step <= cut, columns], check_exact=True)
    assert not original.loc[original.step == cut + 1, columns].equals(changed.loc[changed.step == cut + 1, columns])


def test_altering_data_before_a_step_changes_its_features():
    rng = np.random.default_rng(1)
    inputs = _rich_inputs(rng)
    kwargs = {"n_steps": 50, "history_start": -300.0, "release_time": 600.0, "network_names": ("a", "b")}
    original = features.build_features(**inputs, **kwargs)
    altered = {name: value.copy() for name, value in inputs.items()}
    altered["event_times"] = np.append(altered["event_times"], 599.0)
    altered["event_times"].sort()
    changed = features.build_features(**altered, **kwargs)
    assert original.lag_count_1.iloc[20] != changed.lag_count_1.iloc[20]
