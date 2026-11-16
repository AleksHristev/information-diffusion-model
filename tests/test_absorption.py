"""Absorption times, absorption curves, half-lives, R0 and the absorption clock on hand-made quotes and simulated paths."""

import numpy as np
import pytest

from information_diffusion.epidemic import absorption, compartment_common as common, compartment_sir, compartment_sirs

T0_NS = 1_700_000_000 * absorption.NS
SECOND = absorption.NS


def quotes(rows):
    """Rows of (seconds from t0, bid, ask)."""
    table = np.array(rows, dtype=float)
    return T0_NS + (table[:, 0] * SECOND).astype(np.int64), table[:, 2], table[:, 1]


def test_first_midpoint_outside_the_range_with_boundaries():
    rows = [(-300.0, 99.0, 101.0), (-200.0, 99.5, 100.5), (-1.0, 99.2, 100.8), (0.0, 99.1, 100.9), (0.5, 99.0, 101.0), (1.0, 99.4, 100.8), (2.0, 99.5, 101.5)]
    result = absorption.absorption_time(*quotes(rows), T0_NS)
    assert result["included"] is True and result["quotes_prior"] == 3 and result["quotes_after"] == 4
    assert result["absorption_seconds"] == pytest.approx(1.0)


def test_midpoint_on_the_range_edge_is_not_absorption():
    rows = [(-100.0, 99.9, 100.1), (-50.0, 100.0, 100.2), (10.0, 100.0, 100.2), (20.0, 99.9, 100.1), (30.0, 100.1, 100.3)]
    result = absorption.absorption_time(*quotes(rows), T0_NS)
    assert result["absorption_seconds"] == pytest.approx(30.0)


def test_a_quote_at_the_start_of_the_prior_range_counts_and_one_at_t0_does_not():
    rows = [(-300.0, 99.0, 99.2), (-1.0, 99.0, 99.2), (0.0, 100.0, 100.2)]
    result = absorption.absorption_time(*quotes(rows), T0_NS)
    assert result["quotes_prior"] == 2 and result["absorption_seconds"] == pytest.approx(0.0)
    earlier = absorption.absorption_time(*quotes([(-300.1, 99.0, 99.2), (10.0, 100.0, 100.2)]), T0_NS)
    assert earlier["included"] is False and earlier["quotes_prior"] == 0


@pytest.mark.parametrize("bad", [(0.0, 100.2), (100.0, 0.0), (100.3, 100.1)], ids=["bid zero", "ask zero", "crossed"])
def test_invalid_quotes_do_not_count(bad):
    rows = [(-100.0, 99.9, 100.1), (-50.0, *bad), (5.0, 99.9, 100.1)]
    result = absorption.absorption_time(*quotes(rows), T0_NS)
    assert result["quotes_prior"] == 1 and np.isnan(result["absorption_seconds"])


def test_locked_quotes_are_valid_and_keep_a_midpoint():
    rows = [(-100.0, 100.0, 100.0), (5.0, 101.0, 101.0)]
    assert absorption.absorption_time(*quotes(rows), T0_NS)["absorption_seconds"] == pytest.approx(5.0)


def test_an_instrument_that_does_not_leave_the_range_is_not_absorbed():
    result = absorption.absorption_time(*quotes([(-10.0, 99.0, 101.0), (5.0, 99.5, 100.5)]), T0_NS)
    assert result["included"] is True and np.isnan(result["absorption_seconds"])


def test_quotes_after_the_horizon_are_ignored():
    rows = [(-10.0, 99.9, 100.1), (3599.0, 99.9, 100.1), (3601.0, 110.0, 110.2)]
    result = absorption.absorption_time(*quotes(rows), T0_NS)
    assert np.isnan(result["absorption_seconds"])


def test_no_prior_quote_is_excluded_from_the_denominator():
    result = absorption.absorption_time(*quotes([(10.0, 100.0, 100.2), (20.0, 101.0, 101.2)]), T0_NS)
    assert result["included"] is False and result["quotes_prior"] == 0


def test_simulated_exit_times_are_recovered_exactly():
    """Simulation recovery of the absorption time: prior quotes hold a range, later quotes stay inside it until a known exit time."""
    rng = np.random.default_rng(2)
    for _ in range(30):
        truth = float(rng.uniform(0.0, 3600.0))
        prior = [(-float(rng.uniform(1.0, 299.0)), 99.0, 101.0) for _ in range(5)]
        inside = [(float(s), 99.2, 100.8) for s in np.sort(rng.uniform(0.0, truth, 5))]
        rows = prior + inside + [(truth, 100.9, 101.5)]
        result = absorption.absorption_time(*quotes(sorted(rows)), T0_NS)
        assert result["absorption_seconds"] == pytest.approx(truth, abs=2e-9)


def test_curve_is_the_share_at_or_before_each_grid_point():
    seconds = np.array([0.0, 10.0, 30.0, 3599.0, np.nan])
    curve = absorption.absorption_curve(seconds)
    assert curve[0] == pytest.approx(1 / 5) and curve[1] == pytest.approx(3 / 5)
    assert curve[-1] == pytest.approx(4 / 5)
    assert curve.shape == common.TIME_GRID.shape


def test_curve_marks_grid_points_past_the_window_as_missing():
    curve = absorption.absorption_curve(np.array([10.0, 20.0]), observable_seconds=1800.0)
    assert np.all(np.isfinite(curve[common.TIME_GRID <= 1800.0])) and np.all(np.isnan(curve[common.TIME_GRID > 1800.0]))


def test_curve_over_no_instruments_is_missing():
    assert np.all(np.isnan(absorption.absorption_curve(np.empty(0))))


def test_curve_recovers_known_exit_times_with_excluded_instruments():
    """Instrument-level simulation: 30 instruments, 5 excluded for having no prior quote, known exit times."""
    rng = np.random.default_rng(3)
    exits = rng.uniform(0.0, 3600.0, 30)
    denominator = []
    expected = []
    for k, exit_seconds in enumerate(exits):
        prior = [(-20.0, 99.0, 101.0)] if k >= 5 else []
        rows = prior + [(float(exit_seconds), 101.0, 101.5)]
        result = absorption.absorption_time(*quotes(rows), T0_NS)
        if result["included"]:
            denominator.append(result["absorption_seconds"])
            expected.append(exit_seconds)
    curve = absorption.absorption_curve(np.array(denominator))
    truth = np.array([(np.array(expected) <= t).mean() for t in common.TIME_GRID])
    assert len(denominator) == 25 and np.max(np.abs(curve - truth)) == 0.0


GRID = np.arange(0.0, 3605.0, 5.0)


def test_half_life_of_a_step_up():
    mids = np.full(GRID.size, 100.0)
    mids[GRID >= 100.0] = 100.5
    seconds, status = absorption.half_life(GRID, mids)
    assert status == absorption.HALF_LIFE_OK and seconds == 100.0


def test_half_life_of_a_step_down_uses_the_sign_of_the_move():
    mids = np.full(GRID.size, 100.0)
    mids[GRID >= 200.0] = 99.0
    assert absorption.half_life(GRID, mids) == (200.0, absorption.HALF_LIFE_OK)


def test_half_life_ignores_early_overshoot_in_the_wrong_direction():
    mids = np.full(GRID.size, 100.0)
    mids[(GRID >= 10.0) & (GRID < 50.0)] = 99.0
    mids[GRID >= 500.0] = 101.0
    seconds, status = absorption.half_life(GRID, mids)
    assert status == absorption.HALF_LIFE_OK and seconds == 500.0


def test_half_life_takes_the_first_grid_time_at_half_the_move():
    mids = 100.0 + np.minimum(GRID / 1000.0, 1.0)
    seconds, status = absorption.half_life(GRID, mids)
    assert status == absorption.HALF_LIFE_OK and seconds == 500.0


def test_simulated_half_lives_are_recovered_on_the_grid():
    """Simulation recovery of the half-life: drifting paths whose half-way time is known in closed form."""
    rng = np.random.default_rng(4)
    for _ in range(50):
        duration = rng.uniform(20.0, 3400.0)
        size = float(rng.choice([1, -1])) * float(rng.uniform(0.05, 2.0))
        mids = 100.0 + size * np.minimum(GRID / duration, 1.0)
        seconds, status = absorption.half_life(GRID, mids)
        exact = duration / 2.0
        assert status == absorption.HALF_LIFE_OK
        assert seconds == pytest.approx(5.0 * np.ceil(exact / 5.0 - 1e-9))


def test_half_life_is_missing_for_a_move_below_two_ticks():
    mids = np.full(GRID.size, 100.0)
    mids[GRID >= 100.0] = 100.015
    assert absorption.half_life(GRID, mids) == (pytest.approx(float("nan"), nan_ok=True), absorption.MOVE_BELOW_TWO_TICKS)
    mids[GRID >= 100.0] = 100.02
    assert absorption.half_life(GRID, mids) == (100.0, absorption.HALF_LIFE_OK)


def test_half_life_needs_a_midpoint_at_the_start_and_at_the_horizon():
    mids = np.full(GRID.size, np.nan)
    mids[GRID > 100.0] = 100.0
    assert absorption.half_life(GRID, mids)[1] == absorption.NO_MIDPOINT_BEFORE_T0
    assert absorption.half_life(GRID[:100], np.full(100, 100.0))[1] == absorption.NO_MIDPOINT_AT_HORIZON


def test_r0_is_the_ratio_with_percentile_interval():
    draws = np.log(np.column_stack((np.array([0.02, 0.03, 0.04]), np.array([0.01, 0.01, 0.01]))))
    estimate, lower, upper = absorption.r0_with_interval(0.02, 0.005, draws)
    assert estimate == pytest.approx(4.0) and lower == pytest.approx(np.exp(np.quantile(np.log([2.0, 3.0, 4.0]), 0.025))) and upper == pytest.approx(np.exp(np.quantile(np.log([2.0, 3.0, 4.0]), 0.975)))


def test_r0_interval_is_missing_when_too_many_draws_failed():
    draws = np.array([[np.nan, np.nan], [np.log(0.02), np.log(0.01)]])
    assert np.isnan(absorption.r0_with_interval(0.02, 0.01, draws)[1])


def test_clock_rule_examples():
    assert absorption.clock_share(0.4, 0.2, 0.6) == pytest.approx(1.0 - 0.6 * (0.4 / 0.8))
    assert absorption.clock_share(1.0, 0.2, 0.6) == pytest.approx(1.0)
    assert absorption.clock_share(0.0, 0.0, 0.5) == pytest.approx(0.5)
    assert absorption.clock_share(0.5, 1.0, 1.0) == pytest.approx(0.5)
    assert np.isnan(absorption.clock_share(float("nan"), 0.2, 0.6))


def test_clock_rule_stays_between_the_observed_share_and_one():
    rng = np.random.default_rng(5)
    for _ in range(500):
        y, seen = rng.uniform(0, 1), rng.uniform(0, 0.99)
        ahead = rng.uniform(seen, 1.0)
        prediction = absorption.clock_share(y, seen, ahead)
        assert y - 1e-12 <= prediction <= 1.0 + 1e-12


def test_clock_step_uses_earlier_events_and_recovers_a_noiseless_sir():
    truth = np.array([0.02, 0.005])
    curve = common.mean_field_curve(compartment_sir.SPEC, truth)
    train = np.tile(curve, (12, 1))
    rows = absorption.clock_step(train, curve, mean_model="ode", n_starts=1)
    assert len(rows) == 6
    row = next(item for item in rows if item["observation_seconds"] == 300.0 and item["horizon_seconds"] == 3600.0)
    assert row["beta"] == pytest.approx(0.02, rel=0.02) and row["gamma"] == pytest.approx(0.005, rel=0.02)
    assert row["benchmark_prediction"] == pytest.approx(curve[int(np.searchsorted(common.TIME_GRID, 3600.0))])
    assert row["fit_events"] == 12
    assert row["clock_abs_error"] < 0.02


def test_clock_step_shifts_the_prediction_towards_the_observed_share():
    curve = common.mean_field_curve(compartment_sir.SPEC, np.array([0.02, 0.005]))
    faster = np.clip(curve + 0.1, 0.0, 1.0)
    rows = absorption.clock_step(np.tile(curve, (6, 1)), faster, mean_model="ode", n_starts=1)
    row = next(item for item in rows if item["observation_seconds"] == 900.0 and item["horizon_seconds"] == 1800.0)
    assert row["clock_prediction"] > row["benchmark_prediction"]


def test_paired_mae_difference_matches_means_and_is_reproducible():
    clock, benchmark = np.array([0.1, 0.2, 0.3, 0.2]), np.array([0.2, 0.2, 0.5, 0.3])
    out = absorption.paired_mae_difference(clock, benchmark, rng=np.random.default_rng(6))
    again = absorption.paired_mae_difference(clock, benchmark, rng=np.random.default_rng(6))
    assert out["mae_clock"] == pytest.approx(0.2) and out["mae_benchmark"] == pytest.approx(0.3)
    assert out["mae_difference"] == pytest.approx(-0.1)
    assert out["difference_lower"] <= out["mae_difference"] <= out["difference_upper"]
    assert out == again


def test_dense_exponential_and_uniformization_agree_for_the_real_chain_size():
    """For SIR and SIRS with N = 30 the dense matrix exponential and uniformisation give the same means."""
    sir = common.FiniteChain(compartment_sir.SPEC, common.N_NODES)
    sirs = common.FiniteChain(compartment_sirs.SPEC, common.N_NODES)
    dense_theta = np.array([0.15, 0.002])
    sparse_sir = sir.mean_curve(dense_theta)
    dense_sir = sir._dense_mean_curve(dense_theta, common.TIME_GRID, common.STEP_SECONDS)
    assert np.max(np.abs(sparse_sir - dense_sir)) < 1e-8
    theta_sirs = np.array([0.15, 0.002, 0.001])
    assert np.max(np.abs(sirs.mean_curve(theta_sirs) - sirs._dense_mean_curve(theta_sirs, common.TIME_GRID, common.STEP_SECONDS))) < 1e-8


