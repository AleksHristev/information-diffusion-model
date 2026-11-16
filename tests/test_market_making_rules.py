"""Quoting rules 1 to 6 produce the prices their formulas imply at known times."""

import datetime as dt
import math

import numpy as np
import pytest

from information_diffusion.market_making.quote_terms import QuoteTerms, as_of, quote_ticks
from information_diffusion.market_making.rule1_fixed_spread import FixedSpread
from information_diffusion.market_making.rule2_avellaneda_stoikov import ASParams, AvellanedaStoikov, as_skew_and_half_spread
from information_diffusion.market_making.rule3_hawkes_intensity import intensity_multiplier
from information_diffusion.market_making.rule6_scheduled_avellaneda_stoikov import ScheduledAvellanedaStoikov, fomc_repeat_ns, schedule_multiplier, schedule_times
from information_diffusion.simulation import order_flow as flow


@pytest.mark.parametrize("case", flow.rule_quote_cases(), ids=lambda case: case.name)
def test_each_rule_quotes_the_hand_derived_prices(case):
    terms = case.rule.quote_terms(np.array(case.t_ns, dtype=np.int64), np.array(case.mid), case.window_end_ns)
    bid, ask = quote_ticks(terms, case.inventory, 0.01)
    assert (bid * 10_000).tolist() == [round(x * 1e6) for x in case.bid]
    assert (ask * 10_000).tolist() == [round(x * 1e6) for x in case.ask]


def test_avellaneda_stoikov_terms_match_the_formula():
    params = ASParams(gamma=0.5, sigma=0.02, k=20.0)
    skew, half = as_skew_and_half_spread(np.array([0], dtype=np.int64), 100 * flow.S, params)
    variance_term = 0.5 * 0.02 ** 2 * 100.0
    assert skew[0] == pytest.approx(variance_term)
    assert half[0] == pytest.approx(0.5 * variance_term + math.log(1 + 0.5 / 20.0) / 0.5)


def test_time_to_horizon_is_not_negative_after_the_window():
    params = ASParams(gamma=0.5, sigma=0.02, k=20.0)
    skew, half = as_skew_and_half_spread(np.array([200 * flow.S], dtype=np.int64), 100 * flow.S, params)
    assert skew[0] == 0.0 and half[0] == pytest.approx(math.log1p(0.5 / 20.0) / 0.5)


def test_rounding_snaps_float_noise_to_the_grid():
    terms = QuoteTerms(np.array([100.01]), np.zeros(1), np.array([0.01]))
    bid, ask = quote_ticks(terms, 0, 0.01)
    assert bid.tolist() == [10000] and ask.tolist() == [10002]


def test_half_spread_floor_keeps_the_ask_above_the_bid():
    bid, ask = quote_ticks(QuoteTerms(np.array([100.0]), np.zeros(1), np.array([0.0])), 0, 0.01)
    assert ask[0] > bid[0]


def test_as_of_never_looks_ahead_and_uses_the_default_before_the_first_value():
    series_t = np.array([100, 200], dtype=np.int64)
    values = np.array([1.5, 2.5])
    result = as_of(series_t, values, np.array([50, 100, 150, 200, 250], dtype=np.int64), default=9.0)
    assert result.tolist() == [9.0, 1.5, 1.5, 2.5, 2.5]


def test_as_of_rejects_non_finite_input_values():
    with pytest.raises(ValueError):
        as_of(np.array([100], dtype=np.int64), np.array([np.nan]), np.array([200], dtype=np.int64), default=1.0)


def test_intensity_multiplier_maps_to_sqrt_with_floor_cap_and_baseline():
    t = np.array([0, 10, 20, 30], dtype=np.int64)
    series_t = np.array([10, 20, 30], dtype=np.int64)
    multiplier = intensity_multiplier(t, series_t, np.array([4.0, 16.0, 0.25]), baseline=1.0, cap=3.0)
    assert multiplier.tolist() == [1.0, 2.0, 3.0, 1.0]


def test_schedule_multiplier_boundary_values_by_hand():
    s = flow.S
    t0 = 500 * s
    times = np.array([t0 - 60 * s - 1, t0 - 60 * s, t0, t0 + 120 * s, t0 + 360 * s, t0 + 600 * s], dtype=np.int64)
    assert schedule_multiplier(times, [t0]).tolist() == [1.0, 2.0, 2.0, 2.0, 1.5, 1.0]


def test_fomc_repeat_is_1430_new_york_in_both_dst_states():
    jan = int(dt.datetime(2026, 1, 28, 19, 0, tzinfo=dt.UTC).timestamp()) * flow.S
    jun = int(dt.datetime(2026, 6, 17, 18, 0, tzinfo=dt.UTC).timestamp()) * flow.S
    assert fomc_repeat_ns(jan) == int(dt.datetime(2026, 1, 28, 19, 30, tzinfo=dt.UTC).timestamp()) * flow.S
    assert fomc_repeat_ns(jun) == int(dt.datetime(2026, 6, 17, 18, 30, tzinfo=dt.UTC).timestamp()) * flow.S
    assert schedule_times(jan, is_fomc=False) == (jan,)
    assert len(schedule_times(jan, is_fomc=True)) == 2


@pytest.mark.parametrize("case", flow.fomc_schedule_cases(), ids=lambda case: case["label"])
def test_fomc_quotes_follow_the_repeated_schedule(case):
    t0s = schedule_times(case["t0_ns"], is_fomc=True)
    times = np.array(case["times"], dtype=np.int64)
    assert schedule_multiplier(times, t0s).tolist() == pytest.approx(list(case["expected_multiplier"]), abs=1e-12)
    terms = ScheduledAvellanedaStoikov(case["params"], t0s).quote_terms(times, np.full(len(times), 100.0), case["t0_ns"] + 7_200 * flow.S)
    bid, ask = quote_ticks(terms, 0, 0.01)
    assert (bid * 10_000).tolist() == [round(x * 1e6) for x in case["bid"]]
    assert (ask * 10_000).tolist() == [round(x * 1e6) for x in case["ask"]]


def test_rule1_ignores_inventory_and_rule2_skews_it():
    t, mid = np.array([0], dtype=np.int64), np.array([100.0])
    fixed = FixedSpread(0.03).quote_terms(t, mid, 100 * flow.S)
    assert quote_ticks(fixed, 400, 0.01)[0].tolist() == quote_ticks(fixed, 0, 0.01)[0].tolist()
    params = ASParams(gamma=1e-3, sigma=0.1, k=50.0)
    skewed = AvellanedaStoikov(params).quote_terms(t, mid, 100 * flow.S)
    assert quote_ticks(skewed, 400, 0.01)[0][0] == quote_ticks(skewed, 0, 0.01)[0][0] - 40


