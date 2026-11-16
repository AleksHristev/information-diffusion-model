"""Exact reproduction of hand-derived synthetic scenarios: fills, inventory, mark-outs, cash and order prices."""

import pytest

from information_diffusion.market_making.engine import run_backtest
from information_diffusion.market_making.market import MarketData
from information_diffusion.simulation import order_flow as flow

SCENARIOS = flow.scenarios()


def _run(scenario):
    market = MarketData.from_arrays(**scenario.flow.as_arrays())
    return run_backtest(market, scenario.rule, scenario.fill_model, scenario.latency_ns, scenario.params, window=scenario.resolved_window(), record_orders=True)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda scenario: scenario.name)
def test_scenario_reproduces_exactly(scenario):
    result = _run(scenario)
    checks = flow.check_scenario(result, scenario)
    assert all(checks.values()), f"{scenario.name} failed: {checks}"
    assert all(flow.reference_agrees(result, scenario.flow, scenario.rule, scenario.params).values())


def _result(name):
    return _run(next(s for s in SCENARIOS if s.name == name))


def test_queue_case_fills_optimistic_and_not_pessimistic():
    assert len(_result("queue_not_exhausted_optimistic").fills) == 1
    assert len(_result("queue_not_exhausted_pessimistic").fills) == 0


def test_stale_pickoff_differs_at_500us_and_1ms():
    assert len(_result("latency_stale_pickoff_pessimistic_500000").fills) == 0
    assert len(_result("latency_stale_pickoff_pessimistic_1000000").fills) == 1


def test_maker_and_taker_fee_signs():
    maker = _result("fees_maker_base")
    assert maker.fills.fee_flow.tolist() == [200_000, 200_000]
    taker = _result("fees_taker_buy")
    assert taker.fills.fee_flow.tolist() == [-300_000]
    assert taker.fills.maker.tolist() == [False]


def test_markout_signs_are_opposite_for_buy_and_sell():
    buy = _result("markout_buy")
    sell = _result("markout_sell")
    for horizon in (1, 5, 30):
        assert buy.markout_ticks[horizon][0] == -sell.markout_ticks[horizon][0]


def test_markout_horizons_past_the_data_are_nan_and_later_quotes_are_used():
    past = _result("markout_past_end_of_data")
    assert past.markout_ticks[1][0] == 0.0
    assert past.markout_ticks[5][0] != past.markout_ticks[5][0]
    defined = _result("markout_past_window_end_defined")
    assert defined.markout_ticks[5][0] == pytest.approx(-10.0)


def test_markout_in_basis_points_is_ticks_scaled_by_the_fill_mid():
    buy = _result("markout_buy")
    assert buy.markout_bps[1][0] == pytest.approx(10_000.0 * 0.05 / 100.01)


def test_inventory_limit_never_exceeded_and_quotes_stop():
    capped = _result("limit_in_flight_order_capped")
    assert capped.fills.inventory_after.tolist() == [100]
    long_limit = _result("limit_long_stops_bids")
    assert long_limit.fills.inventory_after.tolist() == [100, 200, 100]
    short_limit = _result("limit_short_stops_asks")
    assert short_limit.fills.inventory_after.tolist() == [-100, 0]


def test_spread_captured_is_half_the_spread_for_a_touch_fill():
    result = _result("fees_maker_base")
    assert result.spread_captured_ticks().tolist() == [1.0, 1.0]
    assert result.spread_captured_bps()[0] == pytest.approx(1e4 * 0.01 / 100.01)
