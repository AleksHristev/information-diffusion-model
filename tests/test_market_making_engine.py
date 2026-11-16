"""Engine input handling, windowing and output contracts."""

import numpy as np
import pytest

from information_diffusion.market_making.engine import default_window, run_backtest
from information_diffusion.market_making.market import MarketData
from information_diffusion.market_making.params import EngineParams
from information_diffusion.market_making.rule1_fixed_spread import FixedSpread
from information_diffusion.simulation import order_flow as flow


def _flat_market(trades=()):
    build = flow.Flow().quote(0, 100.00, 100.02)
    for t, price, size in trades:
        build.trade(t, price, size)
    return MarketData.from_arrays(**build.as_arrays())


def test_invalid_quotes_are_dropped_and_counted():
    market = MarketData.from_arrays(np.array([0, 1, 2, 3], dtype=np.int64), np.array([100.0, 100.0, 0.0, 100.0]), np.array([100.02, 100.0, 100.02, 99.99]),
                                    np.array([100, 100, 100, 100]), np.array([100, 100, 100, 100]), np.array([], dtype=np.int64), np.array([]), np.array([], dtype=np.int64))
    assert market.dropped_quotes == 3 and market.quote_t.tolist() == [0]


def test_prices_become_integer_micro_dollars_and_seconds_are_converted():
    market = MarketData.from_arrays([0.5], [100.0], [100.02], [100], [200], [0.75], [100.0037], [300], ts_unit="s")
    assert market.quote_t.tolist() == [500_000_000]
    assert market.trade_t.tolist() == [750_000_000]
    assert market.bid.tolist() == [100_000_000] and market.ask.tolist() == [100_020_000]
    assert market.trade_price.tolist() == [100_003_700]


def test_unsorted_timestamps_are_rejected():
    with pytest.raises(ValueError):
        MarketData.from_arrays([1, 0], [100.0, 100.0], [100.02, 100.02], [1, 1], [1, 1], [], [], [])


def test_trades_outside_the_window_do_not_fill():
    market = _flat_market([(5 * flow.MS, 100.00, 100)])
    result = run_backtest(market, FixedSpread(0.01), "optimistic", 0, window=(0, 4 * flow.MS))
    assert len(result.fills) == 0


def test_window_decisions_only_inside_the_window():
    build = flow.Flow().quote(0, 100.00, 100.02).quote(5 * flow.MS, 100.05, 100.07).trade(6 * flow.MS, 100.05, 100)
    market = MarketData.from_arrays(**build.as_arrays())
    result = run_backtest(market, FixedSpread(0.01), "optimistic", 0, window=(0, 4 * flow.MS), record_orders=True)
    assert [row[0] for row in result.orders] == [0, 0]


def test_default_window_covers_every_update():
    market = _flat_market()
    assert default_window(market) == (0, 1)


def test_summary_contains_no_cash_amounts():
    market = _flat_market([(1 * flow.MS, 100.00, 100)])
    result = run_backtest(market, FixedSpread(0.01), "optimistic", 0, window=(0, 10 * flow.MS))
    summary = result.summary()
    assert not any("cash" in key or "pnl" in key.lower() or "dollar" in key.lower() for key in summary)
    assert summary["fill_rate"] == 1.0


def test_fill_rate_is_filled_shares_over_market_volume():
    market = _flat_market([(1 * flow.MS, 100.00, 100), (2 * flow.MS, 100.00, 400)])
    result = run_backtest(market, FixedSpread(0.01), "optimistic", 0, window=(0, 10 * flow.MS))
    assert result.summary()["fill_rate"] == pytest.approx(100 / 500)


def test_time_weighted_inventory_by_hand():
    market = _flat_market([(5 * flow.MS, 100.00, 100)])
    result = run_backtest(market, FixedSpread(0.01), "optimistic", 0, window=(0, 10 * flow.MS))
    mean, variance = result.time_weighted_inventory()
    assert mean == pytest.approx(50.0) and variance == pytest.approx(2500.0)


def test_negative_latency_is_rejected():
    with pytest.raises(ValueError):
        run_backtest(_flat_market(), FixedSpread(0.01), "optimistic", -1)


def test_engine_params_reject_invalid_values():
    with pytest.raises(ValueError):
        EngineParams(inventory_limit=0)
    with pytest.raises(ValueError):
        EngineParams(take_fee_cents=-0.1)
