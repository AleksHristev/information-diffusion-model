"""The run grid on seeded random flow: every run against a separate reference replay, sensitivity settings and chunk boundaries."""

import pytest

from information_diffusion.market_making import engine as engine_module
from information_diffusion.market_making import params as pm
from information_diffusion.market_making.engine import FILL_MODELS, run_backtest
from information_diffusion.market_making.grid import run_grid
from information_diffusion.market_making.market import MarketData
from information_diffusion.market_making.rule6_scheduled_avellaneda_stoikov import ScheduledAvellanedaStoikov
from information_diffusion.simulation import order_flow as flow


@pytest.fixture(scope="module")
def grid_runs():
    random_flow = flow.random_flow(7, n_quotes=1_500)
    market = MarketData.from_arrays(**random_flow.as_arrays())
    window = (random_flow.quotes[0][0], random_flow.end_ns())
    params = pm.EngineParams(inventory_limit=300)
    rules = flow.standard_rules(random_flow, t0_ns=(window[0] + window[1]) // 2)
    return run_grid(market, rules, window, params), random_flow, rules, params


def test_every_run_reconciles_exactly(grid_runs):
    runs, _, _, params = grid_runs
    for run in runs:
        assert run.reconciles(), (run.rule, run.fill_model, run.latency_ns)
        assert len(run.fills) == 0 or abs(run.fills.inventory_after).max() <= params.inventory_limit


def test_every_run_matches_the_reference_replay(grid_runs):
    runs, random_flow, rules, params = grid_runs
    per_rule = len(FILL_MODELS) * len(pm.LATENCY_GRID_NS)
    for index, run in enumerate(runs):
        rule = rules[index // per_rule]
        checks = flow.reference_agrees(run, random_flow, rule, params)
        assert all(checks.values()), (run.rule, run.fill_model, run.latency_ns, checks)


def test_at_least_one_fill_on_most_runs(grid_runs):
    runs, *_ = grid_runs
    assert sum(len(run.fills) > 0 for run in runs) > 60


def test_pessimistic_never_fills_more_shares_than_optimistic_on_rule1_0ms(grid_runs):
    runs, *_ = grid_runs
    pessimistic = next(r for r in runs if r.rule == "rule1_fixed_spread" and r.fill_model == "pessimistic" and r.latency_ns == 0)
    optimistic = next(r for r in runs if r.rule == "rule1_fixed_spread" and r.fill_model == "optimistic" and r.latency_ns == 0)
    assert int(pessimistic.fills.qty.sum()) <= int(optimistic.fills.qty.sum())


def test_sensitivity_parameters_are_expressible():
    random_flow = flow.random_flow(9, n_quotes=800)
    market = MarketData.from_arrays(**random_flow.as_arrays())
    window = (random_flow.quotes[0][0], random_flow.end_ns())
    rules = flow.standard_rules(random_flow, t0_ns=(window[0] + window[1]) // 2)
    rule1 = rules[0]
    for params in (pm.EngineParams(inventory_limit=500), pm.EngineParams(inventory_limit=2_000), pm.EngineParams(take_fee_cents=0.25), pm.EngineParams(add_rebate_cents=0.15), pm.EngineParams(add_rebate_cents=0.25)):
        run = run_backtest(market, rule1, "pessimistic", pm.PRIMARY_LATENCY_NS, params, window=window)
        assert run.reconciles()
    rule6 = rules[-1]
    for factor in pm.SENSITIVITY_RULE6_FACTORS:
        for decay in pm.SENSITIVITY_RULE6_DECAY_ENDS_S:
            variant = ScheduledAvellanedaStoikov(rule6.params, rule6.t0s_ns, factor=factor, decay_end_s=decay)
            assert run_backtest(market, variant, "pessimistic", pm.PRIMARY_LATENCY_NS, window=window).reconciles()


@pytest.mark.parametrize("first,smallest,largest", [(1, 1, 1), (2, 1, 3), (5, 2, 11)])
def test_chunk_boundaries_do_not_change_any_result(monkeypatch, first, smallest, largest):
    monkeypatch.setattr(engine_module, "_FIRST_CHUNK", first)
    monkeypatch.setattr(engine_module, "_SMALLEST_CHUNK", smallest)
    monkeypatch.setattr(engine_module, "_LARGEST_CHUNK", largest)
    random_flow = flow.random_flow(13, n_quotes=600)
    market = MarketData.from_arrays(**random_flow.as_arrays())
    window = (random_flow.quotes[0][0], random_flow.end_ns())
    params = pm.EngineParams(inventory_limit=300)
    rules = flow.standard_rules(random_flow, t0_ns=(window[0] + window[1]) // 2)
    for run, rule in zip(run_grid(market, rules, window, params), [r for r in rules for _ in range(len(FILL_MODELS) * len(pm.LATENCY_GRID_NS))]):
        checks = flow.reference_agrees(run, random_flow, rule, params)
        assert all(checks.values()) and run.reconciles(), (first, run.rule, run.fill_model, run.latency_ns, checks)
