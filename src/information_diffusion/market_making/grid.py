"""Grid of backtest runs over quoting rules, fill models and latencies for one window."""

from __future__ import annotations

from collections.abc import Sequence

from information_diffusion.market_making.engine import FILL_MODELS, QuotingRule, rule_terms, simulate
from information_diffusion.market_making.market import MarketData
from information_diffusion.market_making.params import LATENCY_GRID_NS, EngineParams
from information_diffusion.market_making.results import RunResult


def run_grid(market: MarketData, rules: Sequence[QuotingRule], window: tuple[int, int], params: EngineParams = EngineParams(), *, latencies_ns: Sequence[int] = LATENCY_GRID_NS, fill_models: Sequence[str] = tuple(FILL_MODELS)) -> list[RunResult]:
    """Run every rule under every fill model and latency, evaluating each rule's quote terms once.

    Arguments: market is the MarketData; rules are the quoting rules; window is (start, end) in nanoseconds; params
    are the EngineParams; latencies_ns are the latencies in nanoseconds (LATENCY_GRID_NS by default); fill_models are
    the fill model names (all by default).
    Returns: the list of RunResult, rule by rule, then fill model, then latency.
    """
    runs = []
    for rule in rules:
        terms = rule_terms(market, rule, window)
        for fill_model in fill_models:
            for latency_ns in latencies_ns:
                runs.append(simulate(market, terms, rule.name, fill_model, latency_ns, params, window))
    return runs
