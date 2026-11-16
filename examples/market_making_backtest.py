"""Backtest rule 2 (Avellaneda-Stoikov) against the pause rule on simulated order flow around scheduled releases.

Each release window lasts ten minutes. The NBBO is a one-tick market on a random walk; one millisecond after the
release the midpoint jumps ten ticks in a random direction, and a burst of trades at the new touch follows within
50 milliseconds. Rule 2 quotes throughout and its resting quotes are exposed to the jump until its 5 ms latency has
passed; the pause rule is rule 2 with its quotes pulled from one second before each release to five seconds after.
Both run under the two fill models. Results are per share, in basis points of the midpoint: spread captured is the
distance of the fill price from the midpoint at the fill, the 5 s mark-out is the later move of the midpoint in the
direction of the fill, and the edge is their sum. Negative values are adverse selection.
"""

import numpy as np

from information_diffusion.market_making.engine import run_backtest
from information_diffusion.market_making.market import MarketData
from information_diffusion.market_making.pause_rule import PauseRule
from information_diffusion.market_making.rule2_avellaneda_stoikov import ASParams, AvellanedaStoikov

NS, MS = 1_000_000_000, 1_000_000
TICK = 0.01
rng = np.random.default_rng(5)
params = ASParams(gamma=0.01, sigma=0.01, k=400.0)
n_releases, half_window_s, latency_ns = 12, 300, 5 * MS


def release_flow(t0_ns: int) -> dict:
    """NBBO updates and trades for one window centred on a release at t0_ns, as arguments of MarketData.from_arrays."""
    start, end = t0_ns - half_window_s * NS, t0_ns + half_window_s * NS
    t = start + np.cumsum(rng.exponential(50 * MS, 40 * half_window_s).astype(np.int64) + 1)
    t = np.sort(np.append(t[t < end], t0_ns + 1 * MS))
    steps = np.where(rng.random(t.size) < 0.05, rng.choice([-1, 1], t.size), 0)
    jump = int(rng.choice([-10, 10]))
    bid_ticks = 10_000 + np.cumsum(steps) + np.where(t >= t0_ns + 1 * MS, jump, 0)
    bid, ask = bid_ticks * TICK, (bid_ticks + 1) * TICK
    size = rng.integers(1, 6, t.size) * 100

    background = np.sort(rng.uniform(t[0], end, 2 * half_window_s)).astype(np.int64)
    burst = np.sort(t0_ns + 1 * MS + rng.integers(1, 50 * MS, 30)).astype(np.int64)
    trade_t = np.concatenate([background, burst])
    k = np.searchsorted(t, trade_t, side="right") - 1
    buy = np.concatenate([rng.random(background.size) < 0.5, np.full(burst.size, jump > 0)])
    through = np.concatenate([rng.random(background.size) < 0.3, np.zeros(burst.size, dtype=bool)])
    price = np.where(buy, ask[k] + through * TICK, bid[k] - through * TICK)
    order = np.argsort(trade_t, kind="stable")
    return {"quote_t": t, "bid": np.round(bid, 2), "ask": np.round(ask, 2), "bid_size": size, "ask_size": size[::-1].copy(),
            "trade_t": trade_t[order], "trade_price": np.round(price[order], 2), "trade_size": np.full(trade_t.size, 300)}


def measures(result, t0_ns: int) -> dict:
    """Share-weighted sums of one run, over all fills and over the fills from 1 s before to 5 s after the release."""
    f = result.fills
    qty = f.qty.astype(float)
    spread, markout = result.spread_captured_bps(), result.markout_bps[5]
    defined = np.isfinite(markout)
    near = (f.t_ns >= t0_ns - 1 * NS) & (f.t_ns < t0_ns + 5 * NS)
    edge = np.where(defined, spread + np.nan_to_num(markout), 0.0)
    return {"fills": len(f), "near": int(near.sum()), "shares": qty.sum(), "spread": spread @ qty,
            "markout": np.nan_to_num(markout) @ qty, "defined": qty[defined].sum(), "edge": edge @ qty,
            "near_edge": edge[near] @ qty[near], "near_defined": qty[near & defined].sum(), "inventory": int(np.abs(f.inventory_after).max()) if len(f) else 0}


def ratio(rows: list[dict], key: str, weight: str) -> float:
    """Share-weighted mean over the releases in rows."""
    total = sum(r[weight] for r in rows)
    return sum(r[key] for r in rows) / total if total else float("nan")


def shown(value: float, width: int) -> str:
    """The value to two decimals, or "no fills" when it is undefined."""
    return f"{value:>{width}.2f}" if np.isfinite(value) else f"{'no fills':>{width}}"


t0s = [(1000 + 2 * half_window_s * i) * NS for i in range(n_releases)]
markets = [MarketData.from_arrays(**release_flow(t0)) for t0 in t0s]
print(f"{n_releases} simulated releases, windows of {2 * half_window_s} s, latency {latency_ns / MS:.0f} ms; per share, basis points")
for fill_model in ("pessimistic", "optimistic"):
    runs, label = {}, ""
    for name in ("rule 2", "pause rule"):
        runs[name] = []
        for t0, market in zip(t0s, markets):
            rule = AvellanedaStoikov(params) if name == "rule 2" else PauseRule(params, (t0,), -1 * NS, 5 * NS)
            result = run_backtest(market, rule, fill_model, latency_ns, window=(t0 - half_window_s * NS, t0 + half_window_s * NS))
            runs[name].append(measures(result, t0))
            label = result.fill_label
    print(f"fill model: {label if label.startswith(fill_model) else f'{fill_model} ({label})'}")
    print(f"  {'rule':<12}{'fills':>7}{'near release':>14}{'spread captured':>17}{'5 s mark-out':>14}{'edge':>8}{'edge near release':>19}{'max |inventory|':>17}")
    for name, rows in runs.items():
        print(f"  {name:<12}{sum(r['fills'] for r in rows):>7}{sum(r['near'] for r in rows):>14}{shown(ratio(rows, 'spread', 'shares'), 17)}"
              f"{shown(ratio(rows, 'markout', 'defined'), 14)}{shown(ratio(rows, 'edge', 'defined'), 8)}{shown(ratio(rows, 'near_edge', 'near_defined'), 19)}"
              f"{max(r['inventory'] for r in rows):>17}")
    difference = ratio(runs["pause rule"], "edge", "defined") - ratio(runs["rule 2"], "edge", "defined")
    draws = []
    for _ in range(4999):
        pick = rng.integers(0, n_releases, n_releases)
        draws.append(ratio([runs["pause rule"][i] for i in pick], "edge", "defined") - ratio([runs["rule 2"][i] for i in pick], "edge", "defined"))
    lower, upper = np.nanpercentile(draws, [2.5, 97.5])
    print(f"  edge per share, pause rule minus rule 2: {difference:+.3f} bp   95% bootstrap interval over releases [{lower:+.3f}, {upper:+.3f}]")
