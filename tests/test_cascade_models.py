"""Recovery and consistency checks of the cascade hazard, background cascade and spread estimators on simulated cascades."""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from information_diffusion.predictive import cascade_comparison as cc
from information_diffusion.predictive import cascade_data as cd
from information_diffusion.predictive import cascade_hazard as hz
from information_diffusion.predictive import cascade_influence as ci
from information_diffusion.predictive import independent_cascade as ic

GRID = cd.grid(1.0, 60.0)
ACTIVE = np.arange(12)
FAMILY = "CPI and payrolls"


def simulate(rng, cascades: int, *, beta_share=0.0, weights=None, edge_p=None, edges=None, coefficients=None):
    """Cascades from a known model: M0 plus a share-active term (beta_share), M4 edge weights, or M3 edge probabilities."""
    step_cols = cd.step_columns(FAMILY, GRID)
    if coefficients is None:
        coefficients = np.zeros(cd.columns(step_cols, 0))
        coefficients[0] = -2.5
        coefficients[1: 1 + step_cols] = np.linspace(0.8, -0.8, step_cols)
        coefficients[1 + step_cols + ACTIVE] = np.linspace(-0.6, 0.6, ACTIVE.size)
    out = []
    nodes = np.zeros(cd.NODES, dtype=bool)
    nodes[ACTIVE] = True
    for _ in range(cascades):
        a = np.full(cd.NODES, GRID.steps)
        for s in range(GRID.steps):
            at_risk = nodes & (a >= GRID.steps)
            if not at_risk.any():
                break
            eta = coefficients[0] + coefficients[1 + GRID.bin_of_step[s]] + coefficients[1 + step_cols: 1 + step_cols + cd.NODES]
            eta = eta + beta_share * ((nodes & (a < s)).sum() / (nodes.sum() - 1))
            if weights is not None:
                for (v, w), b in zip(edges, weights):
                    if 1 <= s - a[v] <= 4:
                        eta[w] += b
            h = expit(eta)
            if edge_p is not None:
                fail = 1 - h
                for (v, w), p in zip(edges, edge_p):
                    if a[v] == s - 1:
                        fail[w] *= 1 - p
                h = 1 - fail
            fired = at_risk & (rng.random(cd.NODES) < h)
            a[fired] = s
        tau = np.where(nodes & (a < GRID.steps), a + 0.5, np.nan)
        out.append(cd.Cascade(FAMILY, f"e{len(out)}", "CPI", 0, nodes.copy(), tau))
    return out, coefficients


def test_rows_match_a_step_by_step_count():
    rng = np.random.default_rng(1)
    cascades, _ = simulate(rng, 5, beta_share=1.0)
    for c in cascades:
        r = cd.rows(c, GRID)
        a = cd.activation_steps(c, GRID)
        expected_trials = sum(min(a[w], GRID.steps - 1) + 1 for w in ACTIVE)
        assert r.trials == expected_trials
        assert r.base[3].sum() == (a[ACTIVE] < GRID.steps).sum()
        for model in ("M1", "M2"):
            column, node, value, trials, events = r.shares[model]
            assert trials.sum() == expected_trials and events.sum() == r.base[3].sum()
        column, node, value, trials, events = r.shares["M1"]
        for w in ACTIVE[:3]:
            for s in range(min(a[w], GRID.steps - 1) + 1):
                count = (a[ACTIVE] < s).sum()
                hit = (node == w) & (column == GRID.bin_of_step[s]) & np.isclose(value, count / (ACTIVE.size - 1))
                assert hit.sum() == 1


def test_hazard_recovers_the_share_coefficient():
    rng = np.random.default_rng(2)
    cascades, truth = simulate(rng, 600, beta_share=1.5)
    step_cols = cd.step_columns(FAMILY, GRID)
    rows = [cd.rows(c, GRID) for c in cascades]
    d = cd.design_share(rows, step_cols, "M1")
    sd, lower = cd.prior(step_cols, 1, "share")
    fit = hz.fit(d.matrix, d.events, d.trials, sd, lower)
    assert fit.converged
    assert abs(fit.coefficients[-1] - 1.5) < 0.25
    estimated = fit.coefficients[1 + step_cols + ACTIVE] - fit.coefficients[1 + step_cols + ACTIVE].mean()
    assert np.max(np.abs(estimated - (truth[1 + step_cols + ACTIVE] - truth[1 + step_cols + ACTIVE].mean()))) < 0.25


def test_hazard_recovers_non_negative_edge_weights():
    rng = np.random.default_rng(3)
    edges = np.array([[0, 5], [1, 6], [2, 7]])
    cascades, _ = simulate(rng, 800, weights=[2.0, 0.0, 1.0], edges=edges)
    step_cols = cd.step_columns(FAMILY, GRID)
    rows = [cd.rows(c, GRID) for c in cascades]
    d = cd.design_threshold(rows, step_cols, edges)
    sd, lower = cd.prior(step_cols, len(edges), "edge_weight")
    fit = hz.fit(d.matrix, d.events, d.trials, sd, lower)
    b = fit.coefficients[-3:]
    assert fit.converged and np.all(b >= 0)
    assert abs(b[0] - 2.0) < 0.3 and b[1] < 0.3 and abs(b[2] - 1.0) < 0.3


def test_background_cascade_recovers_edge_probabilities():
    rng = np.random.default_rng(4)
    edges = np.array([[0, 5], [1, 6], [2, 7]])
    cascades, truth = simulate(rng, 1500, edge_p=[0.6, 0.3, 0.0], edges=edges)
    step_cols = cd.step_columns(FAMILY, GRID)
    rows = [cd.rows(c, GRID) for c in cascades]
    attempts = cd.design_attempts(rows, step_cols, edges)
    fit = ic.fit_with_background(attempts.incidence, expit(attempts.background @ truth), attempts.activated)
    assert fit.converged and fit.identified.all()
    assert np.max(np.abs(fit.probability - np.array([0.6, 0.3, 0.0]))) < 0.06


def test_m3_and_m4_with_no_link_score_as_m0():
    rng = np.random.default_rng(5)
    cascades, _ = simulate(rng, 14)
    rows = [cd.rows(c, GRID) for c in cascades]
    result = cc.fit_window(rows[:12], rows[12:], FAMILY, GRID, {"empty": np.zeros((0, 2), dtype=np.int64)})
    assert result.losses[("M3", "empty")] == result.losses[("M0", "")] == result.losses[("M4", "empty")]


def test_m3_with_zero_probabilities_scores_as_m0():
    rng = np.random.default_rng(6)
    cascades, _ = simulate(rng, 3)
    rows = [cd.rows(c, GRID) for c in cascades]
    step_cols = cd.step_columns(FAMILY, GRID)
    d0 = cd.design_m0(rows, step_cols)
    b0 = np.zeros(d0.matrix.shape[1])
    b0[0] = -2.0
    edges = np.array([[0, 5], [1, 6]])
    attempts = cd.design_attempts(rows, step_cols, edges)
    h = expit(attempts.background @ b0)
    assert np.allclose(ic.noisy_or(attempts.incidence, h, np.zeros(2)), h)


def test_spread_of_a_chain_matches_its_closed_form():
    rng = np.random.default_rng(7)
    p = np.zeros((3, 3))
    p[0, 1], p[1, 2] = 0.6, 0.5
    live = ci.live_edges(p, 40000, rng)
    sigma = ci.spread(live, np.array([[True, False, False]]), steps=10)[0]
    assert abs(sigma - (1 + 0.6 + 0.6 * 0.5)) < 0.02
    assert abs(ci.spread(live, np.array([[True, False, False]]), steps=2)[0] - 1.6) < 0.02
    chosen, values = ci.greedy(live, 10, np.ones(3, dtype=bool), 1)
    assert chosen == [0]
