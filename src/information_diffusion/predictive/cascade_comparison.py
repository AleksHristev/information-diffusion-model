"""Rolling out-of-sample comparison of the cascade models M0 to M4 by mean Bernoulli log-loss per event."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit

from information_diffusion.predictive import cascade_data as cd
from information_diffusion.predictive import cascade_hazard as hz
from information_diffusion.predictive import independent_cascade as ic
from information_diffusion.statistical_framework import giacomini_white as gw

PROBABILITY_FLOOR = 1e-06
EM_TOLERANCE = 1e-09
EM_MAX_ITERATIONS = 20000
FAMILIES = {'CPI and payrolls': {'groups': ('CPI', 'Payrolls'), 'window': 12, 'shocks': ('08:30',)}, 'FOMC': {'groups': ('FOMC 14:00', 'FOMC 14:30'), 'window': 6, 'shocks': ('14:00', '14:30')}}
SIZE_HORIZON_SECONDS = 60.0


@dataclass(frozen=True)
class WindowResult:
    """Result of one rolling window.

    Fields: losses maps (model, adjacency set) to (summed log-loss, at-risk steps) on the test cascades; converged maps the
    same keys to convergence flags; parameters lists the edge and share parameters when kept; background holds the M0
    coefficients.
    """

    losses: dict
    converged: dict
    parameters: list
    background: np.ndarray


def _loss(probabilities, events, trials) -> float:
    return float(-hz.cell_log_likelihood(probabilities, events, trials, PROBABILITY_FLOOR).sum())


def fit_window(train: list[cd.CascadeRows], test: list[cd.CascadeRows], family: str, g: cd.Grid, adjacency: dict[str, np.ndarray], *, keep_parameters: bool = False) -> WindowResult:
    """Fit M0 to M4 on the cascades in train and score them on the cascades in test.

    M0 is the background hazard, M1 and M2 add the share of active or recently active nodes, M3 is the Independent Cascade
    with the M0 background (noisy-OR on attempt steps, M0 elsewhere) and M4 the Linear Threshold hazard. With an
    adjacency set that has no edges, M3 and M4 score as M0.
    Arguments: train and test are the at-risk rows (cascade_data.rows) of the fitted and the scored cascades; family is
    a key of FAMILIES; g is the grid; adjacency maps a set name to its (edges, 2) parent and child node indices, which
    index cascade_data.INSTRUMENTS; keep_parameters keeps the fitted edge and share parameters.
    Returns: a WindowResult.
    """
    step_cols = cd.step_columns(family, g)
    trials = float(sum(r.trials for r in test))
    losses, parameters = {}, []
    d0 = cd.design_m0(train, step_cols)
    sd, lower = cd.prior(step_cols, 0, None)
    m0 = hz.fit(d0.matrix, d0.events, d0.trials, sd, lower)
    b0 = m0.coefficients
    t0 = cd.design_m0(test, step_cols)
    losses[("M0", "")] = (_loss(hz.probability(t0.matrix, b0), t0.events, t0.trials), trials)
    converged = {("M0", ""): m0.converged}
    for model in ("M1", "M2"):
        d = cd.design_share(train, step_cols, model)
        sd, lower = cd.prior(step_cols, 1, "share")
        fit = hz.fit(d.matrix, d.events, d.trials, sd, lower, start=np.concatenate([b0, [0.0]]))
        t = cd.design_share(test, step_cols, model)
        losses[(model, "")] = (_loss(hz.probability(t.matrix, fit.coefficients), t.events, t.trials), trials)
        converged[(model, "")] = fit.converged
        if keep_parameters:
            parameters.append({"model": model, "adjacency": "", "parent": "", "child": "", "parameter": "share coefficient", "value": float(fit.coefficients[-1]), "identified": True, "attempts": np.nan, "converged": fit.converged})
    for name, edges in adjacency.items():
        edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        if edges.shape[0] == 0:
            losses[("M3", name)] = losses[("M0", "")]
            losses[("M4", name)] = losses[("M0", "")]
            converged[("M3", name)] = converged[("M4", name)] = m0.converged
            continue
        d4 = cd.design_threshold(train, step_cols, edges)
        sd, lower = cd.prior(step_cols, len(edges), "edge_weight")
        m4 = hz.fit(d4.matrix, d4.events, d4.trials, sd, lower, start=np.concatenate([b0, np.zeros(len(edges))]))
        t4 = cd.design_threshold(test, step_cols, edges)
        losses[("M4", name)] = (_loss(hz.probability(t4.matrix, m4.coefficients), t4.events, t4.trials), trials)
        converged[("M4", name)] = m4.converged
        a3 = cd.design_attempts(train, step_cols, edges)
        if a3.activated.size:
            m3 = ic.fit_with_background(a3.incidence, np.minimum(expit(a3.background @ b0), 1.0 - 1e-12), a3.activated, tolerance=EM_TOLERANCE, max_iterations=EM_MAX_ITERATIONS)
            p = m3.probability
        else:
            m3, p = None, np.zeros(len(edges))
        t3 = cd.design_attempts(test, step_cols, edges)
        total = losses[("M0", "")][0]
        if t3.activated.size:
            h = expit(t3.background @ b0)
            total += -_loss(h, t3.activated, np.ones(h.size)) + _loss(ic.noisy_or(t3.incidence, h, p), t3.activated, np.ones(h.size))
        losses[("M3", name)] = (total, trials)
        converged[("M3", name)] = m0.converged and (m3 is None or m3.converged)
        if keep_parameters:
            for e, (v, w) in enumerate(edges):
                parameters.append({"model": "M3", "adjacency": name, "parent": cd.INSTRUMENTS[v], "child": cd.INSTRUMENTS[w], "parameter": "edge probability p",
                                   "value": float(p[e]), "identified": bool(m3.identified[e]) if m3 is not None else False, "attempts": float(m3.attempts[e]) if m3 is not None else 0.0, "converged": bool(m3.converged) if m3 is not None else True})
                parameters.append({"model": "M4", "adjacency": name, "parent": cd.INSTRUMENTS[v], "child": cd.INSTRUMENTS[w], "parameter": "edge weight b",
                                   "value": float(m4.coefficients[-len(edges) + e]), "identified": True, "attempts": np.nan, "converged": m4.converged})
    return WindowResult(losses, converged, parameters, b0)


def rolling(cascade_list: list[cd.Cascade], family: str, g: cd.Grid, adjacency: dict[str, dict[str, np.ndarray]], *, keep_parameters: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Compute the loss of every model on every test event of a family, each fitted on its rolling window of earlier events.

    Arguments: cascade_list is the list of cascades, oldest event first (as from cascade_data.cascades); family is a
    key of FAMILIES, whose window gives the number of earlier events each fit uses; g is the grid; adjacency maps a
    test event id to a set name to (edges, 2) parent and child node indices; keep_parameters keeps the edge
    parameters.
    Returns: the event-loss table (mean log-loss per at-risk step of each model on each test event), the
    edge-parameter table (empty unless kept) and the M0 coefficients by test event.
    """
    events = list(dict.fromkeys(c.event_id for c in cascade_list if c.family == family))
    window = FAMILIES[family]["window"]
    by_event = {e: [cd.rows(c, g) for c in cascade_list if c.event_id == e] for e in events}
    loss_rows, parameter_rows, backgrounds = [], [], {}
    for k, test_event in enumerate(events[window:]):
        train_events = events[k: k + window]
        train = [r for e in train_events for r in by_event[e]]
        result = fit_window(train, by_event[test_event], family, g, adjacency[test_event], keep_parameters=keep_parameters)
        backgrounds[test_event] = result.background
        for (model, name), (total, trials) in result.losses.items():
            loss_rows.append({"family": family, "step_seconds": g.step, "horizon_seconds": g.horizon, "window_index": k, "test_event": test_event, "model": model, "adjacency": name, "loss": total / trials, "at_risk_steps": trials, "converged": result.converged[(model, name)]})
        for p in result.parameters:
            parameter_rows.append({"family": family, "step_seconds": g.step, "horizon_seconds": g.horizon, "window_index": k, "test_event": test_event, **p})
    return pd.DataFrame(loss_rows), pd.DataFrame(parameter_rows), backgrounds


def comparisons(losses: pd.DataFrame) -> pd.DataFrame:
    """Compare each model with M0 by the Giacomini-White test on the event loss differentials.

    Arguments: losses is the event-loss table from rolling.
    Returns: one row per family, step, horizon, model and adjacency set with the mean losses, the mean differential
    (M0 minus model), the counts of improved and equal events, and the Giacomini-White statistic and p-value.
    """
    rows = []
    keys = ["family", "step_seconds", "horizon_seconds"]
    for key, g in losses.groupby(keys, sort=False):
        base = g[g.model == "M0"].set_index("test_event").loss
        for (model, name), m in g[g.model != "M0"].groupby(["model", "adjacency"], sort=False):
            loss = m.set_index("test_event").loss.reindex(base.index)
            d = (base - loss).to_numpy()
            row = dict(zip(keys, key)) | {"model": model, "adjacency": name, "test_events": int(d.size), "mean_loss_m0": float(base.mean()), "mean_loss_model": float(loss.mean()),
                                          "mean_differential": float(d.mean()), "events_improved": int((d > 0).sum()), "events_equal": int((d == 0).sum())}
            try:
                test = gw.giacomini_white(d)
                row |= {"gw_t": test.statistic, "gw_p": test.p_value, "estimable": True}
            except ValueError as error:
                row |= {"gw_t": np.nan, "gw_p": np.nan, "estimable": False, "not_estimable_reason": str(error)}
            rows.append(row)
    return pd.DataFrame(rows)


def size_replicate(seed: np.random.SeedSequence, cascade_list: list[cd.Cascade], family: str, step: float, coefficients: np.ndarray, adjacency: dict) -> list[dict]:
    """Run one size-check replication: redraw every cascade of the family from M0 (no diffusion) and compare the models.

    The cascades are drawn on a grid with the given step and a horizon of SIZE_HORIZON_SECONDS (60 s).
    Arguments: seed is the seed sequence; cascade_list is the list of template cascades; family is a key of FAMILIES;
    step is the step length in seconds; coefficients are the M0 coefficients; adjacency is as in rolling.
    Returns: one record per model and adjacency set with the mean differential, improved events and the test result.
    """
    g = cd.grid(step, SIZE_HORIZON_SECONDS)
    step_cols = cd.step_columns(family, g)
    rng = np.random.default_rng(seed)
    simulated = [cd.simulate_m0(c, g, step_cols, coefficients, rng) for c in cascade_list if c.family == family]
    losses, _, _ = rolling(simulated, family, g, adjacency)
    out = comparisons(losses)
    return out[["model", "adjacency", "mean_differential", "events_improved", "gw_t", "gw_p", "estimable"]].to_dict("records")
