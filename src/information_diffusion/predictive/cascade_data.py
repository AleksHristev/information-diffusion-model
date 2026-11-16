"""Cascades built from absorption times, and the at-risk rows and design matrices of the cascade models M0 to M4."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

INSTRUMENTS = ['SPY', 'QQQ', 'IWM', 'XLF', 'XLK', 'XLE', 'XLU', 'XLRE', 'XLY', 'XLP', 'XLV', 'XLI', 'SMH', 'KRE', 'XBI', 'XOP', 'IYR', 'XLB', 'SOXX', 'VNQ', 'XRT', 'XLC', 'IGV', 'XHB', 'VGT', 'ITB', 'IBB', 'OIH', 'XME', 'KBE']

FAMILIES = {'CPI and payrolls': {'groups': ('CPI', 'Payrolls'), 'window': 12, 'shocks': ('08:30',)}, 'FOMC': {'groups': ('FOMC 14:00', 'FOMC 14:30'), 'window': 6, 'shocks': ('14:00', '14:30')}}
SHOCK_OF_GROUP = {'CPI': 0, 'Payrolls': 0, 'FOMC 14:00': 0, 'FOMC 14:30': 1}
BIN_EDGES_SECONDS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0)
RECENT_STEPS = 4
PRIOR_SD = {'step': 10.0, 'instrument': 2.5, 'share': 10.0, 'edge_weight': 2.5}

NODES = len(INSTRUMENTS)
INDEX = {symbol: i for i, symbol in enumerate(INSTRUMENTS)}
_KEY_BASE = 64


@dataclass(frozen=True)
class Cascade:
    """One cascade: an event and shock with the absorption time of each node.

    Fields: family, event_id, group, shock index, nodes (boolean mask of included instruments) and tau (absorption time in
    seconds, NaN when not absorbed or not included).
    """

    family: str
    event_id: str
    group: str
    shock: int
    nodes: np.ndarray
    tau: np.ndarray


@dataclass(frozen=True)
class Grid:
    """Discrete time grid of a cascade.

    Fields: step and horizon in seconds, the number of steps, the number of step bins used and the bin of each step.
    """

    step: float
    horizon: float
    steps: int
    bins: int
    bin_of_step: np.ndarray


def family_of(group: str) -> str:
    """Return the name of the event family that contains the given group."""
    return next(name for name, spec in FAMILIES.items() if group in spec["groups"])


def cascades(absorption: pd.DataFrame) -> list[Cascade]:
    """Build one cascade per event and shock group, ordered by date, then event_id, then shock index.

    The date is read from the last ten characters of event_id, which must end in an ISO date (YYYY-MM-DD); the shock
    index of each group is given by SHOCK_OF_GROUP.
    Arguments: absorption is a table with event_id, group (a key of SHOCK_OF_GROUP), symbol (one of INSTRUMENTS),
    included (whether the instrument is a node of the cascade) and absorption_seconds columns.
    Returns: the list of cascades.
    """
    out = []
    frame = absorption.assign(date=absorption.event_id.str[-10:], shock=absorption.group.map(SHOCK_OF_GROUP))
    for (date, event_id, shock, group), g in frame.groupby(["date", "event_id", "shock", "group"], sort=True):
        nodes = np.zeros(NODES, dtype=bool)
        tau = np.full(NODES, np.nan)
        for r in g.itertuples():
            i = INDEX[r.symbol]
            nodes[i] = bool(r.included)
            tau[i] = r.absorption_seconds if bool(r.included) else np.nan
        out.append(Cascade(family_of(group), event_id, group, int(shock), nodes, tau))
    return out


def grid(step: float, horizon: float) -> Grid:
    """Build the grid of a step length and horizon in seconds; the horizon must be a whole number of steps.

    Arguments: step is the step length and horizon the horizon, both in seconds.
    Returns: a Grid whose steps are assigned to the bins of BIN_EDGES_SECONDS that they start in, numbering only the
    bins in use.
    """
    steps = int(round(horizon / step))
    if not np.isclose(steps * step, horizon):
        raise ValueError("the horizon must be a whole number of steps")
    starts = np.arange(steps) * step
    edges = np.asarray(BIN_EDGES_SECONDS)
    raw = np.searchsorted(edges, starts + 1e-12, side="right") - 1
    used, bin_of_step = np.unique(raw, return_inverse=True)
    return Grid(step, horizon, steps, int(used.size), bin_of_step.astype(np.int64))


def step_columns(family: str, g: Grid) -> int:
    """Return the number of shock and step-bin columns of a family on grid g."""
    return len(FAMILIES[family]["shocks"]) * g.bins


def activation_steps(c: Cascade, g: Grid) -> np.ndarray:
    """Compute the activation step of each node.

    Arguments: c is the cascade; g is the grid.
    Returns: floor(tau / step) when tau is below the horizon, otherwise the number of steps (never activated or not a node).
    """
    a = np.full(NODES, g.steps, dtype=np.int64)
    hit = c.nodes & np.isfinite(c.tau) & (c.tau < g.horizon)
    a[hit] = np.minimum(np.floor(c.tau[hit] / g.step).astype(np.int64), g.steps - 1)
    return a


def _runs(key: np.ndarray, last: np.ndarray, event_step: np.ndarray, nodes: np.ndarray):
    change = np.flatnonzero(np.diff(key)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [key.size]])
    node_ids = np.flatnonzero(nodes)
    trials = np.clip(np.minimum(ends[None, :], last[node_ids, None] + 1) - starts[None, :], 0, None)
    events = (starts[None, :] <= event_step[node_ids, None]) & (event_step[node_ids, None] < ends[None, :])
    w, j = np.nonzero(trials)
    keys, nodes_out = key[starts[j]], node_ids[w]
    combined = keys * NODES + nodes_out
    unique, inverse = np.unique(combined, return_inverse=True)
    return unique // NODES, unique % NODES, np.bincount(inverse, trials[w, j]), np.bincount(inverse, events[w, j].astype(float))


@dataclass
class CascadeRows:
    """At-risk rows of one cascade on one grid.

    Fields: the cascade and grid, activation steps a, last at-risk step of each node, cascade size, the shock and step-bin
    column of each step, the M0 binomial cells (column, node, trials, events) and the M1 and M2 share cells.
    """

    cascade: Cascade
    grid: Grid
    a: np.ndarray
    last: np.ndarray
    size: int
    column_of_step: np.ndarray
    base: tuple
    shares: dict = field(default_factory=dict)

    @property
    def trials(self) -> float:
        """Total number of at-risk node-steps of the cascade."""
        return float(self.base[2].sum())


def rows(c: Cascade, g: Grid) -> CascadeRows:
    """Build the at-risk rows of a cascade, counted by runs of equal covariates rather than step by step.

    M1 uses the count of nodes active before step s, M2 the count activated in the last RECENT_STEPS steps, each divided
    by the cascade size less one.
    Arguments: c is the cascade; g is the grid.
    Returns: a CascadeRows.
    """
    a = activation_steps(c, g)
    last = np.minimum(a, g.steps - 1)
    event_step = np.where(a < g.steps, a, -1)
    column = c.shock * g.bins + g.bin_of_step
    size = int(c.nodes.sum())
    base = _runs(column, last, event_step, c.nodes)
    counts = np.bincount(a[c.nodes & (a < g.steps)], minlength=g.steps)[: g.steps]
    active_before = np.concatenate([[0], np.cumsum(counts)[:-1]])
    cumulative = np.concatenate([[0], np.cumsum(counts)])
    s = np.arange(g.steps)
    recent = cumulative[s] - cumulative[np.maximum(s - RECENT_STEPS, 0)]
    shares = {}
    for model, count in (("M1", active_before), ("M2", recent)):
        k, node, trials, events = _runs(column * _KEY_BASE + count, last, event_step, c.nodes)
        shares[model] = (k // _KEY_BASE, node, (k % _KEY_BASE) / max(size - 1, 1), trials, events)
    return CascadeRows(c, g, a, last, size, column, base, shares)


def _edge_rows(r: CascadeRows, edges: np.ndarray, offsets: range):
    a, last, nodes = r.a, r.last, r.cascade.nodes
    w_all, s_all, e_all = [], [], []
    for e, (v, w) in enumerate(edges):
        if not (nodes[v] and nodes[w]) or a[v] >= r.grid.steps:
            continue
        s = a[v] + np.asarray(offsets)
        s = s[(s <= last[w]) & (s < r.grid.steps)]
        w_all.append(np.full(s.size, w)), s_all.append(s), e_all.append(np.full(s.size, e))
    if not w_all:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), sparse.csr_matrix((0, len(edges))), np.zeros(0)
    w_all, s_all, e_all = np.concatenate(w_all), np.concatenate(s_all), np.concatenate(e_all)
    combined = w_all * r.grid.steps + s_all
    unique, inverse = np.unique(combined, return_inverse=True)
    incidence = sparse.csr_matrix((np.ones(e_all.size), (inverse, e_all)), shape=(unique.size, len(edges)))
    incidence.data[:] = 1.0
    node, step = unique // r.grid.steps, unique % r.grid.steps
    activated = (r.a[node] == step).astype(float)
    return node, step, incidence, activated


def exposure_rows(r: CascadeRows, edges: np.ndarray):
    """Find the M4 rows: at-risk (w, s) with a parent v activated within the last RECENT_STEPS steps.

    Arguments: r is the cascade's rows; edges is an (edges, 2) array of parent and child node indices.
    Returns: the node, step, sparse edge incidence and activation indicator of each row.
    """
    return _edge_rows(r, edges, range(1, RECENT_STEPS + 1))


def attempt_rows(r: CascadeRows, edges: np.ndarray):
    """Find the M3 rows: at-risk (w, s) with a parent v activated in the previous step.

    Arguments: r is the cascade's rows; edges is an (edges, 2) array of parent and child node indices.
    Returns: the node, step, sparse edge incidence and activation indicator of each row.
    """
    return _edge_rows(r, edges, range(1, 2))




@dataclass(frozen=True)
class Design:
    """Binomial design of a cascade model.

    Fields: the sparse design matrix, events and trials of each row, and the index of the cascade that owns each row.
    """

    matrix: sparse.csr_matrix
    events: np.ndarray
    trials: np.ndarray
    owner: np.ndarray


def _base_columns(column, node, step_cols):
    return np.column_stack([np.zeros(column.size, np.int64), 1 + column, 1 + step_cols + node])


def _matrix(core: np.ndarray, extra: sparse.csr_matrix | None, step_cols: int, extra_cols: int) -> sparse.csr_matrix:
    n = core.shape[0]
    base = sparse.csr_matrix((np.ones(3 * n), (np.repeat(np.arange(n), 3), core.ravel())), shape=(n, 1 + step_cols + NODES))
    if extra_cols == 0:
        return base
    return sparse.hstack([base, extra if extra is not None else sparse.csr_matrix((n, extra_cols))], format="csr")


def columns(step_cols: int, extra: int) -> int:
    """Return the number of design columns: the level, step_cols step columns, one per instrument and extra columns."""
    return 1 + step_cols + NODES + extra


def prior(step_cols: int, extra: int, extra_kind: str | None) -> tuple[np.ndarray, np.ndarray]:
    """Build the Gaussian prior standard deviations and lower bounds of the design columns.

    Arguments: step_cols is the number of step columns; extra is the number of diffusion columns; extra_kind names their
    prior ("share" or "edge_weight"), or None for unit scale.
    Returns: the prior standard deviations (infinite for the level) and lower bounds (zero for edge weights).
    """
    sd = np.concatenate([[np.inf], np.full(step_cols, PRIOR_SD["step"]), np.full(NODES, PRIOR_SD["instrument"]),
                         np.full(extra, PRIOR_SD[extra_kind] if extra_kind else 1.0)])
    lower = np.full(sd.size, -np.inf)
    if extra_kind == "edge_weight":
        lower[-extra:] = 0.0
    return sd, lower


def design_m0(rows_list: list[CascadeRows], step_cols: int) -> Design:
    """Build the M0 design from the binomial cells (shock and step bin, node) of each cascade.

    Arguments: rows_list is the list of cascade rows; step_cols is the number of step columns.
    Returns: a Design.
    """
    parts = [(_base_columns(r.base[0], r.base[1], step_cols), r.base[3], r.base[2], np.full(r.base[0].size, i)) for i, r in enumerate(rows_list)]
    core, events, trials, owner = (np.concatenate(x) for x in zip(*parts))
    return Design(_matrix(core, None, step_cols, 0), events, trials, owner)


def design_share(rows_list: list[CascadeRows], step_cols: int, model: str) -> Design:
    """Build the M1 or M2 design: the M0 columns plus the share of active nodes as one extra column.

    Arguments: rows_list is the list of cascade rows; step_cols is the number of step columns; model is "M1" or "M2".
    Returns: a Design.
    """
    parts = []
    for i, r in enumerate(rows_list):
        column, node, value, trials, events = r.shares[model]
        parts.append((_base_columns(column, node, step_cols), value, events, trials, np.full(column.size, i)))
    core, value, events, trials, owner = (np.concatenate(x) for x in zip(*parts))
    extra = sparse.csr_matrix(value[:, None])
    return Design(_matrix(core, extra, step_cols, 1), events, trials, owner)


def design_threshold(rows_list: list[CascadeRows], step_cols: int, edges: np.ndarray) -> Design:
    """Build the M4 design: the M0 cells less the exposed rows, plus the exposed rows with their edge indicators.

    Arguments: rows_list is the list of cascade rows; step_cols is the number of step columns; edges is an (edges, 2)
    array of parent and child node indices.
    Returns: a Design.
    """
    blocks, events_all, trials_all, owner_all = [], [], [], []
    for i, r in enumerate(rows_list):
        node, step, incidence, activated = exposure_rows(r, edges)
        cells_t = np.zeros((step_cols, NODES))
        cells_y = np.zeros((step_cols, NODES))
        np.add.at(cells_t, (r.base[0], r.base[1]), r.base[2])
        np.add.at(cells_y, (r.base[0], r.base[1]), r.base[3])
        if node.size:
            np.add.at(cells_t, (r.column_of_step[step], node), -1.0)
            np.add.at(cells_y, (r.column_of_step[step], node), -activated)
        c, w = np.nonzero(cells_t > 0.5)
        blocks.append(_matrix(_base_columns(c, w, step_cols), None, step_cols, len(edges)))
        events_all.append(cells_y[c, w]), trials_all.append(cells_t[c, w]), owner_all.append(np.full(c.size, i))
        if node.size:
            blocks.append(_matrix(_base_columns(r.column_of_step[step], node, step_cols), incidence, step_cols, len(edges)))
            events_all.append(activated), trials_all.append(np.ones(node.size)), owner_all.append(np.full(node.size, i))
    return Design(sparse.vstack(blocks, format="csr"), np.concatenate(events_all), np.concatenate(trials_all), np.concatenate(owner_all))


@dataclass(frozen=True)
class Attempts:
    """Attempt rows of the Independent Cascade model M3.

    Fields: the M0 background design of each row, the sparse edge incidence of attempting parents, the activation
    indicator and the index of the owning cascade.
    """

    background: sparse.csr_matrix
    incidence: sparse.csr_matrix
    activated: np.ndarray
    owner: np.ndarray


def design_attempts(rows_list: list[CascadeRows], step_cols: int, edges: np.ndarray) -> Attempts:
    """Build the M3 attempt rows of every cascade.

    Arguments: rows_list is the list of cascade rows; step_cols is the number of step columns; edges is an (edges, 2)
    array of parent and child node indices.
    Returns: an Attempts.
    """
    backgrounds, incidences, activated_all, owner_all = [], [], [], []
    for i, r in enumerate(rows_list):
        node, step, incidence, activated = attempt_rows(r, edges)
        backgrounds.append(_matrix(_base_columns(r.column_of_step[step], node, step_cols), None, step_cols, 0))
        incidences.append(incidence), activated_all.append(activated), owner_all.append(np.full(node.size, i))
    return Attempts(sparse.vstack(backgrounds, format="csr"), sparse.vstack(incidences, format="csr"), np.concatenate(activated_all), np.concatenate(owner_all))


def simulate_m0(template: Cascade, g: Grid, step_cols: int, coefficients: np.ndarray, rng: np.random.Generator) -> Cascade:
    """Draw a cascade from M0 on the template's nodes and shock, with no diffusion.

    Each node activates in each step independently with the M0 hazard; tau is the middle of the first activation step,
    NaN when the node never activates.
    Arguments: template is the cascade to copy; g is the grid; step_cols is the number of step columns; coefficients are
    the M0 coefficients; rng is the random generator.
    Returns: the simulated Cascade.
    """
    column = template.shock * g.bins + g.bin_of_step
    eta = coefficients[0] + coefficients[1 + column][:, None] + coefficients[1 + step_cols: 1 + step_cols + NODES][None, :]
    hazard = 1.0 / (1.0 + np.exp(-eta))
    fired = rng.random(hazard.shape) < hazard
    first = np.where(fired.any(axis=0), fired.argmax(axis=0), g.steps)
    tau = np.where(template.nodes & (first < g.steps), (first + 0.5) * g.step, np.nan)
    return Cascade(template.family, template.event_id, template.group, template.shock, template.nodes.copy(), tau)
