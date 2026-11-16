"""Stability selection of sparse VAR links with whole-event resampling."""

from __future__ import annotations

import numpy as np

from information_diffusion.influence.sparse_var import EventMoments, VarFit, lasso_support
from information_diffusion.statistical_framework import block_bootstrap

DEFAULT_DRAWS = 100
DEFAULT_THRESHOLD = 0.8


def selection_frequencies(moments: EventMoments, fit: VarFit, draws: int, rng: np.random.Generator) -> np.ndarray:
    """Selection frequency Pi_hat_jk = (1 / D) sum_d 1{beta_jk^(d) != 0} of each coefficient over D resamples.

    Each resample draws floor(E / 2) whole events with replacement and fits the Lasso at the full-sample penalties.
    Arguments: moments are the EventMoments; fit the full-sample VarFit; draws is D; rng the generator.
    Returns: frequencies of shape (m, nodes), regressor by equation.
    Reference: Meinshausen and Buhlmann (2010), JRSS B.
    """
    events = moments.n.size
    size = events // 2
    selected = np.zeros((moments.nodes * fit.lag_order, moments.nodes))
    for _ in range(draws):
        draw = block_bootstrap.event_indices(events, rng)[:size]
        selected += lasso_support(moments, fit, np.bincount(draw, minlength=events).astype(float))
    return selected / draws


def stable_links(frequencies: np.ndarray, nodes: int, *, threshold: float = DEFAULT_THRESHOLD, lag: int = 0) -> np.ndarray:
    """Stable links at one lag: frequency at least threshold (default 0.8).

    Arguments: frequencies from selection_frequencies; nodes the number of nodes; threshold the smallest frequency
    kept; lag the zero-based lag index.
    Returns: a boolean (nodes, nodes) mask whose entry [i, j] says node j at that lag enters equation i; diagonal excluded.
    """
    mask = frequencies[lag * nodes : (lag + 1) * nodes].T >= threshold
    np.fill_diagonal(mask, False)
    return mask
