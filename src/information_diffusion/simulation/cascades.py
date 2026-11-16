"""Simulators of Independent Cascade and Linear Threshold diffusion on a directed network, in discrete steps."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.special import expit

NEVER = -1


def _seed_matrix(seeds: np.ndarray | Sequence[int], n_cascades: int, n_nodes: int) -> np.ndarray:
    seeds = np.asarray(seeds)
    if seeds.ndim == 2:
        if seeds.shape != (n_cascades, n_nodes):
            raise ValueError("per-cascade seeds must have shape (n_cascades, n_nodes)")
        return seeds.astype(bool)
    matrix = np.zeros((n_cascades, n_nodes), dtype=bool)
    matrix[:, seeds.astype(int)] = True
    return matrix


def random_digraph(n_nodes: int, n_links: int, seeds: Sequence[int], rng: np.random.Generator) -> np.ndarray:
    """Random directed graph with exactly n_links links, no links into the seeds, and every node reachable from the seeds.

    Each non-seed node, in random order, first gets one parent among the nodes already reached; the remaining links are
    drawn uniformly from the other ordered pairs with a non-seed child.
    Arguments: n_nodes the number of nodes; n_links the number of links; seeds the seed node indices; rng the generator.
    Returns: a boolean (parent, child) adjacency matrix.
    """
    seeds = np.asarray(seeds, dtype=int)
    is_seed = np.zeros(n_nodes, dtype=bool)
    is_seed[seeds] = True
    others = rng.permutation(np.flatnonzero(~is_seed))
    if n_links < len(others):
        raise ValueError("too few links to reach every non-seed node")
    adjacency = np.zeros((n_nodes, n_nodes), dtype=bool)
    reached = list(seeds)
    for node in others:
        adjacency[rng.choice(reached), node] = True
        reached.append(int(node))
    candidates = np.argwhere(~adjacency & ~np.eye(n_nodes, dtype=bool) & ~is_seed[None, :])
    extra = n_links - int(adjacency.sum())
    if extra > len(candidates):
        raise ValueError("more links requested than ordered pairs available")
    chosen = candidates[rng.choice(len(candidates), size=extra, replace=False)]
    adjacency[chosen[:, 0], chosen[:, 1]] = True
    return adjacency


def simulate_independent_cascade(probabilities: np.ndarray, seeds, n_cascades: int, n_steps: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate Independent Cascade diffusion: a node activated at step s has one chance, at s + 1, to activate each inactive child w with probability p_vw.

    Seeds are active at step 0.
    Arguments: probabilities is the (parent, child) matrix p_vw; seeds are node indices or a boolean (cascades, nodes)
    matrix; n_cascades the number of cascades; n_steps the number of steps; rng the generator.
    Returns: activation steps of shape (cascades, nodes), NEVER (-1) if not activated within n_steps.
    Reference: Kempe, Kleinberg and Tardos (2003), KDD.
    """
    n_nodes = probabilities.shape[0]
    seed_matrix = _seed_matrix(seeds, n_cascades, n_nodes)
    activation = np.full((n_cascades, n_nodes), NEVER, dtype=np.int64)
    for c in range(n_cascades):
        active = seed_matrix[c].copy()
        activation[c, active] = 0
        frontier = active.copy()
        for step in range(1, n_steps):
            if not frontier.any():
                break
            attempts = probabilities[frontier]
            success = rng.random(attempts.shape) < attempts
            newly = success.any(axis=0) & ~active
            activation[c, newly] = step
            active |= newly
            frontier = newly
    return activation


def simulate_linear_threshold(weights: np.ndarray, thresholds: np.ndarray, seeds, n_cascades: int, n_steps: int, window: int | None, rng: np.random.Generator) -> np.ndarray:
    """Simulate a stochastic Linear Threshold model: inactive node w activates at step t >= 1 if S_w(t) + e_wt > theta_w.

    S_w(t) sums the weights b_vw of parents activated in the last L steps and e_wt is i.i.d. standard logistic; seeds are
    active at step 0.
    Arguments: weights is the (parent, child) matrix b_vw; thresholds the theta_w; seeds are node indices or a boolean
    (cascades, nodes) matrix; n_cascades and n_steps as for the cascade model; window is L in steps, None meaning an
    active parent counts for ever; rng the generator.
    Returns: activation steps of shape (cascades, nodes), NEVER (-1) if not activated.
    """
    n_nodes = weights.shape[0]
    seed_matrix = _seed_matrix(seeds, n_cascades, n_nodes)
    activation = np.full((n_cascades, n_nodes), NEVER, dtype=np.int64)
    for c in range(n_cascades):
        time_active = np.full(n_nodes, NEVER, dtype=np.int64)
        time_active[seed_matrix[c]] = 0
        for step in range(1, n_steps):
            inactive = np.flatnonzero(time_active < 0)
            if inactive.size == 0:
                break
            exposed = time_active >= 0 if window is None else (time_active >= 0) & (time_active >= step - window)
            pressure = weights[exposed].sum(axis=0)[inactive] if exposed.any() else np.zeros(inactive.size)
            fires = rng.random(inactive.size) < expit(pressure - thresholds[inactive])
            time_active[inactive[fires]] = step
        activation[c] = time_active
    return activation
