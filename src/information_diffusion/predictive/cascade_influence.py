"""Expected spread and greedy influence maximisation under an Independent Cascade, by live-edge simulation."""

from __future__ import annotations

import numpy as np


def live_edges(probability: np.ndarray, simulations: int, rng: np.random.Generator) -> np.ndarray:
    """Draw live-edge graphs, keeping each edge from parent v to child w with probability p_vw.

    Arguments: probability is the (N, N) edge probability matrix (NaN counts as zero); simulations is the number of
    graphs; rng is the random generator.
    Returns: a float 0/1 array of shape (simulations, N, N).
    Reference: Kempe, Kleinberg and Tardos (2003).
    """
    p = np.nan_to_num(np.asarray(probability, dtype=float), nan=0.0)
    if p.ndim != 2 or p.shape[0] != p.shape[1] or np.any((p < 0) | (p > 1)):
        raise ValueError("probability must be a square matrix of values in [0, 1]")
    return (rng.random((simulations, *p.shape)) < p[None]).astype(np.float32)


def spread(live: np.ndarray, seeds: np.ndarray, steps: int, nodes: np.ndarray | None = None) -> np.ndarray:
    """Estimate the expected spread sigma(S) of each seed set on common live-edge draws.

    sigma(S) is the mean over the draws of the number of nodes within graph distance steps - 1 of S in the live-edge
    graph, reached through nodes that can activate only.
    Arguments: live is the (simulations, N, N) array from live_edges; seeds is a (sets, N) boolean array; steps is the
    horizon in steps; nodes is an optional (N,) boolean mask of the nodes that can activate (all by default).
    Returns: the estimated spread of each seed set.
    """
    seeds = np.asarray(seeds, dtype=bool)
    n = live.shape[1]
    allowed = np.ones(n, dtype=bool) if nodes is None else np.asarray(nodes, dtype=bool)
    active = np.broadcast_to(seeds[None, :, :] & allowed, (live.shape[0], *seeds.shape)).copy()
    frontier = active.astype(np.float32)
    for _ in range(max(steps - 1, 0)):
        reached = np.matmul(frontier, live) > 0
        new = reached & ~active & allowed
        if not new.any():
            break
        active |= new
        frontier = new.astype(np.float32)
    return active.sum(axis=2).mean(axis=0)


def individual(live: np.ndarray, steps: int, nodes: np.ndarray) -> np.ndarray:
    """Estimate the spread sigma({v}) of every single node v.

    Arguments: live is the array of live-edge graphs; steps is the horizon in steps; nodes is the boolean node mask.
    Returns: the spread of each node, NaN for a node outside the mask.
    """
    n = live.shape[1]
    out = spread(live, np.eye(n, dtype=bool), steps, nodes)
    return np.where(nodes, out, np.nan)


def greedy(live: np.ndarray, steps: int, nodes: np.ndarray, k: int) -> tuple[list[int], list[float]]:
    """Choose a seed set by greedy influence maximisation, adding in each round the node that gives the largest spread.

    Ties go to the earlier node.
    Arguments: live is the array of live-edge graphs; steps is the horizon in steps; nodes is the boolean node mask;
    k is the size of the seed set.
    Returns: the chosen nodes in order and the estimated spread after each addition.
    Reference: Kempe, Kleinberg and Tardos (2003).
    """
    n = live.shape[1]
    chosen: list[int] = []
    values: list[float] = []
    for _ in range(k):
        candidates = [v for v in range(n) if nodes[v] and v not in chosen]
        sets = np.zeros((len(candidates), n), dtype=bool)
        sets[:, chosen] = True
        sets[np.arange(len(candidates)), candidates] = True
        sigma = spread(live, sets, steps, nodes)
        best = int(np.argmax(sigma))
        chosen.append(candidates[best])
        values.append(float(sigma[best]))
    return chosen, values
