"""Network communities by Louvain modularity optimisation, with modularity and the adjusted Rand index."""

from __future__ import annotations

import numpy as np

MAX_PASSES = 100
DEFAULT_RESTARTS = 20


def _one_level(adjacency: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = adjacency.shape[0]
    degree = adjacency.sum(axis=1)
    total = degree.sum()
    labels = np.arange(n)
    sigma = degree.copy()
    for _ in range(MAX_PASSES):
        moved = False
        for node in rng.permutation(n):
            own = labels[node]
            weights = np.bincount(labels, weights=adjacency[node], minlength=n)
            sigma[own] -= degree[node]
            gains = weights - sigma * degree[node] / total
            weights_self = adjacency[node, node]
            gains[own] -= weights_self
            best = int(np.argmax(gains))
            if gains[best] <= gains[own] + 1e-15:
                best = own
            sigma[best] += degree[node]
            if best != own:
                labels[node] = best
                moved = True
        if not moved:
            break
    return labels


def _relabel(labels: np.ndarray) -> np.ndarray:
    return np.unique(labels, return_inverse=True)[1]


def louvain(adjacency, rng: np.random.Generator, *, restarts: int = DEFAULT_RESTARTS) -> np.ndarray:
    """Communities of a weighted undirected graph by Louvain modularity optimisation, best of several restarts.

    Each restart moves nodes in a random order to the neighbouring community with the largest modularity gain, then merges
    communities into nodes, repeating until modularity stops rising; the partition with the highest modularity is kept.
    Arguments: adjacency is a symmetric non-negative weighted adjacency matrix; rng the generator; restarts the number
    of restarts (default 20).
    Returns: community labels 0, 1, ... for each node.
    Reference: Blondel, Guillaume, Lambiotte and Lefebvre (2008), J. Stat. Mech.
    """
    graph = np.asarray(adjacency, dtype=float)
    if not np.allclose(graph, graph.T):
        raise ValueError("adjacency must be symmetric")
    if np.any(graph < 0):
        raise ValueError("adjacency must be non-negative")
    candidates = [_louvain_once(graph, rng) for _ in range(restarts)]
    return max(candidates, key=lambda labels: modularity(graph, labels))


def _louvain_once(graph: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = graph.shape[0]
    membership = np.arange(n)
    current = graph.copy()
    best_q = modularity(graph, membership)
    while True:
        labels = _relabel(_one_level(current, rng))
        size = labels.max() + 1
        membership = labels[membership]
        q = modularity(graph, membership)
        if size == current.shape[0] or q <= best_q + 1e-12:
            return _relabel(membership)
        best_q = q
        aggregated = np.zeros((size, size))
        np.add.at(aggregated, (labels[np.repeat(np.arange(labels.size), labels.size)], labels[np.tile(np.arange(labels.size), labels.size)]), current.ravel())
        current = aggregated


def modularity(adjacency, labels) -> float:
    """Modularity Q = (1 / 2m) sum_ij (A_ij - k_i k_j / 2m) 1{c_i = c_j} of a partition.

    Arguments: adjacency is the weighted adjacency matrix A; labels the community of each node.
    Returns: Q, or 0 for a graph with no weight.
    Reference: Newman (2004), Physical Review E.
    """
    graph = np.asarray(adjacency, dtype=float)
    labels = np.asarray(labels)
    degree = graph.sum(axis=1)
    total = degree.sum()
    if total <= 0:
        return 0.0
    same = labels[:, None] == labels[None, :]
    return float(np.sum((graph - np.outer(degree, degree) / total) * same) / total)


def adjusted_rand_index(labels_a, labels_b) -> float:
    """Adjusted Rand index of two partitions of the same items, ARI = (a - expected) / (0.5 (b_1 + b_2) - expected).

    Arguments: labels_a and labels_b are the labels of each item under the two partitions.
    Returns: the index (1 for identical partitions; 1 when the denominator is zero).
    Reference: Hubert and Arabie (1985), Journal of Classification.
    """
    a, b = np.asarray(labels_a), np.asarray(labels_b)
    _, a_codes = np.unique(a, return_inverse=True)
    _, b_codes = np.unique(b, return_inverse=True)
    contingency = np.zeros((a_codes.max() + 1, b_codes.max() + 1), dtype=np.int64)
    np.add.at(contingency, (a_codes, b_codes), 1)

    def choose2(values):
        return values * (values - 1) / 2.0

    index = float(choose2(contingency).sum())
    row = float(choose2(contingency.sum(axis=1)).sum())
    column = float(choose2(contingency.sum(axis=0)).sum())
    total = float(choose2(np.array([a.size]))[0])
    expected = row * column / total
    denominator = 0.5 * (row + column) - expected
    return float((index - expected) / denominator) if denominator != 0 else 1.0
