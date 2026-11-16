"""Sensor placement on a diffusion network: the detection objective, greedy placement and the exhaustive optimum."""

from __future__ import annotations

import itertools

import numpy as np


def objective(rewards, sensors) -> float:
    """Placement objective f(S) = (1 / R) sum_r max_{v in S} C_rv, with f(empty) = 0.

    Arguments: rewards is the (cascades, nodes) matrix of detection rewards C_rv >= 0; sensors the chosen node indices.
    Returns: f(S).
    """
    chosen = np.asarray(list(sensors), dtype=np.intp)
    if chosen.size == 0:
        return 0.0
    return float(np.mean(np.max(np.asarray(rewards)[:, chosen], axis=1)))


def greedy(rewards, k: int) -> list[int]:
    """Greedy placement: add argmax_v f(S + v) - f(S) until k sensors are chosen; within 1 - 1/e of the optimum.

    Arguments: rewards is the (cascades, nodes) reward matrix; k the number of sensors.
    Returns: the chosen node indices in the order added.
    Reference: Nemhauser, Wolsey and Fisher (1978), Math. Programming; Kempe, Kleinberg and Tardos (2003), KDD; Krause,
    Singh and Guestrin (2008), JMLR.
    """
    reward_matrix = np.asarray(rewards, dtype=float)
    best_so_far = np.zeros(reward_matrix.shape[0])
    chosen: list[int] = []
    for _ in range(k):
        gain = np.maximum(reward_matrix, best_so_far[:, None]).mean(axis=0)
        gain[chosen] = -np.inf
        pick = int(np.argmax(gain))
        chosen.append(pick)
        best_so_far = np.maximum(best_so_far, reward_matrix[:, pick])
    return chosen


def exhaustive_optimum(rewards, k: int) -> float:
    """Exact optimum max_{|S| = k} f(S) by enumeration of all subsets of k nodes, given the (cascades, nodes) reward matrix."""
    reward_matrix = np.asarray(rewards, dtype=float)
    subsets = np.fromiter(itertools.chain.from_iterable(itertools.combinations(range(reward_matrix.shape[1]), k)), dtype=np.intp).reshape(-1, k)
    best = 0.0
    for start in range(0, subsets.shape[0], 512):
        block = subsets[start : start + 512]
        values = reward_matrix[:, block].max(axis=2).mean(axis=0)
        best = max(best, float(values.max()))
    return best
