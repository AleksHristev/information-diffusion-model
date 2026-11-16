"""Model Confidence Set of Hansen, Lunde and Nason, by block bootstrap."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from information_diffusion.statistical_framework import block_bootstrap


@dataclass(frozen=True)
class ModelConfidenceSet:
    """Estimated Model Confidence Set.

    Fields: included holds the indices of the models with MCS p-value at least alpha; p_values are the MCS p-values;
    elimination_order lists the models in the order eliminated, the survivor last; statistic, alpha and replications
    record the settings.
    """

    included: np.ndarray
    p_values: np.ndarray
    elimination_order: np.ndarray
    statistic: str
    alpha: float
    replications: int


def _indices(n: int, block_length: float, rng: np.random.Generator, scheme: str) -> np.ndarray:
    if scheme == "stationary":
        return block_bootstrap.stationary_indices(n, block_length, rng)
    if scheme == "moving":
        return block_bootstrap.moving_block_indices(n, int(block_length), rng)
    raise ValueError("scheme must be stationary or moving")


def _bootstrap_means(losses: np.ndarray, block_length: float, replications: int, rng: np.random.Generator, scheme: str) -> np.ndarray:
    n, m = losses.shape
    means = np.empty((replications, m))
    for b in range(replications):
        means[b] = losses[_indices(n, block_length, rng, scheme)].mean(axis=0)
    return means


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def _step_range(mean_loss: np.ndarray, zeta: np.ndarray) -> tuple[float, int]:
    diff = mean_loss[:, None] - mean_loss[None, :]
    zeta_pair = zeta[:, :, None] - zeta[:, None, :]
    sd = np.sqrt(np.mean(zeta_pair**2, axis=0))
    t = _safe_divide(diff, sd)
    statistic = np.max(np.abs(t))
    t_star = np.max(np.abs(_safe_divide(zeta_pair, np.broadcast_to(sd, zeta_pair.shape))).reshape(zeta.shape[0], -1), axis=1)
    return float(np.mean(t_star > statistic)), int(np.argmax(np.max(t, axis=1)))


def _step_max(mean_loss: np.ndarray, zeta: np.ndarray) -> tuple[float, int]:
    diff = mean_loss - mean_loss.mean()
    zeta_diff = zeta - zeta.mean(axis=1, keepdims=True)
    sd = np.sqrt(np.mean(zeta_diff**2, axis=0))
    t = _safe_divide(diff, sd)
    statistic = np.max(t)
    t_star = np.max(_safe_divide(zeta_diff, np.broadcast_to(sd, zeta_diff.shape)), axis=1)
    return float(np.mean(t_star > statistic)), int(np.argmax(t))


def model_confidence_set(losses, *, block_length: float, rng: np.random.Generator, alpha: float = 0.05, replications: int = 999,
                         statistic: str = "range", scheme: str = "stationary") -> ModelConfidenceSet:
    """Estimate the Model Confidence Set by sequential elimination of the worst model.

    The MCS p-value of the model eliminated at step k is the running maximum of the equal-ability p-values up to k;
    the last surviving model has p-value 1. Pairs with zero bootstrap variance get t = 0.
    Arguments: losses is an n by m loss matrix (larger is worse); block_length is the (mean) block length; rng is the
    random generator; alpha is the level; replications is the number of bootstrap draws; statistic is "range" or
    "max"; scheme is "stationary" or "moving".
    Returns: a ModelConfidenceSet.
    Reference: Hansen, Lunde and Nason (2011).
    """
    L = np.asarray(losses, dtype=float)
    if L.ndim != 2 or L.shape[1] < 2 or L.shape[0] < 2 or not np.all(np.isfinite(L)):
        raise ValueError("losses must be a finite n by m array with at least two models and two rows")
    if statistic not in ("range", "max"):
        raise ValueError("statistic must be range or max")
    step = _step_range if statistic == "range" else _step_max
    mean_loss = L.mean(axis=0)
    zeta = _bootstrap_means(L, block_length, replications, rng, scheme) - mean_loss
    m = L.shape[1]
    remaining = list(range(m))
    p_values = np.ones(m)
    order: list[int] = []
    running = 0.0
    while len(remaining) > 1:
        p, worst_position = step(mean_loss[remaining], zeta[:, remaining])
        running = max(running, p)
        worst = remaining.pop(worst_position)
        p_values[worst] = running
        order.append(worst)
    order.append(remaining[0])
    return ModelConfidenceSet(np.flatnonzero(p_values >= alpha), p_values, np.array(order), statistic, alpha, replications)
