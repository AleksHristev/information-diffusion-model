"""Resampling index generators for the moving-block, stationary and event bootstraps."""

from __future__ import annotations

import numpy as np


def moving_block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Draw moving-block bootstrap indices: uniform block starts, blocks of fixed length, truncated to n.

    Arguments: n is the sample size; block_length is the block length; rng is the random generator.
    Returns: n resampled indices.
    Reference: Kunsch (1989).
    """
    if not 1 <= block_length <= n:
        raise ValueError("block length must lie between 1 and the sample size")
    starts = rng.integers(0, n - block_length + 1, size=int(np.ceil(n / block_length)))
    return (starts[:, None] + np.arange(block_length)).ravel()[:n]


def stationary_indices(n: int, mean_block_length: float, rng: np.random.Generator) -> np.ndarray:
    """Draw stationary bootstrap indices: geometric block lengths with the given mean, wrapping at the end.

    Arguments: n is the sample size; mean_block_length is the mean block length; rng is the random generator.
    Returns: n resampled indices.
    Reference: Politis and Romano (1994).
    """
    if mean_block_length < 1:
        raise ValueError("mean block length must be at least 1")
    jumps = rng.random(n) < 1.0 / mean_block_length
    jumps[0] = True
    block = np.cumsum(jumps) - 1
    starts = rng.integers(0, n, size=block[-1] + 1)
    first_position = np.flatnonzero(jumps)
    return (starts[block] + np.arange(n) - first_position[block]) % n


def event_indices(n_events: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n_events whole-event indices with replacement using rng, and return them."""
    return rng.integers(0, n_events, size=n_events)
