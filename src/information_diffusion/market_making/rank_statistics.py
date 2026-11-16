"""Rank statistics: ranks with missing values, Kendall's W, Spearman's rho, Kendall's tau-b, sign test and weighted median."""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest, kendalltau, rankdata


def ranks_with_missing(values: np.ndarray) -> np.ndarray:
    """Return the average ranks 1 to n of values, with NaN values tied for the last ranks."""
    v = np.asarray(values, dtype=float)
    return rankdata(np.where(np.isnan(v), np.inf, v), method="average")


def kendall_w(ranks: np.ndarray) -> float:
    """Compute Kendall's W with the tie correction, W = 12 S / (m^2 (n^3 - n) - m sum_j T_j).

    Arguments: ranks is an (m releases, n instruments) matrix of within-release average ranks.
    Returns: W, NaN when the denominator is not positive.
    Reference: Kendall and Babington Smith (1939); Kendall and Gibbons (1990).
    """
    r = np.asarray(ranks, dtype=float)
    m, n = r.shape
    s = float(((r.sum(axis=0) - m * (n + 1) / 2.0) ** 2).sum())
    ties = 0.0
    for row in r:
        _, counts = np.unique(row, return_counts=True)
        ties += float((counts ** 3 - counts).sum())
    denominator = m ** 2 * (n ** 3 - n) - m * ties
    return 12.0 * s / denominator if denominator > 0 else float("nan")


def kendall_w_permutation(ranks: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[float, float]:
    """Compute Kendall's W and its permutation p-value (1 + #{W* >= W}) / (1 + draws).

    Arguments: ranks is as in kendall_w; rng is the random generator; draws is the number of permutations, each
    permuting every release's ranks independently.
    Returns: W and the p-value.
    """
    r = np.asarray(ranks, dtype=float)
    w = kendall_w(r)
    m, n = r.shape
    count = 0
    for _ in range(draws):
        perm = np.take_along_axis(r, np.argsort(rng.random((m, n)), axis=1), axis=1)
        count += kendall_w(perm) >= w - 1e-12
    return w, (1 + count) / (1 + draws)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Return Spearman's rho of x and y over the finite pairs, NaN with fewer than 3 pairs or a constant variable."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    a, b = rankdata(x[ok]), rankdata(y[ok])
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman_ranks(a: np.ndarray, b: np.ndarray) -> float:
    """Return Pearson's correlation of two precomputed rank vectors a and b, NaN when either is constant."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman_permutation(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[float, float, int]:
    """Compute Spearman's rho with a two-sided permutation p-value (1 + #{|rho*| >= |rho|}) / (1 + draws).

    Arguments: x and y are the variables (pairs with a missing value are dropped); rng is the random generator; draws
    is the number of permutations of y.
    Returns: rho, the p-value and the number of pairs.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    a, b = rankdata(x[ok]), rankdata(y[ok])
    n = int(ok.sum())
    if n < 3 or a.std() == 0 or b.std() == 0:
        return float("nan"), float("nan"), n
    rho = float(np.corrcoef(a, b)[0, 1])
    perms = np.argsort(rng.random((draws, n)), axis=1)
    bb = b[perms]
    ac = (a - a.mean()) / a.std()
    bc = (bb - bb.mean(axis=1, keepdims=True)) / bb.std(axis=1, keepdims=True)
    rhos = (ac[None, :] * bc).mean(axis=1)
    p = (1 + int((np.abs(rhos) >= abs(rho) - 1e-12).sum())) / (1 + draws)
    return rho, float(p), n


def rank_residuals(y_ranks: np.ndarray, x_ranks: np.ndarray) -> np.ndarray:
    """Return the residuals of the least-squares line of y_ranks on x_ranks (deviations from the mean if x is constant)."""
    y, x = np.asarray(y_ranks, dtype=float), np.asarray(x_ranks, dtype=float)
    if x.std() == 0:
        return y - y.mean()
    slope = np.cov(x, y, bias=True)[0, 1] / x.var()
    return y - (y.mean() + slope * (x - x.mean()))


def tau(a: np.ndarray, b: np.ndarray) -> float:
    """Return Kendall's tau-b of two orders a and b, NaN where undefined."""
    t = kendalltau(np.asarray(a, dtype=float), np.asarray(b, dtype=float)).statistic
    return float(t) if np.isfinite(t) else float("nan")


def sign_test(values: np.ndarray) -> tuple[int, int, float]:
    """Run the exact two-sided sign test over the finite non-zero values.

    Returns: the number of positive values, the number of negative values and the binomial p-value at 1/2.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    pos, neg = int((v > 0).sum()), int((v < 0).sum())
    return pos, neg, float(binomtest(pos, pos + neg, 0.5).pvalue) if pos + neg else float("nan")


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Return the weighted median, the smallest value whose cumulative weight reaches half the total (NaN without weight)."""
    v, w = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & (w > 0)
    if not ok.any():
        return float("nan")
    order = np.argsort(v[ok], kind="mergesort")
    vs, cw = v[ok][order], np.cumsum(w[ok][order])
    return float(vs[np.searchsorted(cw, cw[-1] / 2.0, side="left")])
