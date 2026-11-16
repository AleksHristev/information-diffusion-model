"""Price-discovery shares of two venues from a VECM with cointegrating vector (1, -1): component, information and leadership shares."""

from __future__ import annotations

import numpy as np

from information_diffusion.influence import vecm

MEASURES = ("is_lower_kalshi", "is_upper_kalshi", "is_midpoint_kalshi", "gg_kalshi", "ils_kalshi")


def component_shares(alpha: np.ndarray) -> np.ndarray:
    """Gonzalo-Granger component shares CS_K = alpha_P / (alpha_P - alpha_K) and CS_P = -alpha_K / (alpha_P - alpha_K).

    K is the first venue and P the second, as in vecm.Pair (here and in the rest of this module).
    Arguments: alpha is (alpha_K, alpha_P), the adjustment coefficients on x_K - x_P.
    Returns: (CS_K, CS_P), which sum to one; NaN when alpha_P = alpha_K.
    Reference: Gonzalo and Granger (1995), J. Business and Economic Statistics.
    """
    alpha_k, alpha_p = (float(value) for value in alpha)
    denominator = alpha_p - alpha_k
    if denominator == 0:
        return np.array([np.nan, np.nan])
    return np.array([alpha_p / denominator, -alpha_k / denominator])


def _share_first(weights: np.ndarray, sigma: np.ndarray) -> float:
    factor = np.linalg.cholesky(sigma)
    loading = weights @ factor
    total = float(loading @ loading)
    return float(loading[0] ** 2 / total) if total > 0 else np.nan


def information_share_orderings(alpha: np.ndarray, sigma: np.ndarray) -> tuple[float, float]:
    """Hasbrouck information share of the first venue (K) under each ordering of the two venues.

    With Sigma = F F', F lower triangular, and psi the component-share weights, IS_K = ([psi F]_K)^2 / (psi Sigma psi').
    Arguments: alpha is (alpha_K, alpha_P); sigma the 2 x 2 innovation covariance with K first.
    Returns: (share with K ordered first, share with K ordered second).
    Reference: Hasbrouck (1995), Journal of Finance; Baillie, Booth, Tse and Zabotina (2002), J. Financial Markets;
    Johansen (1995) for the long-run impact matrix.
    """
    weights = component_shares(alpha)
    sigma = np.asarray(sigma, dtype=float)
    if not np.all(np.isfinite(weights)):
        return np.nan, np.nan
    first = _share_first(weights, sigma)
    second = 1.0 - _share_first(weights[::-1], sigma[::-1, ::-1])
    return first, second


def information_share_bounds(alpha: np.ndarray, sigma: np.ndarray) -> tuple[float, float, float]:
    """Bounds of the first venue's Hasbrouck information share over the two orderings, and their midpoint.

    Arguments: alpha is (alpha_K, alpha_P); sigma the 2 x 2 innovation covariance.
    Returns: (lower, upper, midpoint); the second venue's share is the complement.
    """
    first, second = information_share_orderings(alpha, sigma)
    lower, upper = min(first, second), max(first, second)
    return lower, upper, 0.5 * (lower + upper)


def information_leadership_share(is_kalshi: float, cs_kalshi: float) -> float:
    """Information leadership share of the first venue, ILS_K = IL_K / (IL_K + IL_P).

    IL_K = |(IS_K / IS_P) (CS_P / CS_K)| and IL_P = |(IS_P / IS_K) (CS_K / CS_P)|; ILS_K is 0.5 when IS_K = CS_K.
    Arguments: is_kalshi is the first venue's information share; cs_kalshi its component share.
    Returns: ILS_K in [0, 1].
    Reference: Yan and Zivot (2010), J. Financial Markets; Putnins (2013), J. Empirical Finance.
    """
    is_polymarket, cs_polymarket = 1.0 - is_kalshi, 1.0 - cs_kalshi
    with np.errstate(divide="ignore", invalid="ignore"):
        leadership_k = abs((is_kalshi / is_polymarket) * (cs_polymarket / cs_kalshi))
        leadership_p = abs((is_polymarket / is_kalshi) * (cs_kalshi / cs_polymarket))
        share = leadership_k / (leadership_k + leadership_p)
    if np.isnan(share) and np.isinf(leadership_k) and np.isfinite(leadership_p):
        return 1.0
    if np.isnan(share) and np.isinf(leadership_p) and np.isfinite(leadership_k):
        return 0.0
    return float(share)


def measures(alpha: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """All price-discovery measures of the first venue in MEASURES order: IS lower, upper and midpoint, component share and ILS.

    Arguments: alpha is (alpha_K, alpha_P); sigma the 2 x 2 innovation covariance.
    Returns: the five measures; NaN when sigma has no Cholesky factor.
    """
    try:
        lower, upper, midpoint = information_share_bounds(alpha, sigma)
    except np.linalg.LinAlgError:
        return np.full(len(MEASURES), np.nan)
    cs_kalshi = component_shares(alpha)[0]
    return np.array([lower, upper, midpoint, cs_kalshi, information_leadership_share(midpoint, cs_kalshi)])


def bootstrap(design: vecm.Design, draws: int, rng: np.random.Generator, block_length: int | None = None) -> np.ndarray:
    """Moving-block bootstrap of the measures, refitting the VECM in every draw.

    Arguments: design is the vecm.Design; draws the number of draws; rng the generator; block_length the block length
    in distinct time keys (default from vecm.default_block_length).
    Returns: shape (draws, len(MEASURES)); a failed refit gives a row of NaN.
    Reference: Efron and Tibshirani (1993), An Introduction to the Bootstrap.
    """
    blocks = vecm.KeyBlocks(design.key, block_length)
    output = np.full((draws, len(MEASURES)), np.nan)
    for draw in range(draws):
        try:
            refit = vecm.fit(design, blocks.weights(rng))
        except (np.linalg.LinAlgError, ValueError):
            continue
        output[draw] = measures(refit.alpha, refit.sigma)
    return output


def percentile_interval(draws: np.ndarray, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Columnwise percentile interval at the given level over the finite draws. Returns (lower, upper) arrays."""
    tail = 100.0 * (1.0 - level) / 2.0
    lower = np.nanpercentile(draws, tail, axis=0)
    upper = np.nanpercentile(draws, 100.0 - tail, axis=0)
    return lower, upper
