"""Two-venue vector error-correction model with the cointegrating vector (1, -1) known, with lag selection and a moving-block bootstrap."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from information_diffusion.influence.adf import run_position
from information_diffusion.statistical_framework.block_bootstrap import moving_block_indices

DEFAULT_MAX_LAGS = 3


@dataclass(frozen=True)
class Pair:
    """One pair of series pricing the same outcome on two venues, on a common regular grid.

    Fields: kalshi and polymarket, the log-odds on the first venue (K) and the second venue (P); keep, the boolean mask
    of grid points to use; time_key, an integer key per grid point shared by all pairs observed at the same time (used
    by the block bootstrap).
    """

    kalshi: np.ndarray
    polymarket: np.ndarray
    keep: np.ndarray
    time_key: np.ndarray

    def usable_keep(self) -> np.ndarray:
        """Grid points that are kept and finite on both venues."""
        return np.asarray(self.keep, dtype=bool) & np.isfinite(self.kalshi) & np.isfinite(self.polymarket)


@dataclass(frozen=True)
class Design:
    """Stacked VECM design over pairs.

    Fields: x, the regressors (one intercept dummy per pair, the lagged spread x_K - x_P, then the lagged differences);
    y, the differences of both venues; key, the time key of each row; lags, the lag order; pairs, the number of pairs.
    """

    x: np.ndarray
    y: np.ndarray
    key: np.ndarray
    lags: int
    pairs: int

    @property
    def level_column(self) -> int:
        """Column index of the lagged spread in x."""
        return self.pairs


@dataclass(frozen=True)
class Fit:
    """VECM fit. Fields: coefficients (regressors x 2 equations); sigma, the innovation covariance; lags; rows; pairs."""

    coefficients: np.ndarray
    sigma: np.ndarray
    lags: int
    rows: int
    pairs: int

    @property
    def alpha(self) -> np.ndarray:
        """Adjustment coefficients (alpha_K, alpha_P) on the lagged spread."""
        return self.coefficients[self.pairs]

    @property
    def gamma(self) -> np.ndarray:
        """Short-run matrices Gamma_j, j = 1, ..., k, as shape (k, 2, 2) with Gamma_j[equation, variable]."""
        block = self.coefficients[self.pairs + 1:]
        return np.stack([block[2 * j:2 * j + 2].T for j in range(self.lags)]) if self.lags else np.zeros((0, 2, 2))


def pair_rows(pair: Pair, lags: int) -> np.ndarray:
    """Grid indices t of a pair whose row at the given lag order uses only usable points t - lags - 1, ..., t."""
    return np.flatnonzero(run_position(pair.usable_keep()) >= lags + 2)


def build_design(pairs: Sequence[Pair], lags: int, sample_lags: int | None = None) -> Design:
    """Stacked design of dy_t = c_g + alpha (x_K - x_P)_{t-1} + sum_{j=1}^k Gamma_j dy_{t-j} + e_t over the pairs.

    Arguments: pairs are the Pair objects; lags is k; sample_lags (default lags, not below it) sets the rows used,
    those valid at that lag order.
    Returns: a Design; raises ValueError when no rows are usable.
    """
    sample_lags = lags if sample_lags is None else sample_lags
    if sample_lags < lags:
        raise ValueError("the sample must be valid at the lag order")
    blocks = []
    for pair in pairs:
        rows = pair_rows(pair, sample_lags)
        if rows.size:
            blocks.append((pair, rows))
    if not blocks:
        raise ValueError("no usable rows")
    count = len(blocks)
    xs, ys, keys = [], [], []
    for index, (pair, rows) in enumerate(blocks):
        levels = np.column_stack([np.asarray(pair.kalshi, dtype=float), np.asarray(pair.polymarket, dtype=float)])
        difference = np.vstack([np.full((1, 2), np.nan), np.diff(levels, axis=0)])
        dummies = np.zeros((rows.size, count))
        dummies[:, index] = 1.0
        spread = (levels[rows - 1, 0] - levels[rows - 1, 1])[:, None]
        lagged = [difference[rows - j] for j in range(1, lags + 1)]
        xs.append(np.hstack([dummies, spread, *lagged]))
        ys.append(difference[rows])
        keys.append(np.asarray(pair.time_key, dtype=np.int64)[rows])
    return Design(np.vstack(xs), np.vstack(ys), np.concatenate(keys), lags, count)


def fit(design: Design, weights: np.ndarray | None = None) -> Fit:
    """OLS of both equations, B_hat = (X'WX)^(-1) X'WY and Sigma_hat = E'WE / (sum(W) - p).

    With the cointegrating vector known, OLS equation by equation is the Gaussian maximum likelihood estimator.
    Arguments: design is the Design; weights optional row multiplicities (for a bootstrap refit).
    Returns: a Fit; raises ValueError when there are too few rows.
    Reference: Johansen (1995), Likelihood-Based Inference in Cointegrated Vector Autoregressive Models.
    """
    x, y = design.x, design.y
    if weights is None:
        weighted = x
        total = float(x.shape[0])
    else:
        weighted = x * weights[:, None]
        total = float(weights.sum())
    gram = weighted.T @ x
    active = np.diag(gram) > 0
    coefficients = np.zeros((x.shape[1], y.shape[1]))
    coefficients[active] = np.linalg.solve(gram[np.ix_(active, active)], (weighted.T @ y)[active])
    residual = y - x @ coefficients
    degrees = total - int(active.sum())
    if degrees <= 0:
        raise ValueError("too few rows")
    sigma = (residual.T @ (residual * (weights[:, None] if weights is not None else 1.0))) / degrees
    return Fit(coefficients, sigma, design.lags, int(x.shape[0]), design.pairs)


def select_lags(pairs: Sequence[Pair], max_lags: int = DEFAULT_MAX_LAGS) -> int:
    """Lag order minimising the Schwarz criterion ln det(E'E / n) + 2 p_k ln(n) / n over k = 0, ..., max_lags.

    p_k is the number of regressors per equation at order k. All orders are compared on the common rows valid at max_lags.
    Arguments: pairs are the Pair objects; max_lags the largest lag order (default 3).
    Returns: the chosen lag order.
    Reference: Lutkepohl (2005), New Introduction to Multiple Time Series Analysis.
    """
    best, chosen = np.inf, 0
    for lags in range(max_lags + 1):
        design = build_design(pairs, lags, max_lags)
        coefficients = np.linalg.lstsq(design.x, design.y, rcond=None)[0]
        residual = design.y - design.x @ coefficients
        n = design.x.shape[0]
        sign, logdet = np.linalg.slogdet(residual.T @ residual / n)
        if sign <= 0:
            continue
        criterion = logdet + 2 * design.x.shape[1] * np.log(n) / n
        if criterion < best:
            best, chosen = criterion, lags
    return chosen


def default_block_length(keys: np.ndarray) -> int:
    """Default block length ceil(m^(1/3)) for m distinct time keys (Hall, Horowitz and Jing 1995, Biometrika)."""
    return max(1, int(np.ceil(np.unique(keys).size ** (1.0 / 3.0))))


class KeyBlocks:
    """Moving-block bootstrap over the distinct time keys of a design; all rows with a key share its multiplicity.

    Constructor arguments: keys are the row time keys; block_length the number of consecutive keys per block
    (default_block_length when None).
    Reference: Kunsch (1989), Annals of Statistics.
    """

    def __init__(self, keys: np.ndarray, block_length: int | None = None):
        self.unique, self.inverse = np.unique(keys, return_inverse=True)
        self.block_length = default_block_length(keys) if block_length is None else block_length

    def weights(self, rng: np.random.Generator) -> np.ndarray:
        """One bootstrap draw of row multiplicities with the generator rng."""
        drawn = moving_block_indices(self.unique.size, min(self.block_length, self.unique.size), rng)
        return np.bincount(drawn, minlength=self.unique.size).astype(float)[self.inverse]
