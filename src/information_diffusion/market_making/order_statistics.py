"""Agreement and out-of-sample predictability of the order in which a set of instruments reacts across releases."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from information_diffusion.market_making import rank_statistics as rk

REACTION = 'reaction (stored absorption time)'
GAP_RHO = 'Spearman rho of the quote gap and the lag, per release'
GAP_ITSELF = 'quote gap itself'
RESIDUAL = 'reaction after removing the quote gap (rank residuals)'
TAU = 'Kendall tau between the two orders, per release'
SECOND = "second definition (first one-tick move in the index funds' direction)"
OOS_RESIDUAL_LABEL = 'after removing the quote gap'


@dataclass(frozen=True)
class TestRelease:
    """A release whose reaction order is predicted from earlier releases.

    Fields: event_id, its identifier; row, its row in the release matrices; previous_rows, the rows of the earlier
    releases used to predict it; previous_ids, their identifiers joined into one string.
    """

    event_id: str
    row: int
    previous_rows: tuple[int, ...]
    previous_ids: str


def rank_matrix(values: np.ndarray) -> np.ndarray:
    """Return the within-release average ranks of an (m releases, n instruments) matrix, missing values ranked last."""
    return np.vstack([rk.ranks_with_missing(row) for row in values])


def oos(values: np.ndarray, tests: list[TestRelease]) -> pd.DataFrame:
    """Score the out-of-sample reaction order of each test release by Spearman's rho.

    The predicted order ranks each instrument by its median over the previous releases, a missing value counting as
    later than every value; rho compares these ranks with the release's own ranks, missing values ranked last.
    Arguments: values is an (m releases, n instruments) matrix; tests is the list of TestRelease.
    Returns: a DataFrame with one row per test release and the columns test_event, rho and previous (the previous ids).
    Reference: Spearman (1904).
    """
    rows = []
    for t in tests:
        past = values[list(t.previous_rows)]
        past = np.where(np.isnan(past), np.inf, past)
        predicted = np.median(past, axis=0)
        rho = rk.spearman_ranks(rk.ranks_with_missing(np.where(np.isfinite(predicted), predicted, np.nan)), rk.ranks_with_missing(values[t.row]))
        rows.append({"test_event": t.event_id, "rho": rho, "previous": t.previous_ids})
    return pd.DataFrame(rows)


def order_statistics(lag: np.ndarray, gap: np.ndarray, second: np.ndarray, instruments: list[str], columns: list[int], groups: dict[str, str],
                     tests: list[TestRelease], seed: int, permutations: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute statistics of the reaction order of a set of instruments, with every rank recomputed within the set.

    The summary covers agreement across releases (Kendall's W with permutation p-values), the out-of-sample Spearman
    rho with an exact sign test, the quote gap (per-release rho between gap and lag ranks, and the order after removing
    the gap by rank residuals) and a second reaction definition (per-release Kendall's tau-b against the lag order).
    Arguments: lag, gap and second are (m releases, all instruments) matrices in milliseconds, NaN where missing: lag
    is each instrument's reaction time after each release, gap its quote gap (for example the median time between its
    NBBO updates, as from reaction_measures.median_update_gap, in milliseconds) and second its reaction time under a
    second definition (for example the first one-tick move, as in reaction_measures.first_tick_move); instruments are the instrument names; columns are the column indices of the
    set; groups maps an instrument to its group (an empty group when absent); tests is the list of TestRelease; seed
    seeds numpy.random.default_rng for the W permutations (drawn in the order lag, gap, residual, second);
    permutations is the number of permutations.
    Returns: the summary rows, the per-instrument rows (sorted by mean lag rank) and the per-test-release rows, as
    DataFrames.
    Reference: Kendall and Babington Smith (1939); Kendall and Gibbons (1990); Spearman (1904).
    """
    for name, a in (("lag", lag), ("gap", gap), ("second", second)):
        if a.ndim != 2 or a.shape != lag.shape:
            raise ValueError(f"{name}: shape {a.shape}, expected {lag.shape}")
    cols = list(columns)
    names = [instruments[j] for j in cols]
    lag, gap, second = lag[:, cols], gap[:, cols], second[:, cols]
    m, n = lag.shape
    rng = np.random.default_rng(seed)
    r_lag, r_gap, r_second = rank_matrix(lag), rank_matrix(gap), rank_matrix(second)
    residual = np.vstack([rk.rank_residuals(r_lag[i], r_gap[i]) for i in range(m)])
    r_res = rank_matrix(residual)
    rows = []

    def w_row(label, ranks):
        w, p = rk.kendall_w_permutation(ranks, rng, permutations)
        rows.append({"question": "(1) agreement across releases", "measure": label, "statistic": "Kendall's W", "value": w, "p_value": p, "releases": ranks.shape[0], "instruments": ranks.shape[1]})

    def oos_row(label, frame):
        pos, neg, p = rk.sign_test(frame.rho.to_numpy())
        rows.append({"question": "(2) out of sample", "measure": label, "statistic": "mean Spearman rho over test releases", "value": float(frame.rho.mean()), "p_value": p,
                     "releases": len(frame), "positive": pos, "negative": neg, "share_positive": pos / len(frame) if len(frame) else np.nan, "median": float(frame.rho.median()), "instruments": n})

    w_row(REACTION, r_lag)
    first = oos(lag, tests)
    oos_row(REACTION, first)
    rho_gap = np.array([rk.spearman_ranks(r_gap[i], r_lag[i]) for i in range(m)])
    pos, neg, p = rk.sign_test(rho_gap)
    rows.append({"question": "(3) quote gap", "measure": GAP_RHO, "statistic": "mean rho", "value": float(np.nanmean(rho_gap)), "p_value": p,
                 "releases": int(np.isfinite(rho_gap).sum()), "positive": pos, "negative": neg, "share_positive": pos / np.isfinite(rho_gap).sum(), "median": float(np.nanmedian(rho_gap)),
                 "mean_rho_squared": float(np.nanmean(rho_gap ** 2)), "instruments": n})
    w_row(GAP_ITSELF, r_gap)
    w_row(RESIDUAL, r_res)
    second_oos = oos(residual, tests)
    oos_row(RESIDUAL, second_oos)
    taus = np.array([rk.tau(r_lag[i], r_second[i]) for i in range(m)])
    pos, neg, p = rk.sign_test(taus)
    rows.append({"question": "(4) second definition", "measure": TAU, "statistic": "mean tau", "value": float(np.nanmean(taus)), "p_value": p,
                 "releases": int(np.isfinite(taus).sum()), "positive": pos, "negative": neg, "share_positive": pos / np.isfinite(taus).sum(), "median": float(np.nanmedian(taus)), "instruments": n})
    w_row(SECOND, r_second)
    fin = lambda x: float(x) if np.isfinite(x) else np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        funds = _funds(lag, gap, second, r_lag, r_gap, r_second, r_res, names, groups, fin)
    oos_frame = pd.concat([first.assign(measure=REACTION), second_oos.assign(measure=OOS_RESIDUAL_LABEL)], ignore_index=True)
    return pd.DataFrame(rows), pd.DataFrame(funds).sort_values("mean_rank", kind="mergesort").reset_index(drop=True), oos_frame


def _funds(lag, gap, second, r_lag, r_gap, r_second, r_res, names, groups, fin) -> list[dict]:
    funds = []
    for j, s in enumerate(names):
        lj = np.where(np.isnan(lag[:, j]), np.inf, lag[:, j])
        sj = np.where(np.isnan(second[:, j]), np.inf, second[:, j])
        funds.append({"instrument": s, "group": groups.get(s, ""), "median_lag_ms": fin(np.median(lj)), "q25_lag_ms": fin(np.quantile(lj, 0.25)), "q75_lag_ms": fin(np.quantile(lj, 0.75)),
                      "never": int(np.isnan(lag[:, j]).sum()), "mean_rank": float(r_lag[:, j].mean()), "median_quote_gap_ms": float(np.nanmedian(gap[:, j])), "mean_gap_rank": float(r_gap[:, j].mean()),
                      "median_second_reaction_ms": fin(np.median(sj)), "mean_second_rank": float(r_second[:, j].mean()), "mean_residual_rank": float(r_res[:, j].mean())})
    return funds


