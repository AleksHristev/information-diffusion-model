"""Exponential Hawkes fit with the lowest penalised objective over several descents."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from information_diffusion.epidemic.hawkes_exponential import ExponentialHawkes, FitResult

FAST_KERNELS = (np.array([0.5, 200.0]), np.array([0.8, 900.0]))
TIE_TOLERANCE = 1e-4


@dataclass(frozen=True)
class BestDescent:
    """Result of best_fit.

    Fields: fit is the chosen fit; single_descent is the fit from the single default descent; objective_gap is the
    single-descent objective minus the lowest objective found; descents is the number of descents tried; changed says
    whether another descent replaced the single-descent fit.
    """

    fit: FitResult
    single_descent: FitResult
    objective_gap: float
    descents: int
    changed: bool


def best_fit(problem: ExponentialHawkes, *, tolerance: float = TIE_TOLERANCE) -> BestDescent:
    """Fit an exponential Hawkes model from several starts and keep the lowest penalised objective, x_hat = argmin_k J(x_k).

    The descents start from each data-driven start of the model and from two fast-decay kernel starts. The single default
    descent stays the estimate unless another descent lowers the objective by more than tolerance.
    Arguments: problem is the ExponentialHawkes model to fit; tolerance is the improvement needed to replace the single descent.
    Returns: a BestDescent.
    Reference: Byrd, Lu, Nocedal and Zhu (1995), SIAM J. Sci. Comput. (the descent); Hawkes (1971), Biometrika (the
    model).
    """
    single = problem.fit()
    points = problem.start_points()
    fast = [np.concatenate((points[0][: problem.K], problem.encode_kernel(kernel))) for kernel in FAST_KERNELS]
    others = [fit for fit in (problem.fit(point) for point in (*points, *fast)) if np.isfinite(fit.nll)]
    best = min(others, key=lambda fit: fit.nll) if others else single
    gap = float(single.nll - best.nll) if np.isfinite(single.nll) else float("inf")
    changed = bool(others) and gap > tolerance
    return BestDescent(best if changed else single, single, gap if np.isfinite(gap) else float("nan"), len(others) + 1, changed)
