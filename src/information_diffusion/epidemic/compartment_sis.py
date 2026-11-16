"""SIS (susceptible-infected-susceptible) compartment model: specification, logistic mean-field curve and fit to absorption curves."""

from __future__ import annotations

import numpy as np

from information_diffusion.epidemic import compartment_common as common

SPEC = common.CompartmentSpec(
    name="SIS",
    parameters=("beta", "gamma"),
    compartments=("S", "I"),
    transitions=(
        common.Transition("beta", (-1, 1), lambda c, n: c[0] * c[1] / n),
        common.Transition("gamma", (1, -1), lambda c, n: c[1]),
    ),
    observed=(1,),
    seed_compartment=1,
    default_seeds=1,
    start_grid={"beta": tuple(np.geomspace(1e-3, 0.1, 12)), "gamma": tuple(np.geomspace(2e-4, 0.05, 10))},
)


def closed_form(beta: float, gamma: float, times: np.ndarray = common.TIME_GRID, *, initial_share: float = 1 / common.N_NODES) -> np.ndarray:
    """Mean-field activated share of the SIS model, I(t) = K / (1 + (K / I0 - 1) exp(-r t)) with r = beta - gamma, K = r / beta.

    When beta equals gamma the limit I0 / (1 + beta I0 t) is used.
    Arguments: beta and gamma are the infection and recovery rates (per second); times the grid in seconds;
    initial_share is I0 (default 1/30).
    Returns: the activated share at each time.
    Reference: Hethcote (2000), SIAM Review.
    """
    times = np.asarray(times, dtype=float)
    rate = beta - gamma
    if abs(rate) < 1e-14:
        return initial_share / (1.0 + beta * initial_share * times)
    capacity = rate / beta
    return capacity / (1.0 + (capacity / initial_share - 1.0) * np.exp(-rate * times))


def fit(curves: np.ndarray, **kwargs) -> common.CurveFit:
    """Fit the SIS model: S -> I at rate beta S I / N, I -> S at rate gamma I, from one activated node; the observed curve is I.

    Arguments: curves has one row per event and one column per grid point; kwargs are passed to compartment_common.fit_events.
    Returns: a CurveFit with the estimates of beta and gamma (per second) and their bootstrap intervals.
    Reference: Kermack and McKendrick (1927), Proc. R. Soc. A; Hethcote (2000), SIAM Review.
    """
    return common.fit_events(SPEC, curves, **kwargs)
