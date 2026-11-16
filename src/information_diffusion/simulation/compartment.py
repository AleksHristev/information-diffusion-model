"""Exact stochastic simulation of compartment and Bass models, and one simulate-and-fit replicate for parameter recovery."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np

from information_diffusion.epidemic import compartment_common as common


def simulate_event(spec: common.CompartmentSpec, theta: Sequence[float] | Mapping[str, float], rng: np.random.Generator, *, n_nodes: int = common.N_NODES, seeds: int | None = None, times: np.ndarray = common.TIME_GRID) -> np.ndarray:
    """Simulate one event of a compartment model by the Gillespie direct method and return its observed share on the grid.

    The next jump comes after an Exp(sum_m a_m) time, a_m = theta_m c_m(x, N), and is transition m with probability
    a_m / sum_m a_m; the share at grid time t is the state after the last jump at or before t.
    Arguments: spec is the model; theta its parameters (mapping or sequence); rng the generator; n_nodes, seeds and times
    as in compartment_common.
    Returns: the observed share at each grid time.
    Reference: Gillespie (1977), J. Phys. Chem.
    """
    values = common.parameter_vector(spec, theta)
    rate = {name: float(v) for name, v in zip(spec.parameters, values, strict=True)}
    multipliers = [rate[tr.parameter] for tr in spec.transitions]
    coefficients = [tr.coefficient for tr in spec.transitions]
    deltas = [tr.delta for tr in spec.transitions]
    compartments = range(len(spec.compartments))
    counts = list(common.initial_counts(spec, n_nodes, seeds))
    observed = list(spec.observed)
    horizon = float(times[-1])
    jump_times = [0.0]
    jump_counts = [sum(counts[i] for i in observed)]
    uniforms = rng.random(4096).tolist()
    used = 0
    now = 0.0
    while True:
        intensities = [m * c(counts, n_nodes) for m, c in zip(multipliers, coefficients, strict=True)]
        total = sum(intensities)
        if total <= 0.0:
            break
        if used + 2 > len(uniforms):
            uniforms = rng.random(4096).tolist()
            used = 0
        now -= np.log(1.0 - uniforms[used]) / total
        used += 1
        if now > horizon:
            break
        pick = uniforms[used] * total
        used += 1
        chosen = len(intensities) - 1
        cumulative = 0.0
        for index, intensity in enumerate(intensities):
            cumulative += intensity
            if pick <= cumulative:
                chosen = index
                break
        delta = deltas[chosen]
        for j in compartments:
            counts[j] += delta[j]
        jump_times.append(now)
        jump_counts.append(sum(counts[i] for i in observed))
    positions = np.searchsorted(jump_times, times, side="right") - 1
    return np.asarray(jump_counts, dtype=float)[positions] / n_nodes


def simulate_events(spec: common.CompartmentSpec, theta: Sequence[float] | Mapping[str, float], n_events: int, rng: np.random.Generator, *, n_nodes: int = common.N_NODES, seeds: int | None = None, times: np.ndarray = common.TIME_GRID) -> np.ndarray:
    """Simulate n_events independent events with simulate_event; other arguments as there. Returns shape (n_events, T)."""
    return np.stack([simulate_event(spec, theta, rng, n_nodes=n_nodes, seeds=seeds, times=times) for _ in range(n_events)])


def replicate(spec: common.CompartmentSpec, truth: Mapping[str, float], stream: np.random.SeedSequence, *, bootstrap: int, mean_model: str = "chain", interval_method: str = "refit", n_events: int, n_nodes: int = common.N_NODES) -> dict[str, np.ndarray]:
    """Simulate one data set of events at known parameters and fit the model to it.

    Arguments: spec is the model; truth the true parameters; stream the seed sequence of this replicate; bootstrap the
    number of bootstrap draws; mean_model and interval_method as in compartment_common.fit_events; n_events the number
    of events; n_nodes the number of nodes.
    Returns: a dict with estimate, lower, upper, draw_correlations (pairwise correlations of the bootstrap log-parameter
    draws) and condition_number; all NaN when the fit fails or does not converge.
    """
    n_parameters = len(spec.parameters)
    pair_count = n_parameters * (n_parameters - 1) // 2
    failed = {
        "estimate": np.full(n_parameters, np.nan),
        "lower": np.full(n_parameters, np.nan),
        "upper": np.full(n_parameters, np.nan),
        "draw_correlations": np.full(pair_count, np.nan),
        "condition_number": np.array(np.nan),
    }
    rng = np.random.default_rng(stream)
    try:
        curves = simulate_events(spec, truth, n_events, rng, n_nodes=n_nodes)
        fit = common.fit_events(spec, curves, bootstrap=bootstrap, rng=rng, mean_model=mean_model, n_nodes=n_nodes, interval_method=interval_method)
    except (ArithmeticError, ValueError, np.linalg.LinAlgError):
        return failed
    if not fit.converged or not np.all(np.isfinite(fit.estimate)) or not np.all(np.isfinite(fit.lower)):
        return failed
    pairs = _draw_correlations(fit.bootstrap_log)
    return {"estimate": fit.estimate, "lower": fit.lower, "upper": fit.upper, "draw_correlations": pairs, "condition_number": np.array(fit.information_condition_number)}


def _draw_correlations(draws: np.ndarray) -> np.ndarray:
    draws = draws[np.isfinite(draws).all(axis=1)]
    if draws.shape[1] == 1:
        return np.empty(0)
    correlations = np.corrcoef(draws, rowvar=False)
    return np.array([correlations[i, j] for i, j in combinations(range(draws.shape[1]), 2)], dtype=float)












