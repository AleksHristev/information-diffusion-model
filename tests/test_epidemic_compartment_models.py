"""Checks of the compartment and Bass models: ODE, chain, simulator and fit."""

import numpy as np
import pytest

from information_diffusion.epidemic import bass, compartment_common as common, compartment_seir, compartment_si, compartment_sir, compartment_sirs, compartment_sis
from information_diffusion.simulation import compartment as simulator

ALL_MODELS = [(compartment_si.SPEC, (0.02,)), (compartment_sir.SPEC, (0.02, 0.005)), (compartment_sis.SPEC, (0.02, 0.005)), (compartment_sirs.SPEC, (0.02, 0.005, 0.002)), (compartment_seir.SPEC, (0.02, 0.01, 0.005)), (bass.SPEC, (0.001, 0.02))]
TIMES = common.TIME_GRID


def test_grid_is_the_30_second_grid_to_3600_seconds():
    assert TIMES[0] == 0.0 and TIMES[-1] == 3600.0 and np.all(np.diff(TIMES) == 30.0) and TIMES.size == 121


@pytest.mark.parametrize("beta", [0.005, 0.02])
def test_si_ode_matches_logistic_closed_form(beta):
    numeric = common.mean_field_curve(compartment_si.SPEC, np.array([beta]), TIMES, rk4_step=1.0)
    assert np.max(np.abs(numeric - compartment_si.logistic(beta, TIMES))) < 1e-8


@pytest.mark.parametrize("p,q", [(0.001, 0.02), (0.005, 0.01)])
def test_bass_ode_matches_closed_form(p, q):
    numeric = common.mean_field_curve(bass.SPEC, np.array([p, q]), TIMES, rk4_step=1.0)
    assert np.max(np.abs(numeric - bass.closed_form(p, q, TIMES))) < 1e-8


def test_sis_ode_matches_logistic_closed_form():
    numeric = common.mean_field_curve(compartment_sis.SPEC, np.array([0.02, 0.005]), TIMES, rk4_step=1.0)
    assert np.max(np.abs(numeric - compartment_sis.closed_form(0.02, 0.005, TIMES))) < 1e-8


@pytest.mark.parametrize("spec,theta", ALL_MODELS, ids=lambda x: getattr(x, "name", None))
def test_ode_conserves_total_share(spec, theta):
    trajectory = common.mean_field_compartments(spec, np.array(theta), TIMES)
    assert np.max(np.abs(trajectory.sum(axis=1) - 1.0)) < 1e-10


@pytest.mark.parametrize("spec,theta", ALL_MODELS, ids=lambda x: getattr(x, "name", None))
def test_chain_conserves_probability_and_nodes(spec, theta):
    chain = common.chain_for(spec, 12)
    distributions = chain.distributions(np.array(theta), TIMES)
    assert np.max(np.abs(distributions.sum(axis=1) - 1.0)) < 1e-9
    assert np.all(chain.states.sum(axis=1) == 12)


def test_chain_mean_matches_the_two_node_si_formula():
    beta = 0.02
    chain = common.FiniteChain(compartment_si.SPEC, 2)
    expected = (1.0 + (1.0 - np.exp(-0.5 * beta * TIMES))) / 2.0
    assert np.max(np.abs(chain.mean_curve([beta], TIMES) - expected)) < 1e-10


def test_chain_mean_matches_the_one_node_bass_formula():
    p, q = 0.003, 0.02
    chain = common.FiniteChain(bass.SPEC, 1)
    assert np.max(np.abs(chain.mean_curve([p, q], TIMES) - (1.0 - np.exp(-p * TIMES)))) < 1e-10


def test_uniformization_large_chain_matches_a_dense_matrix_exponential():
    from scipy.linalg import expm

    theta = np.array([0.01, 0.002])
    chain = common.FiniteChain(compartment_sir.SPEC, 10)
    assert chain.n_states > common.DENSE_STATE_LIMIT
    generator = np.zeros((chain.n_states, chain.n_states))
    generator[chain._csr_rows, chain._csr_cols] = chain._coef @ theta
    probabilities = np.zeros(chain.n_states)
    probabilities[0] = 1.0
    propagator = expm(generator * 30.0)
    reference = [chain.observed @ probabilities]
    for _ in range(TIMES.size - 1):
        probabilities = propagator @ probabilities
        reference.append(chain.observed @ probabilities)
    assert np.max(np.abs(chain.mean_curve(theta, TIMES) - np.array(reference))) < 1e-8


@pytest.mark.parametrize("spec,theta", ALL_MODELS, ids=lambda x: getattr(x, "name", None))
def test_simulator_mean_matches_the_exact_chain_mean_at_thirty_nodes(spec, theta):
    rng = np.random.default_rng(11)
    curves = simulator.simulate_events(spec, theta, 800, rng)
    exact = common.chain_for(spec).mean_curve(np.array(theta))
    standard_error = curves.std(axis=0, ddof=1) / np.sqrt(curves.shape[0])
    assert np.max(np.abs(curves.mean(axis=0) - exact) / (standard_error + 1e-3)) < 4.5


@pytest.mark.parametrize("spec,theta", ALL_MODELS, ids=lambda x: getattr(x, "name", None))
def test_simulator_mean_matches_the_ode_at_large_n(spec, theta):
    n_nodes = 1200
    seeds = 0 if spec.default_seeds == 0 else 40
    rng = np.random.default_rng(13)
    curves = simulator.simulate_events(spec, theta, 16, rng, n_nodes=n_nodes, seeds=seeds)
    ode = common.mean_field_curve(spec, np.array(theta), TIMES, n_nodes=n_nodes, seeds=seeds)
    assert np.max(np.abs(curves.mean(axis=0) - ode)) < 0.02


@pytest.mark.parametrize("spec,theta", ALL_MODELS, ids=lambda x: getattr(x, "name", None))
def test_simulated_curves_have_real_discreteness(spec, theta):
    curves = simulator.simulate_events(spec, theta, 5, np.random.default_rng(3))
    expected_start = spec.default_seeds / common.N_NODES
    assert curves.shape == (5, TIMES.size)
    assert np.allclose(curves[:, 0], expected_start)
    assert np.allclose(curves * common.N_NODES, np.round(curves * common.N_NODES))
    if spec in (compartment_si.SPEC, compartment_sir.SPEC, compartment_seir.SPEC, bass.SPEC):
        assert np.all(np.diff(curves, axis=1) >= -1e-12)


@pytest.mark.parametrize("spec,theta", [(compartment_si.SPEC, (0.02,)), (compartment_sir.SPEC, (0.01, 0.002)), (compartment_sis.SPEC, (0.02, 0.005)), (compartment_sirs.SPEC, (0.02, 0.005, 0.002)), (compartment_seir.SPEC, (0.01, 0.005, 0.002)), (bass.SPEC, (0.005, 0.01))], ids=lambda x: getattr(x, "name", None))
def test_fit_recovers_parameters_from_a_noiseless_chain_curve(spec, theta):
    target = common.chain_for(spec).mean_curve(np.array(theta))
    start = np.log(np.array(theta)) + np.array([0.4, -0.3, 0.3]) if spec is compartment_seir.SPEC else None
    u, _, ssr, converged = common.fit_curve(spec, target, mean_model="chain", start=start)
    assert converged and ssr < 1e-8
    assert np.max(np.abs(np.exp(u) / np.array(theta) - 1.0)) < 2e-2


@pytest.mark.parametrize("spec,theta", [(compartment_si.SPEC, (0.02,)), (compartment_sir.SPEC, (0.02, 0.005)), (bass.SPEC, (0.001, 0.02))], ids=lambda x: getattr(x, "name", None))
def test_fit_recovers_parameters_from_a_noiseless_ode_curve(spec, theta):
    target = common.mean_field_curve(spec, np.array(theta))
    u, _, ssr, converged = common.fit_curve(spec, target, mean_model="ode")
    assert converged and ssr < 1e-8
    assert np.max(np.abs(np.exp(u) / np.array(theta) - 1.0)) < 1e-3


def test_bootstrap_interval_brackets_the_estimate_and_is_reproducible():
    curves = simulator.simulate_events(compartment_si.SPEC, (0.01,), 33, np.random.default_rng(5))
    first = common.fit_events(compartment_si.SPEC, curves, bootstrap=99, rng=np.random.default_rng(1))
    second = common.fit_events(compartment_si.SPEC, curves, bootstrap=99, rng=np.random.default_rng(1))
    assert first.lower[0] <= first.estimate[0] <= first.upper[0]
    assert first.lower[0] == second.lower[0] and first.upper[0] == second.upper[0]


@pytest.mark.parametrize("method", common.INTERVAL_METHODS)
def test_every_interval_method_brackets_the_estimate(method):
    curves = simulator.simulate_events(bass.SPEC, (0.005, 0.01), 33, np.random.default_rng(8))
    fit = common.fit_events(bass.SPEC, curves, bootstrap=49, rng=np.random.default_rng(2), interval_method=method)
    assert np.all(fit.lower <= fit.estimate) and np.all(fit.estimate <= fit.upper)


def test_one_step_approximates_the_refit_on_a_near_linear_problem():
    curves = simulator.simulate_events(compartment_si.SPEC, (0.01,), 33, np.random.default_rng(21))
    one_step = common.fit_events(compartment_si.SPEC, curves, bootstrap=49, rng=np.random.default_rng(4), interval_method="one_step")
    refit = common.fit_events(compartment_si.SPEC, curves, bootstrap=49, rng=np.random.default_rng(4), interval_method="refit")
    assert np.max(np.abs(one_step.bootstrap_log - refit.bootstrap_log)) < 0.05


def test_unknown_interval_method_is_rejected():
    curves = simulator.simulate_events(compartment_si.SPEC, (0.01,), 5, np.random.default_rng(1))
    with pytest.raises(ValueError):
        common.fit_events(compartment_si.SPEC, curves, bootstrap=9, rng=np.random.default_rng(1), interval_method="bca")


def test_too_many_failed_draws_fails_the_interval(monkeypatch):
    curves = simulator.simulate_events(compartment_si.SPEC, (0.01,), 12, np.random.default_rng(1))
    original = common.fit_curve
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 1 and calls["n"] % 4 == 0:
            raise ValueError("no convergence")
        return original(*args, **kwargs)

    monkeypatch.setattr(common, "fit_curve", flaky)
    fit = common.fit_events(compartment_si.SPEC, curves, bootstrap=39, rng=np.random.default_rng(1), interval_method="refit")
    assert np.isnan(fit.lower).all() and np.isnan(fit.upper).all()


def test_failed_replicate_returns_nan_rather_than_raising(monkeypatch):
    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(common, "fit_events", fail)
    result = simulator.replicate(compartment_si.SPEC, {"beta": 0.02}, np.random.SeedSequence(1), bootstrap=9, n_events=33)
    assert np.isnan(result["estimate"]).all() and np.isnan(result["lower"]).all()


def test_parameter_vector_rejects_a_wrong_length():
    with pytest.raises(ValueError):
        common.parameter_vector(compartment_sir.SPEC, [0.02])
