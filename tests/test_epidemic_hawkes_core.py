"""Checks of the Hawkes estimators, event merging, spline baseline and simulators."""

import numpy as np
import pytest
from scipy import integrate

from information_diffusion.epidemic import branching_ratio as br
from information_diffusion.epidemic import event_merging as em
from information_diffusion.epidemic import hawkes_exponential as he
from information_diffusion.epidemic import hawkes_marked as hm
from information_diffusion.epidemic import hawkes_multivariate as mv
from information_diffusion.epidemic import hawkes_power_law as hp
from information_diffusion.epidemic import hawkes_sum_exponentials as hs
from information_diffusion.epidemic import spline_baseline as sb
from information_diffusion.epidemic import time_rescaling as tr
from information_diffusion.simulation import hawkes as sim

T = 5400.0
RELEASE = 1800.0


def _times(rng, n=800):
    return np.sort(np.concatenate((rng.uniform(0, T, n // 2), 200 + 5 * rng.standard_normal(n // 2) ** 2)))


def test_merge_floors_to_millisecond_and_sums_marks():
    times, marks = em.merge_events([0.0104, 0.0101, 0.0120, 1.2345, 1.2349], np.array([1, 2, 3, 4, 5.0]))
    np.testing.assert_allclose(times, [0.010, 0.012, 1.234])
    np.testing.assert_allclose(marks, [3.0, 3.0, 9.0])


def test_merge_exact_multiples_stay_in_their_cell_and_unmarked_counts():
    times, marks = em.merge_events(np.array([0.003, 0.003, 0.0031]))
    np.testing.assert_allclose(times, [0.003])
    np.testing.assert_allclose(marks, [3.0])
    assert np.all(em.merge_events(np.array([]))[0].size == 0)


def _double_sums(times, beta, weights):
    A = np.zeros(times.size)
    C = np.zeros(times.size)
    for i in range(times.size):
        lag = times[i] - times[:i]
        contribution = weights[:i] * np.exp(-beta * lag)
        A[i] = contribution.sum()
        C[i] = (lag * contribution).sum()
    return A, C


@pytest.mark.parametrize("beta", [0.02, 1.0, 10.0, 200.0, 5000.0])
def test_exponential_states_match_double_sum(beta):
    rng = np.random.default_rng(int(beta * 7) + 1)
    times = _times(rng)
    weights = rng.lognormal(sigma=0.5, size=times.size)
    A, C = he.exp_states(times, beta, weights, deriv=True)
    A_direct, C_direct = _double_sums(times, beta, weights)
    positive = A_direct > 1e-250
    assert np.max(np.abs(A[positive] - A_direct[positive]) / A_direct[positive]) < 1e-9
    positive_c = C_direct > 1e-100
    assert np.max(np.abs(C[positive_c] - C_direct[positive_c]) / C_direct[positive_c]) < 1e-8


def test_exponential_states_use_block_and_doubling_paths_consistently():
    rng = np.random.default_rng(2)
    times = _times(rng, 2000)
    A_blocks = he.exp_states(times, 1.0)
    A_doubling = he._exp_states_doubling(times, 1.0, np.ones(times.size), False)
    positive = A_blocks > 1e-200
    assert np.max(np.abs(A_blocks[positive] - A_doubling[positive]) / A_blocks[positive]) < 1e-11


def test_spline_basis_partition_of_unity_and_exact_integral():
    baseline = sb.SplineBaseline(T, RELEASE, n_events=40_000)
    t = np.linspace(0, T, 5001)
    assert abs(np.asarray(baseline.design(t).sum(axis=1)).ravel() - 1).max() < 1e-12
    rng = np.random.default_rng(3)
    b = rng.normal(0, 0.4, baseline.n_coef) + 1.0
    value, gradient = baseline.integral_and_grad(b)
    fine = 0.0
    edges = np.unique(baseline.knots)
    for left, right in zip(edges[:-1], edges[1:]):
        grid = np.linspace(left, right, 3000)
        fine += integrate.trapezoid(baseline.rate(b, grid), grid)
    assert value == pytest.approx(fine, rel=1e-6)
    for k in range(5):
        delta = np.zeros(baseline.n_coef)
        delta[k] = 1e-5
        numeric = (baseline.integral_and_grad(b + delta)[0] - baseline.integral_and_grad(b - delta)[0]) / 2e-5
        assert numeric == pytest.approx(gradient[k], rel=1e-5)


def test_spline_jump_at_release_and_cumulative():
    baseline = sb.SplineBaseline(T, RELEASE, n_events=2000)
    b = np.full(baseline.n_coef, 0.0)
    b[baseline.n_left:] = 2.0
    rate = baseline.rate(b, np.array([RELEASE - 1e-6, RELEASE]))
    assert rate[0] == pytest.approx(1.0, rel=1e-9)
    assert rate[1] == pytest.approx(np.exp(2.0), rel=1e-9)
    cumulative = baseline.cumulative(b, np.array([RELEASE, T]))
    assert cumulative[0] == pytest.approx(RELEASE, rel=1e-6)
    assert cumulative[1] == pytest.approx(baseline.integral_and_grad(b)[0], rel=1e-9)


def test_spline_represents_the_spiking_shape_closely():
    baseline = sb.SplineBaseline(T, RELEASE, n_events=40_000)
    t = np.linspace(0, T, 40001)
    target = np.where(t >= RELEASE, np.log(1 + 30 * np.exp(-(t - RELEASE) / 60)), 0.0)
    design = baseline.design(t).toarray()
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    fitted = design @ coefficients
    assert np.max(np.abs(fitted - target)) < 0.02


def test_spline_sampler_mean_count_matches_integral():
    baseline = sb.SplineBaseline(T, RELEASE, n_events=2000)
    b = np.full(baseline.n_coef, np.log(0.2))
    b[baseline.n_left] = np.log(1.5)
    sampler = baseline.sampler(b)
    rng = np.random.default_rng(4)
    counts = [sampler.sample(rng).size for _ in range(400)]
    assert np.mean(counts) == pytest.approx(sampler.expected_count(), rel=0.02)


def _brute_force_nll(kernel_function, compensator, baseline, b, times):
    mu = baseline.rate(b, times)
    excitation = np.array([kernel_function(times[i] - times[:i]).sum() if i else 0.0 for i in range(times.size)])
    integral = baseline.integral_and_grad(b)[0]
    penalty = baseline.penalty_and_grad(b)[0]
    return -np.sum(np.log(mu + excitation)) + integral + compensator + penalty


@pytest.mark.parametrize("baseline_kind", ["constant", "spline"])
def test_exponential_likelihood_equals_double_sum(baseline_kind):
    rng = np.random.default_rng(5)
    times = np.sort(rng.uniform(0, 400, 400)) * 13.0
    baseline = sb.ConstantBaseline(T) if baseline_kind == "constant" else sb.SplineBaseline(T, RELEASE, n_events=400)
    b = np.array([-1.0]) if baseline_kind == "constant" else np.full(baseline.n_coef, -1.0)
    problem = he.ExponentialHawkes(times, T, baseline, resolution=None)
    eta, beta = 0.45, 1.7
    x = np.concatenate((b, problem.encode_kernel(np.array([eta, beta]))))
    compensator = eta * np.sum(1 - np.exp(-beta * (T - times)))
    direct = _brute_force_nll(lambda lag: eta * beta * np.exp(-beta * lag), compensator, baseline, b, times)
    assert problem.nll(x) == pytest.approx(direct, rel=1e-10)


def test_sum_exponentials_likelihood_equals_double_sum():
    rng = np.random.default_rng(6)
    times = np.sort(rng.uniform(0, T, 500))
    baseline = sb.ConstantBaseline(T)
    problem = hs.SumExponentialsHawkes(times, T, baseline, resolution=None)
    b = np.array([np.log(0.05)])
    eta, w1, b1, b2 = 0.6, 0.35, 8.0, 0.3
    x = np.concatenate((b, problem.encode_kernel(np.array([eta, w1, b1, b2]))))
    w = np.array([w1, 1 - w1])
    betas = np.array([b1, b2])
    compensator = sum(eta * w[m] * np.sum(1 - np.exp(-betas[m] * (T - times))) for m in range(2))
    kernel = lambda lag: eta * sum(w[m] * betas[m] * np.exp(-betas[m] * lag) for m in range(2))
    direct = _brute_force_nll(kernel, compensator, baseline, b, times)
    assert problem.nll(x) == pytest.approx(direct, rel=1e-10)


def test_power_law_likelihood_close_to_exact_double_sum():
    rng = np.random.default_rng(7)
    times = np.sort(rng.uniform(0, T, 500))
    baseline = sb.ConstantBaseline(T)
    problem = hp.PowerLawHawkes(times, T, baseline, resolution=None)
    b = np.array([np.log(0.05)])
    eta, theta, c = 0.5, 1.5, 0.1
    x = np.concatenate((b, problem.encode_kernel(np.array([eta, theta, c]))))
    compensator = eta * np.sum(1 - (c / (T - times + c)) ** theta)
    kernel = lambda lag: eta * theta * c ** theta * (lag + c) ** (-(1 + theta))
    direct = _brute_force_nll(kernel, compensator, baseline, b, times)
    assert problem.nll(x) == pytest.approx(direct, abs=1e-3)


def test_power_law_kernel_approximation_error_is_within_stated_tolerance():
    worst = 0.0
    for theta in (0.25, 0.7, 1.5, 3.0, 5.0):
        for c in (0.002, 0.03, 0.1, 1.0, 20.0):
            lags = np.geomspace(1e-3, T, 60)
            exact = hp.exact_kernel(theta, c, lags)
            approximate = hp.approximate_kernel(theta, c, lags)
            worst = max(worst, float(np.max(np.abs(approximate - exact) / exact)))
    assert worst < hp.STATED_RELATIVE_ERROR


def test_marked_reduces_to_unmarked_when_gamma_is_zero():
    rng = np.random.default_rng(8)
    times = np.sort(rng.uniform(0, T, 400))
    baseline = sb.ConstantBaseline(T)
    marks = rng.geometric(0.5, times.size).astype(float)
    b = np.array([np.log(0.03)])
    eta, beta = 0.4, 2.0
    marked = hm.MarkedHawkes(times, T, baseline, marks=marks, resolution=None)
    plain = he.ExponentialHawkes(times, T, baseline, resolution=None)
    x_marked = np.concatenate((b, marked.encode_kernel(np.array([eta, 0.0, beta]))))
    x_plain = np.concatenate((b, plain.encode_kernel(np.array([eta, beta]))))
    assert marked.nll(x_marked) == pytest.approx(plain.nll(x_plain), rel=1e-12)


def _fd_gradient(problem, x, step=1e-6):
    gradient = np.empty(x.size)
    for i in range(x.size):
        up = x.copy()
        down = x.copy()
        up[i] += step
        down[i] -= step
        gradient[i] = (problem.nll_grad(up)[0] - problem.nll_grad(down)[0]) / (2 * step)
    return gradient


@pytest.mark.parametrize("resolution", [None, 0.001])
@pytest.mark.parametrize("baseline_kind", ["constant", "spline"])
@pytest.mark.parametrize("model", ["exponential", "sum_exp", "power_law", "marked"])
def test_analytic_gradient_matches_finite_differences(model, baseline_kind, resolution):
    rng = np.random.default_rng(9)
    times = np.sort(rng.uniform(0, T, 600)) + 1e-4
    times, marks = em.merge_events(times, rng.geometric(0.5, times.size).astype(float))
    baseline = sb.ConstantBaseline(T) if baseline_kind == "constant" else sb.SplineBaseline(T, RELEASE, n_events=500)
    b = baseline.initial(times, 0.07)
    if model == "exponential":
        problem = he.ExponentialHawkes(times, T, baseline, resolution=resolution)
        kernel = np.array([0.6, 1.3])
    elif model == "sum_exp":
        problem = hs.SumExponentialsHawkes(times, T, baseline, resolution=resolution)
        kernel = np.array([0.5, 0.4, 9.0, 0.2])
    elif model == "power_law":
        problem = hp.PowerLawHawkes(times, T, baseline, resolution=resolution)
        kernel = np.array([0.5, 1.6, 0.2])
    else:
        problem = hm.MarkedHawkes(times, T, baseline, marks=marks, resolution=resolution)
        kernel = np.array([0.6, 0.4, 1.1])
    x = np.concatenate((b, problem.encode_kernel(kernel)))
    analytic = problem.nll_grad(x)[1]
    numeric = _fd_gradient(problem, x)
    assert np.max(np.abs(analytic - numeric) / (1.0 + np.abs(analytic))) < 5e-5


def test_renewal_expected_count_matches_exponential_closed_form():
    kernel = sim.ExponentialKernel(1.0)
    baseline = sim.FlatRate(2.0, T)
    eta = 0.8
    delta = 1.0 * (1 - eta)
    closed = 2.0 * (T / (1 - eta) - eta * (1 - np.exp(-delta * T)) / ((1 - eta) * delta))
    assert sim.expected_count(baseline, kernel, eta, T, step=0.01) == pytest.approx(closed, rel=2e-3)


@pytest.mark.parametrize("kernel,eta", [(sim.ExponentialKernel(1.0), 0.6), (sim.SumExponentialsKernel((10.0, 0.1), (0.5, 0.5)), 0.5), (sim.PowerLawKernel(0.1, 1.5), 0.6)])
def test_simulator_mean_count_matches_renewal_theory(kernel, eta):
    target = 2000.0
    baseline = sim.scenario_baseline("flat", kernel, eta, 1000.0, 300.0, target)
    rng = np.random.default_rng(11)
    counts = [sim.simulate_cluster(baseline.sample(rng), kernel, eta, 1000.0, rng).size for _ in range(800)]
    assert np.mean(counts) == pytest.approx(sim.expected_count(baseline, kernel, eta, 1000.0), rel=0.03)


def test_simulator_offspring_mean_equals_branching_ratio():
    rng = np.random.default_rng(12)
    eta = 0.7
    sizes = []
    for _ in range(6000):
        times = sim.simulate_cluster(np.array([0.0]), sim.ExponentialKernel(1.0), eta, 1e9, rng)
        sizes.append(times.size)
    assert np.mean(sizes) == pytest.approx(1 / (1 - eta), rel=0.04)


def test_marked_simulator_branching_with_mark_power():
    rng = np.random.default_rng(13)
    law = sim.GeometricMarks(0.5)
    gamma = 0.3
    eta_0 = 0.35
    eta = eta_0 * law.mean_power(gamma)
    sizes = [sim.simulate_cluster(np.array([0.0]), sim.ExponentialKernel(1.0), eta_0, 1e9, rng, marks_law=law, gamma=gamma)[0].size for _ in range(6000)]
    assert np.mean(sizes) == pytest.approx(1 / (1 - eta), rel=0.05)


def test_multivariate_simulator_counts():
    counts = np.array([200.0, 800.0, 400.0])
    mu = np.array([0.03, 0.08, 0.04])
    gamma = np.array([[0.2, 0.0, 0.1], [0.3, 0.2, 0.0], [0.0, 0.2, 0.2]])
    window = 1000.0
    rng = np.random.default_rng(14)
    expected = np.linalg.solve(np.eye(3) - gamma, mu) * window
    samples = [np.array([t.size for t in sim.simulate_multivariate(mu, gamma, 2.0, window, rng)]) for _ in range(600)]
    np.testing.assert_allclose(np.mean(samples, axis=0), expected, rtol=0.05)


def test_basic_logit_interval_reflects_draws_about_the_estimate():
    eta_hat = 0.6
    ceiling = he.ETA_CEILING
    logit = lambda x: np.log(x / ceiling / (1 - x / ceiling))
    draws = ceiling * 1.0 / (1 + np.exp(-(logit(0.55) + np.linspace(-0.3, 0.3, 501))))
    lower, upper = br.interval_from_draws(eta_hat, draws, 0.95, "basic_logit")
    assert lower < eta_hat < upper
    assert 2 * logit(eta_hat) - logit(lower) == pytest.approx(logit(0.55) + 0.3 * 0.95, abs=0.03)


def test_time_rescaling_is_uniform_under_the_true_model():
    rng = np.random.default_rng(15)
    baseline_truth = sim.FlatRate(1.2, T)
    raw = sim.simulate_cluster(baseline_truth.sample(rng), sim.ExponentialKernel(1.0), 0.6, T, rng)
    times, _ = em.merge_events(raw)
    baseline = sb.ConstantBaseline(T)
    problem = he.ExponentialHawkes(times, T, baseline)
    x = np.concatenate(([np.log(1.2)], problem.encode_kernel(np.array([0.6, 1.0]))))
    fit = he.FitResult(problem, x, np.nan, None, None, True, 0.0, 0)
    result = tr.test_fit(fit, rng=rng)
    assert result.pvalue > 0.001 and result.n == times.size


def test_multivariate_target_objective_gradient():
    rng = np.random.default_rng(16)
    times_list = [em.merge_events(np.sort(rng.uniform(0, 500.0, n)))[0] for n in (300, 200, 150)]
    pooled = mv._Pooled(times_list, 500.0, 0.001)
    beta = 1.3
    R, compensators = pooled.states(beta)
    for i in range(3):
        X = mv._prepare_target(pooled, R, i, np.arange(3))
        y = np.array([0.05, 0.3, 0.2, 0.1])
        penalty = np.array([0.5, 0.0, 0.2])
        value, gradient = mv._objective_target(y, X, compensators, beta, 500.0, 0.001, penalty)
        numeric = np.empty(4)
        for j in range(4):
            up = y.copy(); down = y.copy()
            up[j] += 1e-6; down[j] -= 1e-6
            numeric[j] = (mv._objective_target(up, X, compensators, beta, 500.0, 0.001, penalty)[0] - mv._objective_target(down, X, compensators, beta, 500.0, 0.001, penalty)[0]) / 2e-6
        np.testing.assert_allclose(gradient, numeric, rtol=1e-5, atol=1e-4)


def test_multivariate_truth_design_has_exact_spectral_radius_and_counts():
    counts = np.array([63.0, 410.0, 2050.0, 34833.0, 1000.0, 700.0, 3000.0, 150.0])
    mu, gamma, links, shares = mv.simulation_truth(counts, n_links=10, window=T)
    assert np.max(np.abs(np.linalg.eigvals(gamma))) == pytest.approx(0.6, abs=1e-9)
    np.testing.assert_allclose(np.linalg.solve(np.eye(8) - gamma, mu) * T, counts, rtol=1e-9)
    assert links.sum() == 10 and not links[np.diag_indices(8)].any()


def _tied_data(rng, n=500, horizon=T):
    times = np.sort(rng.uniform(0, horizon, n))
    return em.merge_events(np.concatenate((times, times[::9])))


@pytest.mark.parametrize("model", ["exponential", "sum_exp", "power_law"])
def test_print_weighted_likelihood_equals_tied_double_sum(model):
    rng = np.random.default_rng(21)
    times, prints = _tied_data(rng)
    assert prints.max() >= 2
    baseline = sb.ConstantBaseline(T)
    b = np.array([np.log(0.08)])
    d = 0.001
    if model == "exponential":
        eta, beta = 0.5, 1.4
        problem = he.ExponentialHawkes(times, T, baseline, marks=prints, print_weighted=True)
        x = np.concatenate((b, problem.encode_kernel(np.array([eta, beta]))))
        kernel = lambda lag: eta * beta * np.exp(-beta * lag)
        mass = lambda left: eta * (np.exp(-beta * d) - np.exp(-beta * left))
    elif model == "sum_exp":
        eta, w, b1, b2 = 0.5, 0.3, 8.0, 0.4
        problem = hs.SumExponentialsHawkes(times, T, baseline, marks=prints, print_weighted=True)
        x = np.concatenate((b, problem.encode_kernel(np.array([eta, w, b1, b2]))))
        kernel = lambda lag: eta * (w * b1 * np.exp(-b1 * lag) + (1 - w) * b2 * np.exp(-b2 * lag))
        mass = lambda left: eta * (w * (np.exp(-b1 * d) - np.exp(-b1 * left)) + (1 - w) * (np.exp(-b2 * d) - np.exp(-b2 * left)))
    else:
        eta, theta, c = 0.5, 1.5, 0.1
        problem = hp.PowerLawHawkes(times, T, baseline, marks=prints, print_weighted=True)
        x = np.concatenate((b, problem.encode_kernel(np.array([eta, theta, c]))))
        kernel = lambda lag: eta * theta * c ** theta * (lag + c) ** (-(1 + theta))
        mass = lambda left: eta * ((c / (d + c)) ** theta - (c / (left + c)) ** theta)
    excitation = np.array([np.sum(prints[:i] * kernel(times[i] - times[:i])) if i else 0.0 for i in range(times.size)])
    compensator = np.sum(prints * mass(np.maximum(T - times, d)))
    direct = -np.sum(prints * np.log(np.exp(b[0]) + excitation)) + np.exp(b[0]) * T + compensator
    assert problem.nll(x) == pytest.approx(direct, rel=1e-6)


@pytest.mark.parametrize("baseline_kind", ["constant", "spline"])
@pytest.mark.parametrize("model", ["exponential", "sum_exp", "power_law"])
def test_print_weighted_gradient_matches_finite_differences(model, baseline_kind):
    rng = np.random.default_rng(22)
    times, prints = _tied_data(rng, 400)
    baseline = sb.ConstantBaseline(T) if baseline_kind == "constant" else sb.SplineBaseline(T, RELEASE, n_events=400)
    b = baseline.initial(times, 0.05)
    cls, kernel = {"exponential": (he.ExponentialHawkes, [0.6, 1.3]), "sum_exp": (hs.SumExponentialsHawkes, [0.5, 0.4, 9.0, 0.2]), "power_law": (hp.PowerLawHawkes, [0.5, 1.6, 0.2])}[model]
    problem = cls(times, T, baseline, marks=prints, print_weighted=True)
    x = np.concatenate((b, problem.encode_kernel(np.array(kernel))))
    analytic = problem.nll_grad(x)[1]
    assert np.max(np.abs(analytic - _fd_gradient(problem, x)) / (1.0 + np.abs(analytic))) < 5e-5


def test_print_weighting_needs_prints_and_is_refused_by_the_marked_model():
    times = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        he.ExponentialHawkes(times, T, sb.ConstantBaseline(T), print_weighted=True)
    with pytest.raises(ValueError):
        hm.MarkedHawkes(times, T, sb.ConstantBaseline(T), marks=np.ones(3), print_weighted=True)


def test_print_weighted_rescaling_is_uniform_under_the_true_model():
    rng = np.random.default_rng(23)
    raw = sim.simulate_cluster(sim.FlatRate(1.2, T).sample(rng), sim.ExponentialKernel(1.0), 0.6, T, rng)
    times, prints = em.merge_events(raw)
    problem = he.ExponentialHawkes(times, T, sb.ConstantBaseline(T), marks=prints, print_weighted=True)
    x = np.concatenate(([np.log(1.2)], problem.encode_kernel(np.array([0.6, 1.0]))))
    result = tr.test_fit(he.FitResult(problem, x, np.nan, None, None, True, 0.0, 0), rng=rng)
    assert result.pvalue > 0.001


@pytest.mark.parametrize("cls,natural", [(he.ExponentialHawkes, [0.5, 1.0]), (hs.SumExponentialsHawkes, [0.5, 0.5, 10.0, 0.2]), (hp.PowerLawHawkes, [0.5, 1.5, 0.1])])
def test_warm_refit_reaches_the_same_optimum_as_a_fresh_fit(cls, natural):
    rng = np.random.default_rng(31)
    baseline = sb.ConstantBaseline(1500.0)
    start = cls(np.sort(rng.uniform(0, 1500.0, 1500)), 1500.0, baseline, resolution=None)
    x_true = np.concatenate(([np.log(0.5)], start.encode_kernel(np.array(natural))))
    times = em.merge_events(start.simulate(x_true, rng))[0]
    fitted = cls(times, 1500.0, baseline).fit()
    assert fitted.converged
    fresh = em.merge_events(fitted.problem.simulate(fitted.x, rng))[0]
    problem = fitted.problem.spawn(fresh)
    warm = problem.refit_warm(fitted.x, fitted.hessian)
    full = cls(fresh, 1500.0, baseline).fit()
    assert warm is not None
    assert problem.nll(warm) == pytest.approx(full.nll, abs=1e-3)
    assert problem.branching(warm) == pytest.approx(full.branching, abs=2e-3)


def test_multivariate_fit_runs_end_to_end_and_finds_a_strong_link():
    rng = np.random.default_rng(32)
    mu = np.array([0.4, 0.25, 0.3, 0.2])
    gamma = np.array([[0.3, 0.0, 0.0, 0.0], [0.5, 0.2, 0.0, 0.0], [0.0, 0.0, 0.3, 0.0], [0.0, 0.0, 0.0, 0.2]])
    raw = sim.simulate_multivariate(mu, gamma, 1.0, 3000.0, rng)
    fit = mv.fit_multivariate([em.merge_events(t)[0] for t in raw], 3000.0, seed_beta=1.0)
    assert fit.gamma.shape == (4, 4) and fit.support[1, 0]
    assert fit.gamma[1, 0] == pytest.approx(0.5, abs=0.15)
    lower, upper = fit.spectral_interval()
    assert lower < fit.spectral_radius < upper
