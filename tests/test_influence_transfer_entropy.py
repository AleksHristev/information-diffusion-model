"""Transfer entropy against the Gaussian closed form, and surrogate p-values."""

import numpy as np
import pytest

from information_diffusion.influence import transfer_entropy as te


def _pair(rng, a, b, c, steps, events=1):
    xs, ys = [], []
    for _ in range(events):
        x, y = np.zeros(steps + 50), np.zeros(steps + 50)
        eta, eps = rng.standard_normal(steps + 50), rng.standard_normal(steps + 50)
        for t in range(1, steps + 50):
            x[t] = b * x[t - 1] + eta[t]
            y[t] = a * y[t - 1] + c * x[t - 1] + eps[t]
        xs.append(x[50:])
        ys.append(y[50:])
    return xs, ys


def test_closed_form_matches_the_definition_from_stationary_covariance():
    a, b, c = 0.3, 0.2, 0.4
    vx = 1 / (1 - b**2)
    cxy = c * b * vx / (1 - a * b)
    vy = (1 + c**2 * vx + 2 * a * c * cxy) / (1 - a**2)
    conditional = vx - cxy**2 / vy
    assert te.gaussian_closed_form(a, b, c) == pytest.approx(0.5 * np.log(1 + c**2 * conditional))


def test_zero_coupling_has_zero_entropy():
    assert te.gaussian_closed_form(0.3, 0.2, 0.0) == 0.0


def test_estimator_recovers_the_closed_form_on_a_long_sample():
    rng = np.random.default_rng(1)
    a, b, c = 0.2, 0.2, 0.15
    xs, ys = _pair(rng, a, b, c, steps=40000, events=5)
    estimate = te.transfer_entropy(te.event_moments(xs, ys))
    assert estimate == pytest.approx(te.gaussian_closed_form(a, b, c), rel=0.1)


def test_event_weights_pool_as_a_bootstrap_would():
    rng = np.random.default_rng(2)
    xs, ys = _pair(rng, 0.2, 0.2, 0.1, steps=300, events=4)
    moments = te.event_moments(xs, ys)
    assert te.transfer_entropy(moments, weights=np.ones(4)) == pytest.approx(te.transfer_entropy(moments))


def test_uncoupled_pair_has_a_large_surrogate_pvalue_on_average():
    rng = np.random.default_rng(3)
    p = [te.surrogate_pvalue(*_pair(rng, 0.2, 0.2, 0.0, steps=400, events=4), 49, rng) for _ in range(8)]
    assert np.mean(p) > 0.2


def test_coupled_pair_is_rejected():
    rng = np.random.default_rng(4)
    xs, ys = _pair(rng, 0.2, 0.2, 0.3, steps=1000, events=4)
    assert te.surrogate_pvalue(xs, ys, 99, rng) <= 0.02
