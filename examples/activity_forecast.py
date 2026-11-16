"""Forecast trading activity in 30-second steps around a simulated release, and compare two forecasts out of sample.

Each event is a 90-minute window with a release after 30 minutes: trades follow a Hawkes process whose immigrant
rate spikes at the release and whose kernel decays over about 50 seconds. The benchmark forecast is the mean count
of each step over the fitting events. The second forecast is a Poisson regression of the count on the benchmark (as
an offset) and on the counts of the preceding steps, all stamped strictly before the step starts. Both are fitted on
30 events and evaluated on the next 21 by the Poisson deviance, and the two are compared with the Giacomini-White
test.
"""

import numpy as np

from information_diffusion.epidemic import event_merging
from information_diffusion.predictive import features, poisson_regression
from information_diffusion.simulation import hawkes
from information_diffusion.statistical_framework.giacomini_white import giacomini_white

rng = np.random.default_rng(11)
window, release, n_steps = 5400.0, 1800.0, 180
n_fit, n_test = 30, 21
predictors = ["lag_count_1", "lag_count_2", "lag_count_3", "lag_count_4", "roll_count_10"]

frames = []
for _ in range(n_fit + n_test):
    immigrants = hawkes.SpikingRate(0.05, window, release).sample(rng)
    times = event_merging.merge_events(hawkes.simulate_cluster(immigrants, hawkes.ExponentialKernel(0.02), 0.6, window, rng))[0]
    frames.append(features.build_features(times, n_steps, release_time=release))

fit_counts = np.array([f["y_count"].to_numpy() for f in frames[:n_fit]])
profile = np.maximum(fit_counts.mean(axis=0), 1.0 / n_fit)


def design(frame) -> np.ndarray:
    """Intercept and log(1 + x) of each predictor."""
    return np.column_stack([np.ones(len(frame)), np.log1p(frame[predictors].to_numpy())])


X = np.vstack([design(f) for f in frames[:n_fit]])
y = np.concatenate([f["y_count"].to_numpy() for f in frames[:n_fit]])
offset = np.tile(np.log(profile), n_fit)
model = poisson_regression.fit(y, X, offset)


def mean_deviance(counts: np.ndarray, rate: np.ndarray) -> float:
    """Mean Poisson deviance per step of one event."""
    return poisson_regression.deviance(counts, rate) / counts.size


benchmark_loss, model_loss = [], []
for frame in frames[n_fit:]:
    counts = frame["y_count"].to_numpy()
    benchmark_loss.append(mean_deviance(counts, profile))
    model_loss.append(mean_deviance(counts, model.predict(design(frame), np.log(profile))))
benchmark_loss, model_loss = np.array(benchmark_loss), np.array(model_loss)
differentials = benchmark_loss - model_loss
test = giacomini_white(differentials)

ratios = []
for _ in range(9999):
    pick = rng.integers(0, n_test, n_test)
    ratios.append(1.0 - model_loss[pick].mean() / benchmark_loss[pick].mean())
lower, upper = np.percentile(ratios, [2.5, 97.5])

print(f"events: {n_fit} to fit, {n_test} to evaluate; {n_steps} steps of 30 s each")
print("Poisson regression, coefficients and 95% Wald intervals:")
low, high = model.interval()
for name, value, a, b in zip(["intercept"] + [f"log(1 + {p})" for p in predictors], model.coefficients, low, high):
    print(f"  {name:<24} {value:+.3f}   [{a:+.3f}, {b:+.3f}]")
print(f"mean deviance per step, benchmark        {benchmark_loss.mean():.4f}")
print(f"mean deviance per step, regression       {model_loss.mean():.4f}")
print(f"reduction in deviance                    {1.0 - model_loss.mean() / benchmark_loss.mean():.1%}   95% bootstrap interval over events [{lower:.1%}, {upper:.1%}]")
print(f"Giacomini-White t = {test.statistic:.2f}, p = {test.p_value:.4f} ({test.n_events} events, {test.null} null)")
