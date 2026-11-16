# information-diffusion-model

This is the code of the information diffusion model in the paper "Information Diffusion around US Macroeconomic
Releases: Epidemic, Influence and Predictive Models, with an Application to Market Making" (Hristev, 2026). The model
describes how the news in a scheduled macroeconomic release spreads through a set of traded instruments in the
seconds and minutes after it is published.

The model has three parts and one application:

- The epidemic part measures how fast the news is absorbed. Hawkes processes describe the clustering of trades in
  each instrument. Compartment models (SI, SIR, SIS, SIRS, SEIR) and the Bass model describe the share of instruments
  that have absorbed the news over time.
- The influence part measures which instruments lead. It has lead-lag estimators for asynchronous prices, sparse
  vector autoregressions, connectedness, transfer entropy, price-discovery shares, network communities and sensor
  placement.
- The predictive part forecasts the reaction. It forecasts trading activity per time step from features that use only
  earlier data, and it has cascade models (Independent Cascade, Linear Threshold and discrete-time hazards) for which
  instruments react next.
- The market-making backtest applies the model. It is an event-driven engine with two fill models and a latency grid
  that evaluates quoting rules around releases.

The estimators are written directly in NumPy, SciPy and Numba, so each step can be read and checked. The package
also has the statistical tests used to compare models (Diebold-Mariano, Giacomini-White, Model Confidence Set,
Hansen's test of superior predictive ability, Holm and Benjamini-Yekutieli adjustments, HAC and cluster-robust
variances, block bootstraps), and simulators that generate data with known parameters, so that you can check that
an estimator recovers them.

## Install

Python 3.12 or later is required.

```
pip install git+https://github.com/AleksHristev/information-diffusion-model
```

The dependencies are NumPy, pandas, SciPy and Numba. To run the tests, clone the repository and install the test
extra:

```
pip install ".[test]"
pytest
```

## Quick start

Simulate trades from an exponential Hawkes process with a known branching ratio and fit it back:

```python
import numpy as np

from information_diffusion.epidemic import event_merging
from information_diffusion.epidemic.branching_ratio import bootstrap_branching
from information_diffusion.epidemic.hawkes_exponential import ExponentialHawkes
from information_diffusion.epidemic.spline_baseline import ConstantBaseline
from information_diffusion.simulation import hawkes

rng = np.random.default_rng(1)
window = 3600.0
immigrants = hawkes.FlatRate(0.5, window).sample(rng)
times, _ = event_merging.merge_events(hawkes.simulate_cluster(immigrants, hawkes.ExponentialKernel(1.0), 0.6, window, rng))

fit = ExponentialHawkes(times, window, ConstantBaseline(window)).fit()
interval = bootstrap_branching(fit, 99, rng)
print(fit.branching, interval.lower, interval.upper)
```

The `examples/` folder has four short scripts that run on simulated data and print their results. Each takes under a
minute; most of that is Numba compiling on first use, and later runs take a few seconds:

| Script | What it shows |
| --- | --- |
| `hawkes_and_absorption.py` | A Hawkes fit with a bootstrap interval for the branching ratio; absorption times measured from simulated quotes, absorption curves and an SI fit. |
| `lead_lag.py` | The Hoffmann-Rosenbaum-Yoshida lead-lag estimate between two asynchronously observed price series, with a bootstrap interval, and the Hayashi-Yoshida correlation. |
| `activity_forecast.py` | A forecast of trade counts per 30-second step around a release, compared out of sample with a benchmark by the Giacomini-White test. |
| `market_making_backtest.py` | Quoting rule 2 (Avellaneda-Stoikov) against the pause rule on simulated order flow with a price jump at each release, under both fill models. |

Units: times are in seconds unless a module says it uses integer nanoseconds. Prices are in dollars unless a function
says it takes log prices, probabilities or log-odds. The backtest works in integer micro-dollars. Every public function
and class has a docstring with its arguments and what it returns, and each estimator gives its formula and
reference.

## Modules

The package is imported as `information_diffusion` and has six subpackages.

### `epidemic`

| Module | Contents |
| --- | --- |
| `absorption` | Absorption of news across instruments: absorption times and curves, half-life, R0 interval and the absorption clock. |
| `bass` | Bass diffusion model: specification, closed-form adoption curve and fit to absorption curves. |
| `best_descent` | Exponential Hawkes fit with the lowest penalised objective over several descents. |
| `branching_ratio` | Hawkes branching ratio with a parametric bootstrap interval. |
| `compartment_common` | Shared machinery for the compartment and Bass models: Markov jump chain, mean-field ODE, curve fitting and bootstrap intervals. |
| `compartment_seir` | SEIR compartment model (SIR with a latent exposed stage). |
| `compartment_si` | SI (susceptible-infected) compartment model, with its logistic mean-field curve. |
| `compartment_sir` | SIR (susceptible-infected-removed) compartment model. |
| `compartment_sirs` | SIRS compartment model (SIR with waning removal back to susceptible). |
| `compartment_sis` | SIS (susceptible-infected-susceptible) compartment model. |
| `event_merging` | Merging of prints into one marked event per time cell (one millisecond by default). |
| `hawkes_coarse_clock` | Exponential Hawkes models on an event clock of bin width d: left-edge and exact bin-mass likelihoods. |
| `hawkes_exponential` | Univariate Hawkes process with an exponential kernel, and the shared penalised likelihood engine for Hawkes fits. |
| `hawkes_marked` | Marked Hawkes process with an exponential kernel, the mark entering the excitation as m^gamma. |
| `hawkes_multivariate` | Multivariate Hawkes process with a shared exponential decay: weighted L1 link selection, refit on the support and intervals. |
| `hawkes_pooled` | Exponential Hawkes model pooled across instruments and events with instrument fixed effects, and its forecast of counts per step. |
| `hawkes_power_law` | Univariate Hawkes process with a power-law kernel, approximated by a fixed sum of exponentials. |
| `hawkes_sum_exponentials` | Univariate Hawkes process with a sum-of-exponentials kernel. |
| `rescaled_diagnostics` | Goodness-of-fit summaries of rescaled intervals: Ljung-Box test and Q-Q summaries. |
| `spline_baseline` | Hawkes baseline rates: constant, or a log-linear cubic B-spline with a jump at the release time, with samplers. |
| `time_rescaling` | Goodness of fit of Hawkes models by the time-rescaling theorem, in continuous or discrete (merged-event) time. |
| `window_baselines` | Spline baselines for windows with no release jump or with two shocks, and the baseline for each window shape. |

### `influence`

| Module | Contents |
| --- | --- |
| `adf` | Augmented Dickey-Fuller test on a series observed through a keep mask, with MacKinnon and simulated critical values. |
| `community_detection` | Network communities by Louvain modularity optimisation, with modularity and the adjusted Rand index. |
| `diebold_yilmaz` | Diebold-Yilmaz connectedness by generalised forecast-error variance decomposition of a VAR. |
| `grid_cross_correlation` | Lead-lag estimation by cross-products of previous-tick returns on a 1-second grid. |
| `hayashi_yoshida` | Hayashi-Yoshida covariance of two asynchronously observed series. |
| `hoffmann_rosenbaum_yoshida` | Hoffmann-Rosenbaum-Yoshida lead-lag estimator on asynchronous observations, pooled across events. |
| `lagged_lookup` | Lagged leader returns and follower grid returns by exact previous-tick lookups in integer nanoseconds. |
| `link_fdr` | Benjamini-Yekutieli false-discovery control over the ordered links of a network. |
| `post_decision_lag` | Minute-scale lead-lag between two prediction-market venues after a decision, by the centroid of pooled cross-covariances. |
| `price_discovery` | Price-discovery shares of two venues from a VECM with cointegrating vector (1, -1): component, information and leadership shares. |
| `sensor_placement` | Sensor placement on a diffusion network: the detection objective, greedy placement and the exhaustive optimum. |
| `sparse_var` | L1-penalised (Lasso) VAR from per-event moments, with EBIC penalty and lag selection and a post-Lasso refit. |
| `stability_selection` | Stability selection of sparse VAR links with whole-event resampling. |
| `transfer_entropy` | Linear-Gaussian transfer entropy pooled over events, with circular-shift surrogate p-values. |
| `two_scale_variance` | Two-scale realised variance, a noise-robust variance estimator for tick data. |
| `vecm` | Two-venue vector error-correction model with the cointegrating vector (1, -1) known, with lag selection and a moving-block bootstrap. |

### `predictive`

| Module | Contents |
| --- | --- |
| `autoregressive_poisson` | Autoregressive Poisson count model on lagged counts around a spline baseline. |
| `cascade_comparison` | Rolling out-of-sample comparison of the cascade models M0 to M4 by mean Bernoulli log-loss per event. |
| `cascade_data` | Cascades built from absorption times, and the at-risk rows and design matrices of the cascade models M0 to M4. |
| `cascade_hazard` | Discrete-time logistic hazard of the cascade models, fitted by penalised likelihood on binomial cells. |
| `cascade_influence` | Expected spread and greedy influence maximisation under an Independent Cascade, by live-edge simulation. |
| `features` | Step features built only from data stamped before each step starts, free of look-ahead. |
| `independent_cascade` | Independent Cascade model fitted by maximum likelihood with EM, with and without a background hazard. |
| `linear_threshold` | Linear Threshold model with logistic noise and fixed node thresholds, fitted by maximum likelihood. |
| `poisson_regression` | Poisson regression with a log link, fitted by Newton-Raphson. |
| `purged_cv` | Purged and embargoed cross-validation folds grouped by event. |
| `realised_volatility_regression` | Heterogeneous autoregression of log realised variance over intraday steps, with instrument fixed effects. |
| `spline_poisson` | Inhomogeneous Poisson count model whose rate is a penalised B-spline baseline integrated over each step. |

### `simulation`

| Module | Contents |
| --- | --- |
| `cascades` | Simulators of Independent Cascade and Linear Threshold diffusion on a directed network, in discrete steps. |
| `compartment` | Exact stochastic simulation of compartment and Bass models, and one simulate-and-fit replicate for parameter recovery. |
| `hawkes` | Hawkes process simulators by the cluster representation, delay kernels, baseline rates, marks and expected counts. |
| `lead_lag` | Lead-lag network simulator: asynchronously observed leader and follower prices on a variance clock, and VAR paths. |
| `monte_carlo` | Reproducible, low-priority parallel Monte Carlo with one random stream per replicate. |
| `order_flow` | Synthetic order flow for verifying the market-making engine: hand-derived scenarios and an independent reference replay. |
| `price_discovery` | Simulators for two-venue price discovery: a VECM with known shares and minute-scale prices after a decision with venue delays. |

### `statistical_framework`

| Module | Contents |
| --- | --- |
| `benjamini_yekutieli` | Benjamini-Yekutieli false discovery rate control under arbitrary dependence. |
| `block_bootstrap` | Resampling index generators for the moving-block, stationary and event bootstraps. |
| `cluster_robust` | Cluster-robust (CR1) standard errors for OLS coefficients. |
| `diebold_mariano` | Diebold-Mariano test of equal forecast accuracy, with the Harvey-Leybourne-Newbold correction. |
| `freedman_lane` | Freedman-Lane permutation test of a regression coefficient with nuisance covariates. |
| `giacomini_white` | Giacomini-White test of equal predictive ability on event-level loss differentials from rolling windows. |
| `giacomini_white_critical_values` | Stored simulated null critical values of the Giacomini-White statistic at fixed tail probabilities. |
| `giacomini_white_diagnostic` | Diagnostic of autocorrelation and tail weight of loss differentials, with a parametric Giacomini-White p-value. |
| `hac` | Newey-West HAC variance with a Bartlett kernel, for a mean and for OLS coefficients. |
| `hc3` | Ordinary least squares with HC3 heteroscedasticity-consistent standard errors and linear contrasts. |
| `holm` | Holm's step-down adjustment of p-values for family-wise error control. |
| `holm_family` | Holm decisions in a family of directional tests, under the standard rule and the sign-first rule. |
| `model_confidence_set` | Model Confidence Set of Hansen, Lunde and Nason, by block bootstrap. |
| `quantile_standard_error` | Maritz-Jarrett standard error of a sample quantile. |
| `spa` | Hansen's test of superior predictive ability, with White's Reality Check, by block bootstrap. |

### `market_making`

| Module | Contents |
| --- | --- |
| `engine` | Event-driven market-making backtest for one quoting rule, one fill model and one latency. |
| `fill_optimistic` | Optimistic price-touch fill model, an upper bound on maker fills that ignores queue position. |
| `fill_pessimistic` | Pessimistic queue-position fill model behind the displayed NBBO size, which is taken to be single-venue size. |
| `grid` | Grid of backtest runs over quoting rules, fill models and latencies for one window. |
| `market` | NBBO and trade arrays normalised to integer micro-dollars and nanoseconds for the backtest. |
| `markouts` | Mark-outs of fills: the signed NBBO mid move after each fill, in ticks and basis points. |
| `params` | Backtest parameters: price scale, latency grid, mark-out horizons, fees and sensitivity ranges. |
| `quote_terms` | Common quote terms of every quoting rule, rounding to the tick grid and as-of lookup of input series. |
| `results` | Output of one backtest run: the fill ledger, mark-outs, exact integer accounting and a summary in ticks, basis points and shares. |
| `rule1_fixed_spread` | Quoting rule 1: a fixed half-spread around the NBBO mid, with no inventory skew. |
| `rule2_avellaneda_stoikov` | Quoting rule 2: Avellaneda-Stoikov reservation price and spread with constant arrival intensity. |
| `rule3_hawkes_intensity` | Quoting rule 3: the Avellaneda-Stoikov half-spread widened by a multiplier driven by the Hawkes intensity. |
| `rule4_network_fair_value` | Quoting rule 4: Avellaneda-Stoikov quotes centred on a network fair value in place of the NBBO mid. |
| `rule5_hawkes_network` | Quoting rule 5: network fair value centre and Hawkes intensity spread multiplier combined. |
| `rule6_scheduled_avellaneda_stoikov` | Quoting rule 6: constant-intensity Avellaneda-Stoikov quotes with the spread widened on a fixed release schedule. |
| `phase_rules` | Quoting rules 7 to 11: rule 2 quotes with the half-spread scheduled, widened or pulled around release times. |
| `pause_rule` | Rule 2 quotes pulled for a fixed interval after each anchor time, compared in integer nanoseconds. |
| `withdraw_rule` | Rule 2 quotes pulled for a window of any length around each scheduled release. |
| `wide_quote_rule` | Rule 2 quotes widened by a multiplier after each anchor time, then optionally pulled until the span ends. |
| `one_sided_rule` | Rule 2 quotes restricted to one side in given intervals according to a direction series, with optional pauses. |
| `phases` | Classification of prints against the NBBO and the passive edge earned by resting orders around releases. |
| `tape_measures` | Nanosecond-resolution measures of the tape after a release: quote values, move shares, directions, edges, signals and gaps. |
| `reaction_measures` | Per-release reaction measures: print cells, time to half of the move, move size, peak spread, update gap and first tick move. |
| `release_detector` | Real-time release detector: trailing-range exits of each instrument and k-of-n coincidences across instruments. |
| `resting_side` | Direction of a price move, the side of resting orders relative to it, and inventory statistics over an interval. |
| `rank_statistics` | Rank statistics: ranks with missing values, Kendall's W, Spearman's rho, Kendall's tau-b, sign test and weighted median. |
| `order_statistics` | Agreement and out-of-sample predictability of the order in which a set of instruments reacts across releases. |

The module `compiled` is the shared Numba entry point for the compiled loops.

## Data

The repository has no market data, and it does not have the research code that downloads the data and produces the
results of the paper. The package runs on any data in the formats its functions document. The data used in the paper come from the
sources named in the paper. The tests and examples use simulated data only.

## Tests

The tests run on simulated or hand-made inputs. They check each estimator against hand-computed cases or direct
definitions, and check that it recovers known parameters from data simulated by the `simulation` subpackage. The
backtest engine is checked against hand-derived scenarios and against an independent reference replay of the same
order flow. A smoke test runs every script in `examples/`.

## Citation

Aleksandar Hristev (2026), "Information Diffusion around US Macroeconomic Releases: Epidemic, Influence and
Predictive Models, with an Application to Market Making".

```
@misc{hristev2026information,
  author = {Hristev, Aleksandar},
  title  = {Information Diffusion around {US} Macroeconomic Releases: Epidemic, Influence and Predictive Models, with an Application to Market Making},
  year   = {2026}
}
```

## Licence

MIT. See `LICENSE`.
